# transformer_temp_train_stable.py
# ------------------------------------------------------------
# Stable Transformer training for transient thermal prediction
# Fixes "loss explosion" by:
#   1) gradient clipping
#   2) LR scheduler
#   3) best-checkpoint saving + early stopping
#   4) safe metric computation on REAL scale
# ------------------------------------------------------------

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Fixed Min-Max ranges (physical)
# ============================================================
T_MIN, T_MAX = 80.89, 127.9973
P_MIN, P_MAX = 0.1001, 0.6
HTC_MIN, HTC_MAX = 200.0, 1300.0


def minmax_norm(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def minmax_denorm(x, xmin, xmax):
    return x * (xmax - xmin) + xmin


# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # optional deterministic (slower)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# Dataset (seq_len = 5)
# each CSV: (20, 109) = 54(T) + 54(P) + 1(HTC)
# input x: (seq_len, 109) from time [t-seq+1 ... t]
# target y: T at time (t+1) => (54,)
# ============================================================
class TransformerTempDataset(Dataset):
    def __init__(self, folder="lstm_inputs", seq_len=5):
        self.samples = []
        self.seq_len = int(seq_len)

        files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if len(files) == 0:
            raise RuntimeError(f'No csv found in folder="{folder}"')

        for file in files:
            data = pd.read_csv(file, header=None).values  # (20, 109)

            if data.shape[0] < 20 or data.shape[1] < 109:
                raise RuntimeError(f"Bad shape in {file}: {data.shape}, expect (20,109)")

            T = data[:, :54]
            P = data[:, 54:108]
            H = data[:, 108:109]  # (20,1)

            # normalize
            Tn = minmax_norm(T, T_MIN, T_MAX)
            Pn = minmax_norm(P, P_MIN, P_MAX)
            Hn = minmax_norm(H, HTC_MIN, HTC_MAX)

            fname = os.path.splitext(os.path.basename(file))[0]

            # predict t+1, so t can be up to 18
            # input window ends at t, target is t+1
            for t in range(self.seq_len - 1, 19):
                seq = []
                for k in range(t - self.seq_len + 1, t + 1):
                    seq.append(np.concatenate([Tn[k], Pn[k], Hn[k]], axis=0))  # (109,)
                x = np.stack(seq, axis=0)  # (seq_len, 109)
                y = Tn[t + 1]             # (54,)

                self.samples.append({
                    "x": torch.tensor(x, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.float32),
                    "file": fname,
                    "time": t + 1
                })

        print(f"Loaded {len(self.samples)} samples (seq_len={self.seq_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["x"], s["y"], s["file"], s["time"]


# ============================================================
# Transformer Model
# ============================================================
class TempTransformer(nn.Module):
    def __init__(self, input_dim=109, d_model=256, nhead=8, num_layers=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # helps stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, 54)

    def forward(self, x):
        # x: (B, seq_len, 109)
        x = self.input_proj(x)        # (B, seq_len, d_model)
        h = self.encoder(x)           # (B, seq_len, d_model)
        h_last = h[:, -1, :]          # (B, d_model)
        out = self.fc(h_last)         # (B, 54)
        return out


# ============================================================
# Metrics (REAL scale)
# ============================================================
@torch.no_grad()
def eval_real(model, loader):
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0

    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)

        # denorm to real temperature
        y_real = minmax_denorm(y, T_MIN, T_MAX)
        pred_real = minmax_denorm(pred, T_MIN, T_MAX)

        diff = pred_real - y_real
        mse_sum += torch.mean(diff ** 2).item()
        mae_sum += torch.mean(torch.abs(diff)).item()
        n += 1

    return mse_sum / max(n, 1), mae_sum / max(n, 1)


# ============================================================
# Train (Stable)
# ============================================================
def train(
    data_folder="lstm_inputs",
    seq_len=5,
    batch_size=128,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-2,
    grad_clip=1.0,
    patience=15,
    seed=42,
    out_dir="test_outputs_transformer"
):
    seed_everything(seed)
    os.makedirs(out_dir, exist_ok=True)

    dataset = TransformerTempDataset(data_folder, seq_len=seq_len)

    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    train_eval_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_eval_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_vis_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = TempTransformer().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)  # often more stable for transformers
    )

    # Smooth LR decay (more stable than constant LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    criterion = nn.MSELoss()

    # logging
    hist = {
        "epoch": [],
        "train_mse_norm": [],
        "train_mse_real": [],
        "train_mae_real": [],
        "test_mse_real": [],
        "test_mae_real": [],
        "lr": [],
    }

    best_test_mae = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_ckpt_path = os.path.join(out_dir, "best_transformer_temp_model.pth")

    print("\n==== Start Training ====\n")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0

        for x, y, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # ---- KEY FIX: gradient clipping prevents explosion ----
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            loss_sum += loss.item()

        scheduler.step()
        train_mse_norm = loss_sum / max(len(train_loader), 1)

        # eval (every epoch)
        train_mse_real, train_mae_real = eval_real(model, train_eval_loader)
        test_mse_real, test_mae_real = eval_real(model, test_eval_loader)

        cur_lr = optimizer.param_groups[0]["lr"]

        hist["epoch"].append(epoch)
        hist["train_mse_norm"].append(train_mse_norm)
        hist["train_mse_real"].append(train_mse_real)
        hist["train_mae_real"].append(train_mae_real)
        hist["test_mse_real"].append(test_mse_real)
        hist["test_mae_real"].append(test_mae_real)
        hist["lr"].append(cur_lr)

        print(
            f"Epoch {epoch:03d} | "
            f"LR={cur_lr:.2e} | "
            f"Train MSE(norm)={train_mse_norm:.6e} | "
            f"Train MSE(real)={train_mse_real:.6f}, MAE(real)={train_mae_real:.6f} | "
            f"Test MSE(real)={test_mse_real:.6f}, MAE(real)={test_mae_real:.6f}"
        )

        # ---- best checkpoint by test MAE ----
        if test_mae_real < best_test_mae:
            best_test_mae = test_mae_real
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_test_mae": best_test_mae,
                    "config": {
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "grad_clip": grad_clip,
                        "patience": patience,
                        "seed": seed,
                    },
                },
                best_ckpt_path
            )
        else:
            bad_epochs += 1

        # ---- early stopping ----
        if bad_epochs >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs.")
            break

    print(f"\nBest epoch = {best_epoch}, Best Test MAE(real) = {best_test_mae:.6f}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")

    # ========================================================
    # Save history CSV + curves
    # ========================================================
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(os.path.join(out_dir, "train_history.csv"), index=False)

    # curve: MAE(real)
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_mae_real"], label="Train MAE(real)")
    plt.plot(hist_df["epoch"], hist_df["test_mae_real"], label="Test  MAE(real)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mae_real_curve.png"), dpi=300)
    plt.close()

    # curve: MSE(real)
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_mse_real"], label="Train MSE(real)")
    plt.plot(hist_df["epoch"], hist_df["test_mse_real"], label="Test  MSE(real)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (°C^2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mse_real_curve.png"), dpi=300)
    plt.close()

    # curve: Train MSE(norm)
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_mse_norm"], label="Train MSE(norm)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mse_norm_curve.png"), dpi=300)
    plt.close()

    # ========================================================
    # Load best model for final export
    # ========================================================
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # final metrics on best model
    train_mse_real, train_mae_real = eval_real(model, train_eval_loader)
    test_mse_real, test_mae_real = eval_real(model, test_eval_loader)

    pd.DataFrame([{
        "Best_Epoch": best_epoch,
        "Train_MSE_real": train_mse_real,
        "Train_MAE_real": train_mae_real,
        "Test_MSE_real": test_mse_real,
        "Test_MAE_real": test_mae_real
    }]).to_csv(os.path.join(out_dir, "train_test_metrics_best.csv"), index=False)

    # ========================================================
    # Save test PNG & CSV (best model)
    # ========================================================
    with torch.no_grad():
        for x, y, fname, t_idx in test_vis_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)

            y_real = minmax_denorm(y, T_MIN, T_MAX).cpu().numpy().reshape(6, 9)
            pred_real = minmax_denorm(pred, T_MIN, T_MAX).cpu().numpy().reshape(6, 9)
            abs_err = np.abs(pred_real - y_real)

            # CSV: [true | pred | abs_err] => 6 x 27
            combined = np.hstack([y_real, pred_real, abs_err])
            csv_path = os.path.join(out_dir, f"{fname}_t{int(t_idx)}.csv")
            pd.DataFrame(combined).to_csv(csv_path, index=False, header=False)

            # PNG
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            sns.heatmap(y_real, ax=axs[0], cmap="viridis")
            sns.heatmap(pred_real, ax=axs[1], cmap="viridis")
            sns.heatmap(abs_err, ax=axs[2], cmap="hot")
            axs[0].set_title("True (°C)")
            axs[1].set_title("Pred (°C)")
            axs[2].set_title("|Error| (°C)")
            plt.tight_layout()
            png_path = os.path.join(out_dir, f"{fname}_t{int(t_idx)}.png")
            plt.savefig(png_path, dpi=300)
            plt.close()

    print("\ Transformer results saved to:", out_dir)
    print(" History: train_history.csv, mae/mse curves png")
    print(" Best metrics: train_test_metrics_best.csv")
    print(" Best model checkpoint:", best_ckpt_path)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train(
        data_folder="lstm_inputs",
        seq_len=5,
        batch_size=128,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-2,
        grad_clip=1.0,     # <<< key: prevents explosion
        patience=15,       # early stopping
        seed=42,
        out_dir="test_outputs_transformer"
    )
