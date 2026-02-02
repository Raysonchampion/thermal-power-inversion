import os
import glob
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
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Dataset
# ============================================================
class LSTMTempDataset(Dataset):
    def __init__(self, folder="lstm_inputs"):
        self.samples = []

        files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if len(files) == 0:
            raise RuntimeError(" lstm_inputs  CSV ")

        for file in files:
            data = pd.read_csv(file, header=None).values  # (20, 109)

            T = data[:, :54]
            P = data[:, 54:108]
            H = data[:, 108:109]

            Tn = minmax_norm(T, T_MIN, T_MAX)
            Pn = minmax_norm(P, P_MIN, P_MAX)
            Hn = minmax_norm(H, HTC_MIN, HTC_MAX)

            fname = os.path.splitext(os.path.basename(file))[0]

            for t in range(19):
                x = np.concatenate([Tn[t], Pn[t], Hn[t]], axis=0)
                y = Tn[t + 1]

                self.samples.append({
                    "x": torch.tensor(x, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.float32),
                    "file": fname,
                    "time": t + 1
                })

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["x"], s["y"], s["file"], s["time"]

# ============================================================
# LSTM Model
# ============================================================
class TempLSTM(nn.Module):
    def __init__(self, input_dim=109, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 54)

    def forward(self, x):
        x = x.unsqueeze(1)              # (B,1,109)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # (B,54)

# ============================================================
# Train + Evaluate + Save
# ============================================================
def train():
    dataset = LSTMTempDataset("lstm_inputs")

    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=1,   shuffle=False)

  
    train_eval_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    test_eval_loader  = DataLoader(test_set,  batch_size=1, shuffle=False)

    model = TempLSTM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ========================================================
    # helper: eval real MSE / MAE
    # ========================================================
    def eval_real_metrics(loader):
        mse_sum, mae_sum, n = 0.0, 0.0, 0
        model.eval()

        with torch.no_grad():
            for x, y, _, _ in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)

                y_real = minmax_denorm(y, T_MIN, T_MAX)
                pred_real = minmax_denorm(pred, T_MIN, T_MAX)

                diff = pred_real - y_real
                mse_sum += torch.mean(diff ** 2).item()
                mae_sum += torch.mean(torch.abs(diff)).item()
                n += 1

        return mse_sum / n, mae_sum / n

    # ========================================================
    # Training loop
    # ========================================================
    for epoch in range(1, 1001):
        model.train()
        loss_sum = 0.0

        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        train_mse_norm = loss_sum / len(train_loader)

        # ---- evaluate real metrics ----
        train_mse_real, train_mae_real = eval_real_metrics(train_eval_loader)
        test_mse_real,  test_mae_real  = eval_real_metrics(test_eval_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE(norm)={train_mse_norm:.6e} | "
            f"Train MSE(real)={train_mse_real:.6f}, MAE(real)={train_mae_real:.6f} | "
            f"Test MSE(real)={test_mse_real:.6f}, MAE(real)={test_mae_real:.6f}"
        )

    # ========================================================
    # Save model
    # ========================================================
    torch.save(model.state_dict(), "lstm_temp_model_norm3.pth")

    # ========================================================
    # Save metrics CSV (last epoch)
    # ========================================================
    os.makedirs("test_outputs3", exist_ok=True)
    pd.DataFrame([{
        "Train_MSE_real": train_mse_real,
        "Train_MAE_real": train_mae_real,
        "Test_MSE_real":  test_mse_real,
        "Test_MAE_real":  test_mae_real
    }]).to_csv("./train_test_metrics1.csv", index=False)

    # ========================================================
    # Save test visualizations & CSV
    # ========================================================
    model.eval()
    with torch.no_grad():
        for x, y, fname, t_idx in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            y_real = minmax_denorm(y, T_MIN, T_MAX).cpu().numpy().reshape(6, 9)
            pred_real = minmax_denorm(pred, T_MIN, T_MAX).cpu().numpy().reshape(6, 9)
            abs_err = np.abs(pred_real - y_real)

            # CSV: True | Pred | AbsErr
            combined = np.hstack([y_real, pred_real, abs_err])
            pd.DataFrame(combined).to_csv(
                f"test_outputs3/{fname}_t{int(t_idx)}.csv",
                index=False,
                header=False
            )

            # Heatmap
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            sns.heatmap(y_real, ax=axs[0], cmap="viridis")
            sns.heatmap(pred_real, ax=axs[1], cmap="viridis")
            sns.heatmap(abs_err, ax=axs[2], cmap="hot")

            axs[0].set_title("True (°C)")
            axs[1].set_title("Pred (°C)")
            axs[2].set_title("|Error| (°C)")

            plt.tight_layout()
            plt.savefig(f"test_outputs3/{fname}_t{int(t_idx)}.png", dpi=300)
            plt.close()

    print(" All results saved in test_outputs3/")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train()
