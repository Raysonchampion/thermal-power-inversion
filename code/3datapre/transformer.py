import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# 0. CONFIG
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

T_ROOT = "/home/yslu/1121/T1126_split/i7"
P_ROOT = "/home/yslu/1121/powerzuizhong/i7"

OUT_DIR = "./results_transformer_fulltransfoemer"
TEST_OUT = os.path.join(OUT_DIR, "test_results")
os.makedirs(TEST_OUT, exist_ok=True)

HTC_VALUE = 600.0
HTC_MIN, HTC_MAX = 200.0, 1300.0

EPOCHS = 300
BATCH_SIZE = 4
LR = 5e-5
EPS = 1e-8

# ============================================================
# 1. BUILD TRAIN / TEST SAMPLE LIST (8:2 per workload)
# ============================================================

thermal_folders = sorted(
    glob.glob(os.path.join(T_ROOT, "CPU_i7_*_thermal_maps"))
)

print("Detected thermal workloads:",
      [os.path.basename(f) for f in thermal_folders])

train_samples, test_samples = [], []

for tf in thermal_folders:
    wl = os.path.basename(tf).replace("_thermal_maps", "")
    T_dir = os.path.join(tf, "csv")
    P_dir = os.path.join(P_ROOT, wl + "_power_maps")

    if not os.path.isdir(T_dir) or not os.path.isdir(P_dir):
        print(f"[Skip] {wl} missing T or P")
        continue

    t_files = sorted(glob.glob(os.path.join(T_dir, "*.csv")))
    if len(t_files) < 2:
        print(f"[Skip] {wl} not enough frames")
        continue

    indices = sorted([
        int(os.path.basename(f).split("_")[-1].split(".")[0])
        for f in t_files
    ])
    indices = [i for i in indices if i + 1 in indices]

    random.shuffle(indices)
    split = int(0.8 * len(indices))

    for i in indices[:split]:
        train_samples.append((wl, i))
    for i in indices[split:]:
        test_samples.append((wl, i))

    print(f"{wl}: train={split}, test={len(indices)-split}")

print(f"TOTAL train={len(train_samples)}, test={len(test_samples)}")
assert len(train_samples) > 0 and len(test_samples) > 0

# ============================================================
# 2. COMPUTE MIN / MAX (TRAIN ONLY)
# ============================================================

def compute_minmax(samples):
    T_min, T_max = 1e9, -1e9
    P_min, P_max = 1e9, -1e9

    for wl, t in samples:
        T = np.loadtxt(
            f"{T_ROOT}/{wl}_thermal_maps/csv/{wl}_thermal_maps_{t}.csv",
            delimiter=","
        )
        Tn = np.loadtxt(
            f"{T_ROOT}/{wl}_thermal_maps/csv/{wl}_thermal_maps_{t+1}.csv",
            delimiter=","
        )
        P = np.loadtxt(
            f"{P_ROOT}/{wl}_power_maps/{wl}_power_maps_{t}.csv",
            delimiter=","
        )

        T_min = min(T_min, T.min(), Tn.min())
        T_max = max(T_max, T.max(), Tn.max())
        P_min = min(P_min, P.min())
        P_max = max(P_max, P.max())

    return T_min, T_max, P_min, P_max

T_MIN, T_MAX, P_MIN, P_MAX = compute_minmax(train_samples)
print(f"T range: {T_MIN:.3f} ~ {T_MAX:.3f}")
print(f"P range: {P_MIN:.3e} ~ {P_MAX:.3e}")

def norm_T(x):   return (x - T_MIN) / (T_MAX - T_MIN + EPS)
def norm_P(x):   return (x - P_MIN) / (P_MAX - P_MIN + EPS)
def norm_HTC(x): return (x - HTC_MIN) / (HTC_MAX - HTC_MIN + EPS)
def denorm_T(x): return x * (T_MAX - T_MIN) + T_MIN

# ============================================================
# 3. DATASET
# ============================================================

class TempPowerHTCDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wl, t = self.samples[idx]

        T = np.loadtxt(
            f"{T_ROOT}/{wl}_thermal_maps/csv/{wl}_thermal_maps_{t}.csv",
            delimiter=","
        )
        Tn = np.loadtxt(
            f"{T_ROOT}/{wl}_thermal_maps/csv/{wl}_thermal_maps_{t+1}.csv",
            delimiter=","
        )
        P = np.loadtxt(
            f"{P_ROOT}/{wl}_power_maps/{wl}_power_maps_{t}.csv",
            delimiter=","
        )

        x = np.stack([
            norm_T(T),
            norm_P(P),
            norm_HTC(np.full_like(T, HTC_VALUE))
        ])
        y = norm_T(Tn)[None, ...]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            wl,
            t + 1
        )

# ============================================================
# 4. CONV + TRANSFORMER MODEL
# ============================================================

class ConvTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, heads=8, layers=4):
        super().__init__()

        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4),
            nn.ReLU(inplace=True)
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ph = (4 - H % 4) % 4
        pw = (4 - W % 4) % 4
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")

        x = self.patch(x)
        B, C, Hp, Wp = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2).reshape(B, C, Hp, Wp)

        x = self.head(x)
        return x[:, :, :H, :W]

# ============================================================
# 5. TRAIN
# ============================================================

train_loader = DataLoader(
    TempPowerHTCDataset(train_samples),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TempPowerHTCDataset(test_samples),
    batch_size=1, shuffle=False
)

model = ConvTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.MSELoss()

best_score = 1e9
train_log, test_log = [], []

# ============================================================
# 6. TRAIN LOOP
# ============================================================

for ep in range(EPOCHS):
    model.train()
    mae_sum, mse_sum, n = 0.0, 0.0, 0

    for x, y, _, _ in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_T = denorm_T(pred.detach().cpu().numpy())
        true_T = denorm_T(y.cpu().numpy())
        err = np.abs(pred_T - true_T)

        mae_sum += err.mean()
        mse_sum += (err ** 2).mean()
        n += 1

    train_mae = mae_sum / n
    train_mse = mse_sum / n
    score = 0.7 * train_mse + 0.3 * train_mae
    train_log.append([ep, train_mae, train_mse, score])

    # -------- TEST --------
    model.eval()
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, _, _ in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            pred_T = denorm_T(pred.cpu().numpy())
            true_T = denorm_T(y.cpu().numpy())
            err = np.abs(pred_T - true_T)

            mae_sum += err.mean()
            mse_sum += (err ** 2).mean()
            n += 1

    test_mae = mae_sum / n
    test_mse = mse_sum / n
    test_log.append([ep, test_mae, test_mse])

    if score < best_score:
        best_score = score
        torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pth")

    print(
        f"[Epoch {ep:04d}] "
        f"Train MAE={train_mae:.4f}, MSE={train_mse:.4f}, Score={score:.4f} | "
        f"Test MAE={test_mae:.4f}, MSE={test_mse:.4f}"
    )

# ============================================================
# 7. SAVE LOGS
# ============================================================

pd.DataFrame(
    train_log,
    columns=["epoch", "train_MAE", "train_MSE", "score"]
).to_csv(f"{OUT_DIR}/train_metrics.csv", index=False)

pd.DataFrame(
    test_log,
    columns=["epoch", "test_MAE", "test_MSE"]
).to_csv(f"{OUT_DIR}/test_metrics.csv", index=False)

# ============================================================
# 8. TEST OUTPUT (PNG + CSV, SAME AS UNET)
# ============================================================

def save_one_figure(true, pred, err, save_path, title):
    plt.figure(figsize=(14, 4))
    for i, (data, name, cmap) in enumerate([
        (true, "True Temperature", "jet"),
        (pred, "Predicted Temperature", "jet"),
        (err,  "Absolute Error", "hot")
    ]):
        ax = plt.subplot(1, 3, i + 1)
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

model.load_state_dict(torch.load(f"{OUT_DIR}/best_model.pth"))
model.eval()

test_mae_sum, test_mse_sum, cnt = 0.0, 0.0, 0

with torch.no_grad():
    for x, y, wl, frame in test_loader:
        x = x.to(device)

        pred = denorm_T(model(x).cpu().squeeze().numpy())
        true = denorm_T(y.squeeze().numpy())
        err = np.abs(pred - true)

        test_mae_sum += err.mean()
        test_mse_sum += (err ** 2).mean()
        cnt += 1

        wl_dir = os.path.join(TEST_OUT, wl[0])
        os.makedirs(wl_dir, exist_ok=True)

        h, w = true.shape
        rows = [[i, j, true[i,j], pred[i,j], err[i,j]]
                for i in range(h) for j in range(w)]

        pd.DataFrame(
            rows,
            columns=["row", "col", "T_true", "T_pred", "T_abs_error"]
        ).to_csv(os.path.join(wl_dir, f"frame_{frame.item()}.csv"),
                 index=False)

        save_one_figure(
            true, pred, err,
            os.path.join(wl_dir, f"frame_{frame.item()}.png"),
            title=f"{wl[0]} | Frame {frame.item()}"
        )

print("All Transformer test results saved.")
