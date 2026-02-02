import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 0. CONFIG
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

T_ROOT = "/home/yslu/1121/T1126_split/i7"
P_ROOT = "/home/yslu/1121/powerzuizhong/i7"

OUT_DIR = "/home/yslu/0122/cnn_pre1000"
TEST_OUT = os.path.join(OUT_DIR, "test_results")
os.makedirs(TEST_OUT, exist_ok=True)

HTC_VALUE = 600.0
HTC_MIN, HTC_MAX = 200.0, 1300.0

EPOCHS = 1000
BATCH_SIZE = 4
LR = 1e-4
EPS = 1e-8
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 1. UTILS
# ============================================================

def extract_frame_id(path):
    return int(os.path.basename(path).split("_")[-1].replace(".csv", ""))

# ============================================================
# 2. BUILD RANDOM 8:2 SAMPLES (PER WORKLOAD)
# ============================================================

def build_samples():
    train, test = [], []

    for name in sorted(os.listdir(T_ROOT)):
        if not name.endswith("_thermal_maps"):
            continue

        wl = name.replace("_thermal_maps", "")
        T_dir = os.path.join(T_ROOT, name, "csv")
        P_dir = os.path.join(P_ROOT, wl + "_power_maps")

        if not os.path.isdir(T_dir) or not os.path.isdir(P_dir):
            continue

        T_files = sorted(glob.glob(os.path.join(T_dir, "*.csv")), key=extract_frame_id)
        P_files = sorted(glob.glob(os.path.join(P_dir, "*.csv")), key=extract_frame_id)

        pairs = []
        for i in range(len(T_files) - 1):
            t0 = extract_frame_id(T_files[i])
            t1 = extract_frame_id(T_files[i + 1])
            if t1 == t0 + 1:
                # (T(t), P(t), T(t+1), workload, frame_id_of_t+1)
                pairs.append((T_files[i], P_files[i], T_files[i + 1], wl, t1))

        if len(pairs) < 5:
            continue

        random.shuffle(pairs)
        split = int(0.8 * len(pairs))
        train.extend(pairs[:split])
        test.extend(pairs[split:])

        print(f"{wl}: train={split}, test={len(pairs) - split}")

    print(f"TOTAL train={len(train)}, test={len(test)}")
    return train, test

train_list, test_list = build_samples()
assert len(train_list) > 0 and len(test_list) > 0

# ============================================================
# 3. GLOBAL MIN/MAX (TRAIN ONLY)
# ============================================================

def compute_minmax(samples):
    Tmin, Tmax = 1e9, -1e9
    Pmin, Pmax = 1e9, -1e9

    for T0, P0, T1, _, _ in samples:
        T = np.loadtxt(T0, delimiter=",")
        Tn = np.loadtxt(T1, delimiter=",")
        P = np.loadtxt(P0, delimiter=",")

        Tmin = min(Tmin, T.min(), Tn.min())
        Tmax = max(Tmax, T.max(), Tn.max())
        Pmin = min(Pmin, P.min())
        Pmax = max(Pmax, P.max())

    print(f"T range: {Tmin:.3f} ~ {Tmax:.3f}")
    print(f"P range: {Pmin:.3e} ~ {Pmax:.3e}")
    return Tmin, Tmax, Pmin, Pmax

T_MIN, T_MAX, P_MIN, P_MAX = compute_minmax(train_list)

def norm_T(x):   return (x - T_MIN) / (T_MAX - T_MIN + EPS)
def norm_P(x):   return (x - P_MIN) / (P_MAX - P_MIN + EPS)
def norm_HTC(x): return (x - HTC_MIN) / (HTC_MAX - HTC_MIN + EPS)

def denorm_T_torch(x_norm: torch.Tensor) -> torch.Tensor:
    # x_norm: tensor
    return x_norm * (T_MAX - T_MIN) + T_MIN

def denorm_T_np(x_norm: np.ndarray) -> np.ndarray:
    return x_norm * (T_MAX - T_MIN) + T_MIN

# ============================================================
# 4. DATASET
# ============================================================

class TPDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        T0, P0, T1, wl, frame = self.samples[idx]

        T  = norm_T(np.loadtxt(T0, delimiter=","))
        P  = norm_P(np.loadtxt(P0, delimiter=","))
        Tn = norm_T(np.loadtxt(T1, delimiter=","))

        htc = norm_HTC(np.full_like(T, HTC_VALUE))

        x = torch.tensor(np.stack([T, P, htc]), dtype=torch.float32)   # [3,H,W]
        y = torch.tensor(Tn, dtype=torch.float32).unsqueeze(0)         # [1,H,W]

        return x, y, wl, frame

# ============================================================
# 5. CNN MODEL (replace UNet)
# ============================================================

class SimpleCNN(nn.Module):
    """
    A compact fully-convolutional CNN:
    in:  [B,3,H,W]
    out: [B,1,H,W]
    Keeps spatial size same.
    """
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base, base//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base//2, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# 6. TRAIN (loss on normalized, metrics on real-scale)
# ============================================================

train_loader = DataLoader(TPDataset(train_list), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TPDataset(test_list), batch_size=1, shuffle=False)

model = SimpleCNN().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()  # normalized MSE

best_score = 1e18
log_rows = []

@torch.no_grad()
def eval_real(loader):
    model.eval()
    mae, mse, n = 0.0, 0.0, 0
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        pred_real = denorm_T_torch(pred)
        y_real    = denorm_T_torch(y)

        diff = torch.abs(pred_real - y_real)
        mae += diff.mean().item()
        mse += (diff ** 2).mean().item()
        n += 1
    return mae / n, mse / n

for ep in range(EPOCHS):
    model.train()
    tr_mae, tr_mse, tr_n = 0.0, 0.0, 0
    tr_loss_norm, tr_ln = 0.0, 0

    for x, y, _, _ in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # normalized loss for backprop
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        tr_loss_norm += loss.item()
        tr_ln += 1

        # real-scale metrics (no grad needed but we already have tensors)
        with torch.no_grad():
            pred_real = denorm_T_torch(pred)
            y_real    = denorm_T_torch(y)
            diff = torch.abs(pred_real - y_real)
            tr_mae += diff.mean().item()
            tr_mse += (diff ** 2).mean().item()
            tr_n += 1

    tr_loss_norm /= max(tr_ln, 1)
    tr_mae /= max(tr_n, 1)
    tr_mse /= max(tr_n, 1)

    te_mae, te_mse = eval_real(test_loader)

    score = 0.7 * te_mse + 0.3 * te_mae  
    if score < best_score:
        best_score = score
        torch.save(model.state_dict(), f"{OUT_DIR}/cnn_best.pth")

    log_rows.append({
        "epoch": ep,
        "train_norm_mse_loss": tr_loss_norm,
        "train_real_mae": tr_mae,
        "train_real_mse": tr_mse,
        "test_real_mae": te_mae,
        "test_real_mse": te_mse,
        "score(0.7*mse+0.3*mae)": score
    })

    print(
        f"[Epoch {ep:03d}] "
        f"Train: normMSE(loss)={tr_loss_norm:.6f} | real MAE={tr_mae:.4f}, MSE={tr_mse:.4f} || "
        f"Test:  real MAE={te_mae:.4f}, MSE={te_mse:.4f} | score={score:.4f}"
    )

# save logs
df = pd.DataFrame(log_rows)
csv_path = os.path.join(OUT_DIR, "train_test_metrics_real_and_norm.csv")
df.to_csv(csv_path, index=False)
print("Saved metrics:", csv_path)
print("Best model:", f"{OUT_DIR}/cnn_best.pth")

# ============================================================
# 7. TEST OUTPUT (CSV + PNG) using BEST MODEL
# ============================================================

model.load_state_dict(torch.load(f"{OUT_DIR}/cnn_best.pth", map_location=device))
model.eval()

def save_png(true, pred, err, path, title):
    plt.figure(figsize=(14,4))
    for i, (data, name, cmap) in enumerate([
        (true, "True T(t+1)", "jet"),
        (pred, "Pred T(t+1)", "jet"),
        (err,  "Abs Error", "hot")
    ]):
        ax = plt.subplot(1,3,i+1)
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

with torch.no_grad():
    for x, y, wl, frame in test_loader:
        x = x.to(device)
        pred = model(x).cpu().squeeze().numpy()       # normalized
        true = y.squeeze().numpy()                    # normalized

        pred = denorm_T_np(pred)
        true = denorm_T_np(true)
        err  = np.abs(pred - true)

        wl_dir = os.path.join(TEST_OUT, wl[0])
        os.makedirs(wl_dir, exist_ok=True)

        fid = frame.item()

        rows = [
            [i, j, true[i, j], pred[i, j], err[i, j]]
            for i in range(true.shape[0])
            for j in range(true.shape[1])
        ]

        pd.DataFrame(
            rows,
            columns=["row", "col", "T_true", "T_pred", "T_abs_error"]
        ).to_csv(os.path.join(wl_dir, f"frame_{fid}.csv"), index=False)

        save_png(
            true, pred, err,
            os.path.join(wl_dir, f"frame_{fid}.png"),
            f"{wl[0]} | Frame {fid}"
        )

print("ALL DONE (CNN, Random 8:2)")
