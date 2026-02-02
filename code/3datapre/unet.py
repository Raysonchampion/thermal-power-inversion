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

OUT_DIR = "/home/yslu/1121/unet_prenorm1"
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
def denorm_T(x): return x * (T_MAX - T_MIN) + T_MIN

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

        x = torch.tensor(np.stack([T, P, htc]), dtype=torch.float32)
        y = torch.tensor(Tn, dtype=torch.float32).unsqueeze(0)

        return x, y, wl, frame

# ============================================================
# 5. UNET
# ============================================================

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.e1 = CBR(3, 32)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = CBR(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = CBR(64, 128)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = CBR(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = CBR(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        ph = (4 - H % 4) % 4
        pw = (4 - W % 4) % 4
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")

        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        out = self.out(d1)

        return out[:, :, :H, :W]

# ============================================================
# 6. TRAIN + NORMALIZED METRICS
# ============================================================

train_loader = DataLoader(TPDataset(train_list), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TPDataset(test_list), batch_size=1, shuffle=False)

model = UNet().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

log_rows = []
best_norm_score = float("inf")

def eval_norm(loader):
    model.eval()
    mae, mse, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            diff = torch.abs(pred - y)
            mae += diff.mean().item()
            mse += (diff ** 2).mean().item()
            n += 1
    return mae / n, mse / n

for ep in range(EPOCHS):
    model.train()
    train_mae, train_mse, n = 0.0, 0.0, 0

    for x, y, _, _ in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_mse += loss.item()
        train_mae += torch.abs(pred - y).mean().item()
        n += 1

    train_mse /= n
    train_mae /= n

    test_mae, test_mse = eval_norm(test_loader)

    log_rows.append({
        "epoch": ep,
        "train_norm_mse": train_mse,
        "train_norm_mae": train_mae,
        "test_norm_mse": test_mse,
        "test_norm_mae": test_mae
    })

    norm_score = test_mse  

    if norm_score < best_norm_score:
        best_norm_score = norm_score
        torch.save(model.state_dict(), f"{OUT_DIR}/unet_best_norm.pth")

    print(
        f"[Epoch {ep:04d}] "
        f"Train(norm) MSE={train_mse:.6f}, MAE={train_mae:.6f} | "
        f"Test(norm) MSE={test_mse:.6f}, MAE={test_mae:.6f}"
    )

# ============================================================
# 7. SAVE CSV
# ============================================================

df = pd.DataFrame(log_rows)
csv_path = os.path.join(OUT_DIR, "norm_train_test_metrics.csv")
df.to_csv(csv_path, index=False)

print(f"Saved normalized metrics to: {csv_path}")
print("Best normalized model saved as: unet_best_norm.pth")

# ============================================================
# 8. TEST OUTPUT (CSV + PNG, using BEST NORM MODEL)
# ============================================================

model.load_state_dict(torch.load(f"{OUT_DIR}/unet_best_norm.pth"))
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
        pred = denorm_T(model(x).cpu().squeeze().numpy())
        true = denorm_T(y.squeeze().numpy())
        err = np.abs(pred - true)

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

print("ALL DONE (Best model selected by normalized loss)")
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

OUT_DIR = "/home/yslu/1121/unet_pre"
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
def denorm_T(x): return x * (T_MAX - T_MIN) + T_MIN

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

        x = torch.tensor(np.stack([T, P, htc]), dtype=torch.float32)
        y = torch.tensor(Tn, dtype=torch.float32).unsqueeze(0)

        return x, y, wl, frame

# ============================================================
# 5. UNET
# ============================================================

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.e1 = CBR(3, 32)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = CBR(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = CBR(64, 128)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = CBR(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = CBR(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        ph = (4 - H % 4) % 4
        pw = (4 - W % 4) % 4
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")

        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        out = self.out(d1)

        return out[:, :, :H, :W]

# ============================================================
# 6. TRAIN + NORMALIZED METRICS
# ============================================================

train_loader = DataLoader(TPDataset(train_list), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TPDataset(test_list), batch_size=1, shuffle=False)

model = UNet().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

log_rows = []
best_norm_score = float("inf")

def eval_norm(loader):
    model.eval()
    mae, mse, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            diff = torch.abs(pred - y)
            mae += diff.mean().item()
            mse += (diff ** 2).mean().item()
            n += 1
    return mae / n, mse / n

for ep in range(EPOCHS):
    model.train()
    train_mae, train_mse, n = 0.0, 0.0, 0

    for x, y, _, _ in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_mse += loss.item()
        train_mae += torch.abs(pred - y).mean().item()
        n += 1

    train_mse /= n
    train_mae /= n

    test_mae, test_mse = eval_norm(test_loader)

    log_rows.append({
        "epoch": ep,
        "train_norm_mse": train_mse,
        "train_norm_mae": train_mae,
        "test_norm_mse": test_mse,
        "test_norm_mae": test_mae
    })

    norm_score = test_mse 

    if norm_score < best_norm_score:
        best_norm_score = norm_score
        torch.save(model.state_dict(), f"{OUT_DIR}/unet_best_norm.pth")

    print(
        f"[Epoch {ep:04d}] "
        f"Train(norm) MSE={train_mse:.6f}, MAE={train_mae:.6f} | "
        f"Test(norm) MSE={test_mse:.6f}, MAE={test_mae:.6f}"
    )

# ============================================================
# 7. SAVE CSV
# ============================================================

df = pd.DataFrame(log_rows)
csv_path = os.path.join(OUT_DIR, "norm_train_test_metrics.csv")
df.to_csv(csv_path, index=False)

print(f"Saved normalized metrics to: {csv_path}")
print("Best normalized model saved as: unet_best_norm.pth")

# ============================================================
# 8. TEST OUTPUT (CSV + PNG, using BEST NORM MODEL)
# ============================================================

model.load_state_dict(torch.load(f"{OUT_DIR}/unet_best_norm.pth"))
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
        pred = denorm_T(model(x).cpu().squeeze().numpy())
        true = denorm_T(y.squeeze().numpy())
        err = np.abs(pred - true)

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

print("ALL DONE (Best model selected by normalized loss)")
