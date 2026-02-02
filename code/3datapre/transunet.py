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

OUT_DIR = "/home/yslu/0122/transunet_pre1000"
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# (optional) make training more deterministic
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

        if len(T_files) == 0 or len(P_files) == 0:
            continue

        pairs = []
        for i in range(min(len(T_files), len(P_files)) - 1):
            t0 = extract_frame_id(T_files[i])
            t1 = extract_frame_id(T_files[i + 1])
            if t1 == t0 + 1:
                # (T(t), P(t), T(t+1), workload, frame_id_of_(t+1))
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
        T  = np.loadtxt(T0, delimiter=",")
        Tn = np.loadtxt(T1, delimiter=",")
        P  = np.loadtxt(P0, delimiter=",")

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

        T  = norm_T(np.loadtxt(T0, delimiter=","))   # (H,W)
        P  = norm_P(np.loadtxt(P0, delimiter=","))   # (H,W)
        Tn = norm_T(np.loadtxt(T1, delimiter=","))   # (H,W)

        htc = norm_HTC(np.full_like(T, HTC_VALUE))   # (H,W)

        x = torch.tensor(np.stack([T, P, htc]), dtype=torch.float32)  # (3,H,W)
        y = torch.tensor(Tn, dtype=torch.float32).unsqueeze(0)        # (1,H,W)

        return x, y, wl, frame

# ============================================================
# 5. TransUNet
# ============================================================

class TransUNet(nn.Module):
    """
    Encoder (2 downsamples) + Transformer bottleneck + Decoder (2 upsamples).
    For arbitrary H,W, pad to (4*patch_size) multiple and crop back.
    """
    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        base=32,
        patch_size=8,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        self.patch_size = patch_size

        def CBR(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.e1 = CBR(in_ch, base)           # [B, base, H, W]
        self.p1 = nn.MaxPool2d(2)
        self.e2 = CBR(base, base * 2)       # [B, 2b, H/2, W/2]
        self.p2 = nn.MaxPool2d(2)
        self.e3 = CBR(base * 2, base * 4)   # [B, 4b, H/4, W/4]

        # Patch embedding on e3
        self.proj = nn.Conv2d(base * 4, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.unproj = nn.ConvTranspose2d(embed_dim, base * 4, kernel_size=patch_size, stride=patch_size)

        # Decoder
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = CBR(base * 4, base * 2)

        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = CBR(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        mult = 4 * self.patch_size
        ph = (mult - H % mult) % mult
        pw = (mult - W % mult) % mult
        x_pad = F.pad(x, (0, pw, 0, ph), mode="replicate")
        Hp, Wp = x_pad.shape[-2], x_pad.shape[-1]

        # Encoder
        e1 = self.e1(x_pad)         # [B, b, Hp, Wp]
        e2 = self.e2(self.p1(e1))   # [B, 2b, Hp/2, Wp/2]
        e3 = self.e3(self.p2(e2))   # [B, 4b, Hp/4, Wp/4]

        # Transformer bottleneck
        z = self.proj(e3)           # [B, embed, Bh, Bw]
        Bh, Bw = z.shape[-2], z.shape[-1]
        tokens = z.flatten(2).transpose(1, 2)  # [B, N, embed]
        tokens = self.pos_drop(tokens)
        tokens = self.transformer(tokens)      # [B, N, embed]
        z2 = tokens.transpose(1, 2).reshape(B, -1, Bh, Bw)  # [B, embed, Bh, Bw]
        e3t = self.unproj(z2)                  # [B, 4b, Hp/4, Wp/4]

        # Decoder + skips
        d2 = self.u2(e3t)                      # [B, 2b, Hp/2, Wp/2]
        d2 = self.d2(torch.cat([d2, e2], dim=1))

        d1 = self.u1(d2)                       # [B, b, Hp, Wp]
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        out = self.out(d1)                     # [B, 1, Hp, Wp]
        return out[:, :, :H, :W]

# ============================================================
# 6. TRAIN
# ============================================================

train_loader = DataLoader(TPDataset(train_list), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TPDataset(test_list), batch_size=1, shuffle=False)

model = TransUNet(
    in_ch=3,
    out_ch=1,
    base=32,
    patch_size=8,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    dropout=0.0
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

best_score = 1e18
log_rows = []

for ep in range(EPOCHS):
    model.train()
    mae, mse, n = 0.0, 0.0, 0
    train_norm_mse, nn_count = 0.0, 0

    for x, y, _, _ in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # normalized loss (backprop)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_norm_mse += loss.item()
        nn_count += 1

        # real-scale metrics for printing/selection
        err = np.abs(
            denorm_T(pred.detach().cpu().numpy()) -
            denorm_T(y.detach().cpu().numpy())
        )
        mae += err.mean()
        mse += (err ** 2).mean()
        n += 1

    mae /= max(n, 1)
    mse /= max(n, 1)
    train_norm_mse /= max(nn_count, 1)

    score = 0.7 * mse + 0.3 * mae

    if score < best_score:
        best_score = score
        torch.save(model.state_dict(), f"{OUT_DIR}/transunet_best.pth")

    log_rows.append({
        "epoch": ep,
        "train_norm_mse_loss": train_norm_mse,
        "train_real_mae": mae,
        "train_real_mse": mse,
        "score(0.7*mse+0.3*mae)": score
    })

    print(f"[Epoch {ep:03d}] Train(normMSEloss)={train_norm_mse:.6e} | MAE(real)={mae:.4f}, MSE(real)={mse:.4f} | score={score:.4f}")

# Save training log
os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(log_rows).to_csv(os.path.join(OUT_DIR, "train_metrics.csv"), index=False)
print("Saved train metrics:", os.path.join(OUT_DIR, "train_metrics.csv"))
print("Best model:", os.path.join(OUT_DIR, "transunet_best.pth"))

# ============================================================
# 7. TEST OUTPUT (CSV + PNG)
# ============================================================

model.load_state_dict(torch.load(f"{OUT_DIR}/transunet_best.pth", map_location=device))
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

print("ALL DONE (TransUNet, Random 8:2)")
