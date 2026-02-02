import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

OUT_DIR = "/home/yslu/0122/results_patch_lstmampere"
TEST_OUT = os.path.join(OUT_DIR, "test_results")
os.makedirs(TEST_OUT, exist_ok=True)

HTC_VALUE = 600.0
HTC_MIN, HTC_MAX = 200.0, 1300.0

# ===== sequence length =====
SEQ_LEN = 5
EPOCHS = 300
BATCH_SIZE = 4
LR = 1e-4
EPS = 1e-8
SEED = 42

# ===== Patch-LSTM model =====
PATCH = 4            # patch size (like your conv+transformer)
EMBED_DIM = 128
HIDDEN_DIM = 128     # LSTM hidden per patch token
NUM_LAYERS = 1
DROPOUT = 0.1
CLIP_NORM = 1.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# 1. UTILS
# ============================================================

def extract_frame_id(path):
    return int(os.path.basename(path).split("_")[-1].replace(".csv", ""))

def list_workloads():
    wls = []
    for name in sorted(os.listdir(T_ROOT)):
        if name.endswith("_thermal_maps"):
            wls.append(name.replace("_thermal_maps", ""))
    return wls

def t_dir(wl):
    return os.path.join(T_ROOT, f"{wl}_thermal_maps", "csv")

def p_dir(wl):
    return os.path.join(P_ROOT, f"{wl}_power_maps")

# ============================================================
# 2. BUILD SEQUENCE SAMPLES (workload split, avoid leakage)
#    sample: input frames [t-SEQ_LEN+1 ... t], target frame t+1
# ============================================================

def build_sequences_for_workload(wl, seq_len):
    Td = t_dir(wl)
    Pd = p_dir(wl)
    if (not os.path.isdir(Td)) or (not os.path.isdir(Pd)):
        return []

    T_files = sorted(glob.glob(os.path.join(Td, "*.csv")), key=extract_frame_id)
    P_files = sorted(glob.glob(os.path.join(Pd, "*.csv")), key=extract_frame_id)
    if len(T_files) == 0 or len(P_files) == 0:
        return []

    T_map = {extract_frame_id(p): p for p in T_files}
    P_map = {extract_frame_id(p): p for p in P_files}
    frames = sorted(set(T_map.keys()) & set(P_map.keys()))
    if len(frames) < seq_len + 1:
        return []

    samples = []
    for t in frames:
        in_ids = list(range(t - seq_len + 1, t + 1))
        tgt_id = t + 1

        if any(fid not in T_map for fid in in_ids) or any(fid not in P_map for fid in in_ids):
            continue
        if tgt_id not in T_map:
            continue
        if in_ids[-1] - in_ids[0] != seq_len - 1:
            continue

        samples.append({
            "wl": wl,
            "in_T": [T_map[fid] for fid in in_ids],
            "in_P": [P_map[fid] for fid in in_ids],
            "tgt_T": T_map[tgt_id],
            "frame": tgt_id
        })
    return samples



all_wls = list_workloads()
assert len(all_wls) > 0, "No workloads found."

train_samples, test_samples = [], []

for wl in all_wls:
    ss = build_sequences_for_workload(wl, SEQ_LEN)
    if len(ss) == 0:
        continue

    
    ss = sorted(ss, key=lambda x: x["frame"])

    split = int(0.8 * len(ss))

    train_samples.extend(ss[:split])
    test_samples.extend(ss[split:])

    print(f"[wl split] {wl}: total={len(ss)}, train={split}, test={len(ss)-split}")

print("TOTAL samples | train:", len(train_samples), "| test:", len(test_samples))

assert len(train_samples) > 0 and len(test_samples) > 0, "No sequence samples built."
print("TOTAL samples | train:", len(train_samples), "| test:", len(test_samples))

# ============================================================
# 3. GLOBAL MIN/MAX (TRAIN ONLY)
# ============================================================

def compute_minmax_from_sequences(samples):
    Tmin, Tmax = 1e9, -1e9
    Pmin, Pmax = 1e9, -1e9

    for s in samples:
        for tp in s["in_T"]:
            T = np.loadtxt(tp, delimiter=",")
            Tmin = min(Tmin, float(T.min()))
            Tmax = max(Tmax, float(T.max()))
        Tt = np.loadtxt(s["tgt_T"], delimiter=",")
        Tmin = min(Tmin, float(Tt.min()))
        Tmax = max(Tmax, float(Tt.max()))

        for pp in s["in_P"]:
            P = np.loadtxt(pp, delimiter=",")
            Pmin = min(Pmin, float(P.min()))
            Pmax = max(Pmax, float(P.max()))

    print(f"[TRAIN MINMAX] T: {Tmin:.3f} ~ {Tmax:.3f}")
    print(f"[TRAIN MINMAX] P: {Pmin:.6e} ~ {Pmax:.6e}")
    return Tmin, Tmax, Pmin, Pmax

T_MIN, T_MAX, P_MIN, P_MAX = compute_minmax_from_sequences(train_samples)

def norm_T(x):   return (x - T_MIN) / (T_MAX - T_MIN + EPS)
def norm_P(x):   return (x - P_MIN) / (P_MAX - P_MIN + EPS)
def norm_HTC(v): return (v - HTC_MIN) / (HTC_MAX - HTC_MIN + EPS)
def denorm_T(x): return x * (T_MAX - T_MIN) + T_MIN

HTC_NORM = float(norm_HTC(HTC_VALUE))

# ============================================================
# 4. DATASET (sequence of frames)
#    returns:
#      x_seq: (K, 3, H, W)
#      y:     (1, H, W)
# ============================================================

class SeqMapDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        T0 = np.loadtxt(samples[0]["in_T"][0], delimiter=",")
        self.H, self.W = T0.shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        frames = []
        for tp, pp in zip(s["in_T"], s["in_P"]):
            T = norm_T(np.loadtxt(tp, delimiter=",")).astype(np.float32)
            P = norm_P(np.loadtxt(pp, delimiter=",")).astype(np.float32)
            htc = np.full_like(T, HTC_NORM, dtype=np.float32)
            x = np.stack([T, P, htc], axis=0)  # (3,H,W)
            frames.append(x)

        x_seq = torch.tensor(np.stack(frames, axis=0), dtype=torch.float32)  # (K,3,H,W)

        Tt = norm_T(np.loadtxt(s["tgt_T"], delimiter=",")).astype(np.float32)
        y = torch.tensor(Tt[None, ...], dtype=torch.float32)  # (1,H,W)

        return x_seq, y, s["wl"], s["frame"]

# ============================================================
# 5. Patch-LSTM Model (spatial tokens + temporal LSTM)
# ============================================================

class PatchLSTMTemp(nn.Module):
    """
    x_seq: (B,K,3,H,W)
    1) patch conv per frame -> (B,K,E,Hp,Wp)
    2) per-patch LSTM over time: (B*Hp*Wp, K, E) -> last -> (B,E,Hp,Wp)
    3) deconv upsample -> (B,1,H,W)
    """
    def __init__(self, in_ch=3, embed_dim=128, patch=4, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.patch = patch
        self.embed_dim = embed_dim

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.to_embed = nn.Linear(hidden_dim, embed_dim)

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=patch, stride=patch),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_seq):
        # x_seq: (B,K,3,H,W)
        B, K, C, H, W = x_seq.shape

        # pad to multiple of patch
        ph = (self.patch - H % self.patch) % self.patch
        pw = (self.patch - W % self.patch) % self.patch
        if ph != 0 or pw != 0:
            x = x_seq.reshape(B * K, C, H, W)
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            Hp0, Wp0 = H + ph, W + pw
            x_seq = x.reshape(B, K, C, Hp0, Wp0)
        else:
            Hp0, Wp0 = H, W

        # patch embed each frame
        x2 = x_seq.reshape(B * K, C, Hp0, Wp0)
        feat = self.patch_embed(x2)  # (B*K, E, Hp, Wp)
        E, Hp, Wp = feat.shape[1], feat.shape[2], feat.shape[3]
        feat = feat.reshape(B, K, E, Hp, Wp)  # (B,K,E,Hp,Wp)

        # per-patch LSTM over time
        feat = feat.permute(0, 3, 4, 1, 2).contiguous()   # (B,Hp,Wp,K,E)
        feat = feat.view(B * Hp * Wp, K, E)               # (B*Hp*Wp,K,E)

        out, _ = self.lstm(feat)                          # (B*Hp*Wp,K,Hid)
        h_last = out[:, -1, :]                            # (B*Hp*Wp,Hid)
        e_last = self.to_embed(h_last)                    # (B*Hp*Wp,E)

        e_map = e_last.view(B, Hp, Wp, E).permute(0, 3, 1, 2).contiguous()  # (B,E,Hp,Wp)

        y = self.decode(e_map)                            # (B,1,Hp0,Wp0)
        return y[:, :, :H, :W]

# ============================================================
# 6. TRAIN
# ============================================================

train_loader = DataLoader(SeqMapDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(SeqMapDataset(test_samples),  batch_size=1, shuffle=False, num_workers=0)

model = PatchLSTMTemp(
    in_ch=3,
    embed_dim=EMBED_DIM,
    patch=PATCH,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.MSELoss()

best_score = float("inf")
train_log, test_log = [], []

@torch.no_grad()
def eval_real(loader):
    model.eval()
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    for x_seq, y, _, _ in loader:
        x_seq = x_seq.to(device)
        y = y.to(device)
        pred = model(x_seq)

        pred_T = denorm_T(pred.detach().cpu().numpy())
        true_T = denorm_T(y.detach().cpu().numpy())
        err = np.abs(pred_T - true_T)

        mae_sum += err.mean()
        mse_sum += (err ** 2).mean()
        n += 1

    return mae_sum / max(n, 1), mse_sum / max(n, 1)

for ep in range(EPOCHS):
    model.train()
    loss_sum, n = 0.0, 0

    for x_seq, y, _, _ in train_loader:
        x_seq = x_seq.to(device)
        y = y.to(device)

        pred = model(x_seq)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        loss_sum += loss.item()
        n += 1

    train_norm_mse = loss_sum / max(n, 1)

    train_mae, train_mse = eval_real(train_loader)
    test_mae,  test_mse  = eval_real(test_loader)

    score = 0.7 * test_mse + 0.3 * test_mae
    train_log.append([ep, train_mae, train_mse, train_norm_mse])
    test_log.append([ep, test_mae, test_mse, score])

    if score < best_score:
        best_score = score
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))

    print(
        f"[Epoch {ep:04d}] "
        f"Train MAE={train_mae:.4f}, MSE={train_mse:.4f}, normMSE={train_norm_mse:.6e} | "
        f"Test MAE={test_mae:.4f}, MSE={test_mse:.4f}, Score={score:.4f}"
    )

# ============================================================
# 7. SAVE LOGS
# ============================================================

pd.DataFrame(train_log, columns=["epoch", "train_MAE", "train_MSE", "train_normMSE"]) \
  .to_csv(os.path.join(OUT_DIR, "train_metrics.csv"), index=False)
pd.DataFrame(test_log, columns=["epoch", "test_MAE", "test_MSE", "score"]) \
  .to_csv(os.path.join(OUT_DIR, "test_metrics.csv"), index=False)

# ============================================================
# 8. TEST OUTPUT (PNG + CSV)
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

model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth"), map_location=device))
model.eval()

with torch.no_grad():
    for x_seq, y, wl, frame in test_loader:
        x_seq = x_seq.to(device)
        pred = denorm_T(model(x_seq).cpu().squeeze().numpy())
        true = denorm_T(y.squeeze().numpy())
        err = np.abs(pred - true)

        wl_dir = os.path.join(TEST_OUT, wl[0])
        os.makedirs(wl_dir, exist_ok=True)

        h, w = true.shape
        rows = [[i, j, true[i, j], pred[i, j], err[i, j]]
                for i in range(h) for j in range(w)]

        pd.DataFrame(rows, columns=["row", "col", "T_true", "T_pred", "T_abs_error"]) \
          .to_csv(os.path.join(wl_dir, f"frame_{int(frame.item())}.csv"), index=False)

        save_one_figure(
            true, pred, err,
            os.path.join(wl_dir, f"frame_{int(frame.item())}.png"),
            title=f"{wl[0]} | Frame {int(frame.item())} | SEQ_LEN={SEQ_LEN}"
        )

print("All Patch-LSTM test results saved.")
