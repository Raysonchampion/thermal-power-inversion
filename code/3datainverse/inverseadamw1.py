import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import laplace

# ============================================================
# 0. CONFIG
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

T_ROOT = "/home/yslu/1121/T1126_split/i7"
P_ROOT = "/home/yslu/1121/powerzuizhong/i7"

MODEL_PATH = "/home/yslu/1121/unet_prenorm1/unet_best_norm.pth"
OUT_ROOT = "/home/yslu/0127/invers-adamw/"
os.makedirs(OUT_ROOT, exist_ok=True)

FRAME_START, FRAME_END = 0, 99   # will use f -> f+1

DIE_W_MM, DIE_H_MM = 14.0, 9.0
TOTAL_POWER_W = 15.0

GAUSS_KERNEL = (49, 49)
GAUSS_SIGMA = 5.0
VMAX_PERCENTILE = 99.5

HTC_VALUE = 600.0
HTC_MIN, HTC_MAX = 200.0, 1300.0

EPOCHS = 400
LR = 5e-2
WEIGHT_DECAY = 1e-4
EPS = 1e-8

# 2 + 3:
ALPHA_PRIOR = 0.3          # adaptive prior ratio
LAMBDA_POWER_SUM = 1.0
LAMBDA_SMOOTH = 0.02

# score weights (as you requested)
W_MSE = 0.7
W_MAE = 0.3

# ============================================================
# 1. HELPERS
# ============================================================

def list_workloads():
    return [d.replace("_thermal_maps", "")
            for d in os.listdir(T_ROOT)
            if d.endswith("_thermal_maps")]

def t_file(wl, f):
    return f"{T_ROOT}/{wl}_thermal_maps/csv/{wl}_thermal_maps_{f}.csv"

def p_file(wl, f):
    return f"{P_ROOT}/{wl}_power_maps/{wl}_power_maps_{f}.csv"

# ============================================================
# 2. GLOBAL MIN/MAX (ALL workloads)
# ============================================================

print("[1/4] Computing GLOBAL min/max ...")

workloads = list_workloads()
assert len(workloads) > 0, "No workloads found under T_ROOT."

T_MIN, T_MAX = 1e9, -1e9
P_MIN, P_MAX = 1e9, -1e9

for wl in workloads:
    for f in range(FRAME_START, FRAME_END + 1):
        Tf = t_file(wl, f)
        Pf = p_file(wl, f)
        if not (os.path.exists(Tf) and os.path.exists(Pf)):
            continue
        T = np.loadtxt(Tf, delimiter=",")
        P = np.loadtxt(Pf, delimiter=",")
        T_MIN, T_MAX = min(T_MIN, float(T.min())), max(T_MAX, float(T.max()))
        P_MIN, P_MAX = min(P_MIN, float(P.min())), max(P_MAX, float(P.max()))

print(f"GLOBAL T range: {T_MIN:.3f} ~ {T_MAX:.3f}")
print(f"GLOBAL P range: {P_MIN:.6e} ~ {P_MAX:.6e}")

T_MIN_T = torch.tensor(T_MIN, dtype=torch.float32, device=device)
T_MAX_T = torch.tensor(T_MAX, dtype=torch.float32, device=device)
P_MIN_T = torch.tensor(P_MIN, dtype=torch.float32, device=device)
P_MAX_T = torch.tensor(P_MAX, dtype=torch.float32, device=device)

def norm_T_np(x_np: np.ndarray) -> np.ndarray:
    return (x_np - T_MIN) / (T_MAX - T_MIN + EPS)

def norm_P_np(x_np: np.ndarray) -> np.ndarray:
    return (x_np - P_MIN) / (P_MAX - P_MIN + EPS)

def denorm_T_np(x_np: np.ndarray) -> np.ndarray:
    return x_np * (T_MAX - T_MIN) + T_MIN

def denorm_P_torch(x_t: torch.Tensor) -> torch.Tensor:
    # x_t is normalized [0,1]
    return x_t * (P_MAX_T - P_MIN_T) + P_MIN_T

def denorm_T_torch(x_t: torch.Tensor) -> torch.Tensor:
    # x_t is normalized [0,1]
    return x_t * (T_MAX_T - T_MIN_T) + T_MIN_T

def norm_HTC_value(htc: float) -> float:
    return float((htc - HTC_MIN) / (HTC_MAX - HTC_MIN + EPS))

# ============================================================
# 3. Frozen UNet
# ============================================================

print("[2/4] Loading Frozen UNet ...")

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
        x = x.float()
        B, C, H, W = x.shape
        ph, pw = (4 - H % 4) % 4, (4 - W % 4) % 4
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)[:, :, :H, :W]

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
for p in model.parameters():
    p.requires_grad = False
print("Frozen UNet loaded.")

# ============================================================
# 4. Laplace prior + utilities
# ============================================================

def laplace_prior_from_T(T_np: np.ndarray) -> np.ndarray:
    """
    Input:  T in degC, (H,W)
    Output: P_lap in W/mm^2, (H,W), non-negative, normalized to TOTAL_POWER_W
    """
    T_np = T_np.astype(np.float32)
    H, W = T_np.shape
    dx = DIE_W_MM / W
    dy = DIE_H_MM / H

    Ts = cv2.GaussianBlur(
        T_np.astype(np.float64),
        GAUSS_KERNEL,
        GAUSS_SIGMA,
        borderType=cv2.BORDER_REPLICATE
    )
    raw = -laplace(Ts)
    raw[raw < 0] = 0.0

    integral = raw.sum() * dx * dy
    if integral <= 0:
        uni = np.ones_like(T_np, dtype=np.float64)
        integral2 = uni.sum() * dx * dy
        return (TOTAL_POWER_W * uni / integral2).astype(np.float32)

    return (TOTAL_POWER_W * raw / integral).astype(np.float32)

def power_sum_W_np(P_wmm2: np.ndarray) -> float:
    H, W = P_wmm2.shape
    dx = DIE_W_MM / W
    dy = DIE_H_MM / H
    return float(P_wmm2.sum() * dx * dy)

def laplacian_torch(x_hw: torch.Tensor) -> torch.Tensor:
    """
    x_hw: (H,W) float32
    return: (H,W) float32
    """
    x = x_hw.unsqueeze(0).unsqueeze(0)
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                     dtype=x.dtype, device=x.device).view(1,1,3,3)
    y = F.conv2d(x, k, padding=1)
    return y.squeeze(0).squeeze(0)

def save_triplet_png(a, b, c, titles, out_png, cmaps=("jet","jet","hot"), suptitle=None):
    plt.figure(figsize=(12,4))
    for i, (data, title, cmap) in enumerate(zip([a,b,c], titles, cmaps)):
        plt.subplot(1,3,i+1)
        if i == 2:
            vmax = np.percentile(data, VMAX_PERCENTILE)
            if vmax <= 0:
                vmax = data.max() if data.max() > 0 else 1.0
            plt.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        else:
            plt.imshow(data, cmap=cmap)
        plt.title(title)
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ============================================================
# 5. Inversion
# ============================================================

print("[3/4] Start inversion ...")

TOTAL_POWER_T = torch.tensor(TOTAL_POWER_W, dtype=torch.float32, device=device)
HTC_NORM = norm_HTC_value(HTC_VALUE)

for wl in workloads:
    print(f"\n========== WORKLOAD: {wl} ==========")
    wl_out = os.path.join(OUT_ROOT, wl)
    os.makedirs(wl_out, exist_ok=True)

    summary_rows = []

    for f in range(FRAME_START, FRAME_END):
        Tf0, Tf1 = t_file(wl, f), t_file(wl, f+1)
        Pf0 = p_file(wl, f)

        if not (os.path.exists(Tf0) and os.path.exists(Tf1)):
            print(f"[Skip] {wl} frame {f}: missing T")
            continue

        frame_dir = os.path.join(wl_out, f"frame_{f:03d}")
        os.makedirs(frame_dir, exist_ok=True)

        # load temps (degC)
        T0 = np.loadtxt(Tf0, delimiter=",").astype(np.float32)
        T1 = np.loadtxt(Tf1, delimiter=",").astype(np.float32)

        H, W = T0.shape
        dx_t = torch.tensor(DIE_W_MM / W, dtype=torch.float32, device=device)
        dy_t = torch.tensor(DIE_H_MM / H, dtype=torch.float32, device=device)

        # Laplace prior (W/mm^2)
        P_lap = laplace_prior_from_T(T0)  # float32
        P_lap_sum = power_sum_W_np(P_lap)

        # save prior
        pd.DataFrame(P_lap).to_csv(os.path.join(frame_dir, "P_prior_laplace.csv"),
                                   header=False, index=False)

        # load true power (optional)
        P_true = None
        if os.path.exists(Pf0):
            try:
                P_true = np.loadtxt(Pf0, delimiter=",").astype(np.float32)
                pd.DataFrame(P_true).to_csv(os.path.join(frame_dir, "P_true_loaded.csv"),
                                            header=False, index=False)
            except Exception:
                P_true = None

        # build tensors (normalized)
        Tn  = torch.tensor(norm_T_np(T0), dtype=torch.float32, device=device)
        Tgt = torch.tensor(norm_T_np(T1), dtype=torch.float32, device=device)
        htc_map = torch.full_like(Tn, HTC_NORM, dtype=torch.float32)

        P_prior_n = torch.tensor(norm_P_np(P_lap), dtype=torch.float32, device=device)

        # =========================
        # (3) multiplicative correction:
        # Pn = clamp(P_prior_n * softplus(S), 0, 1)
        # (2) adaptive lambda for prior term
        # =========================
        S = nn.Parameter(torch.zeros_like(P_prior_n))
        opt = torch.optim.AdamW([S], lr=LR, weight_decay=WEIGHT_DECAY)
        mse = nn.MSELoss()

        lambda_prior_eff = torch.tensor(1.0, dtype=torch.float32, device=device)

        # IMPORTANT: now we save EVERY epoch (full curve)
        curve_rows = []

        for ep in range(EPOCHS):
            opt.zero_grad()

            Pn = torch.clamp(P_prior_n * F.softplus(S), 0.0, 1.0).float()

            x = torch.stack([Tn, Pn, htc_map]).unsqueeze(0).float()
            Tpred = model(x).squeeze(0).squeeze(0).float()

            # normalized-domain temperature loss (MSE on normalized temperature)
            loss_T = mse(Tpred, Tgt)

            # (prior term in S-space)
            loss_prior = (S ** 2).mean()

            # (2) adaptive lambda: keep prior contribution proportional to loss_T
            if ep % 10 == 0:
                lambda_prior_eff = (ALPHA_PRIOR * loss_T.detach() / (loss_prior.detach() + EPS)).float()

            # power sum constraint (physical)
            Pphys = denorm_P_torch(Pn).float()  # W/mm^2
            Psum = (Pphys.sum() * dx_t * dy_t).float()
            loss_sum = (Psum - TOTAL_POWER_T) ** 2

            # smoothness (physical)
            lapP = laplacian_torch(Pphys)
            loss_smooth = (lapP ** 2).mean()

            # total objective (mixed terms, used for optimization)
            loss_total = (
                loss_T
                + lambda_prior_eff * loss_prior
                + LAMBDA_POWER_SUM * loss_sum
                + LAMBDA_SMOOTH * loss_smooth
            )

            loss_total.backward()
            opt.step()

            # =========================================================
            # NEW: per-epoch MAE/MSE/Score (normalized + denormalized)
            # =========================================================
            with torch.no_grad():
                # normalized metrics
                diff_n = Tpred - Tgt
                mae_n = diff_n.abs().mean()
                mse_n = (diff_n ** 2).mean()
                score_n = W_MSE * mse_n + W_MAE * mae_n

                # denormalized metrics (degC)
                Tpred_real = denorm_T_torch(Tpred)
                Tgt_real   = denorm_T_torch(Tgt)
                diff_r = Tpred_real - Tgt_real
                mae_r = diff_r.abs().mean()
                mse_r = (diff_r ** 2).mean()
                score_r = W_MSE * mse_r + W_MAE * mae_r

            # save EVERY epoch
            curve_rows.append([
                ep,
                float(loss_T.item()),          # normalized-domain MSE loss (same as mse_n but computed via nn.MSELoss)
                float(mae_n.item()),
                float(mse_n.item()),
                float(score_n.item()),
                float(mae_r.item()),
                float(mse_r.item()),
                float(score_r.item()),
                float(loss_prior.item()),
                float(lambda_prior_eff.item()),
                float(loss_sum.item()),
                float(loss_smooth.item()),
                float(loss_total.item()),
                float(Psum.item()),
            ])

            # print occasionally
            if ep % 50 == 0 or ep == EPOCHS - 1:
                print(f"{wl} frame {f} [ep {ep:04d}] "
                      f"lossT={loss_T.item():.3e} "
                      f"norm(MAE={mae_n.item():.3e}, MSE={mse_n.item():.3e}, S={score_n.item():.3e}) "
                      f"real(MAE={mae_r.item():.3e}C, MSE={mse_r.item():.3e}C2, S={score_r.item():.3e}) "
                      f"lam={lambda_prior_eff.item():.2e} sum={loss_sum.item():.3e} "
                      f"smooth={loss_smooth.item():.3e} total={loss_total.item():.3e} "
                      f"Psum={Psum.item():.4f}W")

        # save per-epoch curve CSV (FULL)
        pd.DataFrame(curve_rows, columns=[
            "epoch",
            "loss_T_normMSE",
            "T_MAE_norm",
            "T_MSE_norm",
            "Score_norm_0.7MSE_0.3MAE",
            "T_MAE_denorm_C",
            "T_MSE_denorm_C2",
            "Score_denorm_0.7MSE_0.3MAE",
            "loss_prior_S2",
            "lambda_prior_eff",
            "loss_sum",
            "loss_smooth",
            "loss_total",
            "P_sum_W"
        ]).to_csv(os.path.join(frame_dir, "inverse_curve.csv"), index=False)

        # final outputs
        with torch.no_grad():
            Pn = torch.clamp(P_prior_n * F.softplus(S), 0.0, 1.0).float()
            x = torch.stack([Tn, Pn, htc_map]).unsqueeze(0).float()
            T_pred_n = model(x).squeeze().detach().cpu().numpy().astype(np.float32)

            P_pred = denorm_P_torch(Pn).detach().cpu().numpy().astype(np.float32)
            T_pred = denorm_T_np(T_pred_n).astype(np.float32)

        T_err = np.abs(T_pred - T1).astype(np.float32)

        # save CSVs
        pd.DataFrame(P_pred).to_csv(os.path.join(frame_dir, "P_inverted.csv"), header=False, index=False)
        pd.DataFrame(T_pred).to_csv(os.path.join(frame_dir, "T_pred.csv"), header=False, index=False)
        pd.DataFrame(T_err).to_csv(os.path.join(frame_dir, "T_abs_error.csv"), header=False, index=False)

        # power compare saving + png
        if P_true is not None:
            P_err_true = np.abs(P_pred - P_true).astype(np.float32)
            pd.DataFrame(P_err_true).to_csv(os.path.join(frame_dir, "P_abs_error_vs_true.csv"),
                                            header=False, index=False)
            save_triplet_png(
                P_true, P_pred, P_err_true,
                ["Power True (loaded)", "Power Inverted", "Power Abs Error"],
                os.path.join(frame_dir, "power_true_vs_inverted.png"),
                suptitle=f"{wl} | Frame {f} Power"
            )
        else:
            P_err_prior = np.abs(P_pred - P_lap).astype(np.float32)
            pd.DataFrame(P_err_prior).to_csv(os.path.join(frame_dir, "P_abs_error_vs_prior.csv"),
                                             header=False, index=False)
            save_triplet_png(
                P_lap, P_pred, P_err_prior,
                ["Power Prior (Laplace)", "Power Inverted", "Abs Diff (Inv-Prior)"],
                os.path.join(frame_dir, "power_prior_vs_inverted.png"),
                suptitle=f"{wl} | Frame {f} Power"
            )

        # temp compare png
        save_triplet_png(
            T1, T_pred, T_err,
            ["Temp True T(t+1)", "Temp Pred T(t+1)", "Temp Abs Error"],
            os.path.join(frame_dir, "temp_compare.png"),
            suptitle=f"{wl} | Frame {f} Temperature"
        )

        # metrics (physical domain)
        P_sum_final = power_sum_W_np(P_pred)
        T_mae = float(T_err.mean())
        T_mse = float((T_err**2).mean())

        P_diff_prior = float(np.abs(P_pred - P_lap).mean())
        P_diff_prior_mse = float(((P_pred - P_lap) ** 2).mean())

        if P_true is not None:
            P_err = np.abs(P_pred - P_true)
            P_mae = float(P_err.mean())
            P_mse = float((P_err**2).mean())
        else:
            P_mae, P_mse = np.nan, np.nan

        summary_rows.append([
            f,
            float(P_lap_sum),
            float(P_sum_final),
            float(P_diff_prior),
            float(P_diff_prior_mse),
            float(P_mae), float(P_mse),
            float(T_mae), float(T_mse),
        ])

    # per-workload summary
    df = pd.DataFrame(summary_rows, columns=[
        "frame",
        "P_prior_sum_W",
        "P_inverted_sum_W",
        "mean_abs_diff_inv_prior",
        "mse_diff_inv_prior",
        "P_MAE_vs_true",
        "P_MSE_vs_true",
        "T_MAE",
        "T_MSE",
    ])
    if len(df) > 0:
        df.loc["mean"] = ["mean"] + [df[c].mean() for c in df.columns[1:]]
    df.to_csv(os.path.join(wl_out, "summary.csv"), index=False)

print("[4/4] DONE. Saved to:", OUT_ROOT)
