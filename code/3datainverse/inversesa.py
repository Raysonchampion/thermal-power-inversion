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
OUT_ROOT = "/home/yslu/0127/inverse_saepoch/"
os.makedirs(OUT_ROOT, exist_ok=True)

FRAME_START, FRAME_END = 0, 99   # f -> f+1

DIE_W_MM, DIE_H_MM = 14.0, 9.0
TOTAL_POWER_W = 15.0

GAUSS_KERNEL = (49, 49)
GAUSS_SIGMA = 5.0
VMAX_PERCENTILE = 99.5

HTC_VALUE = 600.0
HTC_MIN, HTC_MAX = 200.0, 1300.0

# SA hyperparams
SA_MAX_ITERS = 5000
SA_N_STARTS = 3
SA_ALPHA = 0.995
SA_FINAL_TEMP = 1e-3
SA_STEP_SCALE = 0.05

# SA on coarse control grid
CTRL_H, CTRL_W = 24, 20
PERTURB_K = 20

EPS = 1e-8

# loss weights
ALPHA_PRIOR = 0.3
LAMBDA_POWER_SUM = 1.0
LAMBDA_SMOOTH = 0.02

# score weights
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

def power_sum_W_np(P_wmm2: np.ndarray) -> float:
    H, W = P_wmm2.shape
    dx = DIE_W_MM / W
    dy = DIE_H_MM / H
    return float(P_wmm2.sum() * dx * dy)

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

def laplacian_torch(x_hw: torch.Tensor) -> torch.Tensor:
    x = x_hw.unsqueeze(0).unsqueeze(0)
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                     dtype=x.dtype, device=x.device).view(1,1,3,3)
    y = F.conv2d(x, k, padding=1)
    return y.squeeze(0).squeeze(0)

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

def denorm_T_np(x_np: np.ndarray) -> np.ndarray:
    return x_np * (T_MAX - T_MIN) + T_MIN

def denorm_T_torch(x_t: torch.Tensor) -> torch.Tensor:
    return x_t * (T_MAX_T - T_MIN_T) + T_MIN_T

def norm_P_np(x_np: np.ndarray) -> np.ndarray:
    return (x_np - P_MIN) / (P_MAX - P_MIN + EPS)

def denorm_P_torch(x_t: torch.Tensor) -> torch.Tensor:
    return x_t * (P_MAX_T - P_MIN_T) + P_MIN_T

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
# 4. Laplace prior
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

# ============================================================
# 5. SA (returns best + curve of the best start)
# ============================================================

def evaluate_on_file_sa(
    power_model,
    file_path,
    max_iters=5000,
    init_temp=1.0,
    final_temp=1e-3,
    alpha=0.995,
    step_scale=0.05,
    n_starts=3,
    target_accept=0.4,
    adapt_every=50,
    step_min=0.005,
    step_max=0.2,
    reheat_factor=1.5,
    seed=123
):
    """
    power_model(ctrl_grid_np) -> (loss_total_float, pack_dict)
      pack_dict MUST contain:
        loss_T, loss_prior, lambda_prior_eff, loss_sum, loss_smooth, loss_total, Psum_W
        T_MAE_norm, T_MSE_norm, Score_norm, T_MAE_denorm, T_MSE_denorm, Score_denorm
        Pn, Pphys, Tpred_n
    Return:
      best_ctrl, best_pack, best_curve_rows (length=max_iters)
    """
    rng = np.random.RandomState(seed)

    best_global_loss = float("inf")
    best_global_ctrl = None
    best_global_pack = None
    best_global_curve = None

    for start in range(n_starts):
        ctrl = rng.randn(CTRL_H, CTRL_W).astype(np.float32) * 0.01
        cur_loss, cur_pack = power_model(ctrl)

        best_loss = cur_loss
        best_ctrl = ctrl.copy()
        best_pack = cur_pack

        Tcur = float(init_temp)
        accept_cnt = 0
        trial_cnt = 0
        step = float(step_scale)

        curve_rows = []

        for it in range(max_iters):
            trial_cnt += 1

            prop = ctrl.copy()
            idx = rng.choice(CTRL_H * CTRL_W, size=min(PERTURB_K, CTRL_H * CTRL_W), replace=False)
            noise = rng.randn(idx.shape[0]).astype(np.float32) * step
            prop.flat[idx] += noise

            loss2, pack2 = power_model(prop)

            dE = loss2 - cur_loss
            if dE <= 0:
                accept = True
            else:
                accept = (rng.rand() < np.exp(-dE / max(Tcur, 1e-12)))

            if accept:
                ctrl = prop
                cur_loss = loss2
                cur_pack = pack2
                accept_cnt += 1

                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_ctrl = ctrl.copy()
                    best_pack = cur_pack

            # log CURRENT state each iter (treat iter as epoch)
            curve_rows.append([
                it,  # epoch
                cur_pack["loss_T"],
                cur_pack["T_MAE_norm"],
                cur_pack["T_MSE_norm"],
                cur_pack["Score_norm"],
                cur_pack["T_MAE_denorm_C"],
                cur_pack["T_MSE_denorm_C2"],
                cur_pack["Score_denorm"],
                cur_pack["loss_prior"],
                cur_pack["lambda_prior_eff"],
                cur_pack["loss_sum"],
                cur_pack["loss_smooth"],
                cur_pack["loss_total"],
                cur_pack["Psum_W"],
            ])

            # adapt step
            if (it + 1) % adapt_every == 0:
                acc_rate = accept_cnt / max(trial_cnt, 1)
                if acc_rate < target_accept:
                    step *= 0.7
                else:
                    step *= 1.3
                step = float(np.clip(step, step_min, step_max))
                accept_cnt = 0
                trial_cnt = 0

            # cool down + optional reheat
            Tcur *= alpha
            if Tcur < final_temp:
                Tcur = final_temp * reheat_factor

        print(f"[SA] {file_path} start={start+1}/{n_starts} best_loss={best_loss:.6e}")

        if best_loss < best_global_loss:
            best_global_loss = best_loss
            best_global_ctrl = best_ctrl.copy()
            best_global_pack = best_pack
            best_global_curve = curve_rows  # curve of this (winning) start

    return best_global_ctrl, best_global_pack, best_global_curve

# ============================================================
# 6. Inversion loop using SA
# ============================================================

print("[3/4] Start inversion (SA) ...")

TOTAL_POWER_T = torch.tensor(TOTAL_POWER_W, dtype=torch.float32, device=device)
HTC_NORM = norm_HTC_value(HTC_VALUE)
mse_fn = nn.MSELoss()

for wl in workloads:
    print(f"\n========== WORKLOAD: {wl} ==========")
    wl_out = os.path.join(OUT_ROOT, wl)
    os.makedirs(wl_out, exist_ok=True)

    summary_rows = []

    for f in range(FRAME_START, FRAME_END):
        Tf0, Tf1 = t_file(wl, f), t_file(wl, f + 1)
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
        dx = DIE_W_MM / W
        dy = DIE_H_MM / H
        dx_t = torch.tensor(dx, dtype=torch.float32, device=device)
        dy_t = torch.tensor(dy, dtype=torch.float32, device=device)

        # Laplace prior (W/mm^2)
        P_lap = laplace_prior_from_T(T0).astype(np.float32)
        P_lap_sum = power_sum_W_np(P_lap)

        pd.DataFrame(P_lap).to_csv(os.path.join(frame_dir, "P_prior_laplace.csv"),
                                   header=False, index=False)

        # optional true power
        P_true = None
        if os.path.exists(Pf0):
            try:
                P_true = np.loadtxt(Pf0, delimiter=",").astype(np.float32)
                pd.DataFrame(P_true).to_csv(os.path.join(frame_dir, "P_true_loaded.csv"),
                                            header=False, index=False)
            except Exception:
                P_true = None

        # tensors (normalized temperature)
        Tn = torch.tensor(norm_T_np(T0), dtype=torch.float32, device=device)
        Tgt = torch.tensor(norm_T_np(T1), dtype=torch.float32, device=device)
        htc_map = torch.full_like(Tn, HTC_NORM, dtype=torch.float32)

        # prior in normalized P-channel space (for UNet input)
        P_prior_n = norm_P_np(P_lap).astype(np.float32)
        P_prior_n_t = torch.tensor(P_prior_n, dtype=torch.float32, device=device)

        # ---------- define SA objective ----------
        def power_model(ctrl_grid_np: np.ndarray):
            # upsample ctrl to HxW
            ctrl_up = cv2.resize(ctrl_grid_np, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            S_map = torch.tensor(ctrl_up, dtype=torch.float32, device=device)

            # multiplicative correction in normalized space
            Pn = torch.clamp(P_prior_n_t * F.softplus(S_map), 0.0, 1.0)

            x = torch.stack([Tn, Pn, htc_map]).unsqueeze(0)

            with torch.no_grad():
                Tpred = model(x).squeeze(0).squeeze(0)

                # normalized metrics
                diff_n = Tpred - Tgt
                mae_n = diff_n.abs().mean()
                mse_n = (diff_n ** 2).mean()
                score_n = W_MSE * mse_n + W_MAE * mae_n

                # denormalized metrics (degC)
                Tpred_real = denorm_T_torch(Tpred)
                Tgt_real = denorm_T_torch(Tgt)
                diff_r = Tpred_real - Tgt_real
                mae_r = diff_r.abs().mean()
                mse_r = (diff_r ** 2).mean()
                score_r = W_MSE * mse_r + W_MAE * mae_r

            loss_T = float(mse_fn(Tpred, Tgt).item())

            # prior term (S^2 on coarse ctrl grid)
            loss_prior = float((ctrl_grid_np ** 2).mean())

            # power sum constraint in physical unit
            Pphys = denorm_P_torch(Pn)
            Psum = (Pphys.sum() * dx_t * dy_t)
            loss_sum = float(((Psum - TOTAL_POWER_T) ** 2).item())

            # smoothness
            lapP = laplacian_torch(Pphys)
            loss_smooth = float(((lapP ** 2).mean()).item())

            # adaptive lambda prior
            lambda_prior_eff = float(ALPHA_PRIOR * loss_T / (loss_prior + EPS))

            loss_total = float(loss_T + lambda_prior_eff * loss_prior +
                               LAMBDA_POWER_SUM * loss_sum +
                               LAMBDA_SMOOTH * loss_smooth)

            pack = {
                "Pn": Pn.detach().cpu().numpy().astype(np.float32),
                "Pphys": Pphys.detach().cpu().numpy().astype(np.float32),
                "Tpred_n": Tpred.detach().cpu().numpy().astype(np.float32),
                "Psum_W": float(Psum.item()),

                "loss_T": float(loss_T),
                "loss_prior": float(loss_prior),
                "lambda_prior_eff": float(lambda_prior_eff),
                "loss_sum": float(loss_sum),
                "loss_smooth": float(loss_smooth),
                "loss_total": float(loss_total),

                "T_MAE_norm": float(mae_n.item()),
                "T_MSE_norm": float(mse_n.item()),
                "Score_norm": float(score_n.item()),

                "T_MAE_denorm_C": float(mae_r.item()),
                "T_MSE_denorm_C2": float(mse_r.item()),
                "Score_denorm": float(score_r.item()),
            }
            return loss_total, pack

        # ---------- run SA ----------
        file_tag = f"{wl}_frame{f:03d}"
        best_ctrl, best_pack, best_curve = evaluate_on_file_sa(
            power_model=power_model,
            file_path=file_tag,
            max_iters=SA_MAX_ITERS,
            init_temp=1.0,
            final_temp=SA_FINAL_TEMP,
            alpha=SA_ALPHA,
            step_scale=SA_STEP_SCALE,
            n_starts=SA_N_STARTS,
            target_accept=0.4,
            adapt_every=50,
            step_min=0.005,
            step_max=0.2,
            reheat_factor=1.5,
            seed=123 + f  # different seed per frame (optional)
        )

        # ---------- save inverse_curve.csv (same schema as others) ----------
        pd.DataFrame(best_curve, columns=[
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

        # Also save best summary (optional)
        pd.DataFrame([{
            "best_loss_total": best_pack["loss_total"],
            "loss_T": best_pack["loss_T"],
            "loss_prior": best_pack["loss_prior"],
            "lambda_prior_eff": best_pack["lambda_prior_eff"],
            "loss_sum": best_pack["loss_sum"],
            "loss_smooth": best_pack["loss_smooth"],
            "Psum_W": best_pack["Psum_W"],
            "P_prior_sum_W": float(P_lap_sum),
            "CTRL_H": CTRL_H,
            "CTRL_W": CTRL_W,
            "max_iters": SA_MAX_ITERS,
            "n_starts": SA_N_STARTS
        }]).to_csv(os.path.join(frame_dir, "sa_best_summary.csv"), index=False)

        # ---------- final outputs ----------
        P_pred = best_pack["Pphys"]
        T_pred_n = best_pack["Tpred_n"]
        T_pred = denorm_T_np(T_pred_n).astype(np.float32)
        T_err = np.abs(T_pred - T1).astype(np.float32)

        pd.DataFrame(P_pred).to_csv(os.path.join(frame_dir, "P_inverted.csv"), header=False, index=False)
        pd.DataFrame(T_pred).to_csv(os.path.join(frame_dir, "T_pred.csv"), header=False, index=False)
        pd.DataFrame(T_err).to_csv(os.path.join(frame_dir, "T_abs_error.csv"), header=False, index=False)

        # power compare png
        if P_true is not None:
            P_err_true = np.abs(P_pred - P_true).astype(np.float32)
            pd.DataFrame(P_err_true).to_csv(os.path.join(frame_dir, "P_abs_error_vs_true.csv"),
                                            header=False, index=False)
            save_triplet_png(
                P_true, P_pred, P_err_true,
                ["Power True (loaded)", "Power Inverted (SA)", "Power Abs Error"],
                os.path.join(frame_dir, "power_true_vs_inverted.png"),
                suptitle=f"{wl} | Frame {f} Power (SA)"
            )
        else:
            P_err_prior = np.abs(P_pred - P_lap).astype(np.float32)
            pd.DataFrame(P_err_prior).to_csv(os.path.join(frame_dir, "P_abs_error_vs_prior.csv"),
                                             header=False, index=False)
            save_triplet_png(
                P_lap, P_pred, P_err_prior,
                ["Power Prior (Laplace)", "Power Inverted (SA)", "Abs Diff (Inv-Prior)"],
                os.path.join(frame_dir, "power_prior_vs_inverted.png"),
                suptitle=f"{wl} | Frame {f} Power (SA)"
            )

        # temp compare png
        save_triplet_png(
            T1, T_pred, T_err,
            ["Temp True T(t+1)", "Temp Pred T(t+1)", "Temp Abs Error"],
            os.path.join(frame_dir, "temp_compare.png"),
            suptitle=f"{wl} | Frame {f} Temperature (SA)"
        )

        # metrics
        T_mae = float(T_err.mean())
        T_mse = float((T_err ** 2).mean())

        P_diff_prior = float(np.abs(P_pred - P_lap).mean())
        P_diff_prior_mse = float(((P_pred - P_lap) ** 2).mean())

        if P_true is not None:
            P_err = np.abs(P_pred - P_true)
            P_mae = float(P_err.mean())
            P_mse = float((P_err ** 2).mean())
        else:
            P_mae, P_mse = np.nan, np.nan

        summary_rows.append([
            f,
            float(P_lap_sum),
            float(best_pack["Psum_W"]),
            float(P_diff_prior),
            float(P_diff_prior_mse),
            float(P_mae), float(P_mse),
            float(T_mae), float(T_mse),
            float(best_pack["loss_total"])
        ])

        print(f"[SA DONE] {wl} frame {f} | Psum={best_pack['Psum_W']:.4f}W | T_MAE={T_mae:.4f} | loss={best_pack['loss_total']:.3e}")

    # per-workload summary
    df = pd.DataFrame(summary_rows, columns=[
        "frame",
        "P_prior_sum_W",
        "P_inverted_sum_W(denormP)",
        "mean_abs_diff_inv_prior",
        "mse_diff_inv_prior",
        "P_MAE_vs_true",
        "P_MSE_vs_true",
        "T_MAE",
        "T_MSE",
        "loss_total_best",
    ])
    if len(df) > 0:
        df.loc["mean"] = ["mean"] + [df[c].mean() for c in df.columns[1:]]
    df.to_csv(os.path.join(wl_out, "summary.csv"), index=False)

print("[4/4] DONE. Saved to:", OUT_ROOT)
