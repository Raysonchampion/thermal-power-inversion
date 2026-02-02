import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import math

# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Constants (normalization)
# =====================================================
T_MIN, T_MAX = 80.89, 127.9973
P_MIN, P_MAX = 0.1, 0.6

def denorm_T(x):
    return x * (T_MAX - T_MIN) + T_MIN


# =====================================================
# UNet (must match best_model1.pth)
# =====================================================
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        def conv_block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)

        self.conv_final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(e3)

        d3 = self.upconv3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = nn.functional.interpolate(d3, e3.shape[2:], mode="bilinear")
        d3 = self.dec3(torch.cat([d3, e3], 1))

        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = nn.functional.interpolate(d2, e2.shape[2:], mode="bilinear")
        d2 = self.dec2(torch.cat([d2, e2], 1))

        d1 = self.upconv1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = nn.functional.interpolate(d1, e1.shape[2:], mode="bilinear")
        d1 = self.dec1(torch.cat([d1, e1], 1))

        return self.conv_final(d1)  # [1,1,6,9]


# =====================================================
# HTC Model (single scalar HTC)
# =====================================================
class HTCModel(nn.Module):
    def __init__(self, init_htc=750.0, unet=None):
        super().__init__()
        self.htc_min, self.htc_max = 200.0, 1300.0
        self.htc_norm = torch.tensor(
            (init_htc - self.htc_min) / (self.htc_max - self.htc_min),
            device=device
        )
        self.unet = unet

    def forward(self, T, P):
        htc_map = self.htc_norm.expand(6, 9).unsqueeze(0)
        x = torch.stack([T, P, htc_map], dim=1)
        pred_T = self.unet(x)
        real_htc = self.htc_norm * (self.htc_max - self.htc_min) + self.htc_min
        return pred_T, real_htc


# =====================================================
# Data loader
# =====================================================
def load_data(csv_path):
    data = pd.read_csv(csv_path, header=None).values
    T = (torch.tensor(data[:, :9]) - T_MIN) / (T_MAX - T_MIN)
    P = (torch.tensor(data[:, 9:18]) - P_MIN) / (P_MAX - P_MIN)
    Tgt = (torch.tensor(data[:, 27:36]) - T_MIN) / (T_MAX - T_MIN)
    return T.float(), P.float(), Tgt.float()


# =====================================================
# Simulated Annealing for HTC
# =====================================================
def sa_htc(folder, htc_model,
           max_iter=3000,
           T0=1.0,
           Tend=1e-3,
           alpha=0.995,
           step=0.05):

    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    mse_fn, mae_fn = nn.MSELoss(), nn.L1Loss()

    def eval_loss(htc_val):
        htc_model.htc_norm = htc_val.clamp(0, 1)
        Lm, La = 0.0, 0.0
        with torch.no_grad():
            for f in files:
                T, P, Tgt = load_data(os.path.join(folder, f))
                T, P, Tgt = T.unsqueeze(0).to(device), P.unsqueeze(0).to(device), Tgt.unsqueeze(0).to(device)
                pred, _ = htc_model(T, P)
                pred = pred.squeeze(1)

                pred_real = denorm_T(pred)
                tgt_real = denorm_T(Tgt)

                Lm += mse_fn(pred_real, tgt_real).item()
                La += mae_fn(pred_real, tgt_real).item()

        Lm /= len(files)
        La /= len(files)
        return 0.7 * Lm + 0.3 * La, Lm, La

    cur = torch.rand(1, device=device)
    cur_loss, cur_mse, cur_mae = eval_loss(cur)
    best = cur.clone()
    best_loss = cur_loss

    log = []
    T = T0
    start = time.time()

    for it in range(1, max_iter + 1):
        prop = (cur + step * torch.randn_like(cur)).clamp(0, 1)
        new_loss, new_mse, new_mae = eval_loss(prop)
        delta = new_loss - cur_loss

        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            cur, cur_loss = prop, new_loss
            if new_loss < best_loss:
                best, best_loss = prop.clone(), new_loss

        log.append([it, T, cur_loss, new_mse, new_mae, cur.item()])
        T *= alpha
        if T < Tend:
            break

    htc_real = best.item() * (htc_model.htc_max - htc_model.htc_min) + htc_model.htc_min
    return htc_real, best.clone(), log, time.time() - start


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":

    BASE = "SAHTC700"
    os.makedirs(BASE, exist_ok=True)

    for sub in ["sa_traj700", "loss_csv700", "loss_plot700", "temp_csv700", "heatmaps700"]:
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)

    unet = ImprovedUNet().to(device)
    unet.load_state_dict(torch.load("best_model1.pth", map_location=device))
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    htc_model = HTCModel(init_htc=750.0, unet=unet)

    summary_csv = os.path.join(BASE, "htc_summary700_sa.csv")
    if os.path.exists(summary_csv):
        os.remove(summary_csv)

    for i in range(1, 31):
        for j in range(1, 21):
            folder = f"3/{i}-{j}-HTC700"
            if not os.path.exists(folder):
                continue

            print(f"\n[SA] {folder}")
            htc_pred, best_norm, log, t = sa_htc(folder, htc_model)
            case = folder.replace("3/", "")

            # ---- trajectory & loss ----
            df = pd.DataFrame(log, columns=["iter","T","total_loss","mse","mae","htc_norm"])
            df.to_csv(f"{BASE}/sa_traj700/sa_traj_{case}.csv", index=False)
            df.to_csv(f"{BASE}/loss_csv700/loss_log_{case}.csv", index=False)

            plt.figure(figsize=(8,6))
            plt.plot(df["iter"], df["total_loss"], label="Total")
            plt.plot(df["iter"], df["mse"], label="MSE")
            plt.plot(df["iter"], df["mae"], label="MAE")
            plt.legend(); plt.grid()
            plt.savefig(f"{BASE}/loss_plot700/loss_plot_{case}.png")
            plt.close()

            # ---- HTC summary ----
            with open(summary_csv, "a", newline="") as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(["Folder","Pred_HTC","True_HTC","Error","Time_s"])
                w.writerow([case, htc_pred, 700, abs(htc_pred-700), t])

            # ---- temperature prediction for ALL time CSVs ----
            os.makedirs(f"{BASE}/temp_csv700/{case}", exist_ok=True)
            os.makedirs(f"{BASE}/heatmaps700/{case}", exist_ok=True)

            for csv_name in sorted(os.listdir(folder)):
                if not csv_name.endswith(".csv"):
                    continue

                T, P, Tgt = load_data(os.path.join(folder, csv_name))
                T, P, Tgt = T.unsqueeze(0).to(device), P.unsqueeze(0).to(device), Tgt.unsqueeze(0).to(device)

                with torch.no_grad():
                    htc_model.htc_norm = best_norm
                    pred, _ = htc_model(T, P)
                pred = pred.squeeze(1)

                pred_real = denorm_T(pred).squeeze(0).cpu()
                true_real = denorm_T(Tgt).squeeze(0).cpu()
                diff = torch.abs(pred_real - true_real)

                rows = []
                for r in range(6):
                    for c in range(9):
                        rows.append([r, c,
                                     float(true_real[r,c]),
                                     float(pred_real[r,c]),
                                     float(diff[r,c])])

                pd.DataFrame(rows, columns=["row","col","true_temp","pred_temp","abs_error"]) \
                    .to_csv(f"{BASE}/temp_csv700/{case}/{csv_name}", index=False)

                fig, axs = plt.subplots(1,3,figsize=(18,5))
                sns.heatmap(pred_real, ax=axs[0], cmap="viridis", annot=True, fmt=".2f")
                sns.heatmap(true_real, ax=axs[1], cmap="viridis", annot=True, fmt=".2f")
                sns.heatmap(diff, ax=axs[2], cmap="viridis", annot=True, fmt=".2f")
                plt.tight_layout()
                plt.savefig(f"{BASE}/heatmaps700/{case}/{csv_name.replace('.csv','.png')}")
                plt.close()

    print("\n=== SA HTC700 finished ===")
