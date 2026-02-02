import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------- Make folder ---------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# --------------------- UNet model ---------------------
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ImprovedUNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
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
        if d3.size()[2:] != e3.size()[2:]:
            d3 = nn.functional.interpolate(d3, size=e3.size()[2:], mode='bilinear')
        d3 = self.dec3(torch.cat((d3, e3), 1))

        d2 = self.upconv2(d3)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = nn.functional.interpolate(d2, size=e2.size()[2:], mode='bilinear')
        d2 = self.dec2(torch.cat((d2, e2), 1))

        d1 = self.upconv1(d2)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = nn.functional.interpolate(d1, size=e1.size()[2:], mode='bilinear')
        d1 = self.dec1(torch.cat((d1, e1), 1))

        return self.conv_final(d1)


# --------------------- POWER Model ---------------------
class POWERModel(nn.Module):
    def __init__(self, unet_model=None):
        super(POWERModel, self).__init__()
        self.power_normalized = nn.Parameter(torch.rand(6, 9))
        self.unet = unet_model

    def forward(self, temp_input, htc_input):
        power_matrix = self.power_normalized.unsqueeze(0)
        x = torch.stack((temp_input, power_matrix, htc_input), dim=1)
        out = self.unet(x).squeeze(0)
        return out, power_matrix


# --------------------- Load CSV ---------------------
def load_data(file_path):
    data = pd.read_csv(file_path, header=None).values

    temp_input = torch.tensor(data[:, :9], dtype=torch.float32)
    real_power = torch.tensor(data[:, 9:18], dtype=torch.float32).unsqueeze(0)
    htc_input = torch.tensor(data[:, 18:27], dtype=torch.float32)
    target_temp = torch.tensor(data[:, 27:36], dtype=torch.float32)

    return temp_input, htc_input, target_temp, real_power


# --------------------- SA Evaluate ---------------------
def evaluate_on_file_sa(
    power_model, file_path, device,
    max_iters=5000,
    init_temp=None,
    final_temp=1e-3,
    alpha=0.995,
    step_scale=0.05,
    n_starts=5,
    target_accept=0.4,
    adapt_every=50,
    step_min=0.005,
    step_max=0.2,
    clamp_min=0.0,
    clamp_max=1.0,
    reheat_factor=1.5
):

    file_prefix = os.path.splitext(os.path.basename(file_path))[0]

    # ---------- Output directories ----------
    base = "SAA"
    ensure_dir(base)
    dirs = {
        "loss": f"{base}/loss_plots",
        "epoch": f"{base}/epoch_csv",
        "heatmap": f"{base}/heatmaps",
        "pred_temp": f"{base}/predictions_temp",
        "pred_power": f"{base}/predictions_power"
    }
    for d in dirs.values():
        ensure_dir(d)

    # ---------- Load data ----------
    temp_input, htc_input, target_temp, real_power = load_data(file_path)
    temp_input = temp_input.unsqueeze(0).to(device)
    htc_input = htc_input.unsqueeze(0).to(device)
    target_temp = target_temp.unsqueeze(0).to(device)
    real_power = real_power.to(device)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # ---------- SA variables ----------
    current_power = power_model.power_normalized.data.clone()
    best_power = current_power.clone()

    def calc_loss(power_tensor):
        with torch.no_grad():
            power_model.power_normalized.data = power_tensor.to(device)
            pred_temp, _ = power_model(temp_input, htc_input)
            mse = criterion_mse(pred_temp, target_temp).item()
            mae = criterion_mae(pred_temp, target_temp).item()
            total = 0.6*mse + 0.4*mae
        return total, mse, mae

    current_total, current_mse, current_mae = calc_loss(current_power)
    best_total, best_mse, best_mae = current_total, current_mse, current_mae

    # ---------- Logs ----------
    epoch_logs = []
    total_losses, mse_losses, mae_losses = [], [], []

    # ---------- track total iter ----------
    iter_idx = 0
    start_time = time.time()

    # ---------- Multi-start SA ----------
    best_run_losses = None
    best_run_logs = None

    for start_id in range(n_starts):

        if start_id > 0:
            current_power = torch.rand_like(current_power)
            current_total, current_mse, current_mae = calc_loss(current_power)

        # Initialize temperature
        T = init_temp
        if T is None:
            samples = []
            for _ in range(20):
                s = torch.clamp(current_power + step_scale * torch.randn_like(current_power), clamp_min, clamp_max)
                nt, _, _ = calc_loss(s)
                d = nt - current_total
                if d > 0:
                    samples.append(d)
            T = 1.0 if len(samples) == 0 else float(np.median(samples)) / math.log(1.0/0.8)
        T0 = T

        accept_cnt = 0
        local_losses, local_mse, local_mae = [], [], []
        local_logs = []
        local_iter = 0  # ------------------ Local iter start from 1 ------------------

        while T > final_temp and local_iter < max_iters:
            iter_idx += 1
            local_iter += 1  

            proposal = torch.clamp(current_power + step_scale * torch.randn_like(current_power), clamp_min, clamp_max)
            new_total, new_mse, new_mae = calc_loss(proposal)
            delta = new_total - current_total

            acc_prob = math.exp(-delta / T) if delta > 0 else 1.0
            accept = (delta < 0) or (torch.rand(1).item() < acc_prob)

            if accept:
                current_power = proposal
                current_total, current_mse, current_mae = new_total, new_mse, new_mae
                accept_cnt += 1
                if current_total < best_total:
                    best_total = current_total
                    best_mse = current_mse
                    best_mae = current_mae
                    best_power = current_power.clone()
                    best_run_losses = (local_losses.copy(), local_mse.copy(), local_mae.copy())
                    best_run_logs = local_logs.copy()

            # -------------- Save local_iter instead of iter_idx --------------
            local_losses.append(current_total)
            local_mse.append(current_mse)
            local_mae.append(current_mae)
            local_logs.append([local_iter, T, current_total, current_mse, current_mae])

            # Adapt step
            if iter_idx % adapt_every == 0:
                rate = accept_cnt / adapt_every
                if rate < target_accept * 0.8:
                    step_scale = max(step_min, step_scale * 0.7)
                    T = min(T0, T * reheat_factor)
                elif rate > target_accept * 1.2:
                    step_scale = min(step_max, step_scale * 1.3)
                accept_cnt = 0

            T *= alpha

        if best_run_losses is None:
            best_run_losses = (local_losses, local_mse, local_mae)
            best_run_logs = local_logs

    # -------- outputs for best run ----------
    total_losses, mse_losses, mae_losses = best_run_losses
    epoch_logs = best_run_logs

    elapsed_time = time.time() - start_time

    # ---------- Predict with best power ----------
    with torch.no_grad():
        power_model.power_normalized.data = best_power.to(device)
        predicted_temp, predicted_power = power_model(temp_input, htc_input)

    temp_diff = torch.abs(predicted_temp - target_temp).cpu()
    power_diff = torch.abs(predicted_power - real_power).cpu()

    # ---------- Save Loss Plot ----------
    plt.figure(figsize=(10,6))
    plt.plot(total_losses, label="Total")
    plt.plot(mse_losses,  label="MSE")
    plt.plot(mae_losses,  label="MAE")
    plt.legend()
    plt.grid()
    plt.title(f"SA Loss - {file_prefix}")
    plt.savefig(f"{dirs['loss']}/{file_prefix}_loss_sa.png")
    plt.close()

    # ---------- Save Epoch CSV ----------
    pd.DataFrame(epoch_logs,
                 columns=["iter","T","total","mse","mae"]
    ).to_csv(f"{dirs['epoch']}/{file_prefix}_epoch_sa.csv", index=False)

    # ---------- Save Heatmaps ----------
    fig, axs = plt.subplots(2,3,figsize=(18,10))
    sns.heatmap(predicted_temp.squeeze().detach().cpu().numpy(),
            annot=True, cmap="viridis", ax=axs[0,0])
    sns.heatmap(target_temp.squeeze().detach().cpu().numpy(),
            annot=True, cmap="viridis", ax=axs[0,1])
    sns.heatmap(temp_diff.squeeze().detach().cpu().numpy(),
            annot=True, cmap="viridis", ax=axs[0,2])

    sns.heatmap(predicted_power.squeeze().detach().cpu().numpy(),
            annot=True, cmap="coolwarm", ax=axs[1,0])
    sns.heatmap(real_power.squeeze().detach().cpu().numpy(),
            annot=True, cmap="coolwarm", ax=axs[1,1])
    sns.heatmap(power_diff.squeeze().detach().cpu().numpy(),
            annot=True, cmap="coolwarm", ax=axs[1,2])

    plt.tight_layout()
    plt.savefig(f"{dirs['heatmap']}/{file_prefix}_heatmap_sa.png")
    plt.close()

    # ---------- Save Predictions ----------
    np.savetxt(f"{dirs['pred_temp']}/{file_prefix}_pred_temp.csv",
               predicted_temp.squeeze().detach().cpu().numpy(), delimiter=",")

    np.savetxt(f"{dirs['pred_power']}/{file_prefix}_pred_power.csv",
               predicted_power.squeeze().detach().cpu().numpy(), delimiter=",")

    np.savetxt(f"{dirs['pred_power']}/{file_prefix}_best_power_matrix.csv",
               best_power.detach().cpu().numpy(), delimiter=",")

    # ---------- Return results ----------
    return {
        "file": file_prefix,
        "temp_diff_max": temp_diff.max().item(),
        "temp_diff_mean": temp_diff.mean().item(),
        "power_diff_max": power_diff.max().item(),
        "power_diff_mean": power_diff.mean().item(),
        "best_total": best_total,
        "best_mse": best_mse,
        "best_mae": best_mae,
        "total_iters": iter_idx,    
        "time_sec": elapsed_time    
    }


# --------------------- Load UNet ---------------------
def load_best_unet(weight_path="best_model.pth"):
    model = ImprovedUNet(3,1)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


# --------------------- Main ---------------------
if __name__ == "__main__":

    unet_model = load_best_unet().to(device)
    power_model = POWERModel(unet_model).to(device)

    test_files = sorted(glob.glob(os.path.join("test", "*.csv")))

    summary_csv = "test_results_summary_sa.csv"
    if os.path.exists(summary_csv):
        os.remove(summary_csv)

    results = []

    for file_path in test_files:

        print(f"\n=== SA evaluating: {file_path} ===")
        with torch.no_grad():
            power_model.power_normalized.data = torch.rand(6,9).to(device)

        result = evaluate_on_file_sa(power_model, file_path, device)
        results.append(result)

        print(f"Finished {file_path}, time = {result['time_sec']:.2f} sec, iters = {result['total_iters']}")

        pd.DataFrame([result]).to_csv(
            summary_csv,
            mode="a",
            index=False,
            header=not os.path.exists(summary_csv)
        )

    print("\nSA all finished.")
