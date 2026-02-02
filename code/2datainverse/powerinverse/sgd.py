import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import glob

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
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(e3)
        d3 = self.upconv3(b)
        if d3.size()[2:] != e3.size()[2:]:
            d3 = nn.functional.interpolate(d3, size=e3.size()[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat((d3, e3), dim=1))
        d2 = self.upconv2(d3)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = nn.functional.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = self.upconv1(d2)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = nn.functional.interpolate(d1, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        return self.conv_final(d1)

class POWERModel(nn.Module):
    def __init__(self, unet_model=None):
        super(POWERModel, self).__init__()
        self.power_normalized = nn.Parameter(torch.rand(6, 9))
        self.unet = unet_model

    def forward(self, temp_input, htc_input):
        power_matrix = self.power_normalized.unsqueeze(0)
        inputs = torch.stack((temp_input, power_matrix, htc_input), dim=1)
        output_temp = self.unet(inputs).squeeze(0)
        return output_temp, power_matrix

def load_data(file_path):
    data = pd.read_csv(file_path, header=None).values
    temp_input = torch.tensor(data[:, :9], dtype=torch.float32)
    htc_input = torch.tensor(data[:, 18:27], dtype=torch.float32)
    target_temp = torch.tensor(data[:, 27:36], dtype=torch.float32)
    real_power = torch.tensor(data[:, 9:18], dtype=torch.float32).unsqueeze(0)
    return temp_input, htc_input, target_temp, real_power

def evaluate_on_file(power_model, file_path, device, epochs=500, lr=0.01, loss_plot_dir="test_loss_plots", epoch_csv_dir="epoch_csv", heatmap_dir="heatmaps"):
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]
    temp_input, htc_input, target_temp, real_power = load_data(file_path)
    temp_input = temp_input.unsqueeze(0).to(device)
    htc_input = htc_input.unsqueeze(0).to(device)
    target_temp = target_temp.unsqueeze(0).to(device)
    real_power = real_power.to(device)

    optimizer = optim.SGD([power_model.power_normalized], lr=lr)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    total_losses = []
    mse_losses = []
    mae_losses = []
    epoch_logs = []

    epsilon = 1e-5
    stable_counter = 0
    stable_N = 20
    stability_time = None
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted_temp, predicted_power = power_model(temp_input, htc_input)
        loss_mse = criterion_mse(predicted_temp, target_temp)
        loss_mae = criterion_mae(predicted_temp, target_temp)
        loss = 0.6 * loss_mse + 0.4 * loss_mae
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predicted_temp, predicted_power = power_model(temp_input, htc_input)
            loss_mse = criterion_mse(predicted_temp, target_temp)
            loss_mae = criterion_mae(predicted_temp, target_temp)

        total_losses.append(loss.item())
        mse_losses.append(loss_mse.item())
        mae_losses.append(loss_mae.item())
        epoch_logs.append({"Epoch": epoch+1, "Total Loss": loss.item(), "MSE": loss_mse.item(), "MAE": loss_mae.item()})

        if epoch > 0 and abs(total_losses[-1] - total_losses[-2]) < epsilon:
            stable_counter += 1
            if stable_counter >= stable_N and stability_time is None:
                stability_time = time.time() - start_time
        else:
            stable_counter = 0

    if stability_time is None:
        stability_time = time.time() - start_time

    with torch.no_grad():
        predicted_temp, predicted_power = power_model(temp_input, htc_input)

    temp_diff = torch.abs(predicted_temp - target_temp).cpu()
    power_diff = torch.abs(predicted_power - real_power).cpu()

    os.makedirs(loss_plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(total_losses, label="Total Loss", linewidth=2)
    plt.plot(mse_losses, label="MSE Loss", linestyle='--')
    plt.plot(mae_losses, label="MAE Loss", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves - {file_prefix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(loss_plot_dir, f"{file_prefix}_loss_curve.png"))
    plt.close()

    # 保存 epoch 曲线
    os.makedirs(epoch_csv_dir, exist_ok=True)
    pd.DataFrame(epoch_logs).to_csv(os.path.join(epoch_csv_dir, f"{file_prefix}_epoch_loss.csv"), index=False)

    # 保存热力图
    os.makedirs(heatmap_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    sns.heatmap(predicted_temp.squeeze(0).cpu().detach().numpy(), annot=True, cmap="viridis", ax=axs[0, 0])
    axs[0, 0].set_title("Predicted Temperature")
    sns.heatmap(target_temp.squeeze(0).cpu().numpy(), annot=True, cmap="viridis", ax=axs[0, 1])
    axs[0, 1].set_title("True Temperature")
    sns.heatmap(temp_diff.squeeze(0).numpy(), annot=True, cmap="viridis", ax=axs[0, 2])
    axs[0, 2].set_title("Temperature Error")
    sns.heatmap(predicted_power.squeeze(0).detach().cpu().numpy(), annot=True, cmap="coolwarm", ax=axs[1, 0])
    axs[1, 0].set_title("Predicted Power")
    sns.heatmap(real_power.squeeze(0).cpu().numpy(), annot=True, cmap="coolwarm", ax=axs[1, 1])
    axs[1, 1].set_title("True Power")
    sns.heatmap(power_diff.squeeze(0).detach().numpy(), annot=True, cmap="coolwarm", ax=axs[1, 2])
    axs[1, 2].set_title("Power Error")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, f"{file_prefix}_heatmap.png"))
    plt.close()

    result = {
        "File Name": file_prefix,
        "Temp Diff Max": temp_diff.max().item(),
        "Temp Diff Mean": temp_diff.mean().item(),
        "Power Diff Max": power_diff.max().item(),
        "Power Diff Mean": power_diff.mean().item(),
        "MSE": mse_losses[-1],
        "MAE": mae_losses[-1],
        "Total Loss": total_losses[-1],
        "Stable Time (s)": stability_time,
        "Total Time (s)": time.time() - start_time
    }
    return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_best_unet_model(weight_path='best_model.pth'):
    model = ImprovedUNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

unet_model = load_best_unet_model().to(device)
power_model = POWERModel(unet_model=unet_model).to(device)

test_folder = "test"
test_files = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
output_csv = "test_results_summary.csv"
if os.path.exists(output_csv):
    os.remove(output_csv)

for file_path in test_files:
    print(f"Evaluating: {file_path}")
    result = evaluate_on_file(power_model, file_path, device)
    pd.DataFrame([result]).to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

print("test_results_summary.csv 已保存，loss 图保存在 test_loss_plots，epoch loss 保存于 epoch_csv，热图在 heatmaps 文件夹中")
