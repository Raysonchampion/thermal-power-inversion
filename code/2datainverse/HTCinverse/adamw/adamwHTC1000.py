import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv

# Define UNet
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
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = nn.functional.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = nn.functional.interpolate(d1, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.conv_final(d1)

# HTC Model
class HTCModel(nn.Module):
    def __init__(self, init_htc_value=750.0, unet_model=None):
        super(HTCModel, self).__init__()
        self.htc_min, self.htc_max = 200, 1300
        self.htc_normalized = nn.Parameter(torch.tensor((init_htc_value - self.htc_min) / (self.htc_max - self.htc_min)))
        self.unet = unet_model

    def forward(self, temp_input, power_input):
        temp_input = temp_input.to(self.htc_normalized.device)
        power_input = power_input.to(self.htc_normalized.device)
        htc_value = self.htc_normalized
        htc_input = htc_value.expand(6, 9).unsqueeze(0)
        inputs = torch.stack([temp_input, power_input, htc_input], dim=1)
        output_temp = self.unet(inputs)
        return output_temp, htc_value * (self.htc_max - self.htc_min) + self.htc_min

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path, header=None).values
    temp_min, temp_max = 80.89, 127.9973
    power_min, power_max = 0.1, 0.6
    temp_input = (torch.tensor(data[:, :9], dtype=torch.float32) - temp_min) / (temp_max - temp_min)
    power_input = (torch.tensor(data[:, 9:18], dtype=torch.float32) - power_min) / (power_max - power_min)
    target_temp = (torch.tensor(data[:, 27:36], dtype=torch.float32) - temp_min) / (temp_max - temp_min)
    return temp_input, power_input, target_temp

# Save loss log and plot

def save_loss_log(loss_log, folder_name):
    os.makedirs("loss_csv1000", exist_ok=True)
    os.makedirs("loss_plot1000", exist_ok=True)
    cleaned_name = folder_name.replace("3/", "").replace("3-", "")
    csv_path = os.path.join("loss_csv1000", f'loss_log_{cleaned_name}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss', 'MSE', 'MAE'])
        writer.writerows(loss_log)

    epochs = [row[0] for row in loss_log]
    loss_vals = [row[1] for row in loss_log]
    mse_vals = [row[2] for row in loss_log]
    mae_vals = [row[3] for row in loss_log]

    plt.figure()
    plt.plot(epochs, loss_vals, label='Loss')
    plt.plot(epochs, mse_vals, label='MSE')
    plt.plot(epochs, mae_vals, label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Loss, MSE, MAE - {cleaned_name}')
    plt.legend()
    plt.savefig(os.path.join("loss_plot1000", f'loss_plot_{cleaned_name}.png'))
    plt.close()

# Save HTC summary

def append_htc_summary(folder_name, predicted_htc, true_htc, htc_error, stable_time, train_time):
    output_csv = 'htc_summary1000.csv'
    cleaned_name = folder_name.replace("3/", "").replace("3-", "")
    file_exists = os.path.exists(output_csv)
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Folder', 'Predicted HTC', 'True HTC', 'HTC Error', 'Stability Time (s)', 'Training Time (s)'])
        writer.writerow([cleaned_name, predicted_htc, true_htc, htc_error, stable_time, train_time])
def train_htc_model(htc_model, data_folder, epochs=200, htc_lr=0.1):
    optimizer = optim.AdamW([htc_model.htc_normalized], lr=htc_lr)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5)

    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    loss_log = []
    min_loss = float('inf')
    stable_time = None
    stable_count = 0
    stable_threshold = 1e-4
    stable_N = 20

    start_time = time.time()

    for epoch in range(epochs):
        total_loss, total_mae, total_mse = 0, 0, 0
        for file_path in file_paths:
            temp_input, power_input, target_temp = load_data(file_path)
            temp_input = temp_input.unsqueeze(0).to(device)
            power_input = power_input.unsqueeze(0).to(device)
            target_temp = target_temp.unsqueeze(0).to(device)

            optimizer.zero_grad()
            predicted_temp, _ = htc_model(temp_input, power_input)
            predicted_temp = predicted_temp.squeeze(0)

            loss_mse = criterion_mse(predicted_temp, target_temp)
            loss_mae = criterion_mae(predicted_temp, target_temp)
            loss = 0.7 * loss_mse + 0.3 * loss_mae
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mae += loss_mae.item()
            total_mse += loss_mse.item()

        avg_loss = total_loss / len(file_paths)
        avg_mae = total_mae / len(file_paths)
        avg_mse = total_mse / len(file_paths)
        scheduler.step(avg_loss)

        loss_log.append([epoch + 1, avg_loss, avg_mse, avg_mae])
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}")

        if abs(avg_loss - min_loss) < stable_threshold:
            stable_count += 1
            if stable_count >= stable_N and stable_time is None:
                stable_time = time.time() - start_time
        else:
            stable_count = 0
        min_loss = min(min_loss, avg_loss)

    training_time = time.time() - start_time
    final_htc_value = htc_model.htc_normalized.item() * (htc_model.htc_max - htc_model.htc_min) + htc_model.htc_min
    return final_htc_value, loss_log, stable_time, training_time

# Save heatmaps

def save_heatmaps(htc_model, data_folder, htc_true, folder_name):
    os.makedirs("heatmaps1000", exist_ok=True)
    temp_min, temp_max = 80.89, 127.9973
    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    for file_path in file_paths:
        temp_input, power_input, target_temp = load_data(file_path)
        temp_input = temp_input.unsqueeze(0).to(device)
        power_input = power_input.unsqueeze(0).to(device)
        target_temp = target_temp.unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_temp, predicted_htc = htc_model(temp_input, power_input)
            predicted_temp = predicted_temp.squeeze().cpu() * (temp_max - temp_min) + temp_min
            true_temp = target_temp.squeeze().cpu() * (temp_max - temp_min) + temp_min
            diff = torch.abs(predicted_temp - true_temp)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.heatmap(predicted_temp.numpy(), annot=True, cmap="viridis", ax=axs[0], fmt=".2f")
        axs[0].set_title("Predicted Temperature")
        sns.heatmap(true_temp.numpy(), annot=True, cmap="viridis", ax=axs[1], fmt=".2f")
        axs[1].set_title("True Temperature")
        sns.heatmap(diff.numpy(), annot=True, cmap="viridis", ax=axs[2], fmt=".2f")
        axs[2].set_title("Error")
        basename = os.path.splitext(os.path.basename(file_path))[0]
        cleaned_name = folder_name.replace("3/", "").replace("3-", "")
        folder_path = os.path.join("heatmaps1000", cleaned_name)
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, f"{basename}_heatmap.png"))
        plt.close()
# Load best UNet model
def freeze_unet_parameters(unet_model):
    for param in unet_model.parameters():
        param.requires_grad = False

def load_best_unet_model():
    model = ImprovedUNet()
    model.load_state_dict(torch.load('best_model1.pth'))
    model.eval()
    freeze_unet_parameters(model)
    return model

# Main
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_unet_model = load_best_unet_model().to(device)
    htc_model = HTCModel(init_htc_value=750.0, unet_model=best_unet_model).to(device)

    for i in range(1, 31):
        for j in range(1, 21):
            folder_name = f'3/{i}-{j}-HTC1000'
            if not os.path.exists(folder_name):
                print(f"Folder not found: {folder_name}, skipping.")
                continue
            print(f"Training on folder: {folder_name}")
            final_htc_value, loss_log, stable_time, train_time = train_htc_model(htc_model, folder_name)
            save_loss_log(loss_log, folder_name)
            htc_error = abs(final_htc_value - 1000)
            append_htc_summary(folder_name, final_htc_value, 1000, htc_error, stable_time, train_time)
            save_heatmaps(htc_model, folder_name, 1000, folder_name)