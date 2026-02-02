import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time


# 定义 UNet 模型
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


# HTC 学习模型
class HTCModel(nn.Module):
    def __init__(self, init_htc_value=750.0, unet_model=None):
        super(HTCModel, self).__init__()
        # HTC 范围定义
        self.htc_min, self.htc_max = 200, 1300
        # HTC 初始化归一化值
        self.htc_normalized = nn.Parameter(
            torch.tensor((init_htc_value - self.htc_min) / (self.htc_max - self.htc_min)))
        self.unet = unet_model  # 加载的 UNet 模型

    def forward(self, temp_input, power_input):
        # 反归一化 HTC 值并扩展为 6x9
        htc_value = self.htc_normalized
        htc_input = htc_value.expand(6, 9).unsqueeze(0)

        # 堆叠输入通道
        inputs = torch.stack([temp_input, power_input, htc_input], dim=1)

        # 通过 UNet 模型预测
        output_temp = self.unet(inputs)
        return output_temp, htc_value


# 数据加载函数
def load_data(file_path):
    data = pd.read_csv(file_path, header=None).values

    # 定义归一化参数
    temp_min, temp_max = 80.89, 127.9973
    power_min, power_max = 0.1, 0.6

    # 数据归一化
    temp_input = (torch.tensor(data[:, :9], dtype=torch.float32) - temp_min) / (temp_max - temp_min)
    power_input = (torch.tensor(data[:, 9:18], dtype=torch.float32) - power_min) / (power_max - power_min)
    target_temp = (torch.tensor(data[:, 27:36], dtype=torch.float32) - temp_min) / (temp_max - temp_min)
    return temp_input, power_input, target_temp


# 定义热力图绘制函数
def plot_and_save_heatmaps(predicted_temp, true_temp, htc_pred, htc_true, file_name_prefix, output_dir='output'):
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 去掉 batch 维度，确保传递的是 2D 数据
    predicted_temp = predicted_temp.squeeze(0)
    true_temp = true_temp.squeeze(0)

    # 定义反归一化参数
    temp_min, temp_max = 80.89, 127.9973

    # 反归一化温度
    predicted_temp = predicted_temp * (temp_max - temp_min) + temp_min
    true_temp = true_temp * (temp_max - temp_min) + temp_min

    # 计算温度误差
    temp_diff = torch.abs(predicted_temp - true_temp)

    # 创建一个 1x3 的图表布局
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 预测温度热力图
    sns.heatmap(predicted_temp.cpu().numpy(), annot=True, cmap="viridis", ax=axs[0], fmt=".2f")
    axs[0].set_title("Predicted Temperature")

    # 真实温度热力图
    sns.heatmap(true_temp.cpu().numpy(), annot=True, cmap="viridis", ax=axs[1], fmt=".2f")
    axs[1].set_title("True Temperature")

    # 温度误差热力图
    sns.heatmap(temp_diff.cpu().numpy(), annot=True, cmap="viridis", ax=axs[2], fmt=".2f")
    axs[2].set_title("Temperature Difference (Error)")

    # 保存温度预测热力图
    temp_heatmap_file = os.path.join(output_dir, f'{file_name_prefix}_temperature.png')
    plt.savefig(temp_heatmap_file)
    print(f"温度预测结果和误差图已保存到 {temp_heatmap_file}")
    plt.close(fig)

    # 绘制 HTC 的预测、真实值及误差
    fig, ax = plt.subplots(figsize=(6, 5))

    # 计算 HTC 的误差
    htc_diff = abs(htc_pred - htc_true)

    # 画出预测 HTC、真实 HTC 和误差
    ax.bar(['Predicted HTC', 'True HTC', 'HTC Error'], [htc_pred.item(), htc_true, htc_diff.item()],
           color=['blue', 'green', 'red'])
    ax.set_title('HTC Prediction vs True')

    # 保存 HTC 图像
    htc_file = os.path.join(output_dir, f'{file_name_prefix}_htc.png')
    plt.savefig(htc_file)
    print(f"HTC 预测和误差图已保存到 {htc_file}")
    plt.close(fig)


# 保存 HTC 预测结果到 CSV 文件
def save_htc_predictions_to_csv(predictions, output_csv_file):
    # 创建存放 CSV 文件的目录（如果不存在）
    output_dir = os.path.dirname(output_csv_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将预测结果保存为 CSV 格式
    df = pd.DataFrame(predictions, columns=["File Name", "Predicted HTC", "True HTC", "HTC Error"])
    df.to_csv(output_csv_file, index=False)
    print(f"HTC 预测结果已保存到 {output_csv_file}")


# 保存 HTC 预测结果和生成 CSV 文件
def save_predictions_and_csv(model, data_folder, htc_true, final_htc_value, output_csv_file):
    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    htc_results = []

    for file_path in file_paths:
        temp_input, power_input, true_temp = load_data(file_path)

        # 增加 batch 维度并将数据移动到 GPU
        temp_input = temp_input.unsqueeze(0).to(device)
        power_input = power_input.unsqueeze(0).to(device)

        # 进行 HTC 预测
        with torch.no_grad():
            _, predicted_htc = model(temp_input, power_input)  # 只关心 HTC 值
        # 使用 final_htc_value 而不是模型预测的 HTC
        predicted_htc = final_htc_value  # 将最终的 HTC 值作为预测值
        # 计算 HTC 的误差
        htc_error = abs(predicted_htc - htc_true)

        # 保存结果
        file_name = os.path.basename(file_path)
        htc_results.append([file_name, predicted_htc, htc_true, htc_error])

    # 保存到 CSV 文件
    save_htc_predictions_to_csv(htc_results, output_csv_file)



def train_htc_model_with_sgd(htc_model, data_folder, htc_true, sgd_epochs=50, sgd_lr=0.1,
                             loss_log_dir="GDhtc_loss1000", summary_csv_path="summary1000.csv"):
    import time

    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    folder_name = os.path.basename(data_folder)

    os.makedirs(loss_log_dir, exist_ok=True)
    

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer_sgd = optim.SGD([htc_model.htc_normalized], lr=sgd_lr)

    total_loss_history = []
    mse_history = []
    mae_history = []

    # 稳定性判断
    epsilon = 1e-5
    stable_counter = 0
    stable_N = 5
    stability_time = None

    start_time = time.time()
    prev_loss = None

    def closure():
        optimizer_sgd.zero_grad()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0

        for file_path in file_paths:
            temp_input, power_input, target_temp = load_data(file_path)
            temp_input = temp_input.unsqueeze(0).to(device)
            power_input = power_input.unsqueeze(0).to(device)
            target_temp = target_temp.unsqueeze(0).to(device)

            predicted_temp, _ = htc_model(temp_input, power_input)
            predicted_temp = predicted_temp.squeeze(1)

            loss_mse = criterion_mse(predicted_temp, target_temp)
            loss_mae = criterion_mae(predicted_temp, target_temp)
            loss = 0.7 * loss_mse + 0.3 * loss_mae
            loss.backward()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_mae += loss_mae.item()

        return total_loss, total_mse, total_mae

    for epoch in range(sgd_epochs):
        optimizer_sgd.step(lambda: closure()[0])
        loss_val, mse_val, mae_val = closure()

        total_loss_history.append(loss_val)
        mse_history.append(mse_val)
        mae_history.append(mae_val)

        # 稳定性判断
        if prev_loss is not None and abs(loss_val - prev_loss) < epsilon:
            stable_counter += 1
        else:
            stable_counter = 0
        prev_loss = loss_val

        if stable_counter == stable_N and stability_time is None:
            stability_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{sgd_epochs} - Loss: {loss_val:.6f}, MSE: {mse_val:.6f}, MAE: {mae_val:.6f}")

    total_time = time.time() - start_time
    if stability_time is None:
        stability_time = total_time

    # 反归一化 HTC
    final_htc_value = htc_model.htc_normalized.item() * (htc_model.htc_max - htc_model.htc_min) + htc_model.htc_min
    print(f"Final HTC: {final_htc_value:.2f}")

    # === 保存 loss 到 CSV ===
    loss_csv_path = os.path.join(loss_log_dir, f"{folder_name}_loss.csv")
    df = pd.DataFrame({
        "Epoch": list(range(1, sgd_epochs + 1)),
        "Total Loss": total_loss_history,
        "MSE": mse_history,
        "MAE": mae_history
    })
    df.to_csv(loss_csv_path, index=False)

    # === 保存 loss 曲线图 ===
    plt.figure(figsize=(10, 6))
    plt.plot(df["Epoch"], df["Total Loss"], label="Total Loss")
    plt.plot(df["Epoch"], df["MSE"], label="MSE")
    plt.plot(df["Epoch"], df["MAE"], label="MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"HTC Training Loss - {folder_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(loss_log_dir, f"{folder_name}_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # === 追加保存 summary CSV（含 HTC、误差、时间） ===
    summary_row = {
        "Folder": folder_name,
        "Predicted HTC": final_htc_value,
        "True HTC": htc_true,
        "HTC Error": abs(final_htc_value - htc_true),
        "Total Training Time (s)": total_time,
        "Stability Time (s)": stability_time
    }

    # 如果是第一次写入，创建带表头
    if not os.path.exists(summary_csv_path):
        pd.DataFrame([summary_row]).to_csv(summary_csv_path, index=False)
    else:
        pd.DataFrame([summary_row]).to_csv(summary_csv_path, mode='a', index=False, header=False)

    return final_htc_value




    
    # 训练过程
    for epoch in range(sgd_epochs):
        optimizer_sgd.step(closure)  # 调用优化器的 step 方法
        final_loss = closure()  # 计算当前的损失值
        print(f'SGD Epoch {epoch+1}/{sgd_epochs}, Loss: {final_loss:.6f}')
    
    # 获取最终的 HTC 值 (反归一化)
    final_htc_value = htc_model.htc_normalized.item() * (htc_model.htc_max - htc_model.htc_min) + htc_model.htc_min
    print(f"Final HTC value after SGD: {final_htc_value}")
    return final_htc_value


# 保存预测结果和生成热力图
def save_predictions_and_heatmaps(model, data_folder, htc_true, output_dir='output'):
    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

    for file_path in file_paths:
        temp_input, power_input, true_temp = load_data(file_path)

        # 增加 batch 维度并将数据移动到 GPU
        temp_input = temp_input.unsqueeze(0).to(device)
        power_input = power_input.unsqueeze(0).to(device)
        true_temp = true_temp.unsqueeze(0).to(device)

        # 进行预测
        with torch.no_grad():
            predicted_temp, predicted_htc = model(temp_input, power_input)
            predicted_temp = predicted_temp.squeeze(0)  # 去掉 batch 维度

        # 保存温度预测和 HTC 结果
        file_name_prefix = os.path.splitext(os.path.basename(file_path))[0]

        # 生成并保存热力图
        plot_and_save_heatmaps(predicted_temp.cpu(), true_temp.cpu(), predicted_htc.cpu(), htc_true, file_name_prefix,
                               output_dir)


# 冻结 UNet 的所有参数
def freeze_unet_parameters(unet_model):
    for param in unet_model.parameters():
        param.requires_grad = False  # 冻结 UNet 的所有参数


# 加载最优的 UNet 模型权重
def load_best_unet_model():
    unet_model = ImprovedUNet(in_channels=3, out_channels=1)
    unet_model.load_state_dict(torch.load('best_model1.pth'))  # 加载之前保存的最优模型
    unet_model.eval()  # 设置为评估模式，冻结模型
    freeze_unet_parameters(unet_model)  # 冻结 UNet 的所有参数
    return unet_model


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载最优 UNet 模型
best_unet_model = load_best_unet_model().to(device)

# 初始化 HTC 模型，设置初始 HTC 值
htc_model = HTCModel(init_htc_value=750.0, unet_model=best_unet_model).to(device)



for i in range(1, 31):
    for j in range(1, 21):
        folder_name = f'3/{i}-{j}-HTC1000'
        print(f"Training on folder: {folder_name}")

        # 对每个文件夹单独训练 HTC 模型
        final_htc_value = train_htc_model_with_sgd(htc_model, folder_name, htc_true=1000, sgd_epochs=50, sgd_lr=0.1)


        # 创建文件夹路径，所有 CSV 文件将存储在 'htc_predictions' 文件夹下
        output_dir = 'GDhtc_predictions1000'  # 将所有文件存储到这个文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存预测结果到 CSV 文件
        output_csv_file = os.path.join(output_dir, f'htc_predictions_{i}-{j}-HTC1000.csv')
        save_predictions_and_csv(htc_model, folder_name, htc_true=1000,final_htc_value=final_htc_value, output_csv_file=output_csv_file)
        heatmap_output_dir = 'GDhtc_heatmap1000'  # 可自定义输出目录
        save_predictions_and_heatmaps(htc_model, folder_name, htc_true=1000, output_dir=heatmap_output_dir)
        print(f"Saved predictions for folder {folder_name} to {output_csv_file}") 


