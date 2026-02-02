import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# UNet模型定义
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
            d3 = F.interpolate(d3, size=e3.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = F.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = F.interpolate(d1, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        return self.conv_final(d1)

# 数据预处理函数，按HTC值分类并分割为训练集和测试集
def preprocess_data_by_htc(folder_path, output_file, train_ratio=0.9):
    file_paths = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # 按HTC值分类
    htc_groups = {'200': [], '500': [], '700': [], '1000': [], '1300': []}
    
    for file_path in file_paths:
        data = pd.read_csv(os.path.join(folder_path, file_path), header=None).values
        input_temp = data[:, :9]
        input_power = data[:, 9:18]
        input_htc = data[:, 18:27]
        target_temp = data[:, 27:]
        
        inputs = np.stack([input_temp, input_power, input_htc])
        htc_value = file_path.split('-')[-1].replace('.csv', '').replace('HTC', '')

        if htc_value in htc_groups:
            htc_groups[htc_value].append((inputs, target_temp, file_path))
        else:
            print(f"未识别的HTC值: {htc_value}")
    
    # 初始化训练集和测试集
    train_data = {'inputs': [], 'targets': [], 'file_names': []}
    test_data = {'inputs': [], 'targets': [], 'file_names': []}

    # 记录每个 HTC 值的训练集和测试集的数量
    htc_train_counts = {htc: 0 for htc in htc_groups.keys()}
    htc_test_counts = {htc: 0 for htc in htc_groups.keys()}

    # 按HTC值划分90%训练集，10%测试集
    for htc, items in htc_groups.items():
        random.shuffle(items)  # 打乱顺序
        split_point = int(train_ratio * len(items))  # 90% 作为训练集
        
        for i, (inputs, target_temp, file_name) in enumerate(items):
            if i < split_point:
                train_data['inputs'].append(inputs)
                train_data['targets'].append(target_temp)
                train_data['file_names'].append(file_name)
                htc_train_counts[htc] += 1  # 统计训练集数量
            else:
                test_data['inputs'].append(inputs)
                test_data['targets'].append(target_temp)
                test_data['file_names'].append(file_name)
                htc_test_counts[htc] += 1  # 统计测试集数量
    
    # 转换为tensor
    train_data['inputs'] = torch.tensor(np.array(train_data['inputs']), dtype=torch.float32)
    train_data['targets'] = torch.tensor(np.array(train_data['targets']), dtype=torch.float32)
    test_data['inputs'] = torch.tensor(np.array(test_data['inputs']), dtype=torch.float32)
    test_data['targets'] = torch.tensor(np.array(test_data['targets']), dtype=torch.float32)

    # 保存处理后的数据
    torch.save((train_data, test_data), output_file)
    print(f"数据处理完成并保存到 {output_file}")

    # 打印 HTC 值对应的训练集和测试集的数量
    for htc in htc_groups.keys():
        print(f"HTC值为 {htc} 的训练集文件数: {htc_train_counts[htc]}，测试集文件数: {htc_test_counts[htc]}")

# 数据预处理并划分训练集和测试集
preprocess_data_by_htc("1", "preprocessed_htc_data.pt", train_ratio=0.9)

# 加载处理后的数据
train_data, test_data = torch.load("preprocessed_htc_data.pt")

# 输出训练集和测试集的数量
print(f"训练集数量: {len(train_data['inputs'])}")
print(f"测试集数量: {len(test_data['inputs'])}")

# 使用 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedUNet(in_channels=3, out_channels=1).to(device)

# 损失函数和优化器
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(train_data['inputs'], train_data['targets'])
test_dataset = TensorDataset(test_data['inputs'], test_data['targets'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_mse_losses = []
test_mse_losses = []
train_mae_losses = []
test_mae_losses = []
best_test_mse = float('inf')

# 最佳模型存储文件
best_model_info_file = 'best_model_info.txt'

# 训练循环
for epoch in range(2000):
    model.train()
    train_mse = 0
    train_mae = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze(1)

        mse_loss = criterion_mse(outputs, targets)
        mae_loss = criterion_mae(outputs, targets)

        mse_loss.backward()
        optimizer.step()

        train_mse += mse_loss.item()
        train_mae += mae_loss.item()

    train_mse /= len(train_loader)
    train_mae /= len(train_loader)
    train_mse_losses.append(train_mse)
    train_mae_losses.append(train_mae)

    # 测试阶段
    model.eval()
    test_mse = 0
    test_mae = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            test_mse += criterion_mse(outputs, targets).item()
            test_mae += criterion_mae(outputs, targets).item()

    test_mse /= len(test_loader)
    test_mae /= len(test_loader)
    test_mse_losses.append(test_mse)
    test_mae_losses.append(test_mae)

    # 保存最优模型
    if test_mse < best_test_mse:
        best_test_mse = test_mse
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch+1}: Saving best model with MSE {best_test_mse}")

        # 记录最佳模型信息
        with open(best_model_info_file, 'w') as f:
            f.write(f"Epoch {epoch+1}\n")
            f.write(f"Best Test MSE: {best_test_mse}\n")
            f.write(f"Best Test MAE: {test_mae}\n")
            print(f"Best model information saved to {best_model_info_file}")

    print(f"Epoch {epoch+1}: Train MSE = {train_mse}, Test MSE = {test_mse}, Train MAE = {train_mae}, Test MAE = {test_mae}")

# 绘制并保存MSE和MAE曲线
plt.figure(figsize=(12, 5))

# 绘制MSE曲线
plt.subplot(1, 2, 1)
plt.plot(train_mse_losses, label='Train MSE')
plt.plot(test_mse_losses, label='Test MSE')
plt.title('MSE Loss over Epochs')
plt.legend()

# 绘制MAE曲线
plt.subplot(1, 2, 2)
plt.plot(train_mae_losses, label='Train MAE')
plt.plot(test_mae_losses, label='Test MAE')
plt.title('MAE Loss over Epochs')
plt.legend()

# 保存图像
plt.savefig('loss_curves_mse_mae.png')
print("训练完成，最优模型已保存，损失曲线图已保存为 'loss_curves_mse_mae.png'。")

# 热力图生成及评估结果保存
output_dir = 'figure_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

random_files = random.sample(range(len(test_data['file_names'])), 5900)
output_results = []

for idx in random_files:
    inputs = test_data['inputs'][idx].unsqueeze(0).to(device)
    true_values = test_data['targets'][idx].to(device)
    file_name = test_data['file_names'][idx]

    with torch.no_grad():
        predicted_values = model(inputs).squeeze(0).cpu()

    true_values = true_values.cpu()
    predicted_values = predicted_values.squeeze(0)  # 确保形状为 [6, 9]
    true_values = true_values.view_as(predicted_values)  # 调整 true_values 形状与 predicted_values 一致
    difference = torch.abs(true_values - predicted_values)

    # 计算 MSE 和 MAE
    mse = criterion_mse(predicted_values, true_values).item()
    mae = criterion_mae(predicted_values, true_values).item()
    max_error = difference.max().item()
    min_error = difference.min().item()

    # 保存评估结果
    output_results.append({
        'File Name': file_name,
        'MSE': mse,
        'MAE': mae,
        'Max Abs Error': max_error,
        'Min Abs Error': min_error
    })

    true_values = true_values.numpy()
    predicted_values = predicted_values.numpy()
    difference = difference.numpy()

    # 绘制热力图
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(true_values, annot=True, cmap="viridis", ax=axs[0], fmt=".2f")
    axs[0].set_title("True Temperature")

    sns.heatmap(predicted_values, annot=True, cmap="viridis", ax=axs[1], fmt=".2f")
    axs[1].set_title("Predicted Temperature")

    sns.heatmap(difference, annot=True, cmap="viridis", ax=axs[2], fmt=".2f")
    axs[2].set_title("Difference (Error)")

    plt.tight_layout()

    # 保存热力图
    heatmap_file_path = os.path.join(output_dir, f'{file_name}_heatmap.png')
    plt.savefig(heatmap_file_path)
    print(f"热力图已保存到 {heatmap_file_path}")
    plt.close(fig)

# 将评估结果保存到 CSV 文件
metric_df = pd.DataFrame(output_results, columns=['File Name', 'MSE', 'MAE', 'Max Abs Error', 'Min Abs Error'])
metric_file = 'evaluation_metrics.csv'
metric_df.to_csv(metric_file, index=False)
print(f"评估结果已保存到 {metric_file}")
