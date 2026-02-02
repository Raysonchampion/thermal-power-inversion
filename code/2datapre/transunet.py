import os
import pandas as pd
import numpy as np
import torch
import random
# 数据预处理函数
def preprocess_data(folder_path, output_file):
    file_paths = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    
    all_inputs = []
    all_targets = []

    for file_path in file_paths:
        # 读取CSV文件
        data = pd.read_csv(os.path.join(folder_path, file_path), header=None).values
        
        # 三个输入通道：当前温度、功耗、HTC
        input_temp = data[:, :9]
        input_power = data[:, 9:18]
        input_htc = data[:, 18:27]
        
        # 输出为下一个时间点的温度
        target_temp = data[:, 27:]
        
        # 将输入转换为 numpy 数组并添加到列表中
        inputs = np.stack([input_temp, input_power, input_htc])  # (3, 6, 9)
        all_inputs.append(inputs)
        
        # 目标输出
        all_targets.append(target_temp)
    
    # 将所有输入和输出转换为 PyTorch 张量
    all_inputs = torch.tensor(np.array(all_inputs), dtype=torch.float32)  # (num_files, 3, 6, 9)
    all_targets = torch.tensor(np.array(all_targets), dtype=torch.float32)  # (num_files, 6, 9)
    
    # 将处理好的数据保存为 .pt 文件
    torch.save((all_inputs, all_targets), output_file)
    print(f"数据处理完成，已保存到 {output_file}")

# 调用函数处理数据，修改文件夹路径
preprocess_data("1", "preprocessed_data.pt")

# 加载预处理的数据
all_inputs, all_targets = torch.load("preprocessed_data.pt")

# 数据集划分为训练集和测试集
train_size = int(0.9 * len(all_inputs))
test_size = len(all_inputs) - train_size

train_inputs = all_inputs[:train_size]
train_targets = all_targets[:train_size]
test_inputs = all_inputs[train_size:]
test_targets = all_targets[train_size:]

# 构建 PyTorch 数据集和 DataLoader
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_inputs, train_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)[0]
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加 batch 维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :].clone().detach()  # 确保 pe 和 x 的长度匹配
        return x + pe  # 添加位置编码

# UNet 编码和解码模块
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 编码器部分
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.pool = nn.MaxPool2d(2)

        # 解码器部分
        self.dec1 = CBR(256, 128)
        self.dec2 = CBR(128, 64)
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

        # 用于将 e1_cropped 转换为与 d1 通道数匹配
        self.conv_match = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d1 = self.dec1(F.interpolate(e3, size=(6, 9), mode='bilinear', align_corners=True))
        
        # 使用卷积将 e1_cropped 的通道数从 64 转换为 128
        e1_cropped = self.conv_match(e1[:, :, :, :d1.size(3)])
        
        d2 = self.dec2(d1 + e1_cropped)
        x = self.conv_final(d2)
        return x

# TransUNet 模型
class TransUNet(nn.Module):
    def __init__(self, img_size=6, patch_size=2, in_channels=3, out_channels=1, embed_dim=256, num_heads=4, num_layers=2, forward_expansion=4, dropout=0.1):
        super(TransUNet, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        self.unet = UNet(in_channels, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim, max_len=12)
        self.transformer = TransformerEncoder(embed_dim, num_layers, num_heads, forward_expansion, dropout)

        # 修改 conv_final 的输入通道数为 64
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")

        # 编码器部分
        e1 = self.unet.enc1(x)
        #print(f"Shape after enc1 (e1): {e1.shape}")

        e2 = self.unet.enc2(self.unet.pool(e1))
        #print(f"Shape after enc2 (e2): {e2.shape}")

        e3 = self.unet.enc3(self.unet.pool(e2))
        #print(f"Shape after enc3 (e3): {e3.shape}")

        # Transformer 部分
        batch_size, _, height, width = e3.size()
        num_patches = height * width
        x = e3.permute(0, 2, 3, 1).contiguous().view(batch_size, num_patches, -1)
        #print(f"Shape after preparing for transformer: {x.shape}")

        x = self.position_encoding(x)
        #print(f"Shape after position encoding: {x.shape}")

        x = x.permute(1, 0, 2)  # Transformer 输入需要 [seq_len, batch, embed_dim]
        #print(f"Shape before transformer: {x.shape}")

        x = self.transformer(x)
        #print(f"Shape after transformer: {x.shape}")

        x = x.permute(1, 0, 2).contiguous().view(batch_size, height, width, -1)
        #print(f"Shape after transformer and reshaping: {x.shape}")

        x = x.permute(0, 3, 1, 2)  # [batch_size, embed_dim, height, width]
        #print(f"Shape before decoding: {x.shape}")

        # 解码器部分
        d1 = self.unet.dec1(F.interpolate(x, size=(6, 9), mode='bilinear', align_corners=True))
        #print(f"Shape after dec1 (d1): {d1.shape}")

        e1_cropped = self.unet.conv_match(e1[:, :, :, :d1.size(3)])  # 调整 e1_cropped 的通道数
        #print(f"Shape after cropping e1 (e1_cropped): {e1_cropped.shape}")

        d2 = self.unet.dec2(d1 + e1_cropped)
        #print(f"Shape after dec2 (d2): {d2.shape}")

        # 修改 conv_final 的输入通道数为 64
        x = self.conv_final(d2)
        #print(f"Output shape: {x.shape}")
        x = x.squeeze(1)
        return x

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# 创建一个用于保存最优模型的文件夹
if not os.path.exists('models'):
    os.makedirs('models')

# 使用GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransUNet(in_channels=3, out_channels=1, img_size=6, patch_size=2).to(device)

# 损失函数和优化器
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_mse_losses = []
train_mae_losses = []
test_mse_losses = []
test_mae_losses = []
best_test_mse = float('inf')  # 用于保存最优模型

# 保存结果的 CSV 文件路径
csv_file = "results.csv"
csv_data = []

# 训练循环
for epoch in range(2000):
    model.train()
    train_mse = 0
    train_mae = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # 计算输出
        outputs = model(inputs)
        mse_loss = criterion_mse(outputs, targets)
        mae_loss = criterion_mae(outputs, targets)

        # 反向传播和优化
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
            test_mse += criterion_mse(outputs, targets).item()
            test_mae += criterion_mae(outputs, targets).item()
    
    test_mse /= len(test_loader)
    test_mae /= len(test_loader)
    test_mse_losses.append(test_mse)
    test_mae_losses.append(test_mae)

    # 检查是否需要保存最优模型
    if test_mse < best_test_mse:
        best_test_mse = test_mse
        torch.save(model.state_dict(), 'models/best_model.pth')  # 保存最优模型
        print(f"Epoch {epoch+1}: Saving best model with MSE {best_test_mse}")

    print(f'Epoch {epoch+1}, Train MSE: {train_mse}, Train MAE: {train_mae}, Test MSE: {test_mse}, Test MAE: {test_mae}')



import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 创建保存热力图的文件夹
output_dir = 'figure2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 使用 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 随机选择 50 个文件并输出到 CSV
random_files = random.sample(range(len(test_dataset)), 6000)
output_results = []

# 确保模型处于评估模式
model.eval()

# 定义损失函数
criterion_mse = torch.nn.MSELoss()
criterion_mae = torch.nn.L1Loss()

for idx in random_files:
    # 获取输入和真实值
    inputs, true_values = test_dataset[idx]
    inputs = inputs.unsqueeze(0).to(device)  # 增加 batch 维度并传到 GPU
    true_values = true_values.to(device)

    # 模型预测
    with torch.no_grad():
        predicted_values = model(inputs).squeeze(0).cpu()  # 移除 batch 维度，并将结果传到 CPU

    true_values = true_values.cpu()  # 将真实值传到 CPU
    difference = torch.abs(true_values - predicted_values)  # 计算差值

    # 计算误差
    mse = criterion_mse(predicted_values, true_values).item()
    mae = criterion_mae(predicted_values, true_values).item()
    max_error = difference.max().item()
    min_error = difference.min().item()

    # 保存每个文件的 MSE、MAE、最大和最小绝对误差，并保存 True 和 Predicted 值以便后续绘制热力图
    output_results.append({
        'File Name': f'file_{idx}',  # 用 idx 作为文件名标识
        'MSE': mse,
        'MAE': mae,
        'Max Abs Error': max_error,
        'Min Abs Error': min_error,
        'True Values': true_values.numpy().flatten().tolist(),
        'Predicted Values': predicted_values.numpy().flatten().tolist(),
        'Difference': difference.numpy().flatten().tolist()
    })

# 将结果整理到 DataFrame 中，并保存到 CSV
metric_df = pd.DataFrame(output_results, columns=['File Name', 'MSE', 'MAE', 'Max Abs Error', 'Min Abs Error'])
metric_file = 'evaluation_metrics2.csv'
metric_df.to_csv(metric_file, index=False)
print(f"预测结果已保存到 {metric_file}")

# 绘制热力图并保存
for i, result in enumerate(output_results):
    true_values = np.array(result['True Values']).reshape(6, 9)
    predicted_values = np.array(result['Predicted Values']).reshape(6, 9)
    difference = np.array(result['Difference']).reshape(6, 9)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 真实值热力图
    cax1 = axs[0].imshow(true_values, cmap='viridis')
    axs[0].set_title('True Temperature')
    fig.colorbar(cax1, ax=axs[0])
    for (j, k), val in np.ndenumerate(true_values):
        axs[0].text(k, j, f'{val:.2f}', ha='center', va='center', color='white')
    
    # 预测值热力图
    cax2 = axs[1].imshow(predicted_values, cmap='viridis')
    axs[1].set_title('Predicted Temperature')
    fig.colorbar(cax2, ax=axs[1])
    for (j, k), val in np.ndenumerate(predicted_values):
        axs[1].text(k, j, f'{val:.2f}', ha='center', va='center', color='white')
    
    # 误差热力图
    cax3 = axs[2].imshow(difference, cmap='viridis')
    axs[2].set_title('Error Temperature')
    fig.colorbar(cax3, ax=axs[2])
    for (j, k), val in np.ndenumerate(difference):
        axs[2].text(k, j, f'{val:.2f}', ha='center', va='center', color='white')
    
    plt.tight_layout()
    
    # 保存热力图
    heatmap_file_path = os.path.join(output_dir, f'{result["File Name"]}_heatmap.png')
    plt.savefig(heatmap_file_path)
    print(f"热力图已保存到 {heatmap_file_path}")

    plt.close(fig)








