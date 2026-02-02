import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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
        pe = pe.unsqueeze(0)  # 增加batch维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)  # 获取x的序列长度
        if seq_len > self.pe.size(1):
            raise RuntimeError(f"Input sequence length {seq_len} exceeds positional encoding length {self.pe.size(1)}.")
        pe = self.pe[:, :seq_len, :].clone().detach()  # 确保pe和x的长度匹配
        return x + pe  # 添加位置编码

# UNet的编码和解码模块
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = CBR(128, 64)
        self.dec2 = CBR(64, out_channels)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.dec1(F.interpolate(e2, scale_factor=2, mode='bilinear', align_corners=True))
        d2 = self.dec2(d1 + e1[:, :, :, :d1.size(3)])  # 裁剪 e1 的宽度以匹配 d1
        return d2

# TransUNet模型
class TransUNet(nn.Module):
    def __init__(self, img_size=6, patch_size=2, in_channels=2, out_channels=3, embed_dim=128, num_heads=4, num_layers=2, forward_expansion=4, dropout=0.1):
        super(TransUNet, self).__init__()
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        
        self.unet = UNet(in_channels, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim, max_len=12)  # 将 max_len 设置为 12，以适应当前的输入
        self.transformer = TransformerEncoder(embed_dim, num_layers, num_heads, forward_expansion, dropout)
        
        self.conv_final = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")

        e1 = self.unet.enc1(x)  # [batch_size, 64, 6, 9]
        #print(f"Shape after enc1: {e1.shape}")
        
        e2 = self.unet.enc2(self.unet.pool(e1))  # [batch_size, 128, 3, 4]
        #print(f"Shape after enc2: {e2.shape}")

        batch_size, _, height, width = e2.size()
        num_patches = height * width
        x = e2.permute(0, 2, 3, 1).contiguous().view(batch_size, num_patches, -1)  # [batch_size, num_patches, embed_dim]
        #print(f"Shape after view for transformer: {x.shape}")
        
        x = self.position_encoding(x)  # [batch_size, num_patches, embed_dim]
        #print(f"Shape after position encoding: {x.shape}")
        
        x = x.permute(1, 0, 2)  # Transformer输入需要 [seq_len, batch, embed_dim]
        #print(f"Shape before transformer: {x.shape}")
        x = self.transformer(x)
        #print(f"Shape after transformer: {x.shape}")
        x = x.permute(1, 0, 2).contiguous().view(batch_size, height, width, -1)
        #print(f"Shape after permute and view: {x.shape}")
        
        x = x.permute(0, 3, 1, 2)  # [batch_size, embed_dim, height, width]
        #print(f"Shape before decoding: {x.shape}")

        d1 = self.unet.dec1(F.interpolate(x, size=(6, 9), mode='bilinear', align_corners=True))  # 确保输出尺寸与输入尺寸匹配
        #print(f"Shape after upsample and dec1: {d1.shape}")

        e1_cropped = e1[:, :, :, :d1.size(3)]  # 裁剪 e1 的宽度以匹配 d1
        #print(f"Shape after cropping e1: {e1_cropped.shape}")

        d2 = self.unet.dec2(d1 + e1_cropped)  # [batch_size, out_channels, 6, 9]
        #print(f"Shape after dec2: {d2.shape}")

        x = self.conv_final(d2)
        #print(f"Output shape: {x.shape}")
        return x

# 数据集定义
class TemperatureDataset(Dataset):
    def __init__(self, temp_data, power_data, temp_5ms, temp_20ms, temp_100ms):
        self.temp_data = temp_data
        self.power_data = power_data
        self.temp_5ms = temp_5ms
        self.temp_20ms = temp_20ms
        self.temp_100ms = temp_100ms
    
    def __len__(self):
        return len(self.temp_data)
    
    def __getitem__(self, idx):
        input_data = np.stack([self.temp_data[idx], self.power_data[idx]], axis=0)
        input_data = torch.tensor(input_data, dtype=torch.float32)
        label = torch.tensor(np.stack([self.temp_5ms[idx], self.temp_20ms[idx], self.temp_100ms[idx]], axis=0), dtype=torch.float32)
        return input_data, label

# 数据加载与预处理
def load_data(data_dir):
    files = os.listdir(data_dir)
    temperature_data = []
    power_data = []
    temperature_5ms = []
    temperature_20ms = []
    temperature_100ms = []

    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path, header=None)
            matrix = df.values.reshape(6, 45)
            
            temp_initial = matrix[:, :9]
            power = matrix[:, 9:18]
            temp_5ms = matrix[:, 18:27]
            temp_20ms = matrix[:, 27:36]
            temp_100ms = matrix[:, 36:45]
            
            temperature_data.append(temp_initial)
            power_data.append(power)
            temperature_5ms.append(temp_5ms)
            temperature_20ms.append(temp_20ms)
            temperature_100ms.append(temp_100ms)

    temperature_data = np.array(temperature_data)
    power_data = np.array(power_data)
    temperature_5ms = np.array(temperature_5ms)
    temperature_20ms = np.array(temperature_20ms)
    temperature_100ms = np.array(temperature_100ms)

    scaler = MinMaxScaler()
    for i in range(len(temperature_data)):
        temperature_data[i] = scaler.fit_transform(temperature_data[i].reshape(-1, 1)).reshape(6, 9)
        power_data[i] = scaler.fit_transform(power_data[i].reshape(-1, 1)).reshape(6, 9)

    dataset = TemperatureDataset(temperature_data, power_data, temperature_5ms, temperature_20ms, temperature_100ms)
    return dataset

# 训练模型
def train_model(model, train_loader, test_loader, criterion_mse, criterion_mae, optimizer, num_epochs=50):
    train_losses_mse = []
    test_losses_mse = []
    train_losses_mae = []
    test_losses_mae = []
    
    best_test_loss_mse = float('inf')  # 初始化最佳测试损失为无穷大
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss_mse = 0.0
        running_train_loss_mae = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_mse = criterion_mse(outputs, targets)
            loss_mae = criterion_mae(outputs, targets)
            loss_mse.backward()
            optimizer.step()
            
            running_train_loss_mse += loss_mse.item()
            running_train_loss_mae += loss_mae.item()
        
        train_losses_mse.append(running_train_loss_mse / len(train_loader))
        train_losses_mae.append(running_train_loss_mae / len(train_loader))
        
        model.eval()
        running_test_loss_mse = 0.0
        running_test_loss_mae = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss_mse = criterion_mse(outputs, targets)
                loss_mae = criterion_mae(outputs, targets)
                running_test_loss_mse += loss_mse.item()
                running_test_loss_mae += loss_mae.item()
        
        test_losses_mse.append(running_test_loss_mse / len(test_loader))
        test_losses_mae.append(running_test_loss_mae / len(test_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss MSE: {train_losses_mse[-1]:.4f}, Test Loss MSE: {test_losses_mse[-1]:.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss MAE: {train_losses_mae[-1]:.4f}, Test Loss MAE: {test_losses_mae[-1]:.4f}')
        
        # 如果当前测试集MSE损失更低，则保存模型
        if test_losses_mse[-1] < best_test_loss_mse:
            best_test_loss_mse = test_losses_mse[-1]
            best_model_state = model.state_dict()
    
    # 返回最佳模型和损失曲线
    return best_model_state, train_losses_mse, test_losses_mse, train_losses_mae, test_losses_mae

# 主函数
def main():
    # 加载数据集
    data_dir = 'wendugonghao/'
    dataset = load_data(data_dir)

    # 划分训练集和测试集（90% 训练, 10% 测试）
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建DataLoader用于批处理
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建TransUNet模型实例
    model = TransUNet(in_channels=2, out_channels=3, img_size=6, patch_size=2, embed_dim=128, num_heads=4).cuda()

    # 定义损失函数和优化器
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 2000
    best_model_state, train_losses_mse, test_losses_mse, train_losses_mae, test_losses_mae = train_model(
        model, train_loader, test_loader, criterion_mse, criterion_mae, optimizer, num_epochs
    )

    # 保存最优模型
    torch.save(best_model_state, 'best_transunet_temperature_model.pth')

    # 绘制并保存损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_mse, label='Train MSE Loss')
    plt.plot(test_losses_mse, label='Test MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing MSE Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses_mae, label='Train MAE Loss')
    plt.plot(test_losses_mae, label='Test MAE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing MAE Loss Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_curves_transunet.png')
    plt.show()

if __name__ == '__main__':
    main()
