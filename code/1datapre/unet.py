import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. 数据加载
data_dir = 'wendugonghao/'
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

dataset = TemperatureDataset(temperature_data, power_data, temperature_5ms, temperature_20ms, temperature_100ms)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = CBR(2, 64)
        self.enc2 = CBR(64, 128)
        
        self.pool = nn.MaxPool2d(2)
        
        self.dec1 = CBR(128, 64)
        self.dec2 = CBR(64, 32)
        
        self.final = nn.Conv2d(32, 3, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        d1 = self.dec1(e2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(d1 + e1)
        
        output = self.final(d2)
        return output

def train_and_evaluate(model, train_loader, test_loader, criterion_mse, criterion_mae, optimizer, num_epochs=2000):
    train_losses_mse = []
    test_losses_mse = []
    train_losses_mae = []
    test_losses_mae = []
    
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
    
    return train_losses_mse, test_losses_mse, train_losses_mae, test_losses_mae

# 训练和测试模型
model = UNet().cuda()
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2000
train_losses_mse, test_losses_mse, train_losses_mae, test_losses_mae = train_and_evaluate(
    model, train_loader, test_loader, criterion_mse, criterion_mae, optimizer, num_epochs)

# 保存模型
torch.save(model.state_dict(), 'unet_temperature_model.pth')

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
plt.savefig('loss_curves.png')  # 保存曲线图
plt.show()
