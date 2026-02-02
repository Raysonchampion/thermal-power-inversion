import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. 数据加载
# 假设所有文件都在 'wendugonghao/' 目录下
data_dir = 'wendugonghao/'
files = os.listdir(data_dir)

# 初始化一个空的列表来存储所有数据
temperature_data = []
power_data = []
temperature_5ms = []
temperature_20ms = []
temperature_100ms = []

# 遍历所有文件并加载数据
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file)
        # 读取文件内容为6x45矩阵
        df = pd.read_csv(file_path, header=None)
        matrix = df.values.reshape(6, 45)
        
        # 初始温度：前6x9
        temp_initial = matrix[:, :9]
        # 功耗：中间6x9
        power = matrix[:, 9:18]
        # 温度输出：5ms、20ms、100ms分别在6x9之后
        temp_5ms = matrix[:, 18:27]
        temp_20ms = matrix[:, 27:36]
        temp_100ms = matrix[:, 36:45]
        
        # 将数据添加到列表中
        temperature_data.append(temp_initial)
        power_data.append(power)
        temperature_5ms.append(temp_5ms)
        temperature_20ms.append(temp_20ms)
        temperature_100ms.append(temp_100ms)

# 转换为NumPy数组
temperature_data = np.array(temperature_data)
power_data = np.array(power_data)
temperature_5ms = np.array(temperature_5ms)
temperature_20ms = np.array(temperature_20ms)
temperature_100ms = np.array(temperature_100ms)

# 2. 数据标准化（可选）

# 可以对输入特征进行归一化处理
scaler = MinMaxScaler()
# 这里假设对每个文件的数据单独进行标准化
for i in range(len(temperature_data)):
    temperature_data[i] = scaler.fit_transform(temperature_data[i].reshape(-1, 1)).reshape(6, 9)
    power_data[i] = scaler.fit_transform(power_data[i].reshape(-1, 1)).reshape(6, 9)

# 3. 自定义数据集创建

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
        # 将初始温度和功耗组合为多通道输入数据
        input_data = np.stack([self.temp_data[idx], self.power_data[idx]], axis=0)
        # 转换为tensor
        input_data = torch.tensor(input_data, dtype=torch.float32)
        label = torch.tensor([self.temp_5ms[idx], self.temp_20ms[idx], self.temp_100ms[idx]], dtype=torch.float32)
        return input_data, label

# 构建数据集实例
dataset = TemperatureDataset(temperature_data, power_data, temperature_5ms, temperature_20ms, temperature_100ms)

# 划分训练集和测试集（90% 训练, 10% 测试）
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建DataLoader用于批处理
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. CNN模型定义

class TempPredictionCNN(nn.Module):
    def __init__(self):
        super(TempPredictionCNN, self).__init__()
        # 假设输入是2x6x9的格式
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 6 * 9, 128)  # 假设卷积之后的特征图仍然是6x9
        self.fc2 = nn.Linear(128, 6 * 9 * 3)  # 输出的大小为 6x9x3 (对应5ms, 20ms, 100ms)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 3, 6, 9)  # 恢复为 3x6x9 的输出形状，3 对应 5ms, 20ms, 100ms
        return x

# 5. 模型训练和测试

# 定义模型、损失函数和优化器
model = TempPredictionCNN()
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 用于存储训练和测试的损失
train_mse_losses = []
train_mae_losses = []
test_mse_losses = []
test_mae_losses = []

# 训练过程
num_epochs = 2000
best_test_mse_loss = float('inf')

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_mse_loss = 0.0
    running_mae_loss = 0.0
    for inputs, labels in train_loader:
        # 将梯度归零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        mse_loss = criterion_mse(outputs, labels)
        mae_loss = criterion_mae(outputs, labels)
        
        # 反向传播
        mse_loss.backward()
        
        # 更新权重
        optimizer.step()
        
        running_mse_loss += mse_loss.item()
        running_mae_loss += mae_loss.item()
    
    # 记录平均训练损失
    train_mse_losses.append(running_mse_loss / len(train_loader))
    train_mae_losses.append(running_mae_loss / len(train_loader))
    
    # 测试阶段
    model.eval()
    test_mse_loss = 0.0
    test_mae_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_mse_loss += criterion_mse(outputs, labels).item()
            test_mae_loss += criterion_mae(outputs, labels).item()

    # 记录平均测试损失
    test_mse_losses.append(test_mse_loss / len(test_loader))
    test_mae_losses.append(test_mae_loss / len(test_loader))
    
    # 打印训练和测试损失
    print(f"Epoch [{epoch+1}/{num_epochs}], Train MSE Loss: {train_mse_losses[-1]:.4f}, Train MAE Loss: {train_mae_losses[-1]:.4f}, "
          f"Test MSE Loss: {test_mse_losses[-1]:.4f}, Test MAE Loss: {test_mae_losses[-1]:.4f}")
    
    # 保存当前模型（如果测试集上的MSE损失降低）
    if test_mse_losses[-1] < best_test_mse_loss:
        best_test_mse_loss = test_mse_losses[-1]
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved at epoch {epoch+1} with Test MSE Loss: {best_test_mse_loss:.4f}")

# 训练完成后保存最终模型
torch.save(model.state_dict(), 'final_model.pth')
print("Final model saved.")

# 6. 绘制损失曲线
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_mse_losses, label='Train MSE Loss')
plt.plot(test_mse_losses, label='Test MSE Loss')
plt.title('MSE Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_mae_losses, label='Train MAE Loss')
plt.plot(test_mae_losses, label='Test MAE Loss')
plt.title('MAE Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
