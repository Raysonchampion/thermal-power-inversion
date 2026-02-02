import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import random

# 数据预处理和保存
def preprocess_and_save_data(input_folder, output_file):
    data_list = []
    label_list = []
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        data = np.loadtxt(file_path, delimiter=',')
        
        temp_input = data[:, :9]    # 第一块：输入温度
        power_input = data[:, 9:18] # 第二块：输入功耗
        htc_input = data[:, 18:27]  # 第三块：HTC 值
        output = data[:, 27:]       # 输出温度

        X = np.stack([temp_input, power_input, htc_input], axis=0)
        y = output

        data_list.append(X)
        label_list.append(y)
    
    X_all = np.array(data_list)
    y_all = np.array(label_list)

    torch.save((X_all, y_all), output_file)
    print(f"数据已保存到 {output_file}")

# 设置输入文件夹和输出文件路径
input_folder = '1'
output_file = 'preprocessed_data.pt'

# 预处理并保存数据
preprocess_and_save_data(input_folder, output_file)

# 加载预处理好的数据
X_all, y_all = torch.load(output_file)

# 将数据转换为 PyTorch 的 Dataset
class PreprocessedDataset(Dataset):
    def __init__(self, X, y, file_names):
        self.X = X
        self.y = y
        self.file_names = file_names

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), self.file_names[idx]

# 获取文件名列表
file_names = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# 创建数据集和数据加载器
dataset = PreprocessedDataset(X_all, y_all, file_names)

# 按 9:1 的比例划分训练集和测试集
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 定义 CNN 模型
class TempPredictionCNN(nn.Module):
    def __init__(self):
        super(TempPredictionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)  # 输出为 [batch_size, 1, 6, 9]
        return x.squeeze(1)  # 去除通道维度，输出为 [batch_size, 6, 9]

# 初始化模型
model = TempPredictionCNN()

# 定义损失函数和优化器
criterion_mae = nn.L1Loss()
criterion_mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 保存最佳模型的初始化
best_test_mae = float('inf')
best_model_path = 'best_model.pth'

# 训练模型
num_epochs = 2000
train_mae_history = []
train_mse_history = []
test_mae_history = []
test_mse_history = []

for epoch in range(num_epochs):
    model.train()
    running_mae = 0.0
    running_mse = 0.0
    
    for inputs, targets, _ in train_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        mae_loss = criterion_mae(outputs, targets)
        mse_loss = criterion_mse(outputs, targets)
        
        loss = mae_loss + mse_loss
        loss.backward()
        optimizer.step()
        
        running_mae += mae_loss.item()
        running_mse += mse_loss.item()
    
    train_mae = running_mae / len(train_loader)
    train_mse = running_mse / len(train_loader)
    
    train_mae_history.append(train_mae)
    train_mse_history.append(train_mse)
    
    model.eval()
    test_mae = 0.0
    test_mse = 0.0
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            outputs = model(inputs)
            test_mae += criterion_mae(outputs, targets).item()
            test_mse += criterion_mse(outputs, targets).item()
    
    test_mae /= len(test_loader)
    test_mse /= len(test_loader)
    
    test_mae_history.append(test_mae)
    test_mse_history.append(test_mse)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train MAE: {train_mae:.4f}, Train MSE: {train_mse:.4f}, Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}")
    
    # 保存最佳模型
    if test_mae < best_test_mae:
        best_test_mae = test_mae
        torch.save(model.state_dict(), best_model_path)
        print(f"保存最佳模型：{best_model_path}，Test MAE: {best_test_mae:.4f}")



# 创建保存热力图的文件夹
output_dir = 'figure2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 随机选择 10 个文件并输出到 CSV
random_files = random.sample(range(len(dataset)), 6000)
output_results = []

for idx in random_files:
    inputs, true_values, file_name = dataset[idx]
    inputs = inputs.unsqueeze(0)  # 增加 batch 维度
    model.eval()
    with torch.no_grad():
        predicted_values = model(inputs).squeeze(0)  # 移除 batch 维度

    difference = np.abs(true_values - predicted_values)
    
    mse = criterion_mse(predicted_values, true_values).item()
    mae = criterion_mae(predicted_values, true_values).item()
    max_error = difference.max().item()
    min_error = difference.min().item()

    # 保存每个文件的MSE、MAE、最大绝对误差和最小绝对误差，并保存True和Predicted值以便后续绘制热力图
    output_results.append({
        'File Name': file_name,
        'MSE': mse,
        'MAE': mae,
        'Max Abs Error': max_error,
        'Min Abs Error': min_error,
        'True Values': true_values.numpy().flatten().tolist(),
        'Predicted Values': predicted_values.numpy().flatten().tolist(),
        'Difference': difference.numpy().flatten().tolist()
    })

# 将结果整理到 DataFrame 中
metric_df = pd.DataFrame(output_results, columns=['File Name', 'MSE', 'MAE', 'Max Abs Error', 'Min Abs Error'])
metric_file = f'evaluation_metrics2.csv'
metric_df.to_csv(metric_file, index=False)
print(f"预测结果已保存到 {metric_file}")

# 绘制热力图并保存
for i, result in enumerate(output_results):
    true_values = np.array(result['True Values']).reshape(6, 9)
    predicted_values = np.array(result['Predicted Values']).reshape(6, 9)
    difference = np.array(result['Difference']).reshape(6, 9)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    cax1 = axs[0].imshow(true_values, cmap='viridis')
    axs[0].set_title('True Temperature')
    fig.colorbar(cax1, ax=axs[0])
    for (j, k), val in np.ndenumerate(true_values):
        axs[0].text(k, j, f'{val:.2f}', ha='center', va='center', color='white')
    
    cax2 = axs[1].imshow(predicted_values, cmap='viridis')
    axs[1].set_title('Predicted Temperature')
    fig.colorbar(cax2, ax=axs[1])
    for (j, k), val in np.ndenumerate(predicted_values):
        axs[1].text(k, j, f'{val:.2f}', ha='center', va='center', color='white')
    
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

   
