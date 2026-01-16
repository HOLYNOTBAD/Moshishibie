"""
项目一：使用 PyTorch 构建 MLP 对鸢尾花数据集进行分类
对应教材：第12章 - Parallelizing Neural Network Training with PyTorch
主要技术点：张量操作、DataLoader、神经网络模块、训练循环、模型评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(1)
np.random.seed(1)

# ==================== 1. 数据准备 ====================
print("=" * 50)
print("步骤1: 加载并预处理鸢尾花数据集")
print("=" * 50)

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# 标准化特征
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train_std)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_std)
y_test_tensor = torch.LongTensor(y_test)

# 创建 DataLoader（实现批量加载和洗牌）
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"特征维度: {X_train_tensor.shape[1]}")
print(f"类别数: {len(np.unique(y))}")

# ==================== 2. 构建 MLP 模型 ====================
print("\n" + "=" * 50)
print("步骤2: 构建多层感知机(MLP)模型")
print("=" * 50)

class MLP(nn.Module):
    """
    多层感知机模型
    结构: 输入层(4) -> 隐藏层(16) -> 隐藏层(8) -> 输出层(3)
    对应教材第12章中的 'Building a multilayer perceptron for classifying flowers'
    """
    def __init__(self, input_dim=4, hidden1=16, hidden2=8, output_dim=3):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.layer3 = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # 轻微正则化
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.layer3(x)  # 输出层不使用激活函数，因为使用CrossEntropyLoss
        return x

# 初始化模型
model = MLP()
print(model)
print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# ==================== 3. 训练配置 ====================
print("\n" + "=" * 50)
print("步骤3: 配置损失函数和优化器")
print("=" * 50)

criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# ==================== 4. 训练循环 ====================
print("\n" + "=" * 50)
print("步骤4: 训练模型")
print("=" * 50)

num_epochs = 50
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 清零梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        total_loss += loss.item()
        
        # 计算训练准确率
        _, predicted = torch.max(output, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()
    
    # 学习率调度
    scheduler.step()
    
    # 评估训练集准确率
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    
    # 测试阶段
    model.eval()
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()
    
    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)
    
    # 打印进度
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {total_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Test Acc: {test_accuracy:.2f}%')

# ==================== 5. 可视化结果 ====================
print("\n" + "=" * 50)
print("步骤5: 可视化训练过程")
print("=" * 50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(train_losses, label='Training Loss', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss over Epochs')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 准确率曲线
ax2.plot(train_accuracies, label='Training Accuracy', color='green')
ax2.plot(test_accuracies, label='Test Accuracy', color='red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_mlp_training.png', dpi=150)
plt.show()

# ==================== 6. 模型评估 ====================
print("\n" + "=" * 50)
print("步骤6: 最终模型评估")
print("=" * 50)

model.eval()
with torch.no_grad():
    # 在测试集上的最终评估
    test_outputs = model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs, 1)
    final_test_accuracy = 100 * (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    
    # 在训练集上的评估
    train_outputs = model(X_train_tensor)
    _, train_predicted = torch.max(train_outputs, 1)
    final_train_accuracy = 100 * (train_predicted == y_train_tensor).sum().item() / len(y_train_tensor)

print(f"最终训练准确率: {final_train_accuracy:.2f}%")
print(f"最终测试准确率: {final_test_accuracy:.2f}%")

# ==================== 7. 保存模型 ====================
print("\n" + "=" * 50)
print("步骤7: 保存和加载模型")
print("=" * 50)

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler,
    'test_accuracy': final_test_accuracy
}, 'iris_mlp_model.pth')

print("模型已保存到 'iris_mlp_model.pth'")

# 加载模型示例
checkpoint = torch.load('iris_mlp_model.pth')
loaded_model = MLP()
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()
print("模型加载成功!")

# ==================== 8. 预测示例 ====================
print("\n" + "=" * 50)
print("步骤8: 预测新样本")
print("=" * 50)

# 创建一些新样本进行预测
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # 预期: Setosa (0)
    [6.5, 3.0, 5.2, 2.0],  # 预期: Virginica (2)
    [5.9, 3.0, 4.2, 1.5],  # 预期: Versicolor (1)
])

new_samples_std = scaler.transform(new_samples)
new_samples_tensor = torch.FloatTensor(new_samples_std)

with torch.no_grad():
    predictions = loaded_model(new_samples_tensor)
    _, predicted_classes = torch.max(predictions, 1)
    
class_names = ['Setosa', 'Versicolor', 'Virginica']
print("\n新样本预测结果:")
for i, (sample, pred) in enumerate(zip(new_samples, predicted_classes)):
    print(f"样本 {i+1}: {sample} -> 预测: {class_names[pred.item()]} (类别 {pred.item()})")

print("\n" + "=" * 50)
print("项目一完成！")
print("=" * 50)