import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以保证可重复性
torch.manual_seed(1)
np.random.seed(1)


# ==================== 第一部分：卷积操作的基础实现 ====================

# 1. 一维卷积实现
def conv1d(x, w, p=0, s=1):
    """一维卷积的朴素实现"""
    w_rot = np.array(w[::-1])  # 旋转滤波器
    x_padded = np.array(x)

    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])

    res = []
    for i in range(0, int((len(x_padded) - len(w_rot))) + 1, s):
        res.append(np.sum(x_padded[i:i + w_rot.shape[0]] * w_rot))

    return np.array(res)


# 测试一维卷积
print("=== 测试一维卷积 ===")
x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]
print('Conv1d Implementation:', conv1d(x, w, p=2, s=1))
print('NumPy Results:', np.convolve(x, w, mode='same'))


# 2. 二维卷积实现
def conv2d_naive(X, W, p=(0, 0), s=(1, 1)):
    """二维卷积的朴素实现"""
    W_rot = np.array(W)[::-1, ::-1]  # 旋转滤波器
    X_orig = np.array(X)

    # 计算填充后的大小
    n1 = X_orig.shape[0] + 2 * p[0]
    n2 = X_orig.shape[1] + 2 * p[1]

    # 创建填充后的矩阵
    X_padded = np.zeros(shape=(n1, n2))
    X_padded[p[0]:p[0] + X_orig.shape[0], p[1]:p[1] + X_orig.shape[1]] = X_orig

    res = []
    for i in range(0, int((X_padded.shape[0] - W_rot.shape[0]) / s[0]) + 1, s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1] - W_rot.shape[1]) / s[1]) + 1, s[1]):
            X_sub = X_padded[i:i + W_rot.shape[0], j:j + W_rot.shape[1]]
            res[-1].append(np.sum(X_sub * W_rot))

    return np.array(res)


# 测试二维卷积
print("\n=== 测试二维卷积 ===")
X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]
print('Conv2d Implementation:\n', conv2d_naive(X, W, p=(1, 1), s=(1, 1)))
print('SciPy Results:\n', convolve2d(X, W, mode='same'))

# ==================== 第二部分：MNIST手写数字识别 ====================

print("\n" + "=" * 60)
print("MNIST手写数字识别CNN实现")
print("=" * 60)

# 1. 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])

# 加载MNIST数据集
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 创建验证集（从训练集中取前10000个样本）
mnist_valid = Subset(mnist_train, torch.arange(10000))
mnist_train = Subset(mnist_train, torch.arange(10000, len(mnist_train)))

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(mnist_valid, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(mnist_train)}")
print(f"验证集大小: {len(mnist_valid)}")
print(f"测试集大小: {len(mnist_test)}")


# 2. 定义CNN模型
class MNIST_CNN(nn.Module):
    """用于MNIST手写数字识别的CNN模型"""

    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 输入:28x28x1 -> 输出:28x28x32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # 输出:14x14x32

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # 输出:14x14x64
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  # 输出:7x7x64

        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 3. 训练函数
def train_model(model, train_loader, valid_loader, num_epochs=10, learning_rate=0.001):
    """训练CNN模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    print("\n开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # 验证阶段
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()

                _, predicted = torch.max(output, 1)
                valid_total += target.size(0)
                valid_correct += (predicted == target).sum().item()

        valid_loss = valid_loss / len(valid_loader)
        valid_acc = 100. * valid_correct / valid_total

        # 记录历史
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)

        print(f'Epoch {epoch + 1:2d}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%')

    return model, train_loss_history, train_acc_history, valid_loss_history, valid_acc_history


# 4. 评估函数
def evaluate_model(model, test_loader):
    """评估模型在测试集上的性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f'\n测试集准确率: {test_acc:.2f}%')

    return test_acc


# 5. 可视化函数
def plot_training_history(train_loss, train_acc, valid_loss, valid_acc):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(train_loss, label='Train Loss', marker='o')
    axes[0].plot(valid_loss, label='Valid Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 准确率曲线
    axes[1].plot(train_acc, label='Train Accuracy', marker='o')
    axes[1].plot(valid_acc, label='Valid Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# 6. 可视化预测结果
def visualize_predictions(model, test_dataset, num_images=12):
    """可视化模型预测结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 随机选择一些测试图像
    indices = np.random.choice(len(test_dataset), num_images, replace=False)

    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        img, target = test_dataset[idx]

        # 预测
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted = predicted.item()

        # 显示图像
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')

        # 显示标签
        color = 'green' if predicted == target else 'red'
        axes[i].set_title(f'True: {target}\nPred: {predicted}', color=color, fontsize=12)

    plt.suptitle('MNIST手写数字识别预测结果', fontsize=16)
    plt.tight_layout()
    plt.show()


# 7. 主程序：训练和评估MNIST CNN模型
print("\n" + "=" * 60)
print("开始训练MNIST CNN模型...")
print("=" * 60)

# 创建模型
mnist_model = MNIST_CNN()
print(f"模型参数量: {sum(p.numel() for p in mnist_model.parameters()):,}")

# 训练模型
mnist_model, train_loss, train_acc, valid_loss, valid_acc = train_model(
    mnist_model, train_loader, valid_loader, num_epochs=10, learning_rate=0.001
)

# 绘制训练历史
plot_training_history(train_loss, train_acc, valid_loss, valid_acc)

# 评估模型
test_accuracy = evaluate_model(mnist_model, test_loader)

# 可视化预测结果
visualize_predictions(mnist_model, mnist_test)

# ==================== 第三部分：CelebA微笑分类 ====================

print("\n" + "=" * 60)
print("CelebA微笑分类CNN实现")
print("=" * 60)


# 注意：CelebA数据集较大，这里我们使用简化版本

class SmileCNN(nn.Module):
    """用于CelebA微笑分类的CNN模型"""

    def __init__(self):
        super(SmileCNN, self).__init__()

        # 卷积块1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.5)

        # 卷积块2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.5)

        # 卷积块3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        # 卷积块4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # 全连接层
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu2(self.conv2(x))))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.relu4(self.conv4(x))
        x = self.flatten(self.global_pool(x))
        x = self.sigmoid(self.fc(x))
        return x


# 数据增强变换
def get_transforms(augment=True):
    """获取数据变换"""
    if augment:
        # 训练时的数据增强
        return transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 验证和测试时的变换
        return transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


print("\n注意：CelebA数据集较大，需要单独下载")
print("数据集下载地址：https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
print("下载后解压到'./data/celeba'目录")


def train_smile_classifier():
    """训练微笑分类器"""
    try:
        # 尝试加载CelebA数据集
        transform_train = get_transforms(augment=True)
        transform_test = get_transforms(augment=False)

        # 加载数据集（这里假设数据集已下载）
        # train_dataset = datasets.CelebA('./data/celeba', split='train',
        #                                 target_type='attr', download=False,
        #                                 transform=transform_train)
        # test_dataset = datasets.CelebA('./data/celeba', split='test',
        #                                target_type='attr', download=False,
        #                                transform=transform_test)

        print("CelebA数据集加载代码已注释，需要下载数据集后取消注释")

        # 创建模型
        smile_model = SmileCNN()
        print(f"微笑分类模型参数量: {sum(p.numel() for p in smile_model.parameters()):,}")

        return smile_model

    except Exception as e:
        print(f"加载CelebA数据集时出错: {e}")
        print("请先下载CelebA数据集并解压到正确目录")
        return None


# 尝试训练微笑分类器（需要先下载数据集）
smile_model = train_smile_classifier()

# ==================== 第四部分：总结和可视化 ====================

print("\n" + "=" * 60)
print("CNN图像分类试验总结")
print("=" * 60)

# 1. 卷积操作可视化
print("\n1. 卷积操作基础实现:")
print("   - 实现了一维卷积和二维卷积的朴素算法")
print("   - 与NumPy和SciPy的结果进行了对比验证")

# 2. MNIST模型总结
print("\n2. MNIST手写数字识别:")
print(f"   - 测试集准确率: {test_accuracy:.2f}%")
print("   - 模型结构: 2个卷积层 + 2个池化层 + 2个全连接层")
print("   - 使用Dropout(0.5)防止过拟合")


# 3. 模型参数量对比
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 对比CNN和全连接网络的参数量
class SimpleMLP(nn.Module):
    """简单的全连接网络，用于对比参数量"""

    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


mlp_model = SimpleMLP()
cnn_params = count_parameters(mnist_model)
mlp_params = count_parameters(mlp_model)

print("\n3. 参数量对比:")
print(f"   - CNN模型参数量: {cnn_params:,}")
print(f"   - 同等容量MLP参数量: {mlp_params:,}")
print(f"   - 参数量减少比例: {(1 - cnn_params / mlp_params) * 100:.1f}%")


# 4. 卷积层输出可视化
def visualize_conv_filters(model, test_dataset):
    """可视化卷积层滤波器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 获取第一个卷积层的权重
    conv1_weights = model.conv1.weight.data.cpu().numpy()

    # 可视化前16个滤波器
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(min(32, len(axes))):
        ax = axes[i]
        filter_img = conv1_weights[i, 0]  # 第一个通道
        ax.imshow(filter_img, cmap='viridis')
        ax.set_title(f'Filter {i + 1}')
        ax.axis('off')

    plt.suptitle('第一个卷积层的滤波器可视化', fontsize=16)
    plt.tight_layout()
    plt.show()


# 可视化滤波器
visualize_conv_filters(mnist_model, mnist_test)


# 5. 特征图可视化
def visualize_feature_maps(model, test_dataset, image_idx=0):
    """可视化卷积层的特征图"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 获取一个测试图像
    img, target = test_dataset[image_idx]
    img_tensor = img.unsqueeze(0).to(device)

    # 创建钩子来获取特征图
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output.detach().cpu().numpy())

    # 注册钩子
    hook1 = model.conv1.register_forward_hook(hook_fn)
    hook2 = model.conv2.register_forward_hook(hook_fn)

    # 前向传播
    with torch.no_grad():
        _ = model(img_tensor)

    # 移除钩子
    hook1.remove()
    hook2.remove()

    # 可视化特征图
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 原始图像
    axes[0, 0].imshow(img.squeeze(), cmap='gray')
    axes[0, 0].set_title(f'Original Image (Label: {target})')
    axes[0, 0].axis('off')

    # 第一个卷积层的特征图（前4个）
    if len(feature_maps) > 0:
        for i in range(min(4, feature_maps[0].shape[1])):
            row = i // 2
            col = i % 2
            if row == 0 and col == 0:
                continue  # 跳过第一个位置（已显示原始图像）
            ax = axes[row, col]
            ax.imshow(feature_maps[0][0, i], cmap='viridis')
            ax.set_title(f'Conv1 Feature Map {i + 1}')
            ax.axis('off')

    plt.suptitle('卷积层特征图可视化', fontsize=16)
    plt.tight_layout()
    plt.show()


# 可视化特征图
visualize_feature_maps(mnist_model, mnist_test)
