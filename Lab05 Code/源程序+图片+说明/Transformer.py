import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =========== 1. 位置编码类 ===========
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in the original paper"""

    def __init__(self, d_model, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if batch_first:
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        else:
            pe = pe.unsqueeze(1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            # x: [batch_size, seq_len, d_model]
            x = x + self.pe[:, :x.size(1), :]
        else:
            # x: [seq_len, batch_size, d_model]
            x = x + self.pe[:x.size(0), :]
        return x


# =========== 2. 缩放点积注意力 ===========
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism"""

    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 计算上下文向量
        context = torch.matmul(attention_weights, value)

        return context, attention_weights


# =========== 3. 多头注意力 ===========
class MultiHeadAttention(nn.Module):
    """Multi-head attention as described in the transformer paper"""

    def __init__(self, d_model, num_heads, dropout=0.1, batch_first=True):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.batch_first = batch_first

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def split_heads(self, x):
        """将输入分割为多个头"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """将多个头合并"""
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        # 残差连接
        residual = query

        # 线性变换并分割为多个头
        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))

        # 如果有掩码，扩展维度以适应多头
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        # 计算注意力
        context, attention_weights = self.attention(query, key, value, mask)

        # 合并多头并应用输出线性变换
        context = self.combine_heads(context)
        output = self.W_o(context)

        # 应用dropout和层归一化
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output, attention_weights


# =========== 4. 前馈网络 ===========
class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


# =========== 5. 编码器层 ===========
class EncoderLayer(nn.Module):
    """单个Transformer编码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, batch_first=True):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, batch_first)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # 多头自注意力
        x, attention_weights = self.self_attention(x, x, x, mask)
        # 前馈网络
        x = self.feed_forward(x)
        return x, attention_weights


# =========== 6. 改进的简化版Transformer分类器 ===========
class TransformerForClassification(nn.Module):
    """简化版Transformer，仅使用编码器部分进行文本分类"""

    def __init__(self, vocab_size, num_classes, d_model=64, num_layers=2,
                 num_heads=4, d_ff=128, max_seq_len=50, dropout=0.3):
        super(TransformerForClassification, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, batch_first=True)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # 分类头 - 修复维度问题
        # 我们使用平均池化 + 最大池化，所以输入维度是 2 * d_model
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * d_model, d_model)  # 修复：输入维度应该是 2*d_model
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """更好的权重初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [batch_size, seq_len]

        # 嵌入和位置编码
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.positional_encoding(embedded)  # [batch_size, seq_len, d_model]

        # 通过编码器层
        for layer in self.encoder_layers:
            embedded, _ = layer(embedded)

        # 使用平均池化 + 最大池化的组合
        avg_pool = embedded.mean(dim=1)  # 平均池化 [batch_size, d_model]
        max_pool, _ = embedded.max(dim=1)  # 最大池化 [batch_size, d_model]
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # 合并两种池化 [batch_size, 2*d_model]

        # 分类头
        pooled = self.dropout1(pooled)
        pooled = F.relu(self.fc1(pooled))  # [batch_size, d_model]
        pooled = self.dropout2(pooled)
        output = self.fc2(pooled)  # [batch_size, num_classes]

        return output


# =========== 7. 更有挑战性的模拟数据集 ===========
class ChallengingTextDataset(Dataset):
    """更有挑战性的文本分类数据集 - 增加难度"""

    def __init__(self, num_samples=5000, vocab_size=100, seq_len=30, num_classes=5):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_classes = num_classes

        # 生成更有挑战性的数据
        self.data = []
        self.labels = []

        # 为每个类别定义复杂的模式
        # 模式之间有重叠，且每个模式包含多个词
        self.class_patterns = {
            0: [[1, 2, 3], [4, 5, 6], [7, 8]],  # 类别0的模式组合
            1: [[4, 5, 6], [7, 8, 9], [10, 11]],  # 类别1的模式组合（与0有重叠）
            2: [[7, 8, 9], [10, 11, 12], [13, 14]],  # 类别2的模式组合（与1有重叠）
            3: [[10, 11, 12], [13, 14, 15], [16, 17]],  # 类别3的模式组合
            4: [[13, 14, 15], [16, 17, 18], [19, 20]]  # 类别4的模式组合（与3有重叠）
        }

        # 所有可能出现的词
        self.all_pattern_words = set()
        for patterns in self.class_patterns.values():
            for pattern in patterns:
                self.all_pattern_words.update(pattern)

        for _ in range(num_samples):
            # 随机选择类别
            label = torch.randint(0, num_classes, (1,)).item()

            # 创建序列：包含少量模式词和大量噪声词
            sequence = []
            patterns = self.class_patterns[label]

            # 确定要插入的模式数量（1-3个模式）
            num_patterns = random.randint(1, 3)

            # 随机选择要插入的模式
            selected_patterns = random.sample(patterns, min(num_patterns, len(patterns)))

            # 确定模式在序列中的位置
            pattern_positions = sorted(random.sample(range(seq_len), len(selected_patterns)))

            # 构建序列
            pattern_idx = 0
            for pos in range(seq_len):
                if pos in pattern_positions and pattern_idx < len(selected_patterns):
                    # 插入模式词
                    pattern = selected_patterns[pattern_idx]
                    # 从模式中随机选择一个词
                    word = random.choice(pattern)
                    sequence.append(word)
                    pattern_idx += 1
                else:
                    # 插入噪声词（不能是模式词）
                    while True:
                        word = random.randint(0, vocab_size - 1)
                        if word not in self.all_pattern_words:
                            break
                    sequence.append(word)

            self.data.append(torch.tensor(sequence))
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# =========== 8. 训练和评估函数 ===========
def train_transformer_classifier(dataset_class=ChallengingTextDataset):
    """训练简化版Transformer分类器"""
    # 设置超参数 - 增加难度
    vocab_size = 100
    num_classes = 5  # 增加类别数
    d_model = 64  # 减少维度
    num_layers = 2  # 减少层数
    num_heads = 4
    d_ff = 128
    max_seq_len = 30  # 增加序列长度
    batch_size = 32
    num_epochs = 20  # 增加epoch
    learning_rate = 0.001
    dropout = 0.3  # 增加dropout

    # 创建数据集和数据加载器 - 增加数据量
    train_dataset = dataset_class(num_samples=8000, vocab_size=vocab_size,
                                  seq_len=max_seq_len, num_classes=num_classes)
    val_dataset = dataset_class(num_samples=2000, vocab_size=vocab_size,
                                seq_len=max_seq_len, num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = TransformerForClassification(vocab_size, num_classes, d_model,
                                         num_layers, num_heads, d_ff, max_seq_len, dropout)
    model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加权重衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)  # 学习率调度

    # 训练循环
    print("开始训练...")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 统计
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = total_val_loss / len(val_loader)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 记录数据用于可视化
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
        print(f'  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')

        # 早停机制（如果验证损失连续5次没有改善）
        if epoch > 10 and avg_val_loss > max(val_losses[-6:-1]):
            print("验证损失没有改善，提前停止训练")
            break

    print("训练完成!")

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    return model, train_loader, val_loader, device, train_dataset, val_dataset


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """绘制训练和验证曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 准确率曲线
    axes[1].plot(train_accuracies, label='Train Accuracy')
    axes[1].plot(val_accuracies, label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('transformer_training_curves.png', dpi=150)
    plt.show()


def evaluate_model(model, dataloader, device, dataset_name="Test"):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"{dataset_name} Accuracy: {accuracy:.2f}%")

    # 生成分类报告
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(all_labels, all_predictions,
                                target_names=[f'Class{i}' for i in range(5)]))

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class{i}' for i in range(5)],
                yticklabels=[f'Class{i}' for i in range(5)])
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'transformer_confusion_matrix_{dataset_name}.png', dpi=150)
    plt.show()

    return accuracy, all_predictions, all_labels


# =========== 9. 改进的消融实验 ===========
def ablation_study(train_dataset, test_dataset, num_classes=5):
    """进行消融实验，分析不同组件的作用"""
    print("\n" + "=" * 50)
    print("Ablation Study - 消融实验")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实验1：基础模型（无注意力）
    class BaselineModel(nn.Module):
        def __init__(self, vocab_size=100, num_classes=5, d_model=64):
            super(BaselineModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.fc = nn.Linear(d_model, num_classes)

        def forward(self, x):
            embedded = self.embedding(x).mean(dim=1)  # 平均池化
            output = self.fc(embedded)
            return output

    # 实验2：只有注意力层（无位置编码）
    class AttentionOnlyModel(nn.Module):
        def __init__(self, vocab_size=100, num_classes=5, d_model=64, num_heads=4):
            super(AttentionOnlyModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.attention = MultiHeadAttention(d_model, num_heads, batch_first=True)
            self.fc = nn.Linear(d_model, num_classes)

        def forward(self, x):
            embedded = self.embedding(x)
            attended, _ = self.attention(embedded, embedded, embedded)
            pooled = attended.mean(dim=1)  # 平均池化
            output = self.fc(pooled)
            return output

    # 实验3：只有位置编码（无注意力）
    class PositionOnlyModel(nn.Module):
        def __init__(self, vocab_size=100, num_classes=5, d_model=64, max_seq_len=30):
            super(PositionOnlyModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, max_seq_len, batch_first=True)
            self.fc = nn.Linear(d_model, num_classes)

        def forward(self, x):
            embedded = self.embedding(x)
            embedded = self.positional_encoding(embedded)
            pooled = embedded.mean(dim=1)  # 平均池化
            output = self.fc(pooled)
            return output

    # 实验4：完整Transformer模型
    # 使用之前定义的TransformerForClassification类

    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    models = {
        "Baseline (No Attention)": BaselineModel(vocab_size=100, num_classes=num_classes).to(device),
        "Attention Only": AttentionOnlyModel(vocab_size=100, num_classes=num_classes).to(device),
        "Position Only": PositionOnlyModel(vocab_size=100, num_classes=num_classes).to(device),
        "Full Transformer": TransformerForClassification(100, num_classes).to(device)
    }

    results = {}
    for name, model in models.items():
        print(f"\n测试模型: {name}")
        # 简单训练一下（为了公平比较）
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 创建训练数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 训练10个epoch
        model.train()
        for epoch in range(10):
            total_loss = 0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 打印每个epoch的损失
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch + 1}/10, Loss: {total_loss / len(train_loader):.4f}")

        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        results[name] = accuracy
        print(f"测试准确率: {accuracy:.2f}%")

    # 绘制对比图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results.keys(), results.values(), color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    plt.xlabel('Model Architecture')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Ablation Study: Impact of Different Components')
    plt.ylim(0, 100)

    # 在柱子上添加数值
    for i, (name, acc) in enumerate(results.items()):
        plt.text(i, acc + 1, f'{acc:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('transformer_ablation_study.png', dpi=150)
    plt.show()

    return results


# =========== 10. 主函数 ===========
def main():
    """主函数：运行所有实验"""
    print("开始Transformer实验 - 挑战性数据集")
    print("=" * 60)

    # 实验1：训练改进的Transformer分类器
    print("\n" + "=" * 60)
    print("实验1：训练改进的Transformer分类器（挑战性数据集）")
    print("=" * 60)

    model, train_loader, val_loader, device, train_dataset, val_dataset = train_transformer_classifier()

    # 创建测试集
    test_dataset = ChallengingTextDataset(num_samples=2000, vocab_size=100,
                                          seq_len=30, num_classes=5)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 评估模型
    print("\n" + "=" * 60)
    print("实验2：模型评估")
    print("=" * 60)

    train_accuracy, train_preds, train_labels = evaluate_model(model, train_loader, device, "Train")
    val_accuracy, val_preds, val_labels = evaluate_model(model, val_loader, device, "Validation")
    test_accuracy, test_preds, test_labels = evaluate_model(model, test_loader, device, "Test")

    # 实验3：消融实验
    print("\n" + "=" * 60)
    print("实验3：消融实验")
    print("=" * 60)

    ablation_results = ablation_study(train_dataset, test_dataset, num_classes=5)

    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)

    # 总结结果
    print("\n实验总结:")
    print(f"训练集准确率: {train_accuracy:.2f}%")
    print(f"验证集准确率: {val_accuracy:.2f}%")
    print(f"测试集准确率: {test_accuracy:.2f}%")

    # 分析过拟合程度
    overfit_degree = train_accuracy - test_accuracy
    print(f"过拟合程度（训练-测试准确率差）: {overfit_degree:.2f}%")
    if overfit_degree > 15:
        print("警告：模型过拟合严重！")
    elif overfit_degree > 10:
        print("注意：模型有一定过拟合。")
    elif overfit_degree > 5:
        print("轻微过拟合。")
    else:
        print("良好：模型泛化能力较好。")

    # 消融实验结果分析
    print("\n消融实验结果:")
    for name, acc in ablation_results.items():
        print(f"{name}: {acc:.2f}%")

    # 性能提升分析
    baseline_acc = ablation_results.get("Baseline (No Attention)", 0)
    full_acc = ablation_results.get("Full Transformer", 0)
    if baseline_acc > 0 and full_acc > 0:
        improvement = full_acc - baseline_acc
        print(f"\nTransformer相对于Baseline的性能提升: {improvement:.2f}%")
        if improvement > 10:
            print("Transformer架构带来了显著的性能提升！")
        elif improvement > 5:
            print("Transformer架构有一定性能提升。")
        else:
            print("在当前任务中，Transformer架构的优势不明显。")


# =========== 运行主函数 ===========
if __name__ == "__main__":
    main()