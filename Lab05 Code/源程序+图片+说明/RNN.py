import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import re
import os
import time
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import math

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# ==================== 实验1：情感分析 ====================
class SentimentDataset(Dataset):
    """情感分析数据集（完全自包含，不依赖外部数据）"""

    def __init__(self, num_samples=2000, seq_length=50):
        self.num_samples = num_samples
        self.seq_length = seq_length

        # 创建词汇表
        self.create_vocabulary()

        # 生成数据
        self.generate_data()

    def create_vocabulary(self):
        """创建词汇表"""
        # 情感相关词汇
        positive_words = [
            'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant',
            'outstanding', 'superb', 'marvelous', 'terrific', 'perfect',
            'great', 'good', 'awesome', 'best', 'love', 'enjoyed', 'liked',
            'enjoyable', 'pleasing', 'satisfying'
        ]

        negative_words = [
            'terrible', 'awful', 'horrible', 'bad', 'worst',
            'poor', 'boring', 'disappointing', 'dull', 'annoying',
            'hate', 'disliked', 'waste', 'rubbish', 'garbage',
            'mediocre', 'average', 'ordinary', 'uninteresting', 'forgettable'
        ]

        neutral_words = [
            'movie', 'film', 'acting', 'story', 'plot',
            'character', 'director', 'cinema', 'scene', 'performance',
            'actor', 'actress', 'screenplay', 'dialogue', 'ending',
            'beginning', 'middle', 'climax', 'resolution', 'theater'
        ]

        # 构建完整词汇表
        all_words = ['<pad>', '<unk>'] + positive_words + negative_words + neutral_words
        self.vocab = {word: idx for idx, word in enumerate(all_words)}
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # 词汇类别信息（用于生成数据）
        self.positive_words = positive_words
        self.negative_words = negative_words
        self.neutral_words = neutral_words

    def generate_data(self):
        """生成模拟数据"""
        self.data = []
        self.labels = []

        print(f"生成 {self.num_samples} 个样本...")

        for i in range(self.num_samples):
            if i < self.num_samples // 2:
                # 正面评论
                label = 1
                word_pool = self.positive_words
            else:
                # 负面评论
                label = 0
                word_pool = self.negative_words

            # 生成评论
            review_words = []

            # 添加情感词（3-6个）
            num_sentiment_words = np.random.randint(3, 7)
            for _ in range(num_sentiment_words):
                review_words.append(np.random.choice(word_pool))

            # 添加中性词（4-8个）
            num_neutral_words = np.random.randint(4, 9)
            for _ in range(num_neutral_words):
                review_words.append(np.random.choice(self.neutral_words))

            # 打乱顺序
            np.random.shuffle(review_words)

            # 转换为索引
            indices = []
            for word in review_words:
                indices.append(self.vocab[word])

            # 截断或填充
            if len(indices) > self.seq_length:
                indices = indices[:self.seq_length]
            else:
                indices = indices + [0] * (self.seq_length - len(indices))

            self.data.append(indices)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

    def decode_review(self, indices):
        """将索引序列解码为文本"""
        words = []
        for idx in indices:
            if idx == 0:  # <pad>
                continue
            if idx in self.idx_to_word:
                words.append(self.idx_to_word[idx])
        return ' '.join(words)


class SentimentRNN(nn.Module):
    """情感分析RNN模型"""

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM层（双向）
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)

        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 嵌入层
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # LSTM层
        lstm_out, (hidden, _) = self.lstm(embedded)

        # 使用双向LSTM的最后一个隐藏状态
        hidden_forward = hidden[-2, :, :]  # 前向LSTM的最后一个隐藏状态
        hidden_backward = hidden[-1, :, :]  # 后向LSTM的最后一个隐藏状态
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)

        # 全连接层
        out = self.fc1(hidden_concat)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out.squeeze()


class SentimentTrainer:
    """情感分析训练器"""

    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 损失函数和优化器
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 记录训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 统计
            total_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # 每10个batch打印一次进度
            if batch_idx % 10 == 0:
                accuracy = total_correct / total_samples if total_samples > 0 else 0
                print(f'  Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.4f}, Acc={accuracy:.4f}')

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        return avg_loss, accuracy

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        return avg_loss, accuracy

    def train(self, num_epochs=10, patience=3):
        """训练模型"""
        print("开始训练模型...")
        print("=" * 60)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.evaluate(self.val_loader)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印结果
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_sentiment_model.pth')
                print(f"  ✓ 模型已保存 (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ⚠ 早停触发，停止训练")
                    break

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_sentiment_model.pth'))

        print("\n" + "=" * 60)
        print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")

        return self.history


def run_sentiment_experiment():
    """运行情感分析实验"""
    print("=" * 60)
    print("实验1: 情感分析（电影评论分类）")
    print("=" * 60)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 准备数据
    print("\n1. 准备数据...")
    dataset = SentimentDataset(num_samples=2000, seq_length=50)

    # 分割数据集
    train_size = int(0.7 * len(dataset))  # 70% 训练
    val_size = int(0.15 * len(dataset))  # 15% 验证
    test_size = len(dataset) - train_size - val_size  # 15% 测试

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"数据集统计:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  词汇表大小: {dataset.vocab_size}")

    # 2. 创建模型
    print("\n2. 创建模型...")
    model = SentimentRNN(
        vocab_size=dataset.vocab_size,
        embed_dim=64,
        hidden_dim=64,
        dropout=0.3
    )

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 3. 训练模型
    print("\n3. 训练模型...")
    trainer = SentimentTrainer(model, train_loader, val_loader, device)
    history = trainer.train(num_epochs=10, patience=3)

    # 4. 测试模型
    print("\n4. 测试模型...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"测试集结果:")
    print(f"  损失: {test_loss:.4f}")
    print(f"  准确率: {test_acc:.4f}")

    # 5. 可视化结果
    print("\n5. 可视化结果...")
    visualize_training(history)

    # 6. 示例预测
    print("\n6. 示例预测...")
    model.eval()

    # 创建一些测试样本
    test_samples = [
        # 正面评论
        ["excellent movie fantastic acting wonderful story"],
        ["great film superb performance amazing direction"],
        ["love this movie best film ever perfect cinema"],

        # 负面评论
        ["terrible movie awful acting boring story"],
        ["bad film worst performance disappointing ending"],
        ["hate this movie waste of time rubbish film"],

        # 混合情感
        ["good acting but poor story average film"],
        ["interesting concept but bad execution mediocre movie"]
    ]

    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            text = sample[0]
            words = text.split()

            # 转换为索引
            indices = []
            for word in words:
                if word in dataset.vocab:
                    indices.append(dataset.vocab[word])
                else:
                    indices.append(dataset.vocab['<unk>'])

            # 填充到固定长度
            if len(indices) < 50:
                indices = indices + [0] * (50 - len(indices))

            input_tensor = torch.tensor([indices]).to(device)
            output = model(input_tensor)

            sentiment = "正面" if output.item() > 0.5 else "负面"
            confidence = output.item() if output.item() > 0.5 else 1 - output.item()

            print(f"\n示例 {i + 1}:")
            print(f"  评论: {text}")
            print(f"  预测: {sentiment} (置信度: {confidence:.2%})")

    print("\n" + "=" * 60)
    print(f"实验完成！最终测试准确率: {test_acc:.2%}")
    print("=" * 60)

    return model, dataset, test_acc


def visualize_training(history):
    """可视化训练过程"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='训练损失', marker='o', linewidth=2)
    axes[0].plot(history['val_loss'], label='验证损失', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练和验证损失曲线', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#f8f9fa')

    # 准确率曲线
    axes[1].plot(history['train_acc'], label='训练准确率', marker='o', linewidth=2)
    axes[1].plot(history['val_acc'], label='验证准确率', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('训练和验证准确率曲线', fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('#f8f9fa')

    # 设置y轴为百分比
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.suptitle('情感分析模型训练过程', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sentiment_training.png', dpi=120, bbox_inches='tight')
    plt.show()


# ==================== 实验2：字符级文本生成 ====================
class TextGenerator:
    """文本生成器"""

    def __init__(self, seq_length=40):
        self.seq_length = seq_length
        self.text = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.chars = None

    def create_text(self):
        """创建训练文本（莎士比亚风格）"""
        text = """
        To be or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles,
        And by opposing end them. To die, to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to: 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep, perchance to dream, ay, there's the rub:
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause. There's the respect
        That makes calamity of so long life.

        For who would bear the whips and scorns of time,
        The oppressor's wrong, the proud man's contumely,
        The pangs of despised love, the law's delay,
        The insolence of office, and the spurns
        That patient merit of the unworthy takes,
        When he himself might his quietus make
        With a bare bodkin? Who would fardels bear,
        To grunt and sweat under a weary life,
        But that the dread of something after death,
        The undiscover'd country from whose bourn
        No traveller returns, puzzles the will,
        And makes us rather bear those ills we have
        Than fly to others that we know not of?
        Thus conscience does make cowards of us all,
        And thus the native hue of resolution
        Is sicklied o'er with the pale cast of thought,
        And enterprises of great pith and moment
        With this regard their currents turn awry,
        And lose the name of action.
        """

        # 重复几次以增加数据量
        self.text = (text * 3).lower()
        return self.text

    def preprocess(self):
        """预处理文本"""
        if self.text is None:
            self.create_text()

        # 清理文本
        text = self.text
        text = re.sub(r'\s+', ' ', text).strip()

        # 获取所有唯一字符
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        print(f"文本统计:")
        print(f"  总字符数: {len(text)}")
        print(f"  唯一字符数: {len(self.chars)}")
        print(f"  字符集: {''.join(self.chars[:30])}...")

        # 将文本转换为索引
        self.text_indices = [self.char_to_idx[ch] for ch in text]

        # 创建训练序列
        self.create_sequences()

        return self

    def create_sequences(self):
        """创建训练序列"""
        sequences = []
        targets = []

        # 使用步长3，减少重叠
        step = 3
        for i in range(0, len(self.text_indices) - self.seq_length, step):
            seq = self.text_indices[i:i + self.seq_length]
            target = self.text_indices[i + 1:i + self.seq_length + 1]

            sequences.append(seq)
            targets.append(target)

        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

        print(f"创建了 {len(self.sequences)} 个训练序列")

        # 创建数据集
        self.dataset = torch.utils.data.TensorDataset(self.sequences, self.targets)

        return self

    def get_dataloader(self, batch_size=64):
        """获取数据加载器"""
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        return dataloader


class CharRNN(nn.Module):
    """字符级RNN模型"""

    def __init__(self, vocab_size, embed_dim=32, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM层
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # 嵌入
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # LSTM
        if hidden is not None:
            lstm_out, hidden = self.lstm(embedded, hidden)
        else:
            lstm_out, hidden = self.lstm(embedded)

        # 只使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 全连接层
        out = self.dropout(last_output)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """初始化隐藏状态"""
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        )


def train_text_generation_model():
    """训练文本生成模型"""
    print("\n" + "=" * 60)
    print("实验2: 字符级文本生成")
    print("=" * 60)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 准备数据
    print("\n1. 准备数据...")
    generator = TextGenerator(seq_length=40)
    generator.preprocess()
    dataloader = generator.get_dataloader(batch_size=64)

    # 2. 创建模型
    print("\n2. 创建模型...")
    vocab_size = len(generator.chars)
    model = CharRNN(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    )

    model = model.to(device)

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  词汇表大小: {vocab_size}")

    # 3. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # 4. 训练模型
    print("\n3. 训练模型...")
    num_epochs = 15
    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_items = 0

        for batch_idx, (seq_batch, target_batch) in enumerate(dataloader):
            seq_batch, target_batch = seq_batch.to(device), target_batch.to(device)

            # 初始化隐藏状态
            hidden = model.init_hidden(seq_batch.size(0), device)

            # 前向传播
            optimizer.zero_grad()
            output, hidden = model(seq_batch, hidden)

            # 计算损失（只使用最后一个字符的预测）
            loss = criterion(output, target_batch[:, -1])

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新权重
            optimizer.step()

            # 统计
            total_loss += loss.item() * seq_batch.size(0)
            total_items += seq_batch.size(0)

            if batch_idx % 10 == 0:
                print(f'  Epoch {epoch + 1}, Batch {batch_idx}: Loss={loss.item():.4f}')

        # 计算平均损失
        avg_loss = total_loss / total_items if total_items > 0 else 0
        losses.append(avg_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}')

        # 每3个epoch生成一些示例文本
        if (epoch + 1) % 3 == 0:
            print("\n生成示例文本:")
            generated = generate_text(
                model, generator,
                "to be ",
                length=80,
                temperature=0.8,
                device=device
            )
            print(generated)
            print()

    # 5. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': generator.char_to_idx,
        'idx_to_char': generator.idx_to_char,
        'chars': generator.chars
    }, 'text_generation_model.pth')

    print("\n模型已保存到 'text_generation_model.pth'")

    # 6. 可视化训练损失
    visualize_text_generation_loss(losses)

    # 7. 生成不同温度下的文本
    print("\n4. 不同温度下的文本生成示例:")
    temperatures = [0.5, 0.8, 1.0, 1.2]
    start_strings = ["to be", "whether", "the heart", "to die"]

    for start_str in start_strings:
        print(f"\n起始字符串: '{start_str}'")
        print("-" * 50)

        for temp in temperatures:
            print(f"\n温度 {temp}:")
            generated = generate_text(
                model, generator,
                start_str,
                length=60,
                temperature=temp,
                device=device
            )
            print(generated)

    return model, generator, losses[-1]


def generate_text(model, generator, start_string, length=100, temperature=1.0, device='cpu'):
    """生成文本"""
    model.eval()

    # 将起始字符串转换为索引
    input_indices = []
    for ch in start_string.lower():
        if ch in generator.char_to_idx:
            input_indices.append(generator.char_to_idx[ch])
        else:
            # 如果字符不在词汇表中，使用空格
            input_indices.append(generator.char_to_idx.get(' ', 0))

    # 初始化隐藏状态
    hidden = model.init_hidden(1, device)

    # 生成文本
    generated_text = start_string

    with torch.no_grad():
        # 首先处理起始字符串
        for i in range(len(input_indices)):
            input_tensor = torch.tensor([[input_indices[i]]]).to(device)
            _, hidden = model(input_tensor, hidden)

        # 最后一个字符作为第一个预测的输入
        last_char = input_indices[-1] if input_indices else generator.char_to_idx.get(' ', 0)

        for i in range(length):
            input_tensor = torch.tensor([[last_char]]).to(device)
            output, hidden = model(input_tensor, hidden)

            # 获取输出
            output = output[0] / temperature

            # 应用softmax获取概率
            probabilities = F.softmax(output, dim=-1).cpu().numpy()

            # 根据概率采样下一个字符
            next_idx = np.random.choice(len(generator.chars), p=probabilities)

            # 添加到生成的文本中
            next_char = generator.idx_to_char[next_idx]
            generated_text += next_char

            # 准备下一个输入
            last_char = next_idx

    return generated_text


def visualize_text_generation_loss(losses):
    """可视化文本生成训练损失"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', linewidth=2, color='#e74c3c')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('文本生成模型训练损失曲线', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.fill_between(range(len(losses)), losses, alpha=0.3, color='#e74c3c')
    plt.xticks(range(len(losses)))
    plt.gca().set_facecolor('#f8f9fa')

    # 添加最小损失标记
    min_loss_idx = np.argmin(losses)
    plt.scatter(min_loss_idx, losses[min_loss_idx], color='#2ecc71', s=100, zorder=5,
                label=f'最小损失: {losses[min_loss_idx]:.4f}')
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('text_generation_loss.png', dpi=120, bbox_inches='tight')
    plt.show()


# ==================== 主程序 ====================
def main():
    """主程序"""
    print("=" * 60)
    print("RNN序列建模实验 - 最终版本")
    print("=" * 60)
    print("本版本特点:")
    print("1. 完全自包含，不依赖外部数据集")
    print("2. 完全兼容PyTorch 2.9.1+")
    print("3. 包含两个完整实验:")
    print("   - 实验1: 情感分析（电影评论分类）")
    print("   - 实验2: 字符级文本生成")
    print("=" * 60)

    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NumPy版本: {np.__version__}")

    # 运行情感分析实验
    sentiment_model, sentiment_dataset, sentiment_acc = run_sentiment_experiment()

    # 运行文本生成实验
    text_model, text_generator, text_loss = train_text_generation_model()

    # 实验总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"1. 情感分析实验:")
    print(f"   - 测试准确率: {sentiment_acc:.2%}")
    print(f"   - 模型已保存: 'best_sentiment_model.pth'")
    print(f"   - 可视化结果: 'sentiment_training.png'")
    print()
    print(f"2. 文本生成实验:")
    print(f"   - 最终训练损失: {text_loss:.4f}")
    print(f"   - 模型已保存: 'text_generation_model.pth'")
    print(f"   - 可视化结果: 'text_generation_loss.png'")
    print()
    print(f"3. RNN原理演示:")
    print(f"   - RNN通过循环连接保持对历史信息的记忆")
    print(f"   - LSTM通过门控机制解决梯度消失问题")
    print(f"   - 双向RNN从两个方向处理序列信息")
    print("=" * 60)

    # 展示模型结构
    print("\n模型结构总结:")
    print("-" * 40)
    print("情感分析模型 (SentimentRNN):")
    print("  输入层 → 嵌入层 → 双向LSTM → 全连接层 → 输出层")
    print(f"  参数数量: {sum(p.numel() for p in sentiment_model.parameters()):,}")
    print()
    print("文本生成模型 (CharRNN):")
    print("  输入层 → 嵌入层 → LSTM → 全连接层 → 输出层")
    print(f"  参数数量: {sum(p.numel() for p in text_model.parameters()):,}")
    print("-" * 40)

    return sentiment_model, text_model


# ==================== 运行程序 ====================
if __name__ == "__main__":
    # 设置中文字体（如果支持）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    # 运行主程序
    main()