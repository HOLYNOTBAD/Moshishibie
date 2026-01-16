"""2.2.2 对鸢尾花数据集训练感知机模型"""
import os
import pandas as pd
## 读取数据

s = 'https://archive.ics.uci.edu/ml/'+'machine-learning-databases/iris/iris.data'
print('From URL:', s)

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 定义 data 文件夹路径
data_dir = os.path.join(script_dir, 'data')
# 定义本地保存路径
local_path = os.path.join(data_dir, 'iris.data')

# 确保 data 目录存在
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 检查本地是否有文件，没有则下载保存，有则直接读取
if not os.path.exists(local_path):
    print('Downloading data to local...')
    # 从 URL 读取
    df = pd.read_csv(s, header=None, encoding='utf-8')
    # 保存到本地 data 文件夹，不保存索引，不保存表头（因为原数据没表头）
    df.to_csv(local_path, index=False, header=None, encoding='utf-8')
    print(f'Data saved to {local_path}')
else:
    print(f'Loading data from local file: {local_path}')
    df = pd.read_csv(local_path, header=None, encoding='utf-8')
    print(f'done')
# print('Dataset shape:', df.shape)
# print(df.tail())
# Dataset shape: (150, 5)
#        0    1    2    3               4
# 145  6.7  3.0  5.2  2.3  Iris-virginica/setosa/versicolor（是Iris的三个品种）
# 146  6.3  2.5  5.0  1.9  Iris-virginica
# 147  6.5  3.0  5.2  2.0  Iris-virginica
# 148  6.2  3.4  5.4  2.3  Iris-virginica
# 149  5.9  3.0  5.1  1.8  Iris-virginica

# 打印前5行数据以确认读取成功
print('First 5 rows of data:')
print(df.head())

# 确保 images 目录存在
images_dir = os.path.join(script_dir, 'results', 'images')
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
    print(f'Created directory: {images_dir}')

import matplotlib.pyplot as plt
import numpy as np
# select setosa and versicolor，数据的前100行
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1) # Iris-setosa标记为0，Iris-versicolor标记为1
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
# plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
# 保存
plt.savefig(os.path.join(images_dir, 'BasicLearningRule-setosa和versicolor鸢尾花的花萼长度及花瓣长度散点图.png'), dpi=300)
plt.clf() # 清除之前的绘图

from Perceptron import Perceptron
ppn = Perceptron(eta=0.3, n_iter=10) # 学习率0.1，迭代10次

## 训练感知机模型,利用Perceptron类的fit方法
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
print(ppn.errors_)
plt.xlabel('Epochs')
plt.ylabel('Number of discorrect predictions')
plt.savefig(os.path.join(images_dir, 'BasicLearningRule-错误归类数对迭代次数的折线图.png'), dpi=300)
plt.clf() # 清除之前的绘图，防止折线图重叠到下一张图上

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.savefig(os.path.join(images_dir, 'BasicLearningRule-感知机决策区域图.png'), dpi=300)
