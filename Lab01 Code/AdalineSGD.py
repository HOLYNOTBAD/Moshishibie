"""2.2.4 随机梯度下降--AdalineSGD算法"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BasicLearningRule import plot_decision_regions
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

# 检查本地是否有文件
if not os.path.exists(local_path):
    print('Downloading data to local...')
    df = pd.read_csv(s, header=None, encoding='utf-8')
    df.to_csv(local_path, index=False, header=None, encoding='utf-8')
else:
    print(f'Loading data from local file: {local_path}')
    df = pd.read_csv(local_path, header=None, encoding='utf-8')

# 确保 images 目录存在
images_dir = os.path.join(script_dir, 'results', 'images')
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
    print(f'Created directory: {images_dir}')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values

class AdalineSGD:
    """ADAptive LInear NEuron classifier.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent 
        cycles.
    random_state : int
        Random number generator seed for random weight 
        initialization.
    
    
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function value averaged over all
        training examples in each epoch.
    
    
    """
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    
    def fit(self, X, y):
        """ Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of 
            examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
        
        Returns
        -------
        self : object
        
        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses) 
            self.losses_.append(avg_loss)
        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                   size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))
                        >= 0.5, 1, 0)

plt.clf() # 清除之前的绘图

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'SGD-Adaline随机梯度下降决策区域图.png'), dpi=300)
plt.clf() # 清除之前的绘图

plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_,
         marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'SGD-Adaline随机梯度下降损失函数变化图.png'), dpi=300)
plt.clf()