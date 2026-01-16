"""2.2.1 面向对象的感知机API"""
import numpy as np
class Perceptron:
    """Perceptron classifier.
    
    Parameters
    ------------
    eta : 浮点数
    学习率（取值范围0.0到1.0）
    n_iter : 整数
    对训练数据集的遍历次数
    random_state : 整数
    用于随机权重初始化的随机数生成器种子
    
    Attributes
    -----------
    w_ : 一维数组
    训练后的权重向量
    b_ : 标量
    训练后的偏置单元
    errors_ : 列表
    每个训练周期中的错误分类次数（权重更新次数）
    
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit training data.
        
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
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0) # 如果发生了更新（即预测错了），这个表达式的值为 1，否则为 0
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
