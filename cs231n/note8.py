# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

# 这一节是一个两层的神经网络的例子

N = 100 # 每个类别的数据
D = 2 # 特征数
K = 3 # 3个类别
step_size = 1e-0 # 学习率
reg = 1e-3 # 正则化程度

def generating_data():
    """
    产生数据
    """
    X = np.zeros((N*K, D)) # 样本数据 每行是一条数据
    y = np.zeros(N*K, dtype='uint8') # label
    for j in xrange(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    # lets visualize the data:
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    return X, y
   
def train_linear(X, y):
    # 从正态分布中随机采样，并且乘以0.01 产生较小的数字
    # 但是这里没有考虑经过每层的输出是否是分布相同的
    # 1/sqrt(n_out)
    W = 0.01 * np.random.rand(D, K)    
    b = np.zeros((1, K))

    num_examples = X.shape[0]
    for i in xrange(200):
        scores = np.dot(X, W) + b # （N, K）
        # 计算概率
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (N, K)
        # 得到正确的概率 索引得到正确的概率
        correct_logprobs = -np.log(probs[range(num_examples), y]) 
        data_loss = np.sum(correct_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(W*W) # 0.5是为了求导消去
        loss = data_loss + reg_loss
        if i % 10 == 0:
            print 'iteration {}: loss {}'.format(i, loss)
        
        # 对x*W的结果进行求导
        # sigmoid求导f'(x) = y(1-y)
        # crossEntropyLoss求导 if j=i f'(x)=y_
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)

        dW += reg*W # 正则化的导数部分

        # 更新参数
        W += -step_size * dW
        b += -step_size * db

    scores = np.dot(X, W) + b
    predicted_class = np.argmax(scores, axis=1)
    print 'training acuracy: {}'.format(np.mean(predicted_class == y))

def train_neural(X, y):
    h = 100 # 隐藏层神经元数目
    W = 0.01 * np.random.rand(D, h)
    b = np.zeros((1, h))
    W2 = 0.01 * np.random.rand(h, K)
    b2 = np.zeros((1,K))
    num_examples = X.shape[0]
    for i in xrange(10000):
  
        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2
  
        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
        # compute the loss: average cross-entropy loss and regularization
        correct_logprobs = -np.log(probs[range(num_examples),y])
        data_loss = np.sum(correct_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            print "iteration %d: loss %f" % (i, loss)
  
        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),y] -= 1
        dscores /= num_examples
  
        # 计算W2 b2的导数
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # ReLU激活函数进行反向传播
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)
  
        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W
  
        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print 'training accuracy: %.2f' % (np.mean(predicted_class == y))
 
if __name__ == '__main__':
    X, y = generating_data()
    train_neural(X, y)