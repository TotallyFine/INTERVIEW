## 使用两层的神经网络来分类

### 关于bias
bias的两种方式：

**第一种，将bias添加到数据中去。**```np.hstack([X, np.ones((X.shape[0], 1))])```将X从(num_train, 3072)变为(num_train, 3072+1)，然后在权重W处同样增加维度由原来的(3072, 10)变为(3072+1, 10)。assignment1中svm和softmax都是使用这种方法的。初始化的时候bias和W一起使用了较小的随机数值进行了初始化。

**第二种，分离式计算bias。**```hidden = np.maximum(0, X.dot(W1) + b1)```其中```b1 = np.zeros(hidden_size); W1.shape=(3072, hidden_size)```。hidden中的每一行对应一条数据，hidden[i, j]就是隐藏层中第i条数据的第j个神经元的值，在计算hidden[i, j]的时候要加上bias偏置，同时hidden[i, j+1]第j+1个神经元的bias偏置与第j个神经元的bias偏置不同。隐藏层有多少个神经元就有多少套连接到这个神经元的权重，同时也就有多少个偏置。里面的那个maximum是ReLU函数。

### 计算loss以及导数
关键就在于将矩阵的size对应就可以，矩阵相乘的时候，还有导数矩阵必须和参数矩阵的shape一样。
```python
dscores = softmax_output.copy() # d(loss)/d(score)  (num_train, 10)
dscores[range(N), list(y)] -= 1
dscores /= N
# h_output: (num_train, hidden_size)
# dW2: (hidden_size, 10)
grads['W2'] = h_output.T.dot(dscores) + reg * W2 # d(loss)/d(W2) = d(loss)/d(score) * d(score)/d(W2)
# bias本身是一个行向量，一整列的值都是一样的，而之前就已经dscores /= N，所以这里就不再需要求平均了
grads['b2'] = np.sum(dscores, axis = 0) # d(score)/d(b2)=1 thus d(loss)/d(b2)=d(loss)/d(score)

# h是隐藏层的输入(num_train, hidden_size)
# h本身并不需要更新，它只是一个中间的数据，只是需要对他求导然后反向传播到上一层
dh = dscores.dot(W2.T) # (num_train, hidden_size)
dh_ReLu = (h_output > 0) * dh # (num_train, hidden_size)对ReLU求导，如果是大于零的导数就是1 如果小于零那么导数为0
# X: (num_train, 3072)
grads['W1'] = X.T.dot(dh_ReLu) + reg * W1 # (3072, hidden_size)
grads['b1'] = np.sum(dh_ReLu, axis = 0)
```

### 前向传播
```python
h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
scores = h.dot(self.params['W2']) + self.params['b2'])
y_pred = np.argmax(scores, axis=1)
```

### 编程技巧
**随机选择batch** 
```python
idx = np.random.choice(num_train, batch_size, replace=True)
X_batch = X[idx]
y_batch = y[idx]
```

**记录学习率、loss、eval准确率的变化**
```python
loss_history.append(loss)
train_acc_history.append(train_acc)
val_acc_history.append(val_acc)

return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
}
```