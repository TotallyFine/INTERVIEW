## 使用softmax进行分类
大体上和线性svm分类差不多，只是loss函数不一样。softmax的公式: $ loss = -log(\frac{e^{score_j}}{\sum_{i}e^{score_i}}) $变形一下$ loss = -score_j + log(\sum_{i}e^{score_i}) $。

至于导数，因为是单层的线性分类器，所以用softmax的输出（一个向量(10,)）对softmax求导得到（一个向量，(10,)），然后用这个向量乘以这条训练数据得到softmax对于权重的导数。$ \frac{\partial{softmax}}{\partial{W}} = \frac{\partial{softmax}}{\partial{Z}} \frac{\partial{Z}}{\partial{W}} (Z = X.dot(W)) $。其中$ \frac{\partial{softmax}}{\partial{Z}} = softmax[j]-1 (j \ is \ correct \ class) $如果不是正确类别的话$ \frac{\partial{softmax}}{\partial{Z}} = softmax[j] (j \ is \ not \ correct \ class) $


### 使用循环来计算softmax的loss和梯度
```python
# 第i个数据
for i in xrange(num_train):
    scores = X[i].dot(W)
    shift_scores = scores - max(scores)
    # 这条数据计算出来的损失
    loss_i = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
    loss += loss_i
    # 对于这条数据得到的score，每个类别的score都计算导数
    # 单个的score计算导数相当于更新这个类别对应的那个分类器的导数所以有[:, j]
    for j in xrange(num_classes):
        softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
        # 如果这个类别是正确的类别
        if j == y[i]:
            dW[:,j] += (-1 + softmax_output) *X[i] 
        else: 
            dW[:,j] += softmax_output *X[i] 

loss /= num_train 
loss +=  0.5* reg * np.sum(W * W)
dW = dW/num_train + reg* W 
```

### 使用矩阵来计算softmax的loss和梯度
```python
num_classes = W.shape[1]
num_train = X.shape[0]
# 计算得到每个训练数据每个类别的score
scores = X.dot(W) # (num_train, 10)
# 下面的reshape是为了让计算得到的max在和score相减的时候进行广播，每行的max都是一样的
shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1) # (num_train, 10)
# 下面的reshape是为了能够进行广播
softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1) # (num_train, 10)
loss = -np.sum(np.log(softmax_output[range(num_train), list(y)])) # 正确类别的占比的负数
loss /= num_train 
loss +=  0.5* reg * np.sum(W * W)
  
dS = softmax_output.copy()
# 正确类别的减一，其他的直接和X矩阵相乘
dS[range(num_train), list(y)] += -1
dW = (X.T).dot(dS)
dW = dW/num_train + reg* W 
```

### 其他问题
**在刚初始化的时候，为什么希望计算出来的loss接近-log(0.1)?** 因为刚初始化的时候，每个类别的概率都应该相同，所以正确类别的概率应该是0.1，那么loss就应该是-log(0.1)。