## 使用KNN进行分类
### NumPy及其他API技巧

**flatnonzero**在第四个代码框中，使用```np.flatnonzero(y_train == y)```来选出特定的样本：
```python
classes = ['planes', 'car', ...]
for y, cls in enumerate(classes):
    # y_train (50000L,)
    idxs = np.flatnonzero(y_train == y)
```

### 矩阵编程技巧
**一个循环计算两个矩阵之间的l2距离** knn需要计算测试数据和训练数据之间的l2距离来决定前k个和测试数据相近的图片。l2距离：两点先做差得到一个向量，然后向量每个值平方，然后求和，然后开方，可以用二维的例子来想想。训练数据：X\_train(num\_train, 3072L)，测试数据：X(num\_test, 3072L)。产生的矩阵的大小就应该是(num\_train, num\_test)，使用一个循环的情况如下：
```python
dists = np.zeros((num_test, num_train))
for i in range(num_test):
    dists[i] = np.sqrt(np.sum(np.square(X_train-X[i]), axis=1))
```

**无需循环计算两个矩阵之间的l2距离。** X\_train(num\_train, 3072L)训练数据X(num\_test, 3072L)。产生的矩阵的大小就应该是(num\_test, num\_train)。l2距离的公式$ \sqrt{(x-y)^2} = \sqrt{x^2 + y^2 - 2xy} $。所以先进行矩阵相乘得到$ -2xy $，然后加上$ x^2 \quad y^2 $：
```python
dists = np.zeros((num_test, num_train))
dists = np.sqrt(-2*np.dot(X, X_train.T) + np.sum(np.square(X_train), axis=1) + np.transpose([np.sum(np.square(X), axis=1)]))
```
其中```np.sum(np.square(X_train), axis=1)```先对训练数据中每个元素都取平方，然后求和得到维度为(num\_train,)的一维向量，在和前面的$ -2xy $相加的时候会自动进行广播变为(num_test, num_train)，其中每行是一样的。同样```np.transpose([np.sum(np.square(X), axis=1)])```每个元素先取平方，然后求和得到(num\_train,)的一维向量，再进行转置变成列向量，相加的时候也会进行广播变为(num\_test, num\_train)其中每列是一样的。

在使用numpy的时候一维的向量转置还是一维的向量，只有二维及以上才能进行转置。
```python
print(np.array([1,2,3]).T)
# array([1,2,3])
print(np.array([[1,2,3]]).T)
# array([[1],
#        [2],
#        [3]])
```