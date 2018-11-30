## 图片分类
**Motivation** ：将图片进行分类是计算机视觉中的一个核心问题，其他的一些高级的计算机视觉的任务也都可以由图片分类延展开。

**Example** ：对于计算机来说图片并不是人眼看到的一样，而是一个3维矩阵，一个RGB模式的400X300的图片，在计算机看来就是一个400X300X3的矩阵。分类的任务就是将这些成千上万的矩阵中的数字转化为一个标签。

**Challenges** ：视角变换Viewpoint variation，尺度不同Scale variation，仿射变换Deformation，覆盖Occlusion，亮度条件Illumination conditions，背景聚集Background clutter，类内不同Intra-class variation

一个好的分类器必须能都处理上述问题，同时对类间的差别保持敏感。

**Data-driven approach**：实现分类问题并不是通过在代码中一个特征一个特征进行判断，而是通过给计算机看很多的图片来训练学习算法，这被称为数据驱动的方式。

**The image classifiaction pipline** ：Input --> Learning --> Evalution

### Nearset Neighbor Classifier
KNN算法没有涉及神经网络，而是通过大量的数据进行对比来得到的结果，一般的KNN算法使用L2距离也就是欧式距离来衡量，也可以使用L1距离，当训练数据和测试数据都进行过标准化的时候L1距离和L2距离的结果相同。

**不使用循环来计算L2距离的方法** ：

两个向量之间的L2距离可以变为d(X, Y) = sqrt((X1-Y1)^2 + (X2-Y2)^2 + ...)= sqrt(X1^2 + X2^2 + ... + Y1^2 + Y2^2 + ... -2\*(X1\*Y1+ X2\*Y2+...))。
dot点积就是矩阵乘法，内积是两个向量求出一个数字。
np.multiply(x1, x2)是将两个矩阵进行元素与元素之间的相乘。
必须矩阵的大小一致或者可以广播。
先求出X和X_train.T 的矩阵乘法。

```python
dists = np.multiply(np.dot(X, self.X_train.T), -2)
# 对X求平方和，求和后的shape(num_test, 1)
sq1 = np.sum(np.square(X), axis=1, keepdim=True)
# 对X_train求平方和
sq2 = np.sum(np.square(self.X_train), axis=1)
dists = np.add(dists, sq1)
dists = np.add(dists, sq2)
dists = np.sqrt(dists)
```
**超参数的设置**：在KNN中的唯一参数就是参考的样本个数也就是K的大小，一般通过5折验证。对于不同的k，将这5个数据子集依次留出一个子集作为验证集计算准确率，这样每个k就得到了5个准确率。但是实际中通常只留出一个数据子集作为固定的验证集，因为训练多个模型的开销太大了。但是不能使用测试集来进行超参数的设置，那样的话就相当于在测试集上进行训练了。

使用KNN的时候，K特别小的时候会导致过拟合，K较大则泛化能力较好。

KNN的优点：
1. 容易实现和理解
2. 不需要时间去训练，在测试的时候直接计算距离即可

KNN的缺点：
1. 测试时间耗费较长
2. 准确度低
3. 高维数据开销巨大，所以不适合图像分类
4. 对于图像的移动、缺失、亮度的变化不能正确分类
5. 需要存储全部的训练数据

在实践中使用KNN：
1. 对数据进行预处理，归一化
2. 如果数据的维度很高，进行降维PCA
3. 将数据随机分为train/val 70-90%的数据一般被分为训练集。如果有很多的超参数，那么应该有一个较大的验证集，最好进行交叉验证，虽然开销较大。
4. 使用不同的距离衡量方法L1 L2 在多个K的情况下建立模型
5. 如果KNN运行时间过长，可以考虑使用ANN
6. 记录给出最好结果的超参数，最后一般不在验证集上再进行训练

ANN：Approximate Nearest Neighbor，可以加速临近数据点的寻找，建立索引或者以空间换时间的方法。
