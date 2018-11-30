## Question:

### NN
1. 推导多层感知机的BP算法？

2. NN中激活函数有哪些？

3. NN中损失函数有哪些？

4. NN中初始化参数如何选择？

5. NN参数更新方式？

### CNN
1. 推导CNN的BP算法？

2. 卷积如何实现？

3. 卷积层权值共享如何实现？

4. 梯度弥散/爆炸问题的原因，如何解决？

5. 池化作用是什么，有那些种类？

6. CNN压缩/加速有哪些方法？

### DL
1. 深度学习中预防过拟合的方法？

2. Drop out 流程？ 有哪些优点/为什么效果好？ 为什么能预防过拟合？

3. 介绍白化预处理？

4. Batch Normalization有哪些优点？

## 代码
1. 写出LR的损失函数，如何推导？

2. 编写函数计算softmax中的cross entropy loss。

### 框架
1. 介绍tensorflow的计算图

2. 介绍tensorflow的自动求导机制

## Answer:
### NN
1. 《机器学习》101-104页

2. Sigmoid, Tanh, ReLU, Leaky ReLU, Maxout。
详见[Neural Networks Part 1: Setting up the Architecture](http://cs231n.github.io/neural-networks-1/#actfun)

3. - 损失函数组成：数据损失+正则化。
 - 分类：SVM / softmax，类别数目巨大用分层Softmax(Hierarchical Softmax)；
 - 回归：L2平方范式或L1范式
 - RCNN： BCE + Smooth L1;
 - 详见[Neural Networks Part 2: Setting up the Data and the Loss](http://cs231n.github.io/neural-networks-2/#losses)

4. w = np.random.randn(n) * sqrt(2.0/n)
 详见[Neural Networks Part 2: Setting up the Data and the Loss](http://cs231n.github.io/neural-networks-2/#init)

5. SGD， Momentum， Nesterov， Adagrad， RMSprop， Adam
详见[Neural Networks Part 3: Learning and Evaluation](http://cs231n.github.io/neural-networks-3/#update)

### CNN
1. [推导CNN的BP算法](https://www.zybuluo.com/hanbingtao/note/485480)，注意卷积与cross-correlation(互相关)的区别。

2. - [caffe im2col](http://blog.csdn.net/jiongnima/article/details/69736844)

 - [Implementation as Matrix Multiplication](http://cs231n.github.io/convolutional-networks/)

3. 给一张输入图片，用一个filter去扫这张图，filter里面的数就叫权重，这张图每个位置是被同样的filter扫的，所以权重是一样的，也就是共享。每个滤波器学习一种feature。

4. - 原因解释：
梯度消失问题和梯度爆炸问题一般随着网络层数的增加会变得越来越明显。
反向传播公式中，hidden1的偏导由hidden2到hiddenL的权重乘积得到。当L较大时，w>1，连乘趋于无穷;w<0，连乘趋于0;
sigmoid function，趋于两级，梯度小。连续乘，BP结果很小。
 - 解决方法：
(1)激活函数的选择sigmoid -> relu-> prelu;(2)权重初始化;(3)逐层训练(4)Batch Normalization，Dropout;(5)RNN->LSTM; (6) Residual shortcut connection; (7)gradient clip

5. - **池化作用**：(1)逐渐降低数据体的空间尺寸，减少网络中参数的数量，使得计算资源耗费变少，(2)控制过拟合。
 - **种类**：max pooling，average pooling，L2-norm pooling。
 - **比较**：max pooling比avg pooling鲁棒性更好、处理过后边缘/平滑。

6. - 首先，CNN有冗余才能压缩/加速。其次，CNN中全连接、卷积层才有参数，主要针对这两个层，池化、ReLU没有参数。
 - (1)合理调超参数，在模型的损失函数中加入惩罚项，包括模型的Density和模型的Diversity。Density指的是模型参数的冗余度，就是零和极小值的多少；Diversity指的是参数的多样性，即如果参数能够聚类成为少数几个类别，那么就是多样性低，反之就是多样性丰富。
 - (2)全连接层参数较多，重点关注。可采用剪枝、量化（聚类存下标、编码、奇异值分解）。
 - (3)卷积层：低秩分解。

### DL
1. (1)数据：标注更多, augmentation; (2)正则化: 损失函数L1, L2 norm, Max norm, Dropout; (3)提前终止; (4)迁移学习; (5)ensemble

2.
- 介绍： 对于某一层中的每个节点，dropout技术使得该节点以一定的概率p不参与到训练的过程中（即前向传导时不参与计算，bp计算时不参与梯度更新）
- Drop out 训练流程： (1)对l层第j个神经元按照伯努利分布，生成一个随机数 (2)该神经元的输入乘上产生的随机数作为这个神经元新的输入 (3)再用该神经元的新的输入，卷积，得到输出 (4)**U1 = (np.random.rand(*H1.shape) < p) / p**, inverted dropout。
- Drop out 测试流程： 不随机失活。
- Drop out 优点： (1)通过dropout，节点之间的耦合度降低了，节点对于其他节点不再那么敏感了，这样就可以促使模型学到更加鲁棒的特征；(2)dropout layer层中的每个节点都没有得到充分的训练（因为它们只有p的出勤率），这样就避免了对于训练样本的过分学习；(3)在测试阶段，dropout layer的所有节点都用上了，这样就起到了ensemble的作用，ensemble能够有效地克服模型的过拟合。

3. 经过白化预处理后，数据满足条件：a、特征之间的相关性降低，这个就相当于pca；b、数据均值、标准差归一化，也就是使得每一维特征均值为0，标准差为1。缺点：然而白化计算量太大了，很不划算，还有就是白化不是处处可微的。

4.
- 训练过程：对于每一个batch：
```
m = K.mean(X, axis=-1, keepdims=True)#计算均值  
std = K.std(X, axis=-1, keepdims=True)#计算标准差  
X_normed = (X - m) / (std + self.epsilon)#归一化  
out = self.gamma * X_normed + self.beta#重构变换
```
- 测试过程： 均值来说直接计算所有batch u值的平均值；然后对于标准偏差采用每个batch σB的无偏估计;

- 优点：减小中间层数据分布发生的改变(Internal  Covariate Shift)
