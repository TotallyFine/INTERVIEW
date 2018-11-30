### Data Preprocessing
假设输入的数据是一个矩阵X，(N, D)N是数据的数目，D是特征数。一般来说有三种方法对数据进行预处理。

**Mean subtraction**
在每个特征上独立减去各自的均值，实现对数据zero-center以零为中心。
```python
# 在numpy中可以直接减去均值
X -= np.mean(X, axis=0)

# 对于图片可以直接减去一个值，或者各个通道自己减去均值
X -= np.mean(X)
```

**Normalization**
两种方法：
1. 当数据变成zero-center之后，```X /= np.std(X, axis=0)```
2. 标准化各个维度的数据使得最大的值为1 最小的值为-1.只有当不同的特征有不同的scale尺度的时候使用这种方法，然而对图片来说都是0-255.所以一般在图像中不适用这种方法。

**注意**标准化只能在训练集上应用，并提取出均值和方差，再应用于测试集和验证集。而不是先在所有数据中提取均值和方差，然后应用于各个集合中。

**PCA以及白化**
数据先中心化zero-center，然后计算相关系数矩阵。
```python
# Assume input data matrix X of size [N x D]
X -= np.mean(X, axis = 0) # zero-center the data (important)
cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
```
cov[i,j]包含了第i个特征和第j个特征之间的相关系数，cov的对角线上是方差，cov是对称矩阵、正定矩阵。然后进行奇异值分解SVD
```python
U,S,V = np.linalg.svd(cov)
```
U中的列向量就是奇异向量，S是一个包含奇异值的一维向量（已经排序），为了去相关性，将中心化之后的数据乘以U。
```python
Xrot = np.dot(X, U) # 去相关性 X rotate (N, D)
```
注意到U中的列向量（奇异向量）是互相正交的，可以看作是一个新的坐标体系。和中心化之后的数据相乘相当于让数据进行旋转。相乘后Xrot矩阵的相关系数组成的矩阵是对角矩阵，这意味着Xrot中的每列已经不再互相关联了。同时可以使用奇异值进行降维，S中的每个奇异值都和U中奇异向量一一对应，只需要少数几个U中的奇异向量就可以代表整个数据。
```python
Xrot_reduced = np.dot(X, U[:, :100]) # (N, 100)
```
PCA是针对方阵的，而SVD是PCA的更普遍版本，对于非方阵也可以处理。

**白化 whitening**
白化操作接受PCA之后的数据并且将每个特征都除以对应的奇异值来统一尺度scale。
```python
Xwhite = Xrot / np.sqrt(S + 1e-5) # (N, D)
```
白化效果：
1. 这个矩阵将变成均值为0，方差都相同的矩阵。加上1e-5是为了防止下溢和除以零，同时进行平滑作用。
2. 可视化白化的效果，就好像所有维度的数据都被压缩squash进了一个圈中（二维看来是一个圈）。

白化的缺点：
1. 会极大的扩大噪音。白化会在所有的特征维度都进行，尤其是那些有微小方差的与其他维度相关性小的维度也会收到影响。这一点可以通过更强的平滑系数，即将1e-5增大来改善。
2. 白化计算量大，对于图像来说协方差矩阵太大计算量太大。所以卷积神经网路中一般不使用白化，但是标准化是需要的。

### Weight Initialization
**不能全部初始化为同样的数字/0**
一开始我们并不知道权重最后会被优化成什么样子，但是预处理过后我们可以假设到初始化的权重大概有一半为负数一半为正数。但是这不代表可以将所有的权重都初始化为零，这会导致所有神经元输出同样的数据，反向传播同样的数据，这导致所有的神经元都是相同的。

**随机的小数值**
我们仍希望初始化的权重接近零，但又不是零。所以使用随机的互相不同的小数值。这也被看作是_产生不同的神经元_。```W = 0.01*np.random.rand(D, H)```rand函数从正态分布(高斯分布)中进行采样。
但是随机的小数值也不一定会产生好的效果，这可能会导致很小的梯度，可能会导致梯度消失。

**使用1/sqrt(n)校准方差**
使用随机的小数值进行初始化的一个缺点就是，每个神经元的输入都是有方差的，并且会随着神经元数目的增多而累计。所以通过除以sqrt(fan\_in)来消除方差(fan\_in是输入的神经元的数目)。
```python
w = np.random.rand(n) / sqrt(n)
```
1. 保证所有神经元的输出都处于一个相同的分布中
2. 提高了收敛的速度
为什么除以sqrt(n)的推导过程，$ s = \sum_i^nw_ix_i $，下面是$ s $的方差推导
$$ % <![CDATA[
\begin{align}
\text{Var}(s) &= \text{Var}(\sum_i^n w_ix_i) \\\\
&= \sum_i^n \text{Var}(w_ix_i) \\\\
&= \sum_i^n [E(w_i)]^2\text{Var}(x_i) + E[(x_i)]^2\text{Var}(w_i) + \text{Var}(x_i)\text{Var}(w_i) \\\\
&= \sum_i^n \text{Var}(x_i)\text{Var}(w_i) \\\\
&= \left( n \text{Var}(w) \right) \text{Var}(x)
\end{align} %]]> $$
第一二步是方差的性质，第三步因为假设输入的x的均值是0所以$ E(x_i) = E(w_i) = 0 $，最后一步是因为假设了$ w_i x_i $是同样的分布。所以如果希望神经元的输出$ s $是都处于同样分布中的，那么就需要使得$ w $的方差是$ 1/n $。且因为$ Var(aX) = a^2Var(X) $(a是标量 X是随机变量)，所以从正态分布中采样之后就就除以$ a = \sqrt(1/n) $。

Glorot Xavier提出了从$ Var(w) = 2/(n_{in} + n_{out}) $中进行初始化，n\_in是输入的神经元的数目 n\_out是输出的神经元的数目。
```python
w=np.random.rand(n_int, n_out) / np.sqrt(2/(n_in+n_out))
```

何凯明的针对ReLU的MSAR初始化方法，他们证明了使用ReLU的神经网络方差应该为$ 2.0/n_{in} $所以他们的初始化方法为：
```python
w = np.random.rand(n) * sqrt(2.0/n)
```

**Sparse Initialization**
稀疏初始化，另一个解决不统一的方差。把所有的权重都设置为随机的小数值，并且随机让他们进行联结固定数目的下一层神经元，一般是10个神经元。

**Initializing the biases**
把bias初始化为0是很常见的方法。使用ReLU的时候很多人会把bias初始化为固定的较小的常数如0.01，这样可以使得一开始ReLU能够被激活，并且得到一些导数值，但是目前并不清楚这种方法有什么好处，好像训练的效果还会变坏。所以一般还是将bias初始化为0.

**Batch Normalization**
BN可以消除很多的初始化问题，在全连接层或者卷积层之后加上激活函数。BN对于差的初始化有更好的鲁棒性。BN也可以看作是每一层都进行了预处理，但是这个预处理以可微分的形式融合进了神经网络本身里。

### Regularization
控制神经网路的容量来防止过拟合。

**L2 Regularization**
L2正则化比较常见，它意味着直接在每个权重$ w $上加上$ \frac{1}{2}\lambdaw^2 $其中$ \lambda $是正则化的程度。前面的1/2是为了求导后得到$ \lambdaw $。L2正则化直观的看出来，它对于某个值较大的权重的惩罚力度很大，并且倾向于更分散的权重，让神经网络的每一层利用更多的输入神经元。而使用L2正则化意味着每个权重都是线性下降的 ```W += -lambda * W```

**L1 Regularization**
另一个常见的正则化方法。对每个权重$ w $都加上$ \lambda|w| $。也有把L1 L2正则化结合起来的$ \lambda_1|w| + \lambda_2w^2 $，这也称作 Elastic net regularization。L1正则化有个很有意思的特性，在优化的过程中，它会导致神经网络的权重变得稀疏（非常接近0），换而言之就是最后只有一部分权重能进行激活，这使得对噪音有着更强的鲁棒性。通常来说使用L2的效果要好于L1。

**Max norm constrains**
另一种正则化就是限制每个权重的最大值，$ \lVert\overrightarrow{x}\rVert_2 < c $。

**Dropout**
对于全连接神经网络单元，按照一定的概率使其暂时失效，故而每个mini-batch相当于在训练不同的神经网络，它强迫一个神经单元和随机挑选出来的其他神经单元共同工作，消除了神经元节点之间的联合适应性，增强了泛化能力，是CNN中防止过拟合的一个重要方法。
```python
p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  """ X contains the data """
  
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1) # 使用ReLU作为激活函数，生成hidden layer
  U1 = np.random.rand(*H1.shape) < p # first dropout mask
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # second dropout mask
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)
  
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
  out = np.dot(W3, H2) + b3
```
在predict的时候不再进行dropout，但是原来进行dropout的层都要乘以概率p。因为在predict的时候所有的神经元都能看到所有的输入，所以他们接受的输入尺度要和训练的时候一样。在训练的时候有dropout，每个神经元的输入为$ px + (1 - p)0 $，期望就是$ px $，所以在predict的时候就需要让输入变为$ px $.

dropout也可以看作是训练了多个神经网络，然后在predict的时候将输入x乘p，在所有的单个神经网络上都遍历一遍，将他们集成起来计算集成后的输出。

dropout有个缺点就是在测试的时候还需要计算，这增加了计算量。也有一种**inverted dropout**，训练的时候和普通dropout相同，只是在测试的时候不乘以p。


在更大的范围来说，dropout属于神经网络中加上随机化的一种方法。在前向传播的时候加上随机的噪音。两种形式的随机噪音：
1. 解析上，如dropout
2. 数值上，几种不同的随机采样方法，然后再对这几种方法进行平均，然后再前向传播。
CNN也利用了这种形式的方法，例如Data argumentation，stochastic pooling等。

**Bias regularization**
通常不对bias进行正则化，因为他们并没有和数据进行直接的交互。但是在实践中对bias进行正则化好像并没有什么坏处，也可能是因为bias相对其他的参数比较少，对整体的影响较小。

**实践总结**
实践中一般使用L2正则化，正则化的强度通过交叉验证来得到。并且在所有层的后面加上p=0.5的dropout层，当然这个p也可以通过交叉验证来调整。

### Loss function
loss一般是一个batch中的各个样本的loss的均值$ L = \frac{1}{N}\sum_iL_i $.

**分类**
SVM loss：$$ L_i = \sum_{j\neq y_i} \max(0, f_j - f_{y_i} + 1) $$，也有人说使用平方效果更好$ \max(0, f_j - f_{yi} + 1)^2 $。

cross entropy loss:$$ L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) $$

**问题：类别太多**
当数据集的类别太多的时候例如ImageNet 22000个类别，可以使用Hierarchical Softmax。它将标签解构成一个树，每一个标签代表了树上的一条路径，一个softmax分类器在每个节点上进行分类，区分左右分支，树的结构对结果性能很影响。

**属性分类**
SVM loss和交叉熵都假设每个样本只有一个标签。但是假如标签是一个二元的向量(只包含0/1)呢，代表每个样本是否含有这个属性。一个方法是对每个属性建立一个二分类器。每个类别独立的二分类器也许是这个样子的：$$ L_i = \sum_i\max(0, 1-u_{ij}f_j) $$
$ f_j $是一个对属性j的二分类score function，$ y_{ij} $取-1或+1.loss在所有需要分类的属性上进行求和。

另一个方法是对每个需要分类的属性训练一个logistic二分类器，只有两个类别输出为(0, 1)。输出这个属性为1的概率为：$$ P(y = 1 \mid x; w, b) = \frac{1}{1 + e^{-(w^Tx +b)}} = \sigma (w^Tx + b) $$。所以如果输出$ \sigma(w^Tx + b) > 0.5 $那么就可以看作是1这个类别。loss函数只需要将似然函数极大化就可以了$$ L_i = \sum_j y_{ij} \log(\sigma(f_j)) + (1 - y_{ij}) \log(1 - \sigma(f_j)) $$。这个loss函数的导函数为$ \partial{L_i} / \partial{f_j} = y_{ij} - \sigma(f_j) $。

**回归**
衡量输出和标签之间的差异，使用L2平方norm$ L_i = \Vert f - y_i \Vert_2^2  $或者L1 norm$ L_i = \Vert f - y_i \Vert_1 = \sum_j \mid f_j - (y_i)_j \mid $，L1 norm在输出的每个维度上进行求差的绝对值和(输出是一个向量)。只看第i个样本输出的第j维度，和标签之间的差异可以记作$ \delta_{ij} $。对于L2 norm来说这个维度上的导数就直接是$ \partial{L_i} / \partial{f_j} $，对于L1 norm来说就是$ sign(\delta_{ij}) $。总之导数很容易求得。

**注意**
相比softmax loss等，L2 norm更难进行训练。并且它的鲁棒性也更差因为离群点会产生更大的导数，影响更大。当遇到一个回归问题的时候，首先思考能不能把它变成一个分类问题，例如给东西达1-5星，最好是训练5个独立的分类器而不是使用回归的方法。如果不能分解为分类问题的话，小心地使用L2 norm并且同时使用使用dropout可能不会很好。

**结构化的预测**
结构化的预测指的是利用结构化的标签如图、树或者其他复杂的结构。并且也通常假设这个结构的空间非常大，没办法进行遍历。进行这类的预测通常梯度下降难以办到。