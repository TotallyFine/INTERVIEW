### Optimize 优化

### 可视化损失函数
任何一个神经网络的参数的维度都是非常高的，所以直接可视化参数是不现实的，但是可以随机产生一个参数矩阵W，然后再沿着一个方向W1，来回移动就可以得到损失函数的一维的可视化$ L(W + a*W1) $。那么沿着两个方向可以得到损失函数的二维可视化$ L(W + a*W1 + b*W2) $，[具体见](http://cs231n.github.io/optimization-1/)。一般来说损失函数可视化之后都是碗状的(bowl)，中心的损失低，四周的损失高。

**为什么是碗状的？** 假设存在一个高维超平面A(维度和参数的维度相同)使得loss函数取得最小值，取一个维度Xi，沿着这个维度移动超平面B，在维度Xi上和超平面A之间的距离就是loss，loss值就是AB之间的距离。同样的分别移动其他不同的维度，可以想象出来损失函数为什么是碗状的了。

当只有一层神经网络也就是线性分类器的时候，损失函数很明显是一个凸函数，可以用凸优化的各种方法来得到最优解。但是当叠加了多层神经网络的时候，损失函数就变成了一个非凸函数，不能再使用凸优化的方法了。

**方法 #1** 随机搜索
```python
# assume X_train is the data where each column is an example (e.g. 3073 x 50,000)
# assume Y_train are the labels (e.g. 1D array of 50,000)
# assume the function L evaluates the loss function

bestloss = float("inf") # Python assigns the highest possible float value
for num in xrange(1000):
  W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
  loss = L(X_train, Y_train, W) # get the loss over the entire training set
  if loss < bestloss: # keep track of the best solution
    bestloss = loss
    bestW = W
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)
```
这种方法不用多说，非常坏，因为参数的维度过高，搜索的范围太大。

**方法 #2** 随机本地搜索Random Local Search
从一个随机的W开始，每次尝试增加不同的δW，如果W+δW使得loss更低那么就记录下来。
```python
W = np.random.randn(10, 3073) * 0.001 # generate random starting W
bestloss = float("inf")
for i in xrange(1000):
  step_size = 0.0001
  Wtry = W + np.random.randn(10, 3073) * step_size
  loss = L(Xtr_cols, Ytr, Wtry)
  if loss < bestloss:
    W = Wtry
    bestloss = loss
  print 'iter %d loss is %f' % (i, bestloss)
```

**方法 #3** 跟随梯度
随机出初始的W之后不用随机的search，只要找到loss下降最快的方向就可以，那就是loss function的梯度，对每个参数求偏导，求出这个参数使得loss下降最快的方向。

### 计算梯度
有两种方法计算梯度
1. 数值解numerical gradient 费时计算量大就是在x附近进行大量得到输出，然后对梯度进行模拟。计算数值解的时候最好用$ [f(x + h) - f(x - h)]/2h $来进行模拟。
2. 解析解 analytic gradient 快速 计算量小，利用微积分得到导函数的表达式直接计算。使用解析解计算梯度的时候通常要检查gradient计算的是否正确，这称为gradient check

计算完梯度，得到每个使loss下降最快的每个参数的变化方向之后就可以更新梯度了。更新梯度的时候需要注意**setp size**也就是learning rate。梯度只告诉我们哪个方向可以让loss下降，但是没有说应该往这个方向走多少的路。如果步子过大的话，反而可能导致loss增加。通常都会使用mini-batch。
```python
while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
```
更进一步，除了mini-batch，还有SGD（or also some on-line gradient descent）。SGD每次只使用一个样本对参数进行更新。