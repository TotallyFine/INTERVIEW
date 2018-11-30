## Fully Connected Neural Network
之前的两层神经网络是用非常简单的方式进行组织的，没有模块化。模块化的前向传播应该是下面的这个样子的，返回的值还应该有中间数据：
```python
def layer_forward(x, w):
    # compute forward
    z = ...
    out = ...
    cache = (x, w, z, out)
    return out, cache
```
一个反向传播应该是类似这样的，返回的值还应该有局部权重的梯度以及再向前的梯度：
```python
def layer_backward(dout, cache):
    # Unpack cache
    x, w, z, out = cache
    dz = ...
    dx = ...
    dw = ...
    return dx, dw
```

### 带ReLU的前向传播和反向传播
前向传播
```python
def affine_relu_forward(x, w, b):
    """
    先进行权重相乘（affine），然后apply一个relu函数
    Inputs:
    - x: 输入的数据(N, D)
    - w, b: 权重相乘的权重(D, M) (M,)

    Returns:
    - out: ReLU之后的输出(N, M)
    - cache: 反向传播需要的中间数据
    """
    N = x.shape[0]
    x_rsp = np.reshape(x, (N, -1)) # (N, D)
    affine = x_rsp.dot(w) + b # (N, M)
    out = affine * (affine >= 0)
    cache = (x, affine)
    return out, cache
```
反向传播
```python
def affine_relu_backforward(dout, cache):
   """
   反向传播
   Inputs:
   - dout: 输出的倒数(N, M)
   - cache: 缓存的中间数据(x, affine)

   Returns:
   - dw: w的导数(D, M)
   - db: b的导数(M,)
   """
   # x: (N, D) affine: (D, M)
   x, affine = cache
   drelu = dout * (affine >=0 ) # (N, M)
   dw = x.T.dot(drelu) # x.T=>(D, N) dot=>(D, M)
   db = np.sum(drelu, axis=0)
   return dw, db
```

### 将优化权重过程集成到一个类中
之前的优化逻辑都是直接写的没有进行封装，现在将其封装在Solver类中。只有针对导数进行优化的代码在Solver中，其他的如计算loss、梯度都是模型本身的功能。模型计算完梯度之后将梯度给Solver类，然后 Solver类选择算法进行优化。