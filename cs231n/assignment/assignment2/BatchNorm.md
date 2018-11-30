## Batch Normalization
### 原理
在机器学习中让输入的数据之间相关性越少越好，最好输入的每个样本都是均值为0方差为1。在输入神经网络之前可以对数据进行处理让数据消除共线性，但是这样的话输入层的激活层看到的是一个分布良好的数据，但是较深的激活层看到的的分布就没那么完美了，分布将变化的很严重。这样会使得训练神经网络变得更加困难。所以添加BatchNorm层，在训练的时候BN层使用batch来估计数据的均值和方差，然后用均值和方差来标准化这个batch的数据，并且随着不同的batch经过网络，均值和方差都在做累计平均。在测试的时候就直接作为标准化的依据。

这样的方法也有可能导致降低神经网络的表示能力，因为某些层的全局最优的特征可能不是均值为0或者方差为1的。所以BN层也是能够进行学习每个特征维度的缩放gamma和平移beta的来避免这样的情况。

### BN层前向传播
```python
def batchnorm_forward(x, gamma, beta, bn_param):
    """先进行标准化再进行平移缩放
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Input:
    - x: (N, D) 输入的数据
    - gamma: (D,) 每个特征维度数据的缩放
    - beta: (D,) 每个特征维度数据的偏移
    - bn_param: 字典，有如下键值:
       - mode: 'train'/'test' 必须指定
       - eps: 一个常量为了维持数值稳定，保证不会除0
       - momentum: 动量
       - running_mean: (D,) 积累的均值
       - running_var: (D,) 积累的方差

    Returns:
    - out: (N,D)
    - cache: 反向传播时需要的数据
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        # 先标准化
        x_hat = (x - sample_mean)/(np.sqrt(sample_var + eps))
        # 再做缩放偏移
        out = gamma * x_hat + beta
        cache = (gamma, x, sample_mean, sample_var, eps, x_hat)
        running_mean = momentum * running_mean + (1-momuntum)*sample_mean
        running_var = momentum * running_var + (1-momentum)*sample_var
    elif mode == 'test':
        # 先标准化
        #x_hat = (x - running_mean)/(np.sqrt(running_var+eps))
        # 再做缩放偏移
        #out = gamma * x_hat + beta
        # 或者是下面的骚写法
        scale = gamma/(np.sqrt(running_var + eps))
        out = x*scale + (beta - running_mean*scale)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache
```

### BN层反向传播
```python
def batchnorm_barckward(out, cache):
    """反向传播的简单写法，易于理解
    Inputs:
    - dout: (N,D) dloss/dout
    - cache: (gamma, x, sample_mean, sample_var, eps, x_hat)

    Returns:
    - dx: (N,D)
    - dgamma: (D,) 每个维度的缩放和平移参数不同
    - dbeta: (D,)
    """
    dx, dgamma, dbeta = None, None, None
    # unpack cache
    gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
    N = x.shape[0]

    dx_1 = gamma * dout # dloss/dx_hat = dloss/dout * gamma (N, D)
    dx_2_b = np.sum((x - u_b) * dx_1, axis=0)
    dx_2_a = ((sigma_squared_b + eps)**-0.5)*dx_1
    dx_3_b = (-0.5) * ((sigma_squared_b + eps)**-1.5)*dx_2_b
    dx_4_b = dx_3_b * 1
    dx_5_b = np.ones_like(x)/N * dx_4_b
    dx_6_b = 2*(x-u_b)*dx_5_b
    dx_7_a = dx_6_b*1 + dx_2_a*1
    dx_7_b = dx_6_b*1 * dx_2_a*1
    dx_8_b = -1*np.sum(dx_7_b, axis=0)
    dx_9_b = np.ones_like(x)/N * dx_8_b
    dx_10 = dx_9_b + dx_7_a

    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = dx_10

    return dx, dgamma, dbeta
```
下面是直接使用公式来计算：
```python
def batchnorm_backward_alt(dout, cache):
    dx, dgamma, dbeta = None, None, None
    # unpack cache
    gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
    N = x.shape[0]
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat* (x - sample_mean) * -0.5 * np.power(sample_var + eps, -1.5), axis = 0)

    dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var +eps), axis = 0) + dvar * np.mean(-2 * (x - sample_mean), axis =0)

    dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dvar * 2.0 / N * (x-sample_mean) + 1.0 / N * dmean

    dgamma = np.sum(x_hat * dout, axis = 0)
    dbeta = np.sum(dout , axis = 0)

    return dx, dgamma, dbeta
```

### BN有什么作用
1. 对于不好的权重初始化有更高的鲁棒性，仍然能得到较好的效果。
2. 能更好的避免过拟合。
3. 解决梯度消失/爆炸问题，BN防止了前向传播的时候数值过大或者过小，这样就能让反向传播时梯度处于一个较好的区间内。

## 卷积神经网络中的BN
### 前向传播
```python
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """利用普通神经网络的BN来实现卷积神经网络的BN
    Inputs:
    - x: (N, C, H, W)
    - gamma: (C,)缩放系数
    - beta: (C,)平移系数
    - bn_param: 包含如下键的字典
       - mode: 'train'/'test'必须的键
       - eps: 数值稳定需要的一个较小的值
       - momentum: 一个常量，用来处理running mean和var的。如果momentum=0 那么之前不利用之前的均值和方差。momentum=1表示不利用现在的均值和方差，一般设置momentum=0.9
       - running_mean: (C,)
       - running_var: (C,)

    Returns:
    - out: (N, C, H, W)
    - cache: 反向传播需要的数据，这里直接使用了普通神经网络的cache
    """
    N, C, H, W = x.shape
    # transpose之后(N, W, H, C) channel在这里就可以看成是特征
    temp_out, cache = batchnorm_forward(x.transpose(0, 3, 2, 1).reshape((N*H*W, C)), gamma, beta, bn_param)
    # 再恢复shape
    out = temp_output.reshape(N, W, H, C).transpose(0, 3, 2, 1)
    return out, cache
```
### 反向传播
```python
def spatial_batchnorm_backward(dout, cache):
    """利用普通神经网络的BN反向传播实现卷积神经网络中的BN反向传播
    Inputs:
    - dout: (N, C, H, W) 反向传播回来的导数
    - cache: 前向传播时的中间数据

    Returns:
    - dx: (N, C, H, W)
    - dgamma: (C,) 缩放系数的导数
    - dbeta: (C,) 偏移系数的导数
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    # 利用普通神经网络的BN进行计算 (N*H*W, C)channel看成是特征维度
    dx_temp, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0, 3, 2, 1).reshape((N*H*W, C)), cache)
    # 将shape恢复
    dx = dx_temp.reshape(N, W, H, C).transpose(0, 3, 2, 1)
    return dx, dgamma, dbeta
```