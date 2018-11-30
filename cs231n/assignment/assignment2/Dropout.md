## Dropout
Dropout(Improving neural networks by preventing co-adaptation of feature detectors)是一个regularization技术,随机让某些神经元进行失效来获得更好的效果。

### Dropout前向传播
```python
def dropout_forward(x, dropout_param):
    """
    Inputs:
    - x: 输入的数据，可以是任何的shape
    - dropout_param: dict包含如下的键：
       - p: dropout概率，每个神经元被失活的概率
       - mode: 'test'/'train'如果是'test'则不进行失活
       - seed: 随机数生成种子，这个是为了梯度检验用，正常使用中不应该指定这个参数
    
    Outputs:
    - out: 输出数据shape同x
    - cache: (dropout_param, mask) 在'train'mode中，mask作用于输入x得到输出，在'test'mode中mask为None
    """
    p, mode = dropout['p'], dropout_param['mode']
    if 'seed' in deopout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None
    if mode == 'train':
        # 注意/(1-p) 前向传播的时候均值得稳定
        mask = (np.random.rand(*x.shape)>=p)/(1-p)
        out = x*mask
    elif mode == 'test':
        out = x
    
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache
```

### Dropout反向传播
```python
def dropout_backward(dout, cache):
    """
    Inputs:
    - dout: 反向传播回来的导数
    - cache: (dropout_param, mask)

    Output:
    - dx: 导数
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        # dloss/dx = dloss/dout * dout/dx = dloss/dout * mask
        # 被失活的神经元的mask处为0，其余为1
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx
```

### Dropout的作用
Dropout可以有效地抑制过拟合，一般来说神经元失活的概率越大在训练集上和在验证集上的区别就越小，但是较大的失活概率会导致神经网络的容量下降，更难拟合数据。