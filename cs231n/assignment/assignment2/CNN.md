## Convolutional Neural Network

### 卷积层前向传播
```python
def conv_forward_naive(x, w, b, conv_param):
    """用循环实现的卷积层
    Inputs:
    - x: 输入的数据(N, C, H, W)
    - w: filter的数值(F, C, HH, WW) F是输出的channel
    - b: 偏置单元 (F,)
    - conv_param: 包含如下键的字典：
       - 'stride': 相邻感受野之间的像素值个数，垂直方向上和水平方向上
       - 'pad': zero-pad的个数

    Returns:
    - out: (N, F, H', W')其中
       H' = (H + 2*pad - HH)/stride + 1
       W' = (W + 2*pad - WW)/stride + 1
    - cache: (x, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2*pad - HH)/stride
    W_out = 1 + (W + 2*pad - WW)/stride
    out = np.zeros((N, F, H_out, W_out))

    # np.pad(data, ((before, after), (before, after)...))
    # 这里pad仍然是两边pad
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            # 确定feature map上的这个位置
            x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] # (N, C, HH, WW)
            # 每个卷积核都对这个位置进行相乘，如果不考虑batch，那么相乘的结果就是一个点
            # 卷积的结果（那个点）输出的就是第k个channel，第i, j的位置
            # sum(..., axis=(1,2,3))让对应位置相乘之后求和
            for k in range(F):
                out[:, k, i, j] = np.sum(x_pad_masked*w[k, :, :, :], axis=(1,2,3))

    out = out + (b)[None, :, None, None]
    cache = (x, w, b, conv_param)
    return out, cache
```
### 卷积层反向传播
```python
def conv_backward_navie(dout, cache):
    """
    Inputs:
    - dout: 反向传播过来的导数
    - cache: (x, w, b, conv_param)

    Return:
    - dx: x的导数
    - dw: w的导数
    - db: b的导数
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = (H + 2*pad - HH)/stride + 1
    W_out = (H + 2*pad - WW)/stride + 1

    # 卷积的中间数据
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    # 先生成各种导数的数据
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for i in range(H_out):
        for j in range(W_out):
            # 对于每个卷积核和feature map相乘的位置
            x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            # 对于每个卷积核，输出的channel为F，那么就有F个卷积核
            for k in range(F):
                # 这个卷积核参数的导数，是每个卷积位置的数据和dout相乘后求和(+=)，那个sum是为了缩减batch
                dw[k, :, :, :] += np.sum(x_pad_masked * (dout[:, k, i, j][:, None, None, None], axis=0))
            # 对于这个位置上数据的每个batch中的导数[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            for n in range(N):
                # 这个位置上的数据的导数就是权重和dout相乘后求和(+=)，sum是为了缩减batch
                dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :]*(dout[n, :, i, j])[:, None, None, None]), axis=0)
    # 使用dx_pad得到dx
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    return dx, dw, db
```