## 最大池化

### 最大池化前向传播
在前向传播的过程中注意保存中间数据，为了反向传播的时候方便计算。
```python
def max_pool_forward_naive(x, pool_param):
    """MaxPool前向传播的一个简单实现版本
    Inputs:
    - x: (N,　C, H, W)
    - pool_param: 包含最大池化参数的字典
       - 'pool_height': 池化区域的高度
       - 'pool_width': 池化区域的宽度
       - 'stride': 相邻池化区域的距离

    Returns:
    - out: 输出的数据
    - cache: (x, pool_param)
    """
    out = None
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width']
    H_out = (H-HH)/stride+1
    W_out = (W-WW)/stride+1
    out = np.zeros((N, C, H_out, W_out))
    # 先确定每个做最大池化的区域
    for i in xrange(H_out):
        for j in xrange(W_out):
            # 取出这个区域的数据
            x_masked = x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] # (N, C, HH, WW)
            # 取最大值 (N, C)
            out[:, :, i, j] = np.max(x_masked, axis=(2,3))
    cache = (x, pool_param)
    return out, cache
```

### 最大池化反向传播
```python
def max_pool_backward_naive(dout, cache):
    """MaxPool反向传播的一个简单版本
    Inputs:
    - dout: 反向传播过来的导数
    - cache: (x, pool_param)

    Returns:
    - dx: 导数
    """
    dx = None
    x, pool_param = cache
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = (H-HH)/stride+1
    W_out = (W-WW)/stride+1
    # 最大值处的导数能够进行传播，其余的地方导数为0
    dx = np.zeros_like(x)

    # 先确定之前池化的每个区域
    for i in xrange(H_out):
        for j in xrange(W_out):
            # 取出池化这个区域的数据
            x_masked = x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] # (N, C, HH, WW)
            # 取得最大值 (N,C)
            max_x_masked = np.max(x_masked, axis=(2,3))
            # 得到最大值的mask (N, C, HH, WW) 最大值为1 其余为0
            temp_binary_mask = (x_masked == (max_x_masked)[:,:,None,None])
            # 最大值处的导数能够进行传播，其余的地方导数为0
            dx[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += temp_binary_mask * (dout[:,:,i,j])[:,:,None,None]
```