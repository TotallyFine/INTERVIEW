### 反向传播
**Problem** 一个函数f(x) 它的输入是一个向量或者矩阵，在这样的情况下进行求导。对于反向传播，X*W，本地对W求偏导，那么就是W的梯度了，然后为了反向，对X求偏导，让对X的导数进行传播。

**Motivation** 进行反向传播的原因是为了降低loss function的值，根据梯度下降最快的方向进行学习。一般来说只对参数W b计算导数，然而某些时候对输入的数据X计算导数也有用，可以进行一些可视化。

#### Simple expressions and interpretation of the gradient
$$ \frac{df(x)}{dx} = \lim{h\ \to 0} \frac{f(x + h) - f(h)}{h} $$值得注意的是，左边的分号不能像右边的分号一样看成是除法，左边的分号代表的是$ \frac{d}{dx} $这个算子被应用在函数$ f $上,并且返回这个函数的微分函数。右边的话，可以将$ h $看成是非常小的数值，右边的函数用一个直线来模拟在$ x $处的导数。

整个方程也可以看做$ f(x + h) = f(x) + h \frac{df(x)}{dx} $，当x变化a的时候，f(x)变化了ah。

**每个变量的导数表示了整个方程对这个变量的敏感程度**

因为输入输出是矩阵，所以求导的结果也是一个矩阵$ \nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x] $。并且使用“对x的导数”来代替标准的说法“对x的偏导”。

**加法求导**
$$ f(x,y) = x + y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = 1 \hspace{0.5in} \frac{\partial f}{\partial y} = 1 $$
对x, y的导数忽略了具体的数值，直接等于1，也就是当反向传播回来一个值的时候，经过加法计算导数不变直接向后传播。Sum operation distributes gradients equally to all its inputs.

**max(x, y)求导**
$$ f(x,y) = \max(x, y) \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial f}{\partial x} = \mathbb{1}(x >= y) \hspace{0.5in} \frac{\partial f}{\partial y} = \mathbb{1}(y >= x) $$
如果x最大的话的那么对x的偏导就是1，对y的偏导就是0。但是这个导数并没有体现出函数随输入的它的变化程度，只体现了函数对哪个变量更敏感一点。反向传播的时候只有最大的那个变量的梯度传播回去了，其他的都是0。Max operation routes the gradient to the higher input.

### 对反向传播的直观理解
反向传播是一个良好的local process，它从别的地方接受一个数字，并且输出两个东西：
1. 经过这个局部计算之后的导数
2. 本地的导数
它不需要知道整个计算图的样子，就可以进行工作。在方向传播的时候，链式法则通过让局部的导数乘以之前的导数来完成通过这个计算的传播。反向传播可以看成是每个local计算或者称作门之间进行相互交流，每个局部计算希望他们的输出上升或者下降，以及他们的意愿有多强烈，最终使得loss下降。

**不那么直观的角度以及微妙的结果**如果输入加法门的值x1很小但是值x2很大，反向传播的时候将会给较小的值x1一个很大的导数，给很大的值x2一个很小的导数。在多层感知机中，通常W*X.T，也就是说X的范围scale决定了权重W的导数值的量级。假如说给X都乘1000，那么影响W的导数也增加了1000倍，这就必须降低学习率来适应。这也体现了预处理的重要性！

对反向传播的直观理解有助于debug！

#### Sigmoid例子
每个可微分的函数都可以看成一个门计算，也可以把一个加法看成一个门计算，或者将一个函数拆分开来每个都看成一个门计算。以sigmoid为例：
$$ f(w,x) = \frac{1}{1+e^{-(w_0x_0 + w_1x_1 + w_2)}} $$
可以被拆分为：
$$ f(x) = \frac{1}{x} 
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = -1/x^2 
\\\\
f_c(x) = c + x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = 1 
\\\\
f(x) = e^x
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = e^x
\\\\
f_a(x) = ax
\hspace{1in} \rightarrow \hspace{1in} 
\frac{df}{dx} = a $$
它的导数为
$$ \sigma(x) = \frac{1}{1+e^{-x}} \\\\
\rightarrow \hspace{0.3in} \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \left( \frac{1 + e^{-x} - 1}{1 + e^{-x}} \right) \left( \frac{1}{1+e^{-x}} \right) 
= \left( 1 - \sigma(x) \right) \sigma(x)) $$
sigmoid的导函数是非常简单的，直接用导函数计算导数而不是每个简单的局部计算来叠加可以减少很多数值问题，所以很多框架中都是直接将每个单元的导函数给保存下来，直接计算。
```python
w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit
```
**实现细节** 如上面的代码所示，一般在实现反向传播的时候，都是分阶段来计算导数的，使得每个阶段各自很容易计算导数。

### 一个较为复杂的例子
$$ f(x,y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2} $$
前向传播
```python
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # denominator                        #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # done!                                 #(8)
```
反向传播
```python
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den 
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !!                  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
```
**注意事项**
1. 缓存几个前向传播时的值，用于反向传播，如上面的num invden xpy等
2. 当函数f包含好几个关于x的部分的时候，分叉的关于x的导数要进行相加

### 矩阵的导数计算
```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```
记住矩阵的导数和这个矩阵的行列数都必须相同！
