## 利用线性SVM进行分类
train_data: (train_num, 3072)
### 训练流程
1. 初始化权重W: (3072, 10) 梯度dW: (3072, 10)
2. train_data和权重相乘得到score(10,)对应每个类别的分数
  2.1 对于每个score中的分数i，如果是正确的类别对应的score跳过
  2.2 如果是其他的类别，计算margin=score[i]-correct_score+1
  2.3 如果其他的margin大于零则```loss+=margin; dW[:, i]+=X[i].T; dW[:, correct]-=X[i].T```
3. 得到平均的loss```loss /= num_train; loss += 0.5 * reg * np.sum(W * W)```
4. 得到最终的梯度```dW /= num_train; dW += reg * W```

### 向量化上述过程
```python
scores = X.dot(W) # (num_train, num_class)
# 之前的correct类的下标是一个一个获取的，这里直接作为列向量
correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(num_train, 1)
# score和correct相减的时候会进行广播，这里要注意正确类别的margin也变成了1，而在上面的流程中是不计算的
margins = np.maximum(0, scores - correct_class_scores +1) # (num_train, num_class)
# 将正确类别的margin置为0
margins[range(num_train), list(y)] = 0
loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

# 用coeff_mat和训练数据矩阵相乘来得到梯度
coeff_mat = np.zeros((num_train, num_classes))
# 那些需要更新参数的位置的系数就是1，上面的流程中是dW[:, i]+=X[i].T，这里系数为1，然后下面再进行矩阵相乘，效果一样
coeff_mat[margins > 0] = 1
coeff_mat[range(num_train), list(y)] = 0
# 正确类别的梯度的是其他类别之和
coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

dW = (X.T).dot(coeff_mat)
dW = dW/num_train + reg*W
```

### 问题
**为什么在梯度检查的时候会出现个别较大的差错？** 因为svm loss不是完全可导的，就像relu函数在0附近一样，越靠近0，分析梯度和数值梯度的差就越多。

**将参数W进行可视化的结果是什么样的？** W: (3072, 10), 重新reshape到(32, 32, 3, 10)最后的10代表10个分类器，前面的就是每个分类器可视化的结果，它像是每个类别的所有图像的一个模板（取了平均值，因为只有数值相近的时候平方才最大），如果某个类别的分数较高，那么它就越接近这个类图片的模板。