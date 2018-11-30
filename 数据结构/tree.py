# coding:utf-8

class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None

def has_subtree(A, B):
	"""
	A B是二叉树
	判断B是不是A的子结构
	空树不是任何树的子结构
	这里的子结构指的是包含数值和结构
	"""
	if B is None or A is None:
		return False
	# 先用A作为根节点的树和B进行比较，然后用左子树 右子树和B进行比较
	return is_subtree(A, B) or has_subtree(A.right, B) or has_subtree(A.left, B)

def is_subtree(A, B):
	"""
	这是个递归的函数，一开始假设输入的是A B的根节点，那么就左子树 右子树一个一个比较下去
	"""
	# B的结构已经结束说明这个子树符合B的样子
	if B is None:
		return True
	# B的结构还没有遍历完 A就结束说明A这个部分不是符合B的结构
	if A is None:
		return False
	if A.val == B.val:
		return is_subtree(A.left, B.left) and is_subtree(A.right, A.right)
	else:
		return False
    
def Mirror(root):
    """
	生成树root的镜像树，只需要前序遍历，把所有左右子树交换即可
	"""
    if root is None:
        return root
    root.left, root.right = root.right, root.left
    if root.left is not None:
        Mirror(root.left)
    if root.right is not None:
        Mirror(root.right)
    return root

def printMatrix(matrix):
    """
    matrix是一个二维的list，是一个矩阵，需要返回逆时针遍历这个矩阵的数值
    例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
    则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
    这个方法会将原始的matrix消解掉
    """
    if matrix is None:
        return None
    ret = []
    # 如果矩阵不为空则继续
    while matrix:
        # 矩阵的第一行
        ret.extend(matrix.pop(0))
        # 遍历矩阵的所有行 将每行的最后一个数值加入
        if matrix and matrix[0]:
            for row in matrix:
                ret.append(row.pop())
        # 矩阵的最后一行
        if matrix and matrix[0]:
            ret.extend(matrix.pop()[::-1])
        # 矩阵每行的第一个元素
        if matrix and matrix[0]:
            for row in matrix:
                ret.append(row.pop())
    return ret
