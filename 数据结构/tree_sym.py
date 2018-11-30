# coding:utf-8

class TreeNode():
    def __init__(self, x):
        self.val = x
        self.right = None
        self.left = None

# 判断形状是否是对称的        
def is_symmetrical(root):
    # 如果输入的树为空则直接返回True
    if root is None:
        return True
    # 如果两个子树都存在 则递归比较两个子树作为根节点的树是否对称
    if root.right and root.left:
        return is_symmetrical(root.right) and is_symmetrical(root.left)
    # 如果两个子树都不存在则返回True 这个分支其实不能直接删除，开头的if无法代替
    elif root.right is None and root.left is None:
        return True
    # 上面两个分支都没有
    return False
        
