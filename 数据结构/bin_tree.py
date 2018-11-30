# coding:utf-8

# 二叉树的每个节点至多只有二颗子树，不存在度大于2的节点，二叉树的子树有左右之分，不能颠倒
# 二叉树的第i层有2^(i-1)个节点
# 深度为k的二叉树至多有2^k-1个节点
# 任何一个二叉树T 如果其叶节点数为N0,度为2的节点数为N2 则N0 = N2 + 1
# https://www.cnblogs.com/freeman818/p/7252041.html 二叉树的各种遍历方式
# 前序遍历就是先根节点 然后左子树 然后右子树
# 中序遍历就是先左子树 然后根节点 然后右子树
# 后序遍历就是先左子树 然后右子树 最后根节点
# 区别就在于根节点在什么时候被访问


# 用递归的形式遍历平均时间复杂度是O(n) 空间复杂度是O(log(n)) 但实际上编译器会做一些优化比如尾递归 所以空间复杂度实际上是小于O(log(n))的 但是逻辑上是O(log(n)) 这个log(n)实际上是二叉树的深度 h = log(n)
# 最坏的情况下 时间复杂度是O(n) 空间复杂度是O(n) 这个树是一条线左边一直下来
# 如果用Morris遍历算法 则空间复杂度只有O(1)

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def construct(lst):
    """
    根据层次遍历来构造二叉树，用'#'来代替空
    完全二叉树中层次遍历的数组中
    节点i的子节点为2i 2i+1 （下标从1开始）
    """
    if lst is None or len(lst) == 0 or lst[0]=='#':
        return None
    # i也是从1开始，作为索引的时候减1即可
    i, root = 2, TreeNode(lst[0])
    root.right = construct_core(lst, 2*i)
    root.left = construct_core(lst, 2*i+1)
    return root

def construct_core(lst, i):
    if i-1 >= len(lst) or lst[i] == '#':
        return None
    node = TreeNode(lst[i])
    node.right = construct_core(lst, 2*i)
    node.left = construct_core(lst, 2*i+1)
    return node

def preorder(root):
    if root == None:
        return
    print(root.val)
    preorder(root.left)
    preorder(root.right)
    
def inorder(root):
    if root == None:
        return
    inorder(root.left)
    print(root.val)
    inorder(root.right)

def postorder(root):
    if root == None:
        return
    postorder(root.left)
    postorder(root.right)
    print(root.val)

# 二叉树中序遍历 分为递归和非递归两种     
def inorder_no_recursion(root):
    """
    中序遍历 非递归的形式
    先沿着左边一路走下去，将所有左子树根节点都压到栈里 直到左子树为空
    然后输出这个子树根节点，然后将右子树作为根节点继续进行遍历
    同样的沿着左边压栈，一遇到空就输出 然后右子树作为根节点
    当栈空的时候并且不再存在右子树的时候就停止
    """
    ret = []
    stack = []
    while root or stack:
        # 沿着左边一路压栈
        while root:
            stack.append(root)
            root = root.left
        # 出栈一个输出并将右子树作为新的根节点
        if stack:
            t = stack.pop()
            ret.append(t.val)
            # 右子树可能是None，但是这不会影响
            root = t.right
    return ret
    
def preorder_no_recursion(root):
    """
    前序遍历 非递归形式
    首先将根节点压栈
    如果栈不为空则 出栈
    首先输出根节点 然后将右子树入栈 左子树入栈
    """
    ret = []
    stack = []
    stack.append(root)
    while len(stack) != 0:
        t = stack.pop()
        if t is not None:
            ret.append(t.val)
        stack.append(t.right)
        stack.append(t.left)
    return ret
        
def postorder_no_recursion(root):
    """
    后序遍历 非递归形式
    首先将根节点压栈
    如果栈不为空则 出栈
    若出栈的节点有左右子树则再将这个节点压栈，并先将右子树压栈 再将左子树压栈
    若出栈的节点没有左右子树则输出
    """
    ret = []
    stack = []
    stack.append(root)
    while len(stack) != 0:
        t = stack.pop()
        if t.right is None and t.left is None:
            ret.append(t.val)
        else:
            stack.append(t)
            if t.right is not None:
                stack.append(t.right)
            if t.left is not None:
                stack.append(t.left)
    return ret
    
def bsf(root):
    """
    广度优先遍历 需要借用到队列
    """
    ret = []
    queue = []
    queue.insert(0, root)
    while queue:
        t = queue.pop(0)
        ret.append(t.val)
        if t.left is not None:
            queue.append(t.left)
        if t.right is not None:
            queue.append(t.right)
    return ret

def convert_no_recurion(root):
    """
    输入一个二叉搜索树，将这个二叉搜索树变为一个排好序的双向链表
    1. 使用非递归中序遍历 遍历的同时进行更改指针
    2. 使用递归的形式 将左子树变为一个双向有序链表 右子树变为一个双向有序链表 然后进行连接
    """
    if root is None:
        return None
    stack = []
    r = None
    flag = True
    pre = None
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        if stack:
            t = stack.pop()
            if flag:
                r = t
                pre = t
                root = root.right
                flag = False
            else:
                t.left = pre
                pre.right = t
                pre = t
                root = root.right
    return r

def convert_recursion(root):
    """
    递归的形式
    """
    if root is None:
        return None
    if root.left is None and root.right is None:
        return root
    # 左子树改造成双向排序链表
    left = convert_recursion(root.left)
    p = left
    # p为左子树变为的链表的最后一个节点
    while p.right:
        p = p.right
    if left:
        p.right = root
        root.left = p
    # 右子树构造成排序双向链表
    right = convert_recursion(root.right)
    if right:
        root.right = right
        right.left = root
    return left if left is not None else root

def tree_depth(root):
    """
    得到树的深度
    """
    if root is None:
        return 0
    left = tree_depth(root.left)
    right = tree_depth(root.right)
    return max(left, right) + 1

def is_balance(root):
    """
    判断是不是AVL 平衡二叉树（左子树和右子树的深度不超过1 递归下去也一样）
    这里利用深度优先遍历 当左右子树的深度只差超过1 的时候就返回-1作为深度，那么无论是加-1 还是减-1 绝对值都超过1 了
    """
    return balance_helper(root) != -1

def balance_helper(root):
    if root is None:
        return 0
    depth_l = balance_helper(root.left)
    if depth_l == -1:
        return -1
    depth_r = balance_helper(root.right)
    if depth_r == -1:
        return -1
    return -1 if abs(depth_l - depth_r) > 1 else max(depth_l, depth_r)+1


# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
def get_next(node):
    """
    给一个节点，找出这个节点的中序遍历的下一个节点
    其中node.next是父节点
    如果这个节点是一个子树的根节点，那么从右子树中找到最左边的那个节点
    如果这个节点是叶子节点并且是根节点的左子树，那么返回根节点
    如果这个节点是叶子节点并且是根节点的右子树，那么往上回溯直到不是右子树
    或者回溯到根节点直接返回父节点
    """
    if node is None:
        return node
    if node.right:
        node = node.right
        while node.left:
            node = node.left
    else:
        if node.next and node.next.left == node:
            return node.next
    while node.next and node.next.right == node:
        node = node.next
    return node.next

def z_print(pRoot):
    """
    z字形遍历二叉树 第一层从左往右 第二层从右往左 需要返回[[第一层]], [第二层]...]
    层次遍历需要借助队列，两种方法
    1. 全部正常遍历，然后某些层再倒过来
    2. 遍历的时候设置两个栈queue是正在被遍历的节点
    l是下一层的节点，从左向右的时候l先append左然后右
    从右向左的时候先右后左，通过flag来控制左右方向
    """
    # write code here
    if pRoot is None:
        return []
    queue, ret, temp, l, flag = [], [], [], [], False
    queue.append(pRoot)
    while queue:
        t = queue.pop()
        temp.append(t.val)
        if flag is False: # 从左 -> 右
            if t.left:
                l.append(t.left)
            if t.right:
                l.append(t.right)
        else:
            if t.right:
                l.append(t.right)
            if t.left:
                l.append(t.left)
        if len(queue) == 0:
            ret.append(temp)
            queue, l, temp, flag = l, [], [], False if flag else True
    return ret

class Serialize:
    """
    对二叉树进行序列化与反序列化
    """
    i = 0
    s = ''

    def Serialize(self, root):
        # write code here
        self.serialize_helper(root)
        s=self.s
        self.s=''
        return s
        
    def serialize_helper(self, root):
        if root is None:
            self.s += '#,'
            return
        self.s += str(root.val)
        self.s += ','
        self.serialize_helper(root.left)
        self.serialize_helper(root.right)
        
    def Deserialize(self, s):
        # write code here
        if s is None or len(s)==0 or s[0] == '#':
            return None
        while s[self.i] != ',':
            self.i += 1
        root = TreeNode(int(s[0:self.i]))
        self.i += 1
        # 每次传入的i都是指向逗号的,所以这里直接指向了逗号的下一个
        root.left = self.deserialize_helper(s)
        print('i', self.i)
        root.right = self.deserialize_helper(s)
        return root
        
    def deserialize_helper(self, s):
        # 传入这里的i指向的不是逗号
        if s[self.i] == '#':
            self.i += 2 # 使i指向下一个数字字符
            return None
        j = self.i
        while s[j] != ',':
            j += 1
        print('i', self.i, 'j', j, 'i:j', s[self.i:j])
        node = TreeNode(int(s[self.i:j]))
        self.i = j + 1
        # 指向逗号下一个字符
        node.left = self.deserialize_helper(s)
        node.right = self.deserialize_helper(s)
        return node

if __name__ == '__main__':
    s = Serialize()
    root = s.Deserialize('8,6,5,#,#,7,#,#,10,9,#,#,11,#,#')
    preorder(root)
    print(s.Serialize(root))