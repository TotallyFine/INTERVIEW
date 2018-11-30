# coding:utf-8

def find(arr, target):
    """
    一个二维数组，从左往右 从上往下都是有序的，在这个二维数组中进行查找目标
    先和右上角的数字进行比较，
    如果相等就结束查找
    如果目标小于这个数字则剔除这一列
    如果目标大于这个数字就剔除这一行
    当然也可以选择左下角的数字
    这样查找每次都能比较整行整列 很快
    """
    row_idx = 0 # 右上角是(0, len(arr[0])-1)
    col_idx = len(arr[0]) - 1
    
    while row_idx <= len(arr)-1 and col_idx >= 0:
        # 找到了 
        if arr[row_idx][col_idx] == target:
            return row_idx, col_idx
        # 目标小于这个数字 剔除掉这一列 col-=1
        elif arr[row_idx][col_idx] > target:
            col_idx -= 1
        # 目标大于这个数字剔除掉这一行 row+=1
        else: # arr[row_idx][col_idx] < target
            row_idx += 1
    return -1, -1 # 没有找到
    
def maxsum_subarr_kadane(lists):
    """
    求最大的连续子串 lists是一个数组，这个数组中哪个连续的子数组的和最大
    数组 A = [-2, -3, 4, -1, -2, 1, 5, -3]， 最大子数组应为[4, -1, -2, 1, 5],其和为7。
    首先如果A中的元素全为正数，那么最大连续子数组就是它本身，如果A中全为负数，那么最大连续子数组就是最小的那个负数构成的数组。一般来说有以下几种方法
    1 暴力破解法
    2 分治法
    3 kadane算法 复杂度O(n)Kadane的算法是基于将一组可能的解决方案分解为互斥（不相交）集合。 
      利用任何子数组（即同一个子数组组中的任何一个成员）将始终具有最后一个元素i（这就是“在位置i结束的和”）的事实
    4 动态规划
    """
    size = len(lists)
    # 最长子串之和
    max_so_far = float('-inf')
    # 以这里为子数组的结束位置的话，最长子串之和是多少
    max_ending_here = 0
    ret_end = 0 # max_so_far代表的子数组的结束位置
    ret_begin = 0 # max_so_far代表的子数组的开始位置
    now_begin = 0 # max_ending_here代表的子数组的开始位置
    
    # 遍历每个位置，max_ending_here都加上
    for i in range(size):
        max_ending_here = max_ending_here + lists[i]
        # 如果是遇到了新的最长子串则进行记录
        # 这个最长子串可能是开头遇到的，也可能是前面都是负数中间遇到了小的负数
        # 这个最长的子串也是基于max_ending_here以这里结尾的子数组进行计算的
        if max_so_far < max_ending_here:
            max_so_far = max_ending_here
            ret_end = i
            ret_begin = now_begin

        # 如果是在开头或者是清零后遇到了负数，或者是加上这个位置上的数字变成了负数那么归零
        # 归零意味着，从下个位置重新开始计数，这个位置是max_ending_here代表的子数组的开始位置
        if max_ending_here < 0:
            max_ending_here = 0
            now_begin = i+1

    return max_so_far, ret_begin, ret_end

# 动态规划
def maxsum_subarr_DP(lists):
    t = len(lists)
    MS = [0]*t
    MS[0] = lists[0]
    # 对每个位置进行叠加 比较增加了这个位置的数据后子数组之和是否变大了
    # 如果加上这个元素 变成了比这个元素还小，那么冲新开始叠加
    for i in range(1, t):
        MS[i] = max(MS[i-1]+lists[i], lists[i])
    
    return MS

def reorder_odd_even(arr):
    """
    一个整数数组，设计方法，让所有奇数在前，所有偶数在后; 
    扩展：三种数，第一种放前，第二种放中间，第三种放最后
    如果是扩展的话可以来两趟，第一第二种看成是一种和第三种进行区分，然后第一第二种再进行区分
    """
    # 如果数组为空或者长度不够
    if arr is None or len(arr)<2:
        return arr
    # 下标左边从0 右边从len(arr)-1开始
    left, right = 0, len(arr)-1
    # 类似于快速排序中的写法
    while left < right:
        while left < right and arr[right]%2==0:
            right -= 1
        while left<right and arr[left]%2==1:
            left += 1
        # 进行交换
        arr[left], arr[right] = arr[right], arr[left]
    return arr

def reOrderArray(array):
    """
    输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
    使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
    并保证奇数和奇数，偶数和偶数之间的相对位置不变。
    """
    # write code here
    odd = []
    even = []
    for i in range(len(array)):
        if array[i]%2==1:
            odd.append(array[i])
        else:
            even.append(array[i])
    a = odd+even
    return a

def FindKthToTail(head, k):
    """
    输入一个链表，输出该链表中倒数第k个结点。
    使用两个指针，先让第一指针到k-1位置处，第二个指针在开始位置
    然后两个指针同时移动，第一个指针指向末尾的时候，第二个指针指向的就是倒数第k个节点
    """
    # write code here
    if head is None or k <= 0:
        return None
    pre = head
    i = 0
    while i<k-1:
        pre = pre.next
        i += 1
        if pre is None:
            return None
    last = head
    while pre.next is not None:
        pre = pre.next
        last = last.next
    return last

def calc_n(n):
    """
    给定一个整数n，编写函数算出有多少个每位都不相同的n位整数
    """
    if n > 10:
        return 0
    sum = 0
    while n > 0:
        sum += n
        n -= 1
    return n

def left_first_bigger(lists):
    """
    给定一个整形数组，数组是无重复随机无序的，打印出所有元素左边第一个大于该元素的值
    使用栈，从左往右，先把0号位置上的数字压栈，然后比较下一个位置的数字和栈顶的数字
    如果栈顶的数字大于下个位置的数字则输出并将这个位置的数字压栈
    如果栈顶的数字小于位置上的数字则出栈,出栈后再用栈顶进行比较
    """
    ret = []
    stack = []
    i = 0
    stack.append(lists[0])
    while i < len(lists):
        while len(stack) != 0 and lists[i] > stack[-1]:
            stack.pop()
        if len(stack) == 0:
            stack.append(lists[i])
        elif lists[i] < stack[-1]: # 必须使用elif
            ret.append(stack[-1])
            stack.append(lists[i])
        i += 1
    return ret

def merge(arr1, arr2):
    """
    归并两个有序数组
    归并n个长度为k的数组：
    1. 先把所有数字复制到长度为nk的数组中然后进行排序O(nlogn) 复杂度为O(nklognk)
    2. 利用最小堆：O(nklogk)
      1. 创建一个大小为n×k的数组保存最后的结果
      2. 创建一个大小为k的最小堆，堆中元素为k个数组中的每个数组的第一个元素
      3. 重复下列步骤n×k次：
        1. 每次从堆中取出最小元素（堆顶元素），将其放入数组中
        2. 用堆顶所在数组的下一元素将堆顶元素替换掉
        3. 如果数组中元素被取光了，将堆顶元素替换为无穷大。每次取出堆顶后进行调整
    """
    i = 0
    j = 0
    ret = []
    while i < len(arr1) and j < len(arr2):
        if arr1[i] > arr2[j]:
            ret.append(arr2[j])
            j += 1
        else:
            ret.append(arr1[i])
            i += 1
    ret += arr1[i:]
    ret += arr1[j:]
    return ret

def arry_split(arr):
    """
    一个长度为n的数组(n一定是偶数个)，将其平均分成两部分，找出能使这两部分的和的乘积最大的数组平分方式.
    这个问题可以看成是动态规划，或者是0-1背包问题
    
    """

def stack_order(pushV, popV):
    """
    pushV是进栈的顺序
    popV是出栈的顺序
    判断进栈出栈是否合理，这里使用一个真实的栈来进行模拟
    """
    i, j = 0, 0
    stack = []
    # 遍历进栈的顺序
    for i in range(len(pushV)):
        # 将数据压入栈中
        stack.append(pushV[i])
        # 判断栈顶是否和下一个出栈的数据相同，如果相同则进行出栈
        while j < len(popV) and popV[i] == stack[-1]:
            stack.pop()
            j += 1
        i += 1
    # 如果栈为空则说明出栈入栈的顺序相同
    return len(stack) == 0

def seq_of_bst(seq):
    """
    输入一个序列，判断是不是二叉排序树（二叉搜索树）的后序遍历结果
    二叉搜索树：左子树上的数字都 < 根节点 < 右子树上的数字
    """
    if seq is None or len(seq)==0:
        return False
    bst_helper(0, len(seq)-1, seq)

def bst_helper(i, j, seq):
    """
    帮助进行检查后序遍历结果
    先从左往右找到左右子树的分届，然后再往右如果出现了比根节点还小的数字则不是后序遍历的结果
    """
    if i >= j:
        return True
    left, right = i, j
    while seq[i] > seq[right]:
        i += 1
    j = i
    while j < right:
        if seq[j] < seq[right]:
            return False
        j += 1
    return bst_helper(left, i-1, seq) and bst_helper(i, right-1, seq)

def more_than_half_num(numbers):
    """
    numbers是一个list，其中存放数字，找出出现次数超过numbers长度二分之一的数字
    没有就返回0
    """
    if numbers is None or len(numbers)==0:
        return 0
    num = numbers[0]
    count = 1
    for i in range(1, len(numbers)):
        if numbers[i] == num:
            count += 1
        else:
            count -= 1
        if count == 0:
            num = numbers[i]
            count = 1
    count = 0
    for i in range(len(numbers)):
        if numbers[i] == num:
            count += 1
    if count > len(numbers)/2:
        return count
    else:
        return 0

if __name__ == '__main__':
    a = [1,2,3,4,5,6,7]
    print(reOrderArray(a))
