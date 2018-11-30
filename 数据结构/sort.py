# coding:utf-8

# 八大排序算法 适用于Python2

def insert_sort(lists):
    """
    插入排序：
    将一个数据插入到已经排好序的有序数据中，从而获得一个新的、个数加一的有序数据
    算法适用于少量数据的排序，时间复杂度为O(n^2) 是稳定排序方法
    插入算法要把排序的数据分成两部分，第一部分包含这个数组的所有元素，但将最后一个元素除外
    而第二部分就只包含这一个元素（待插入元素），在第一部分排序完成后，再将这个最后元素插入到
    已经排好序的第一部分中
    """
    count = len(lists)
    # 首先认为第0个元素已经是排好序的
    for i in range(1, count):
        # 对第i个元素进行排序，lists[:i]都是排好序的
        key = lists[i]
        j = i - 1
        # 在已经排好序的元素中一个一个比较
        # 如果出现插入的元素更小的情况就进行交换，然后一直比较下去
        while j >= 0:
            if lists[j] > key:
                lists[j+1] = lists[j]
                lists[j] = key
            j -= 1
    return lists
  
def shell_sort(lists):
    """
    希尔排序：
    希尔排序是插入排序的一种 也称为缩小增量排序，是直接插入排序算法的一种更高效改进版本
    希尔排序是把记录按下表的一定增量分组，对每组使用直接插入排序算法排序
    随着增量逐渐减少，每组包含的关键词越来越多，当增量减至一，整个列表被分成一组，排序结束
    希尔排序是非稳定排序，最好的情况下时间复杂度为O(n^(1.3))元素已经排好序 最坏的情况下时间复杂度为O(n^2)
    https://www.cnblogs.com/chengxiao/p/6104371.html
    """
    count = len(lists)
    step = 2
    # 进行分组 group的值是组的个数 也是一个组内两个元素的间隔
    group = count / step
    while group > 0:
        for i in range(0, group):
            j = i + group
            # 在组内进行插入排序
            # j之前的组内元素认为是已经排好序了
            while j < count:
                # 组内排好序的需要和待插入元素进行比较的元素
                k = j - group
                # 待插入元素
                key = lists[j]
                # 进行比较来插入
                while k >= 0:
                    if lists[k] > key:
                        lists[k + group] = lists[k]
                        lists[k] = key
                    k -= group
                # 待插入元素切换到下一个
                j += group
        # 缩小group的数目
        group /= step
    return lists

def bubble_sort(lists):
    """
    冒泡排序：
    将每个元素都和列表后面的元素进行比较，如果这个元素大了就放到后边
    这样保证了最外层的for循环每次遍历之后前lists[:i]都是前i个最小的
    平均算法复杂度O(n^2) 最坏也是O(n^2) 最好的情况（已经排好序）下是O(n)
    """
    count = len(lists)
    for i in range(0, count):
        for j in range(i+1, count):
            if lists[i] > lists[j]:
                lists[i], lists[j] = lists[j], lists[i]
                print(lists)
    return lists

def quick_sort(lists, left, right):
    """
    快速排序：
    将要排序的数据分为两部分，其中一部分的所有数据都比另一部分的所有数据要小
    然后再按此方法对这两部分数据分别进行快速排序，整个过程递归进行
    平均算法复杂度为O(n*logn)
    最坏算法复杂度为O(n^2)
    每个排序算法都在努力突破O(n^2)的复杂度
    http://developer.51cto.com/art/201403/430986.htm
    """
    # 两个哨兵的位置已经交错则直接返回
    if left >= right:
        return lists
    # 基数，排好序的左边的部分要小于基数 右边部分要大于基数
    key = lists[left]
    low = left
    high = right
    # 哨兵没有相遇的时候
    while left < right:
        # 右边的哨兵先出发 直到发现一个比基数小的数字
        while left < right and lists[right] >= key:
            right -= 1
        # 基数已经保存在了key中，所以直接将小小于基数的数字保存进基数的位置
        # 如果是再次循环的话，之前lists[left]的数字也已经转移位置了
        lists[left] = lists[right]
        # 左边的哨兵出发 直到发现一个比基数大的数字
        while left < right and lists[left] <= key:
            left += 1
        # 之前lists[right]已经被保存在了key的位置或者其他没有被占用的位置中 所以这里可以直接将lists[left]保存在里面
        lists[right] = lists[left]
    # 左右哨兵相遇 此时相遇位置上的元素已经被放到了其他的位置上 将基数放在这个位置上完成排序
    # right left就是相遇的位置
    assert right == left
    lists[left] = key
    quick_sort(lists, low, left-1)
    quick_sort(lists, left+1, high)
    return lists

def select_sort(lists):
    """
    直接选择排序：
    第一趟，在待排序记录r1～rn中选出最小的记录将它和r1交换
    第二趟，载待排序记录r2～rn中选出最小记录和r2交换
    以此类推经过r趟排序即可完成排序
    """
    count = lists
    for i in range(count):
        min = i
        for j in range(i+1, count):
            if lists[min] > lists[j]:
                min = j
            lists[min], lists[j] = lists[j], lists[min]
    return lists

def adjust_heap(lists, i, size):
	lchild = 2 * i + 1
	rchild = 2 * i + 2
	# max保存的是这个子树中的最大的节点的下标
	max = i
	# 当i是中间节点的时候进行调整
	if i < size // 2:
		# lchild<size保证不会越界
		if lchild < size and lists[lchild] > lists[max]:
			max = lchild
		if rchild < size and lists[rchild] > lists[max]:
			max = rchild
		# 如果这个子树中的最大值不是根节点 交换两个节点
		if max != i:
			lists[max], lists[i] = lists[i], lists[max]
			# 交换过之后 将下沉的节点作为子树的根节点继续进行调整
			adjust_heap(lists, max, size)

def build_heap(lists, size):
	# size//2得到树中的中间节点数
	# 而且这些中间节点都是在lists[:size//2]中
	# 调整的时候从最后的非叶节点开始 这样保证所有的根节点在子树内都是最大的
	for i in range(size//2)[::-1]:
		# 调整每个子树 i就是每个子树的根
		adjust_heap(lists, i, size)

def heap_sort(lists):
	"""
	堆排序：
	堆排序是一种选择排序，最好 最坏 平均时间复杂度为O(n*logn)
	将待排序序列构造成一个大顶堆，此时整个序列的最大值就是堆顶的根节点
	再将根节点和末尾元素进行交换，此时末尾就为最大值，然后将剩余n-1个元素
	重新构造成一个大顶堆，如此反复就能进行排序

	当用数组表示树的时候：
	大顶堆 arr[i] >= arr[2i+1] and arr[i] >= arr[2i+2]
	小顶堆 arr[i] <= arr[2i+1] and arr[i] <= arr[2i+2]
	"""
	size = len(lists)
	# 构建大顶堆
	build_heap(lists, size)
	# 从最后的那个节点开始 交换根节点和它的位置
	for i in range(size)[::-1]:
		lists[0], lists[i] = lists[i], lists[0]
		# 交换过后继续进行调整堆
		adjust_heap(lists, 0, i)

def merge(left, right):
	# left right都是有序子序列
	i, j = 0, 0
	result = []
	while i < len(left) and j < len(right):
		if left[i] <= right[i]:
			result.append(left[i])
			i += 1
		else:
			result.append(right[j])
			j += 1
	result += left[i:]
	result += right[j:]
	return result

def merge_sort(lists):
	"""
	归并排序：
	采用分治法 先使每个子序列有序 再将已经有序的子序列合并
	如果是将两个有序列表合并则是二路归并
	分治法的时间：分解时间 + 解决问题时间 + 合并时间
	分解时间就是把待排序结合分成两部分 O(1)
	解决问题时间是两个递归式：把N的问题规模分成两个 N/2的子问题 时间为2T(N/2)
	构成一个递归树 只要解决树的叶节点上的问题时间复杂度为常数就可以
	合并时间为O(N)
	总时间为T(n) = 2T(n/2)+O(n) 解为O(n*logn)
	时间复杂度最好 平均 最坏都是O(n*logn) 归并排序是稳定排序
	"""
	if len(lists) <= 1:
		return lists
	num = len(lists) // 2
	left = merge_sort(lists[:num])
	right = merge_sort(lists[num:])
	return merge(left, right)

def radix_sort(lists, radix=10):
	"""
	基数排序：
	属于分配式排序 又称桶子法
	https://www.cnblogs.com/ECJTUACM-873284962/p/6935506.html
	"""
	import math
	k = int(math.ceil(math.log(max(lists), radix)))
	bucket = [[] for i in range(radix)]
	for i in range(1, k+1):
		for j in lists:
			bucket[j/(radix**(i-1)) % (radix**i)].append(j)
		del lists[:]
		for z in bucket:
			lists += z
			del z[:]
	return lists

# 海量数据进行排序 或者找出Top K大/小 的数字
# 这种情况下内存肯定是受限制的，不可能一次把所有的数字都加载到内存中
# 1 局部淘汰法：用一个容器放置无序的前K个数字，再逐个将后面的数字与容器中的数字进行比较，交换 O(n+m^2)
# 2 分治法 采用多路归并（外排序）
# 3 hash方法，如果这大量的数据中有很多重复的，那么就可以去掉重复的，提高效率
# 4 大/小顶堆：先读入前K个数字构建大/小顶堆，时间复杂度O(m*logm).然后遍历后续数字，与堆顶的数字进行比较.
#   如果求前K个最小的数字，那么构建大顶堆，将后续数字和堆顶进行比较，如果小于堆顶则替换 调整堆，如果大于则进行下一个
