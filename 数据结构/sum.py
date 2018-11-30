# coding:utf-8

# 从整数数组中找到三个数他们的和为0，找到所有的三元组
# 这个三个元素必须是从小到大排序，这三个数字不能重复
def three_sum(arr):
    """
    arr: list[int]
    return list[list[int]]
    先将问题转换成两个数之和，那么问题就变为了B+C=K-A
    给定一个n个元素的集合S，找出S中满足条件的整数对B C,  使得B+C=K-A。

    假定集合S已经排好序的话，则上面这个问题可以在O（n）的时间内解决。
    使用2个索引值left和right，分别指向第一个元素和最后一个元素，设指向的第一个元素为B，
    则我们的任务就是找到对应于B的元素C，C=K-B。如果arr[left]+arr[right]>target则right-=1
    如果arr[left]+arr[right]<target则left+=1
    """
    result = []
    # 对数组进行排序，方便把三数和转换成两数和的问题
    arr.sort()
    i = 0
    # 遍历每个位置上的数字a，将a+b+c=0 转换成b+c=-a 找到所有和a对应的b c
    while i < len(arr):
        target = 0 - arr[i]
        # 因为后面的数字都比这个数字大，所以如果这个数字小于零，那么不用再找了
        if target < 0:
            break
        left, right = i+1, len(arr)-1
        while left < right:
            # 目前的b c之和
            cur_sum = arr[left] + arr[right]
            # b c之和应该等于target
            # 小于则让左边的下标前进
            # 大于则让右边的下标后退
            if cur_sum < target:
                left += 1
            elif cur_sum > target:
                right -= 1
            else:
                # 构造三元组，同时原来的数组中可能还有重复的数字，这里都进行排除
                triple = [arr[i], arr[left], arr[right]]
                while left < right and arr[left] == triple[1]:
                    left += 1
                while left < right and arr[right] == triple[2]:
                    right -= 1
                result.append(triple)

        # a也有可能重复，这里进行排除
        while i+1 < len(arr) and arr[i+1] == arr[i]:
            i += 1
        i += 1
