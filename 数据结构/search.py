# coding:utf-8

def bin_search(lists, key):
    """
    二分查找相当于构建了一个树，在树上进行查找
    时间复杂度为 O(log2(n))
    """
    low, high = 0, len(lists)-1
    while  low <= high:
        mid = int(low+(high-low)/2)
        if key < lists[mid]:
            high = mid - 1
        elif key > lists[mid]:
            low = mid+1
        else:
            return mid
    return None

def bin_search_recursion(lists, key, low, high):
    if low > high:
        return None
    mid = int(low+(high-low)/2)
    if lists[mid] == key:
        return mid
    elif lists[mid] > key:
        high = mid - 1
    else:
        low = mid + 1
    return bin_search_recursion(lists, key, low, high)

if __name__ == '__main__':
    lists = [1, 3, 5, 6, 12, 15 ,16, 20]
    print(bin_search_recursion(lists, 3, 0, len(lists)-1))