# coding:utf-8

def num_of_1(n):
    """
    求1~n之间的各个数字中1出现的次数
    先设定整数点作为位置i （1 10 100 。。。）对这些点进行分析
    利用整数点进行分割 高位部分n/i 地位部分n%i
    假设i正在表示百位，且百位的数>=2例如n=31456 i=100 则a=314 b=56
      此时百位为1的次数有a/10+1=32个，每一次都包含100个连续的点
      所以共有(a%10+1)*100个点的百位为1
    假设百位的数字为1 例如n=31156 i=100 则a=311 b=56 
      此时百位对应的就是1 则共有a%10次 是包含100个连续点
    当i表示百位，且百位对应的数为0,如n=31056,i=100，
    则a=310,b=56，此时百位为1的次数有a/10=31（最高两位0~30）
    综合以上三种情况，当百位对应0或>=2时，有(a+8)/10次包含所有
      100个点，还有当百位为1(a%10==1)，需要增加局部点b+1
    之所以补8，是因为当百位为0，则a/10==(a+8)/10，
      当百位>=2，补8会产生进位位，效果等同于(a/10+1)
    """
    if n <= 0:
            return 0
    count = 0
    i = 1
    while i <= n:
        a = n/i
        b = n%i
        count=count+(a+8)/10*i+(a%10==1)*(b+1)
        i *= 10
    return count

def min_num(numbers):
    """
    numbers是一个数组，把数组中的所有数字拼接起来排成一个数字
    输出能拼接的数字中最小的那个
    例如输入数组{3，32，321}
    ，则打印出这三个数字能排成的最小数字为321323。
    下面的代码是python2.7的
    """
    if numbers is None or len(numbers) == 0:
        return ""
    s = list(map(str, numbers))
    s.sort(cmp=my_cmp)
    return ''.join(x for x in s)

def my_cmp(x, y):
    """
    一个比较器，输入的x y都是字符串
    比较两者拼接起来 哪个在前面数字更小 
    """
    if x+y > y+x:
        return -1
    elif x+y == y+x:
        return 0
    else:
        return 1

def ugly_num(index):
    """
    寻找丑数
    丑数是只包含质因子 2 3 5的数比如 6 8 是丑数
    但是14因为包含了7质因子不是丑数，第一个丑数默认是1
    解题思路：
      每个丑数都可以由之前的丑数产生p = t2*2 + t3*3 + t5*5
      所以用t2 t3 t5 跟踪丑数，从1开始
      同一个数字*5>*3>*2，但是不同的数字就不一样了
      每次需要从之前的丑数中进行组合找到新的最小的丑数（大于现有丑数）
    """
    if index < 7:
        return index
    res = [0 for i in range(index)]
    res[0] = 1
    t2, t3, t5 = 0, 0, 0
    i = 1
    while i < index:
        # print(res)
        res[i] = min(res[t2]*2, res[t3]*3, res[t5]*5)
        if res[i] == res[t2]*2:
            t2 += 1
        if res[i] == res[t3]*3:
            t3 += 1
        if res[i] == res[t5]*5:
            t5 += 1
        i += 1
        # print(t2, t3, t5, i)
    return res[-1]

def first_not_repeat(s):
    """
    s是一个字符串 从中找出第一个没有重复的字符
    使用字典来保存 如果重复出现就置为-1 第一次出现就置为位置
    然后遍历字典 输出第一个不为-1的值
    """
    if s is None or len(s) == 0:
        return -1
    d = {}
    for i in range(len(s)):
        if s[i] in d:
            d[s[i]] = -1
        else:
            d[s[i]] = i
    for key in d:
        if d[key] != -1:
          return d[key]
    return -1

def get_num_of_k(data, k):
    """
    data是一个排序数组，从中找出k的个数
    先利用二分查找找到位置，然后去前后查找
    """
    if data is None or len(data)==0:
        return 0
    pos = bin_search(data, k, 0, len(data)-1)
    if pos is None:
        return 0
    p = pos
    count = 1
    while p+1 < len(data) and data[p+1] == k:
        p += 1
        count += 1
        print(p, count)
    p = pos
    while p-1 >= 0 and data[p-1] == k:
        p -= 1
        count += 1
        print(p, count)
    return count

def bin_search(data, k, start, end):
    """
    二分查找
    """
    if start > end:
        return
    mid = (start+end)//2
    if data[mid] == k:
        return mid
    elif data[mid] > k:
        return bin_search(data, k, start, mid-1)
    elif data[mid] < k:
        return bin_search(data, k, mid+1, end)

def find_two_once(array):
    """
    数组中其他数字都是出现了两次 找到其中值出现一次的两个数字
    利用异或，两个数字异或的结果是0 那么整个数组进行异或因为两个只出现一次所以结果不是0
    然后异或的结果就相当于是两个只出现一次的数字进行异或 那么肯定某个位数上出现了1
    利用这个1把数组中的所有数字分为两个部分 那么在这两个部分中进行异或结果就是这两个数字
    """
    if array is None or len(array)<2:
        return
    temp = array[0]
    for i in range(1, len(array)):
        temp = temp^array[i]
    index = 0
    while temp&1 == 0:
        temp = temp >> 1
        index += 1
    num1, num2 = 0, 0
    for i in range(len(array)):
        if is_bit(array[i], index):
            num1 = num1^array[i]
        else:
            num2 = num2 ^ array[i]
    return [num1, num2]
                
def is_bit(num, index):
    num = num >> index
    return num&1

def FindContinuousSequence(tsum):
    """
    找出连续的数组 数组之和为tsum
    """
    import math
    if tsum <= 0:
        return []
    ans = []
    # 根据等差数列，数组中从1开始 最长 sum = (1+n)*n / 2 所以 n < sqrt(2*sum)
    n = int(math.sqrt(2 * tsum))
    while n >= 2:
        # n&1==1表示n长度为奇数 这个时候数组最中间的数字就是平均值 那么sum%n==0
        # 当n为偶数的时候
        # m-1, m, m+1, m+2  m-1 m+1抵消 sum%4=2
        # m-2, m-1, m, m+1, m+2, m+3
        # m-3, m-2, m-1, m, m+1, m+2, m+3, m+4
        # 很明显sum对长度取余的结果就是长度的一半
        if (n&1==1 and tsum%n==0) or (tsum%n) == n/2:
            l = []
            # j是l的下标 k是数组中j下标对应的数字
            # tsum/n得到中间值 减去 (n-1)/2得到最小的那个数字
            j, k = 0, tsum/n - (n-1)/2
            while j < n:
                l.append(k)
                j += 1
                k += 1
            ans.append(l)
        n -= 1
    return ans

def last_remain(n, m):
    """
    约瑟夫环问题 在n个人循环数数选出第m个人，
    剔除出去然后重新进行循环选出第m个人，
    求最后选出的那个人在一开始的序列中的下标 从0开始
    两种解法：
    1. 利用数组进行模拟
    2. 利用数学原理进行解：
    设第一次循环选（这次循环中的各个数字下标记序列记为 序列1）出第m个人，这个人的下标是k=n%m，
    然后把这个数字的下一个数字作为0号数字进行计数（各个数字序列记为 序列2）这次有n-1个人
    从n-1个人中循环选出第m个 这个数在序列2中的下标为k'=(n-1)%m
    那么第二个数字在第一个序列中的下标是多少呢?k'+
    """

def str2int(s):
    """
    s是一个数字字符串 不使用库函数转为数字
    """
    if s is None or len(s)==0:
        return 0
    ret = 0
    i = 0
    for n in s[::-1]:
        if n not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '+']:
            return 0
        if n is '-':
            if i == len(s)-1:
                return -ret
            else:
                 return 0
        if n is '+':
            if i == len(s)-1:
                return ret
            else:
                return 0
        ret += (ord(n)-ord('0'))*(10**i)
        i += 1
        print(ret, i)
    return ret

def duplication(numbers):
    """
    从数组中找到第一个重复的数字，数组中的数字大小小于数组长度(数值从0开始)
    两种方法：
    1. 声明一个额外的数组来标记
    2. 利用数组本身进行标记 当遍历到一个数字的时候就把numbers[numbers[i]] += len(numbers)
    利用数字值小于数组长度，将这个数字作为下标，把那个位置上的值加n
    如果下次再加的时候发现大于len(numbers)就是这个数字第一次重复
    如果遍历到一个数字大于len(numbers)那么就减去来得到原本的数值
    """
    for i in numbers:
        index = i
        if index >= len(numbers):
            index = index - len(numbers)
        if numbers[index] >= len(numbers):
            return index
        numbers[index] += len(numbers)
    return -1

def multiply(A):
    """
    构建乘积数组
    A是一个数组，返回一个数组B，使得B[i] = A[1]*A[2]*...*A[i-1] * A[i+1]*...A[len(A)-1]
    除了A[i]之外都乘上，不能用除法
    很明显把B数组展开是一个矩阵
    | 1 | A[1] |...| A[n-1] |
    | A[0] | 1 | A[2] |...| A[n-1] |
    ...
    这样先利用上三角矩阵的性质进行连乘
    然后再在下三角矩阵中进行连乘
    """
    if A is None or len(A) == 0:
        return None
    B = [0 for i in range(len(A))]
    B[0] = 1
    # 先乘上三角
    for i in range(1, len(B)):
        B[i] = B[i-1]*A[i-1]
    temp = 1
    print(B)
    # 下三角矩阵中连乘
    for i in range(len(B)-1)[::-1]:
        temp *= A[i+1]
        B[i] *= temp
        print(B, temp, i)
    return B

def min_in_rotate_arr(arr):
    """
    把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 
    输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 
    例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 
    NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
    使用类似于二分查找法，一个区域内如果出现了非有序的部分那就是存在最小值的部分
    """
    if len(rotateArray) == 0:
        return 0
    i, j = 0, len(rotateArray)-1
    while i < j:
        mid = (i+j)/2
        if rotateArray[mid] > rotateArray[i]:
            # 左边是有序的，忽略左边
            i = mid
        elif rotateArray[mid] < rotateArray[j]:
            # 右边是有序的，mid的位置可能就是最小的值，所以不能忽略
            j = mid
        else:
            # 因为Python2.7除法是地板除，所以只剩下两个数的时候mid==i
            i += 1
    return rotateArray[i]

if __name__ == '__main__':
    #print(ugly_num(8))
    #print(first_not_repeat('google'))
    #print(get_num_of_k([1,2,3,3,3,4,5], 3.5))
    #print(bin_search([3], 3, 0, 0))
    #print(str2int('+123'))
    #print(duplication([2,1,3,1,4]))
    print(multiply([1,2,3,4,5]))