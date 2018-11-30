# coding:utf-8

def jump_floor(n):
	"""
	普通的跳台阶题目是一次可以跳1个或者2个
	这里是一次可以跳1 2 3 4。。。n都可以
	那么从一个台阶开始
	f(1) = 1
	f(2) = 2
	f(3) = f(3-3) + f(3-2) + f(3-1)
	...
	f(n) = f(n-n) + f(n-(n-1)) + ... + f(n-2) + f(n-1)
	n个台阶的最后一步有n种，注意到
	f(n) = f(n-1) + [ f((n-1)-1) + f((n-1)-2) + ... +f((n-1)-(n-2)) + f((n-1)-(n-1)) ]
	f(n) = 2 * f(n-1)
	"""
	if n <= 0:
		return 0
	if n == 1:
		return 1
	return 2*jump_floor(n-1)

def rect_cover(target):
	"""
	使用2*1的方块覆盖target*2的矩形，小方块可以竖着放也可以横着放
	使用递归
	1. 如果矩形只有一个小方块的面积直接返回1
	2. 如果第一个是横着放的，那么剩余的面积就是
	| x  |    |。。。|
	|    |    |。。。|
	那么剩余的放法就是f(n-2)
	3. 如果第一个是竖着放的，那么剩余的面积就是
	|x| |...|
	|x| |...|
	那么剩余的放法就是f(n-1)
	"""
	if target <= 0:
		return 0
	if target == 1:
		return 1
	if target == 2:
		return 2
	return rect_cover(target-1) + rect_cover(target-2)
