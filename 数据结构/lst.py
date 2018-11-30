# coding:utf-8

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def construct(lst):
    if lst is None or len(lst)==0:
        return None
    head = ListNode(lst[0])
    i, p = 1, head
    while i < len(lst):
        p.next = ListNode(lst[i])
        i, p = i+1, p.next
    return head

def show(head):
    while head:
        print(head.val)
        head = head.next

def first_common_node(pHead1, pHead2):
    """
    找到第一个公共节点
    两个链表的样子类似于
    ——————\\____
      ____//
    先算出两个链表的长度，然后长的链表先走一段距离
    使得剩下的部分长度相同，然后一个一个比较判断
    """
    if pHead1 is None or pHead2 is None:
        return None
    len1 = 0
    p = pHead1
    while p:
        len1 += 1
        p = p.next
    len2 = 0
    p = pHead2
    while p:
        len2 += 1
        p = p.next
    if len1 > len2:
        l = len1 - len2
        while l > 0:
            pHead1 = pHead1.next
            l -= 1
    if len2 > len1:
        l = len2 - len1
        while l > 0:
            pHead2 = pHead2.next
            l -= 1
    while pHead1 and pHead2:
        if pHead1 == pHead2:
            return pHead1
        pHead1 = pHead1.next
        pHead2 = pHead2.next
    return None

def entry_of_loop(pHead):
    """
    一个链表尾部有一个环，找到这个环的入口
    思路：设置一个快指针 慢指针 两者会在环内相遇
    然后利用相遇的位置得到环的长度n 设总长度为m
    然后设置两个指针p1 p2 p1先走n步
    然后让p1 p2同时一步步走 那么他们会在环的入口相遇
    因为p1接下来走m-n p2也会走m-n
    """
    meet = meeting(pHead)
    if meet is None :
        return None
    loop_len = 1
    p = meet.next
    while p != meet:
        p = p.next
        loop_len += 1
    p1 = pHead
    i = 0
    while i < loop_len:
        p1 = p1.next
        i += 1
    p2 = pHead
    while p1 != p2:
        p1 = p1.next
        p2 = p2.next
    return p1

def meeting(p):
    if p is None or p.next is None:
        return None
    slow = p
    fast = p.next
    while slow and fast:
        if slow == fast:
            return slow
        slow = slow.next
        fast = fast.next
        if slow != fast:
            fast = slow.next

def deleteDuplication(pHead):
    """
    从链表中去掉重复的节点，先去掉前面的重复节点，然后去掉中间的重复节点
    """
    if pHead is None:
        return None
    pre, now, flag = pHead, pHead.next, False
    while now and pre.val == now.val:
        if now.next and now.next.next:
            pHead, pre, now = now.next, now.next, now.next.next
        elif now.next:
            return now.next if now.val != now.next.val else None
        else:
            return now.next
    while now:   
        while now.next and now.next.val == now.val:
            now, flag = now.next, True
        if flag:
            pre.next, now, flag = now.next, pre.next, False
        else:
            pre, now = now, now.next
        #print(pre.val, now.val)
    return pHead

if __name__ == '__main__':
    head = construct([1,2,3,3,4,4,5])
    show(deleteDuplication(head))