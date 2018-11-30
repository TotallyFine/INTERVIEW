# coding:utf-8

def permutation(ss):
    """
    ss是一个字符串，只包含了大小写的字母
    现在让其中的字母重新进行排列组合 保存到返回的list中，list要有序
    利用回溯法 每次按照顺序交换字母 然后固定 进行下一个字母的交换 如果到最后就放入ret中
    """
    if ss is None or len(ss)==0:
        return []
    ret = []
    ss = list(ss)
    permu_helper(ss, 0, ret)
    return sorted(ret)

def permu_helper(ss, i, ret):
    if i == len(ss)-1:
        s = ''.join(x for x in ss)
        if s not in ret:
            ret.append(s)
    else:
        j = i
        while j < len(ss):
            swap(ss, i, j)
            permu_helper(ss, i+1, ret)
            swap(ss, i, j)
            j += 1

def swap(ss, i, j):
    ss[i], ss[j] = ss[j], ss[i]

def match(s, pattern):
    """
    实现一个能够匹配包括 '.' '*' 的模式串
    思路：模式串中的下一个是不是*，来进行区分
    结束匹配后有四种情况
    1. s还剩有没匹配 pattern还剩有 False
    2. s匹配完 pattern还有 'aaa' 'a*a'
    3. s还有 pattern没有 False
    4. s匹配万 pattern匹配完 True
    """
    # write code here
    if s is None or pattern is None:
        return False
    if s == pattern:
        return True
    i, j = 0, 0
    while j < len(pattern) and i < len(s):
        if  j+1 < len(pattern) and pattern[j+1] == '*':
            if pattern[j] == '.':
                i += 1
            elif pattern[j] != s[i]:
                j += 2
                continue
            else:
                i += 1
            if i == len(s):
                j += 2
        elif pattern[j] == '.':
            i += 1
            j += 1
        else:
            if pattern[j] != s[i]:
                return False
            i += 1
            j += 1
    if i != len(s):
        return False
    # 最后匹配的是x* 再将j+=2
    if j+2 <= len(pattern) and pattern[j+1] == '*':
        j += 2
    print(i, j)
    if i == len(s) and j == len(pattern):
       return True
    if j+1 == len(pattern) and pattern[j]!='.' and pattern[j]==pattern[j-2] and pattern[j-1]=='*':
        return True
    return False

class FirstAppear:
    l_1 = []
    l_2 = []
    # 返回对应char
    def FirstAppearingOnce(self):
        # write code here
        return self.l_1[0] if len(self.l_1)!=0 else '#'
    def Insert(self, char):
        # write code here
        if char in self.l_1:
            self.l_1.pop(self.l_1.index(char))
            self.l_2.append(char)
        if char in self.l_2:
            return
        else:
            self.l_1.append(char)

if __name__ == '__main__':
    #print(match("aaa", "a*a"))
    s = FirstAppear()
    s.Insert('h')
    s.Insert('e')
    s.Insert('l')
    s.Insert('l')
    print(s.FirstAppearingOnce())