import numpy as np

#只剩一个球时，测楼层需要的次数的期望值
def exp1(n):
    return (n*(n+1)/2-1)/n

#两个球时，测楼层需要的次数的期望值
exp2 = {}
exp2[1] = 0
print('{}个楼层的最佳策略下的佳期望次数是{}次，第一次从{}个位置测。'.format(1,0,0))
exp2[2] = 1
print('{}个楼层的最佳策略下的佳期望次数是{}次，第一次从{}个位置测。'.format(2,1,1))
for i in range(3,101):
    s = None
    for j in range(1,i):
        if s is None:
            s = 1 + j/i*exp1(j) + (i-j)/i*exp2[i-j]
            m = 1
        s_temp = (1 + j/i*exp1(j) + (i-j)/i*exp2[i-j])
        if s > s_temp:
            s = s_temp
            m = j
    exp2[i] = s
    print('{}个楼层的最佳策略下的佳期望次数是{}次，第一次从{}个位置测。'.format(i, round(s,2), m))


import numpy as np
import pandas as pd

data1 = {5:-0.966315,6:0.0677072,7:-0.904273,8:0.0013746,9:-0.961071,10:0.0137296,11:-1,13:-1,19:-1,27:-0.601472,28:0.00530575,29:-0.983698,31:-1,35:-1}
data2 = {5:-0.976128,6:0.0238378,7:-0.904273,8:0.000310778,9:-1,10:1.03698e-05,11:-1,13:-1,19:-1,27:-0.601472,28:0.00335127,29:-1,31:-1,35:-1}
data3 = {5:-0.999441,6:-0.00748592,7:-0.903863,8:0.000179295,9:-1,10:0.000321463,11:-1,13:-1,19:-1,27:-0.594021,28:0.0111151,29:-1,31:-1,35:-1}


# 政府比例
def get_rate(d):
    z = 0
    f = 0
    for i in data1:
        if d[i]>=0:
            z+=1
        else:
            f+=1
    print('正例个数：{},负例个数：{},政府比例：{}'.format(z,f,z/f))
    return z/f

# 平均值
def get_mean(d):
    s = 0
    for i in d:
        s += d[i]
    print('平均值：{}'.format(s/len(d)))
    return s/len(d)

# 方差
def get_v(d):
    s = 0
    for i in d:
        s += d[i]
    m = s/len(d)
    std = 0
    for i in d:
        std += (d[i]-m)**2
    std = std/len(d)
    print('方差：{}'.format(std))
    return std

# 标准差
def get_std(d):
    s = 0
    for i in d:
        s += d[i]
    m = s/len(d)
    std = 0
    for i in d:
        std += (d[i]-m)**2
    std = std/len(d)**0.5
    print('标准差：{}'.format(std))
    return std

print(get_rate(data1))
get_mean(data2)
get_mean(data3)
get_v(data2)
get_v(data3)
get_std(data2)
get_std(data3)


def findMedianSortedArrays(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
        import math
        # 找出最大的n个值
        def search(nums1, nums2, L):
            L1 = min(len(nums1), L)
            L2 = min(len(nums2), L)
            num1 = nums1[-L1]
            num2 = nums2[-L2]
            if num1 > num2:
                return (nums1[:-L], nums2, num1, L1)
            else:
                return (nums1, nums2[:-L], num2, L2)
        liminal = (len(nums1) + len(nums2)) // 2
        residue = (len(nums1) + len(nums2)) % 2
        liminal_num = None
        while (liminal > 0) & (len(nums1)>0) & (len(nums2)>0):
            nums1, nums2, liminal_num_temp, L = search(nums1, nums2, int(math.ceil(liminal / 2)))
            liminal_num = liminal_num_temp if liminal_num is None else min(liminal_num_temp,liminal_num)
            liminal = liminal - L
        if (len(nums1)==0):
            if liminal>0:
                liminal_num = nums2[-liminal] if liminal_num is None else min(nums2[-liminal],liminal_num)
                nums2 = nums2[:-liminal]
            if residue == 0:
                return (liminal_num + nums2[-1]) / 2
            else:
                return nums2[-1]
        if (len(nums2)==0):
            if liminal>0:
                liminal_num = nums1[-liminal] if liminal_num is None else min(nums1[-liminal],liminal_num)
                nums1 = nums1[:-liminal]
            if residue == 0:
                return (liminal_num + nums1[-1]) / 2
            else:
                return nums1[-1]
        if residue == 0:
            return (liminal_num + max(nums1[-1], nums2[-1])) / 2
        if residue == 1:
            return max(nums1[-1], nums2[-1])

nums1 = [1,3]
nums2 = [1,2]
findMedianSortedArrays( nums1, nums2)


def findMedianSortedArrays(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    Y = None
    import math
    # 找出最大的n个值
    def search(nums1, nums2, L):
        try:
            num1 = nums1[-L]
            num2 = nums2[-L]
            if num1 > num2:
                return (nums1[:-L], nums2, num1)
            else:
                return (nums1, nums2[:-L], num2)
        except:
            return (-L, -L, -L)

    liminal = (len(nums1) + len(nums2)) // 2
    residue = (len(nums1) + len(nums2)) % 2
    while (liminal > 0):
        return (int(math.ceil(liminal / 2)))
        nums1, nums2, liminal_num = search(nums1, nums2, int(math.ceil(liminal / 2)))
        return (liminal_num)
        liminal = liminal - int(math.ceil(liminal / 2))
    if residue == 0:
        return (liminal_num + max(nums1[-1], nums2[-1])) / 2
    if residue == 1:
        return max(nums1[-1], nums2[-1])


class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """

        q1 = [0]
        q2 = []
        level = 0
        visited = [False] * (n + 1)
        while True:
            level += 1
            for v in q1:
                i = 0
                while True:
                    i += 1
                    t = v + i * i
                    if t == n: return level
                    if t > n: break
                    if visited[t]: continue
                    q2.append(t)
                    visited[t] = True
            q1 = q2
            q2 = []

        return 0

    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = 1
        li_temp = [0]
        while True:
            li = []
            for j in li_temp:
                i = 0
                while True:
                    i += 1
                    k = j + i ** 2
                    if n == k:
                        return r
                    if k > n:
                        break
                    else:
                        li.append(k)
                print(li)
            li_temp = li
            r += 1
            if r == 4:
                return 4
        return r

class Solution:
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        flag1 = [False] * len(height)
        flag2 = [False] * len(height)
        temp = 0
        for i,h in enumerate(height):
            if h>temp:
                temp = h
                flag1[i] = True
        temp = 0
        L = len(height)
        for i,h in enumerate(height[::-1]):
            i = L-i-1
            if h>temp:
                temp = h
                flag2[i] = True
        max_area = 0
        for i,h1 in enumerate(height):
            if flag1[i] is False:
                continue
            for j,(h2,flag) in enumerate(zip(height,flag2)[i+1:]):
                j = j+1
                if flag is False:
                    continue
                max_area = max(max_area,j*min(h1,h2))
        return max_area

def longestCommonPrefix(self, strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    result = ''
    for ss in zip(strs):
        if len(set(ss)) > 1:
            return result
        print(result)
        result += ss
    return result


def isValid(self, s):
    """
    :type s: str
    :rtype: bool
    """
    d = {'(': ')', '{': '}', '[': ']'}
    li = []
    try:
        for i in s:
            if len(li) == 0:
                li.append(i)
            else:
                t = li[-1]
                if d[t] == i:
                    d[t] == i
                li.pop()
            else:
                li.append(i)
    except:
        return False
    return li == []