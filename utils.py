#!/usr/bin/env python
# -*- coding: UTF-8 -*

# function to check if a string is dui chen
import re
import getopt
import random
import operator
from collections import namedtuple
import os

# Given a string, find the length of the longest substring without repeating characters.
'''
Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3.

Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. Note that the answer must be a substring, 
"pwke" is a subsequence and not a substring
'''

def longest_substring(s):
    ans = ""
    length = 0
    queue = ""
    for c in s:
        if c not in queue:
            queue += c
            ans = queue
            length = max(len(ans), length)
        else:
            queue = queue[queue.index(c)+1:] + c
    return length, ans

def isValid(s):
    my_dict = {"{": "}", "<": ">", "(": ")"}
    my_stack = []
    for char in s:
        if char in my_dict.keys():
            my_stack.append(char)
        elif char in my_dict.values():
            if (len(my_stack) != 0) and (char == my_dict[my_stack[-1]]):
                my_stack.pop()
            else:
                return False
        else:
            continue
    return len(my_stack) == 0

# Given a string containing just the characters '(' and ')',
# find the length of the longest valid (well-formed) parentheses substring.
# eg. input: ")()())", output 4  because the string is "()()"
def longestValidParentheses(s):
    my_stack = list()
    my_stack.append(-1) # empty stack and push -1 to it. The first element of stack is used to provide base for next valid string.

    #left = {} # save the ')' matching most left '(' success, both use index (key is right index, value is most left index)
    length = 0
    for i, c in enumerate(s):
        if c == "(":
            my_stack.append(i) # push index
        elif c == ")":
            my_stack.pop() # Pop the previous opening bracket's index
            if len(my_stack) != 0:
                length = max(length, i - my_stack[len(my_stack)-1])
            else:
                my_stack.append(i)
            #r = i
            #if left.has_key(l-1):
            #    left[r] = left[l-1]
            #else:
            #    left[r] = l
            #length = r - left[r] + 1
        else:
            continue
    return length

# function to return the first non-alphabetic order character in a string
def firstNonAlphabeticOrderChar(s):
    result = ""
    if len(s) < 0:
        return result
    # strip special characters, white space and numbers
    input = s.lower().strip().replace(" ", "")
    p1 = re.compile(r'\W+') # or p = re.compile('[^a-zA-Z0-9]+')
    input = re.sub(p1, '', input)
    # another way to strip special characters : input = ''.join([c for c in input if c.isalnum()])
    # we can use regex to replace all white spaces as well, or split string with space and then join it
    p2 = re.compile(r'\s+')
    input = re.sub(p2, '', input)
    input = ''.join([c for c in input if not c.isdigit()]) # strip numbers
    input = ''.join([c for c in input if c.isalnum()])
    print "input string is {0}".format(input)
    sorted_s = "abcdefghijklmnopqrstuvwxyz" # or import string, sorted_s = string.ascii_lowercase
    pre = ""
    for i in range(len(input)):
        if pre == input[i]: # this is a repeated character
            continue
        if input[i] in sorted_s:
            pre = input[i]
            index = sorted_s.index(input[i])
            sorted_s = sorted_s[index+1:]
        else:
            return input[i] # this is the first non alphabetic order character in the string

# function to check if a number is armstrong number, assume input number is valid
# An Armstrong number of three digits is an integer such that the sum of the cubes of its digits is equal to the number itself
# extend this to n digital number
def isArmstrong(n):
    # TODO: exception handle
    sum = 0
    input = n
    power = len(str(n))
    while n > 0:
        digit = n % 10
        sum = sum + digit ** power
        n = n / 10 # or use // operator: n = n // 10
    return sum == input

def firstNonRepreatedCharInString(s):
    result = ""
    if len(s) < 0:
        return result
    s = s.lower()
    # strip while space, special characters
    s = ''.join([c for c in s if c.isalnum()])
    count = {} # key is each character in s, value is the its repeating times in s
    print "input string is %s, we will count %s" % (s, str(count))
    for c in s:
        if c in count.keys():
            count[c] = count[c] + 1
        else:
            count[c] = 1
    for index in range(len(s)):
        if count[s[index]] == 1:
            return s[index]
    # option 2: code can be simplified like this
    count2 = {}.fromkeys(set(s), 0) # or : count2={}.fromkeys(s,0)
    for c in s:
        count2[c] += 1
    for c in s:
        if count2[c] == 1:
            return c

# remove characters in string r from string s
def removeChars1(s, r):
    if len(s) < 0:
        return s
    result = s
    for c in s:
        if c in r:
            result = result.replace(c, "")
    return result

# this function is same as 1, without needing to create a temp string result, not like Java
def removeChars2(s, r):
    if len(s) < 0:
        return s
    for c in s:
        if c in r:
            s = s.replace(c, "")
    return s

def removeChars3(s,r):
    if len(s) < 0:
        return s
    r_dict = {}.fromkeys(r, True)
    for cur in s:
        if (cur in r_dict.keys()) and (r_dict[cur]):
            s = s.replace(cur, "")
    return s

# remove a sub string r from a string s
def remove_sub_string(s, r):
    s = s.replace(r, '')
    return s

# remove sub string r1 and r2 from a list of strings ss, and return the list
def remove_sb_strings(ss, r1, r2):
    new_ss = [s.relace(r1, '').replace(r2, '') for s in ss]
    return new_ss

# remove sub string r1 and r2 from a set of strings ss, and return the new set
def remove_sb_strings(ss, r1, r2):
    new_ss = {s.relace(r1, '').replace(r2, '') for s in ss}
    return new_ss

# function to transpose rows and columns in a matrix
# assume it's n row and m column matrix, each row has same column number
def transposeMatrix(matrix):
    m = len(matrix[0]) # get the column number of this matrix
    transposed = []
    # way 1:
    for j in range(m):
            transposed.append([row[j] for row in matrix])
    return transposed
    # way 2
    for j in range(m):
        transposed_row = []
        for row in matrix:
            transposed_row.append(row[j])
        transposed.append(transposed_row)

    # way 3
    return [[row[i] for row in matrix] for i in range(m)]

    # way 4:  Zip returns an iterator of tuples, where the i-th tuple contains the i-th element from
    # each of the argument sequences or iterables. In this example we unzip our array using * and
    # then zip it to get the transpose.
    return t_matrix= zip(*matrix)

    # use numpy
    return numpy.transpose(matrix)

def convertDecimalToBinary(n):
    """Function to print binary number for the input decimal using recursion"""
    if n > 1:
        convertDecimalToBinary(n//2)
    #print(n % 2, end = '')
    # or: '{0:b}'.format(n)


def twoSum(nums, target):
    """
    function to return indices of the two numbers such that they add up to a specific target.
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    if len(nums) < 2:
        return [(-1,-1)]
    diff_dict = {} # key is num, value is the difference between target and this key
    result = list()
    for index, num in enumerate(nums):
        if num in diff_dict.values():
            # find key by value from the dictionary
            num1_s = diff_dict.keys()[diff_dict.values().index(num)]
            num1 = int(num1_s)
            result.append((nums.index(num1), index))
        else:
            diff_dict[str(num)] = target - num
    if len(result) > 0:
        return result
    else:
        return [(-1,-1)]


# Given a non negative integer number num. For every number between 0 and num,
# calculate the number of 1's in their binary representation and return them as an array.
def countBits(num):
    """
    :type num: int
    :rtype: List[int]
    """
    # key is the current number, value is the count of 1
    my_dict = {}

    for i in range (num + 1):
        bin_s = str(bin(i)[2:]) # '{0:b}'.format(i)
        count_1 = bin_s.count("1")
        my_dict[str(i)] = count_1
    # return my_dict.values() # there is no sequence if return like this
    result = []
    for key in sorted(my_dict): # sort the dict by its key, can be written as: sorted(my_dict.keys())
        result.append(my_dict[key])
    return result

# return the least common multiple
def lcm(x, y):
    if x > y:
        greater = x
    else:
        greater = y
    while True:
        if greater % x == 0 and greater % y == 0:
            break
        else:
            ++greater
    return greater

# return True if string is Palindrome ( string is equal with its reverse)
def isPalindrome(s):
    return s.lower() == "".join(reversed(s.lower())) # or s == s[::-1].lower()

def longestPalindrome1(s):
    if len(s) < 1:
        return ""
    if len(s) == 1:
        return s
    herp = ""
    for idy, item in enumerate(s):
        for idx, item in enumerate(s):
            derp = s[idy:idx+1]
            if isPalindrome(derp) and (len(derp) > len(herp)):
                herp = derp
    return herp

def longestPalindrome2(s):
    if len(s) < 1:
        return ""
    if len(s) == 1:
        return s
    result = []
    for i in range(len(s)):
        for j in range(0,i):
            chunk = s[j:i+1]
            if isPalindrome(chunk):
                result.append(chunk)
    return s.index(max(result, key=len)), result
# text.index() will only find the first occurrence of the longest palindrome,
# so if you want the last, replace it with text.rindex()

# display the Fibonacci sequence up to n-th term using recursive functions
def recur_fibo(n):
    if n <= 1:
        return n
    else:
        return recur_fibo(n-1) + recur_fibo(n-2)

# Change this value for a different result
nterms = 10
# uncomment to take input from the user
#nterms = int(input("How many terms? "))

# check if the number of terms is valid
if nterms <= 0:
    print("Plese enter a positive integer")
else:
    print("Fibonacci sequence:")
    for i in range(nterms):
        print(recur_fibo(i))

def fizzBuzz(n):
    """
    :type n: int
    :rtype: List[str]
    """
    result = []
    for i in range(n+1):
        if i % 3 == 0 and i % 5 == 0:
            result.append("FizzBuzz")
        elif i % 5 == 0:
            result.append("Buzz")
        elif i % 3 == 0:
            result.append("Fizz")
        else:
            result.append(str(i))
    return result

# find the longest path in string like
# "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
def lengthLongestPath(input):
    """
    :type input: str
    :rtype: int
    """
    maxlen = 0
    pathlen = {} # key is the number of tabs(depth), value is the length of current path
    for s in input.splitlines(): # or input.split("\n")
        curname = s.strip("\t") # or item.lstrip("\t")
        depth = len(s) - len(curname) # number of tab
        if '.' in curname:
            total_path_len = 0
            for d in range(depth):
                total_path_len += pathlen[d]
            maxlen = max(maxlen, total_path_len + len(curname))
        else:
            pathlen[depth] = len(curname) + 1 # +1 for "/"
            #pathlen[depth + 1] = pathlen[depth] + len(pathname) + 1 # depth+1 is the total path length before file
    return maxlen

# There is a list of sorted integers from 1 to n. Starting from left to right, remove the first number and
# every other number afterward until you reach the end of the list.
# Repeat the previous step again, but this time from right to left, remove the right most number
# and every other number from the remaining numbers.
def lastRemaining(n):
    nums = range(1, n+1)
    while len(nums) > 1:
        nums = nums[1::2][::-1] # from left, return num starting from index 1 to the end, and step is 2, then ::-1 reverse the list
    return nums[0]

# Find the length of the longest substring T of a given string (consists of lowercase letters only)
# such that every character in T appears no less than k times.
def longestSubstring(s, k):
    for c in set(s):
        if s.count(c) < k:
            return max(longestSubstring(item, k) for item in s.split(c))
    return len(s)

# input string is something like "3[a2[c]]", output is "accaccacc"
def decodeString(s):
    while "[" in s:
        s = re.sub(r"(\d+)\[([a-z]*)\]", lambda m: int(m.group(1)) * m.group(2), s)
    return s

# use recursion
def decodeString2(s):
    found = re.search("(\d+)\[([a-z]+)\]",s)
    return decodeString2(s[:found.start()] + found.group(2)*int(found.group(1))+s[found.end():]) if found else s

# Given an array of numbers nums, in which exactly two elements appear only once and all the other elements
# appear exactly twice. Find the two elements that appear only once
def singleNumber(nums):
    my_set = set(nums)
    result = []
    for num in my_set:
        if nums.count(num) == 1:
            result.append(num)
    return result

# function to find num of arithmetic sub string from nums, nums is int list
def numberOfArithmeticSlices(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    ans = 0
    if len(nums) > 2:
        # difference between https://www.pythoncentral.io/how-to-use-pythons-xrange-and-range/
        # they both create a list
        diff = [nums[i] - nums[i - 1] for i in xrange(1, len(nums))]
        count = 1
        pre = diff[0]
        for i in xrange(1, len(diff)):
            if diff[i] == pre:
                count += 1
            else:
                ans += count * (count - 1) / 2
                count = 1
            pre = diff[i]
        ans += count * (count - 1) / 2
    return ans

# function to find an item in an ordered list nums (binary search): update index
def search(nums,n):
    found = False
    low = 0
    high = len(nums) - 1
    while (not found) and (high >= low):
        mid = (high + low) / 2
        if n == nums[mid]:
            found = True
        elif n < nums[mid]:
            high = mid - 1
        else:
            low = mid + 1

    return found
# method 2: update list
def binarySearch(nums, n):
    found = False
    first = 0
    last = len(nums) - 1
    while (not found) and (first <= last):
        middle = (first + last) / 2
        if n == nums[middle]:
            found = True
        elif n < nums[middle]:
            nums = nums[:middle]
        else:
            nums = nums[middle+1:]
        last = len(nums) - 1
    return found

# shuffle a list
def shuffleList(a_list):
    result = a_list[:]
    length = len(a_list)
    for i in range(length):
        dyn_pos = random.randint(0, length) % (length - i) # make sure the reminder is smaller than length, because the reminder will be used as index
        result[i], result[i + dyn_pos] = swap(result[i], result[i + dyn_pos])
    return result

def swap(a,b):
    temp = a
    a = b
    b = temp
    return a, b

# Given a non-negative integer n, count all numbers with unique digits, x, where 0 ≤ x < 10**n
# way 1: use recursion
def countNumbersWithUniqueDigits(n):
    if n < 0:
        return 0
    if n > 11:
        n = 10 # if n is greater than 10, we use the result of 10
    # this is recursion method to cal n digits
    def count(n):
        if n == 0:
            return 1
        if n == 1:
            return 9
        else:
            return (11 - n) * count (n-1)
    sum = 0
    for i in range(n+1):
        sum += count(i)
    return sum

# way 2: use loop
def countNumbersWithUniqueDigits2(n):
    if n < 0:
        return 0
    if n > 10:
        n = 10
    sum = 0
    result = [] # or use this to initialize the list:result = (n+1) * [None], then update by result[i] = value
    for i in range(n+1):
        if i == 0:
            result.append(1)
        elif i == 1:
            result.append(9)
        else:
            result.append((11-i) * result[i - 1])
        sum += result[i]
    return sum

# Given a collection of intervals, find the minimum number of intervals you need to remove to make the
# rest of the intervals non-overlapping.input is something like [ [1,2], [2,3], [3,4], [1,3] ]
def eraseOverlapIntervals(intervals):
    count = 0
    input = sorted(intervals, key = lambda  item: item[1])
    pre_item_end =  -sys.maxint - 1 # sys.maxint is the max int
    for interval in input:
        if interval[0] >= pre_item_end:
            pre_item_end = interval[1]
        else:
            count += 1
    return count

# input is a string s, check if it's a valid number
def inNumber(s):
    #define DFA state transition tables
    states = [
              {},
              # State (1) - initial state (scan ahead thru blanks)
              {'blank': 1, 'sign': 2, 'digit':3, '.':4},
              # State (2) - found sign (expect digit/dot)
              {'digit':3, '.':4},
              # State (3) - digit consumer (loop until non-digit)
              {'digit':3, '.':5, 'e':6, 'blank':9},
              # State (4) - found dot (only a digit is valid)
              {'digit':5},
              # State (5) - after dot (expect digits, e, or end of valid input)
              {'digit':5, 'e':6, 'blank':9},
              # State (6) - found 'e' (only a sign or digit valid)
              {'sign':7, 'digit':8},
              # State (7) - sign after 'e' (only digit)
              {'digit':8},
              # State (8) - digit after 'e' (expect digits or end of valid input)
              {'digit':8, 'blank':9},
              # State (9) - Terminal state (fail if non-blank found)
              {'blank':9}
              ]
    current_state = 1
    for c in s:
        # If char c is of a known class set it to the class name
        if c in "0123456789":
            c= 'digit'
        elif c in "\t\n":
            c = 'blank'
        elif c in "+-":
            c = 'sign'
        # If char/class is not in our state transition table it is invalid input
        if c not in states[current_state][c]:
            return False
        # State transition
        current_state = states[current_state][c]

    # The only valid terminal states are end on digit, after dot, digit after e, or white space after valid input
    if current_state not in [3,5,8,9]:
        return False
    return True
# function to parse git log output into a dict
def parseGitLogOutput():
    GIT_COMMIT_FIELDS = ['id', 'author_name', 'author_email', 'date', 'message']
    GIT_LOG_FORMAT = ['%H', '%an', '%ae', '%ad', '%s']
    GIT_LOG_FORMAT = "%x1f".join(GIT_LOG_FORMAT) + "%x1e" # "\x1f" (ASCII field separator), "\x1e" (ASCII record separator)
    cmd = "git log --format=" % GIT_LOG_FORMAT
    p = Popen(cmd, shell=TRUE, stdout=PIPE) # run the cmd
    (log, _) = p.communicate() # get the output
    log = log.strip("\n\x1e").split("\x1e")
    log = [row.strip().split("\x1f") for row in log]
    log = [dict(zip(GIT_COMMIT_FIELDS, row)) for row in log]
    return log

# given two words  start and end, and a dictionary  of word list, find all shortest transformation sequences
# from start to end. (only one letter can be changed at a time, ech intermediate word must in the dictionary)
# e.g. start: "hit", end:"cog". dictionary: ["hot", "dot", "dog", "lot", "log"]
# return: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
def findLadders(start, end, dict):
    result = []
    if len(dict) == 0:
        return result
    min = sys.maxint
    queue = list([start]) # store the words from start to end
    ladder = {}.fromkeys(dict, sys.maxint)
    ladder[start] = 0
    dict.append(end)
    my_map = {} # key is word, value is the list of word path
    import string
    # BFS: Dijisktra search, breath first search
    while len(queue) != 0:
        word = queue.pop(-1) # remove and return the last obj from the list
        step = ladder[word] + 1 # step indicates how many steps are need to travel to word
        if step > sys.maxint:
            break
        for i in range(len(word)): # parse and replace each character in the current word
            for c in string.ascii_lowercase:
                new_word = word.repalce(word[i], c, 1) # construct a new word by replacing character
                if new_word in dict:
                    if new_word in ladder.keys():
                        if step > ladder[new_word]: # check if it's the shortest path to one word
                            continue
                        elif step < ladder[new_word]:
                            queue.append(new_word)
                            ladder[new_word, step]
                        else: # it is a KEY line. If one word is already in the one ladder, DO NOT insert it into the queue again
                            pass
                    if new_word in my_map.keys(): # build adjacent graph, so that we can back trace
                        my_map[new_word].append(word)
                    else:
                        my_map[new_word] = list([word])

                    if new_word == end:
                        min = step
    # back tracking, to construct the result, result is list, you can pass it as reference like java, but not string
    backTrace(end, start, result, my_map)
    return result

# word, start are string, res is list
def backTrace(word, start, res, map):
    if word == start:
        res.append(start)
        return
    res.append(word)
    while map[word] is not None:
        for s in map[word]:
            backTrace(s, start, res)


def mergeTwoSortedList(l1, l2):
    m = len(l1)
    n = len(l2)
    l1.extend([-sys.maxint - 1] * n) # extend the result list so that it can hold all nums
    # construct the new list from the end
    while m > 0 and n > 0:
        if l1[m-1] > l2[n-1]:
            l1[m + n - 1] = l1[m - 1]
            m -= 1
        else:
            l1[m + n - 1] = l2[n - 1]
            n -= 1
    while n > 0:
        l1[m + n - 1] = l2[n - 1]
        n -= 1

def findMedianSortedArrays(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    mergeTwoSortedList(nums1, nums2)
    size = len(nums1)
    if size % 2 != 0:
        return nums1[size / 2]
    else:
        index = size / 2
        return (nums1[index - 1] + nums1[index])/ 2.0

# in python 2.6, use module itertools
def findPermutations(s):
    """:type s: string or list"""
    from itertools import permutations
    cons = [''.join(p) for p in permutations(s)]
    print cons # print all permutations
    print set(cons) # print permuted list without duplicates

def all_perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                # elements[0:1] works in both string and list contexts
                yield perm[:i] + elements[0:1] + perm[i:]

# function to compare two version strings
# return 0 if s1 equals s2; -1 if s1 is less than s2; 1  if s1 is greater than s2
# assume input strings include only alphabetic and numerical characters
def compareVersion(s1, s2):
    if s1 is None or s2 is None:
        raise ValueError
    if s1 == s2:
        return 0

    l1 = s1.split(r'.')
    l2 = s2.split(r'.')
    min_len = min(len(l1), len(l2))
    for i in range(min_len):
        if l1[i] == l2[i]:
            continue
        elif l1[i] < l2[i]:
            return -1
        else:
            return 1
    return -1
    # continue to handle the extra elements

# Maximum difference between two elements such that larger element appears after the smaller number
'''Input : arr = {2, 3, 10, 6, 4, 8, 1}
Output : 8
Explanation : The maximum difference is between 10 and 2.

Input : arr = {7, 9, 5, 6, 3, 2}
Output : 2
Explanation : The maximum difference is between 9 and 7.'''
def maxDiff1(arr):
    # exception check of arrary
    maxDiff = arr[1] - arr[0]
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if (arr[j] - arr[i]) > maxDiff:
                maxDiff = arr[j] - arr[i]
    return maxDiff

def maxDiff2(arr):
    maxDiff = arr[1] - arr[0]
    cur_min_num = arr[0]
    for i in range(1, len(arr)):
        if (arr[i] - cur_min_num) > maxDiff:
            maxDiff = arr[i] - cur_min_num
        if arr[i] < cur_min_num:
            cur_min_num = arr[i]
    return maxDiff

# function to invert a binary tree
'''
input:
     4
    / \
  2     7
 / \   / \
1   3 6   9
output:
     4
   /   \
  7     2
 / \   / \
9   6 3   1
'''
# way 1: recursion. This method seems wrong
def invertTree1(root):
    if root is None:
        return
    temp = root.left
    root.left = invertTree1(root.right)
    root.right = invertTree1(temp)
    return root

# way 2: recursion with helper
def invertTree2(root):
    def invertTreeHelper(node):
        if node is None:
            return
        temp = node.right
        node.right = node.left
        node.left = temp
        invertTreeHelper(node.right)
        invertTreeHelper(node.left)

    invertTreeHelper(root)
    return root

# way 3: iterative
def invertTree3(root):
    if root is None:
        raise ValueError
    node_queue = list(root) # queue to store the current node
    while len(node_queue) != 0:
        cur_node = node_queue.pop(0)
        if cur_node.left is not None and cur_node.right is not None:
            cur_node.left, cur_node.right = cur_node.right, cur_node.left
        if cur_node.left is not None:
            node_queue.append(cur_node.left)
        if cur_node.right is not None:
            node_queue.append(cur_node.right)

    return root

# Given a string, sort it in decreasing order based on the frequency of characters.
# if the frequency is same, any order of characters is good
def frequencySort(s):
    if s is None:
        raise ValueError
    if len(s) == 0:
        return ""
    frequency = {}.fromkeys(set(s), 0)
    for c in s:
            frequency[c] += 1

    sorted_tuple = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    return "".join([item[0] * item[1] for item in sorted_tuple])

def frequencySort2(s):
    from collections import Counter
    frequency, c2 = Counter(s), {}
    for k, v in frequency.items():
        c2.setdefault(v, []).append(k * v)  # c2'key is the frequency of each character

    return "".join(["".join(c2[i]) for i in range(len(s), -1, -1) if i in c2])


'''
Binary Search Tree is a node-based binary tree data structure which has the following properties:
The left subtree of a node contains only nodes with keys lesser than the node’s key.
The right subtree of a node contains only nodes with keys greater than the node’s key.
The left and right subtree each must also be a binary search tree.
'''
# Count BST nodes that lie in a given range
'''
Input:
        10
      /    \
    5       50
   /       /  \
 1       40   100
Range: [5, 45]

Output:  3
There are three nodes in range, 5, 10 and 40
'''
class TreeNode(object):
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None

def getCount(root, low, high):
    # base case
    if root is None:
        return 0
    # special case
    if root.value == low and root.value == low:
        return 1

    if low <= root.value <= high: # low <= root.value and root.value <= high:
        return 1 + getCount(root.left, low, high) + getCount(root.right, low, high)
    elif root.value < low:
        return getCount(root.right, low, high)
    else: # root.value > high
        return getCount(root.left, low, high)

# Given a root node reference of a BST and a key (target value), delete the node with the given key in the BST.
# Return the root node reference (possibly updated) of the BST
def deleteNodeFromBST(root, key):
    # node in the BST
    class TreeNode(object):
        def __init__(self, val, n):
            self.value = val  # value is used for sorting
            self.left = None  # left node of this node
            self.right = None # right node of this node
            self.n = n  # number of nodes in subtree

    # helper: find the minimum node in the BST and return it
    def findMin(root):
        if root is None:
            return None
        while root.left is not None:
            root = root.left
        return root
    # helper: find the maximum node in the BST and return it
    def findMax(root):
        if root is None:
            return None
        while root.right is not None:
            root = root.right
        return root

    if root is None:
        return None
    if key < root.value:
        root.left = deleteNodeFromBST(root.left,  key)
    elif key > root.value:
        root.right = deleteNodeFromBST(root.right, key)
    else:
        if (root.left is None) and (root.right is None):  # no left and right subtree
            return None
        elif root.left is None:  # root only has right subtree
            root = root.right
        elif root.right is None:  # root only has left subtree
            root = root.left
        else:  # root has both left and right subtree
            minNodeInRightSubTree = findMin(root.right)
            root.value = minNodeInRightSubTree.value
            root.right = deleteNodeFromBST(root.right, minNodeInRightSubTree.value)

    return root


# l is the first node of the list, this function can remove multiple matched key
def removeNodeFromList(l, key):
    class ListNode(object):  # single list node
        def __init__(self, value):
            self.value = value
            self.next = None

    if l is None:  # handle null list
        return l

    if (l.next is None) and (l.value == key):  # handle the case of only one node in the list and it's the target
        l = l.next
    else:
        pre_node = l
        cur_node = l.next
        while cur_node is not None:
            if cur_node.value == key:  # current node is target node
                pre_node.next = cur_node.next
                return l
            else:
                pre_node = cur_node
            cur_node = cur_node.next
    return l


# Design and implement a data structure for Least Frequently Used (LFU) cache.
# It should support the following operations: get and set. Capacity is applied
# get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
# set(key, value) - Set or insert the value if the key is not already present. When the cache reaches its capacity,
# it should invalidate the least frequently used item before inserting a new item. For the purpose of this problem,
# when there is a tie (i.e., two or more keys that have the same frequency), the least recently used key would be evicted.
def leastFrequencyUsedDesignOld():
    class LFUCacheOld(object):
        def __init__(self, capacity):
            self.capacity = capacity
            self.content = dict()  # store the pair of key, value
            self.lfu_stack = list()  # the least frequently used key is in the beginning, and latest used key is at the end

        def get(self, key):
            if key in self.content.keys():
                self.updateFrequency(key)
                return self.content(key)
            else:
                return -1


        def set(self, key, value):
            if len(self.content) >= self.capacity:
                self.evictLFUItem()

            self.content[key] = value
            self.updateFrequency()

        # update frequency, put latest used key (get/set) at the end
        def updateFrequency(self, key):
            if key in self.lfu_stack:
                self.lfu_stack.remove(key)

            self.lfu_stack.append(key)

        # sort the cache by value, and return sorted dict
        def sortContentByValue(self):
            sorted_tuple = sorted(self.content.items(), key = operator.itemgetter(1))
            sorted_content = {}
            for item in sorted_tuple:
                sorted_content[item(0)] = item[1]
            return sorted_content

        # sort the cache by value, split it into two lists: keys and values
        def convertDictToLists(self):
            return [], []

        def evictLFUItem(self):
            removed_key = self.lfu_stack.pop(0)
            del self.content[removed_key]
    # end of class LFUCache
    # run some tests here
    pass

def parse_parameters_from_cmd_line(proxyconfig):
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "h",
            ["help", "keypregenerated=", "sslkey=", "sslcertreq=", "sslcert=", "sslchaincert=",
             "appip=", "domainname=", "modjkport=", "usetemplate"
             ]
        )
    except getopt.GetoptError, err:
        print err

    argCounter = 0
    requiredArgs = 8
    # don't need to handle the --usetemplate case, that is dealt with in the main
    for o, a in opts:
        if o == "--keypregenerated":
            if a.lower() == "yes" or a.lower() == "no":
                proxyconfig.keyPreGen = a.lower()
                argCounter = argCounter + 1
        elif o == "--sslkey":
            proxyconfig.sslkey = a
            argCounter = argCounter + 1
        elif o == "--sslcertreq":
            proxyconfig.sslcertreq = a
            argCounter = argCounter + 1
        elif o in ("-h", "--help"):
            print "help message"

    if argCounter != requiredArgs and argCounter != 0:
        print 'help message'

def get_mount_point(pathname):
    """Get the mount point of the filesystem containing pathname"""
    pathname= os.path.normcase(os.path.realpath(pathname))
    parent_device= path_device= os.stat(pathname).st_dev
    while parent_device == path_device:
        mount_point= pathname
        pathname= os.path.dirname(pathname)
        if pathname == mount_point: break
        parent_device= os.stat(pathname).st_dev
    return mount_point

def get_mounted_device(pathname):
    """Use /proc/mounts to get the device mounted at pathname"""
    pathname= os.path.normcase(pathname) # might be unnecessary here
    try:
        with open("/proc/mounts", "r") as ifp:
            for line in ifp:
                fields= line.rstrip('\n').split()
                # note that line above assumes that
                # no mount points contain whitespace
                if fields[1] == pathname:
                    return fields[0]
    except EnvironmentError:
        pass
    return None # explicit

def get_fs_freespace(pathname):
    """Get the free space of the filesystem containing pathname"""
    stat= os.statvfs(pathname)
    # use f_bfree for superuser, or f_bavail if filesystem has reserved space for superuser
    return stat.f_bfree*stat.f_bsize


class LFUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {} # {key: cache_node}, key here is same as the key used in cache_node
        self.freq_link_head = None # how many times a node has been accessed (get)

    # get the node value if key exists, and modify this key's frequency (move it forward along the freq_node list)
    def get(self, key):
        if key in self.cache:
            cache_node = self.cache[key]
            freq_node = cache_node.freq_node
            self.move_forward(cache_node,  freq_node)
            return cache_node.value
        else:
            return -1

    def set(self, key, value, freq_node, pre, nxt):
        pass

    # move the cache node along the freq_link (double link), means: increase access time by 1
    def move_forward(self, cache_node, cur_freq_node):
        if cur_freq_node.nxt is not None: # current freq node is not the last node
            if cur_freq_node.freq == cur_freq_node.nxt.freq: # append the cache node to next freq node
                pass
            else: # create a new freq node and insert after cur freq_node
                pass
        else: # create a new freq node and insert after cur freq_node
            pass


class LFUCache2(object):
    import sys
    from datetime import datetime
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.cache = {}
        self.frequence = {}


    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.cache:
            self.increase_frequency(key)
            return self.cache[key]
        else:
            return -1


    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if len(self.cache) >= self.capacity and (key not in self.cache):
            self.remove_lfu_item()
        self.cache[key] = value
        self.increase_frequency(key)


    def increase_frequency(self, key):
        if key in self.frequence:
            self.frequence[key] = (self.frequence[key][0] + 1, datetime.now())
        else:
            self.frequence[key] = (1, datetime.now())

    def remove_lfu_item(self):
        lfu_value = sys.maxint
        lfu_keys= []
        for key, freq in self.frequence.items():
            if freq[0] <= lfu_value:
                lfu_value = freq[0]
                lfu_keys.append(key)

        k = self.find_least_recent_used(lfu_keys)
        del self.frequence[k]
        del self.cache[k]
        self.capacity -= 1

    def find_least_recent_used(self, keys):
        if len(keys) > 1:
            return keys[0]
        else:
            return keys[0]


# double linked list, track which node is access most recently
class CacheNode(object):
    def __init__(self, key, value, freq_node, pre, nxt):
        self.key = key
        self.value = value
        self.freq_node = freq_node
        self.pre = pre
        self.nxt = nxt

    # remove this node from the double-linked list
    def remove_myself(self):
        if self.freq_node.cache_head == self.freq_node.cache_tail:# this node is the only one node in the cache node list
            self.freq_node.cache_head = None
            self.freq_node.cache_tail = None
        elif self.freq_node.cache_head == self:# this node is the head of the link
            self.freq_node.cache_head = self.nxt
            self.nxt.pre = None
        elif self.freq_node.cache_tail == self:# this node is the tail of the the link
            self.freq_node.cache_tail = self.pre
            self.pre.nxt = None
        else: # this node inside the link
            self.pre.nxt = self.nxt
            self.nxt.pre = self.pre

        self.pre = None
        self.nxt = None
        self.freq_node = None

    # remove a node whose key is key, do nothing if key doesn't exit
    def remove_node(self, key):
        if self.key == key:
            self.remove_myself()
        else:
            return

    # add a new node in this link
    def add_node(self, key):
        if self.key == key:
            self.update_node(key)
        else:
            pass

    # update an existing node from this link: update value
    def update_node(self, key):
        pass


# double linked list, track how many times a node has been accessed
class FreqNode(object):
    def __init__(self, freq, pre, nxt, cache_head, cache_tail):
        self.freq = freq # how many times the cache nodes have been accessed
        self.pre = pre
        self.nxt = nxt
        self.cache_head = cache_head  # head of the cache_node list belonging to this FreqNode
        self.cache_tail = cache_tail  # tail of the cache_node list belonging to this FreqNode

    """these functions handle operations of cache node"""
    # append a cache node to tail of cache_node list
    def append_to_cache_node_tail(self, cache_node):
        if freq_node is None:
            return
        cache_node.freq_node = self
        if self.cache_tail is None and self.cache_head is None: # no cache node for this freq_node
            self.cache_head = self.cache_tail = cache_node
        else:
            cache_node.pre = self.cache_tail
            cache_node.nxt = None
            self.cache_tail.nxt = cache_node
            self.cache_tail = cache_node # move the pointer to current node

    # add a cache node to the head of cache_node list
    def add_before_cache_node_head(self, cache_node):
        if cache_node is None:
            return
        cache_node.freq_node = self
        if self.cache_tail is None and self.cache_head is None: # no cache node for this freq_node
            self.cache_head = self.cache_tail = cache_node
        else:
            cache_node.nxt = self.cache_head
            cache_node.pre = None
            self.cache_head.pre = cache_node
            self.cache_head = cache_node

    # count the size of cache node
    def count_cache_node(self):
        if self.cache_head is None and self.cache_tail is None:
            return 0
        else:
            cur_node = self.cache_head
            size = 0
            while cur_node is not None:
                size += 1
                cur_node = cur_node.nxt
            return size

    # pop the cache head node
    def pop_cache_head(self):
        cache_head = self.cache_head
        if self.cache_head is not None:
            if self.cache_head == self.cache_tail: # only one node
                self.cache_head = self.cache_tail = None
            else:
                self.cache_head = self.cache_head.nxt
                self.cache_head.pre = None
        return cache_head
    # remove the cache tail node
    def remove_cache_tail(self):
        cache_tail = self.cache_tail
        if self.cache_tail is not None:
            if self.cache_tail == self.cache_head: # only one node
                self.cache_head = self.cache_tail = None
            else:
                self.cache_tail = self.cache_tail.pre
                self.cache_tail.nxt = None
        return cache_tail
    # remove a cache node whose key is matched, or return None
    def remove_cache_node(self, key):
        if self.cache_head is None and self.cache_tail is None: # no cache node list
            return None
        elif self.cache_head == self.cache_tail: # only one node in the list
            if self.cache_head.key == key:
                cache_node = self.cache_head
                self.cache_head = self.cache_tail = None
                return cache_node
            else:
                return None
        else:
            cache_node = self.cache_head
            while cache_node is not None:
                if cache_node.key == key:
                    cache_node.pre.nxt = cache_node.nxt
                    cache_node.nxt.pre = cache_node.pre
                    return cache_node
                else:
                    cache_node = cache_node.nxt
            return None

    """These functions handle freq_node list"""
    # insert a freq_node in front of me in freq node list
    def insert_before_me(self, freq_node):
        if self.pre is not None: # current node is not the first node
            self.pre.nxt = freq_node
        freq_node.pre = self.pre
        freq_node.nxt = self
        self.pre = freq_node
    # insert a freq_node after me in freq node list
    def insert_after_me(self, freq_node):
        if self.nxt is not None: # current node is not the last node
            self.nxt.pre = freq_node
        freq_node.pre = self
        freq_node.nxt = self.nxt
        self.nxt = freq_node

def disk_partitions(all_partition = False):
    """
    Return all mounted partitions as a namedtuple.
    If all_partition == False then return physical partitions only.
    """
    disk_ntuple = namedtuple('partition',  'device mountpoint fstype')
    phydevs = []
    with open("/proc/filesystems", "r") as f:
        for line in f.readlines():
            if not line.startswith("nodev"):
                phydevs.append(line.strip())

    retlist = list()
    with open("/etc/fstab", "r") as f:
        for line in f.readlines():
            if not all_partition and line.startswith('none'):
                continue
            fields = line.split()
            device = fields[0]
            mountpoint = fields[1]
            fstype = fields[2]
            if not all_partition and fstype not in phydevs:
                continue
            if device == 'none':
                device = ''
            ntuple = disk_ntuple(device, mountpoint, fstype)
            retlist.append(ntuple)
    return retlist

def disk_usage(path):
    """Return disk usage associated with path."""
    usage_ntuple = namedtuple('usage',  'total used free percent')
    st = os.statvfs(path)
    free = (st.f_bavail * st.f_frsize)
    total = (st.f_blocks * st.f_frsize)
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    try:
        percent = ret = (float(used) / total) * 100
    except ZeroDivisionError:
        percent = 0
    # the percentage is -5% than what shown by df due to reserved blocks that we are currently not considering:
    # http://goo.gl/sWGbH
    return usage_ntuple(total, used, free, round(percent, 1))

def is_ipv4(ip):
    ls = ip.split(".")
    if len(ls) == 4 and all(g.isdigit() and str(int(g)) == g and 0 <= int(g) <= 255 for g in ls):
        return True
    return False

def is_ipv6(ip):
    ls = ip.split(":")
    if len(ls) == 8 and all(0<len(g)<=4 and all(c in '0123456789abcdefABCDEF' for c in g)for g in ls):
        return True
    return False

def reverse_words(s):
    s = s.strip()
    l1 = re.split(r'\s+',s)
    l2 = l1[::-1]
    r = " ".join(l2)
    return r

def print_to_text(base_url):
    import requests
    from bs4 import BeautifulSoup
    r = requests.get(base_url)
    soup = BeautifulSoup(r.text, features="html.parser")
    with open("less.txt", "w") as f:
        for paragraph in soup.findall(dir='ltr'):
            f.write(paragraph.text.replace("<span>", ""))

# if this module is called directly like this: python <file name>
if __name__ == '__main__':
    for part in disk_partitions():
        print part
        print "%s\n" % str(disk_usage(part.mountpoint))

