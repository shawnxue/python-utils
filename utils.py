#!/usr/bin/env python
# -*- coding: UTF-8 -*

# function to check if a string is dui chen
from distutils import extension
import re
import getopt
import random
import operator
from collections import namedtuple
import os, sys

# Here’s a function with four required keyword-only arguments
from random import choice, shuffle
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE = UPPERCASE.lower()
DIGITS = "0123456789"
ALL = UPPERCASE + LOWERCASE + DIGITS
def random_password(*, upper, lower, digits, length):
    import random
    chars = [
        *(random.choice(UPPERCASE) for _ in range(upper)),
        *(random.choice(LOWERCASE) for _ in range(lower)),
        *(random.choice(DIGITS) for _ in range(digits)),
        *(random.choice(ALL) for _ in range(length-upper-lower-digits)),
    ]
    random.shuffle(chars)
    return "".join(chars)

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    
    return digits[::-1]

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

def longest_substring_1(s):
  length, ans = 0, ""
  queue = ""
  for c in s:
    if c not in queue:
      queue += c
      if length < len(queue):
        ans = queue
        length = max(len(ans), length)
    else:
      queue = queue[queue.index(c)+1:] + c
  return length, ans

# Find the length of the longest substring T of a given string (consists of lowercase letters only)
# such that every character in T appears no less than k times.
def longestSubstring_2(s, k):
  for c in set(s):
    if s.count(c) < k:
      return max(longestSubstring_2(item, k) for item in s.split(c))
  return len(s)

def isValid(s):
  my_dict = {"{": "}", "<": ">", "(": ")"}
  my_stack = [] # storing keys in my_dict
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
  my_stack = list() # storing index of char in s
  my_stack.append(-1) # empty stack and push -1 to it. The first element of stack is used to provide base for next valid string.
  
  #left = {} # save the ')' matching most left '(' success, both use index (key is right index, value is most left index)
  length = 0
  for i, c in enumerate(s):
    if c == "(":
      my_stack.append(i) # push index
    elif c == ")":
      my_stack.pop() # Pop the previous opening bracket's index (last element in the list)
      if len(my_stack) != 0:
        length = max(length, i - my_stack[len(my_stack)-1])
      else:
        my_stack.append(i)
    else:
      continue
    #r = i
    #if left.has_key(l-1):
    #    left[r] = left[l-1]
    #else:
    #    left[r] = l
    #length = r - left[r] + 1
  return length

# function to return the first non-alphabetic order character in a string
# test your functions with different input
# example
# input: ceggghkbblortv
# output: b
def firstNonAlphabeticOrderChar(s):
  result = ""
  if len(s) < 0:
    return result
  # strip special characters, white space and numbers
  input = s.lower().strip().replace(" ", "")
  p1 = re.compile(r'\W+') # or p = re.compile('[^a-zA-Z0-9]+')
  input = re.sub(p1, '', input) # or: input = re.sub(r'\W+', '', input)
  # another way to strip special characters : input = ''.join([c for c in input if c.isalnum()])
  # we can use regex to replace all white spaces as well, or split string with space and then join it
  p2 = re.compile(r'\s+')
  input = re.sub(p2, '', input)
  input = ''.join([c for c in input if not c.isdigit()]) # strip numbers
  input = ''.join([c for c in input if c.isalnum()])
  print(f"input string is {0}".format(input))
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
    n = int(n / 10) # or use // operator: n = n // 10
  return sum == input

# Given a double, ‘x’, and an integer, ‘n’, write a function to calculate ‘x’ raised to the power ‘n’. For example:
# power (2, 5) = 32, power (3, 4) = 81, power (1.5, 3) = 3.375, power (2, -2) = 0.25
def power_rec(x, n):
  if n == 0:
    return 1
  if n == 1:
    return x
  temp = power_rec(x, n//2)
  if n % 2 == 0:
    return temp * temp
  else:
    return x * temp * temp

def power(x, n):
  is_negative = False
  if n < 0:
    is_negative = True
    n *= -1
  result = power_rec(x, n)
  if is_negative:
    return 1 / result
  return result

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
            num1_s = list(diff_dict.keys())[list(diff_dict.values()).index(num)]
            num1 = int(num1_s)
            result.append((nums.index(num1), index))
        else:
            diff_dict[str(num)] = target - num
    if len(result) > 0:
        return result
    else:
        return [(-1,-1)]

def threeSum(nums, target):
  """
  function to return indices of the three numbers such that they add up to a specific target.
  :type nums: List[int]
  :type target: int
  :rtype: List[int]
  """
  if len(nums) < 2:
    return [(-1,-1)]
  result = list()
  for i in range(len(nums) - 2):
    remain_sum = target - nums[i]
    sum_2 = twoSum(nums, remain_sum)
    print(f"sum_2={sum_2}")
    if sum_2[0][0] != -1:
      result.append((i, sum_2[0][0], sum_2[0][1]))
  print(f"target={target}, nums={nums}")
  if len(result) != 0:
    return result
  else:
    return [(-1,-1,-1)]

# Given a positive integer, target, print all possible combinations of positive integers that sum up to the target number.
# For example, if we are given input ‘5’, these are the possible sum combinations.
# 1, 4
# 2, 3
# 1, 1, 3
# 1, 2, 2
# 1, 1, 1, 2
# 1, 1, 1, 1, 1
import copy
def print_all_sum_rec(target, current_sum, start, output, result):
  if current_sum == target:
    output.append(copy.copy(result))

  for i in range(start, target):
    temp_sum = current_sum + i
    if temp_sum <= target:
      result.append(i)
      print_all_sum_rec(target, temp_sum, i, output, result)
      result.pop()
    else:
      return

def print_all_sum(target):
  output = []
  result = []
  print_all_sum_rec(target, 0, 1, output, result)
  return output

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
            greater += 1
    return greater

# return True if string is Palindrome ( string is equal with its reverse)
def isPalindrome(s):
    return s.lower() == "".join(reversed(s.lower())) # or s.lower() == s[::-1].lower()

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
# time complexity for recurision is O(#branches power of depth). For fib, branch number is 2, depth is n, so time complexity is O(2**n)
# space complexity is O(depth), such as fib, it's O(n)
def recur_fibo(n):
    if n <= 0:
        return 0
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
        """
        t = timeit.Timer(lamda: recur_fibo(i))
        time = t.timeit(1)
        print(f"fib({0}) took {1} time".format(i, time))
        """

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
    nums = [num for num in range(1, n+1)]
    while len(nums) > 1:
        nums = nums[1::2][::-1] # from left, return num starting from index 1 to the end, and step is 2, then ::-1 reverse the list
    return nums[0]

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
def singleNumber(nums: List[int]): 
    my_set = set(nums)
    result = []
    for num in my_set:
        if nums.count(num) == 1:
            result.append(num)
    return result

# function to find num of arithmetic sub string from nums, nums is int list
def numberOfArithmeticSlices(nums: list[int]):
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
        mid = (high + low) // 2
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
        middle = (first + last) // 2
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
    import sys
    count = 0
    removed_items = list()
    input = sorted(intervals, key = lambda  item: item[1])
    pre_item_end =  -sys.maxsize - 1 # sys.maxsize is the max int in Python 3
    for interval in input:
        if interval[0] >= pre_item_end:
            pre_item_end = interval[1]
        else:
            count += 1
            removed_items.append(interval)
    return count, removed_items

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

# given two words  start and end, and a dictionary of word list, find all shortest transformation sequences
# from start to end. (only one letter can be changed at a time, ech intermediate word must in the dictionary)
# e.g. start: "hit", end:"cog". dictionary: ["hot", "dot", "dog", "lot", "log"]
# return: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
def findLadders(start, end, dict):
    result = []
    if len(dict) == 0:
        return result
    queue = list([start]) # store the words from start to end
    ladder = {}.fromkeys(dict, sys.maxsize)
    ladder[start] = 0
    dict.append(end)
    my_map = {} # key is word, value is the list of word path
    import string
    # BFS: Dijisktra search, breath first search
    while len(queue) != 0:
        word = queue.pop(-1) # remove and return the last obj from the list
        step = ladder[word] + 1 # step indicates how many steps are need to travel to word
        if step > sys.maxsize:
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
# l1 and l2 are linked list, each node is an integer
# add them together and return the new list. New node's value is the reminder: (l1.node + l2.node + carry) % 10
class Node:
  def __init__(self, data=None, next=None):
    self.data = data
    self.next  = next
    
def addLinkedListInteger(l1, l2):
  head = None
  nxt = None
  carry = 0 # carry is the integer part of sum devided by 10
  while (l1 is not None or l2 is not None or carry > 0):
    first = (0 if l1 is None else l1.data)
    second = (0 if l2 is None else l2.data)
    sum = first + second + carry
    newNode = Node(sum % 10)
    carry = sum // 10
    if head is None:
      head = newNode
    else:
      nxt.next = newNode
    nxt = newNode
    if l1 is not None:
      l1 = l1.next
    if l2 is not None:
      l2 = l2.next

  return head

# time complexcity: O(1), space complexicity: Linear, O(m + n), m and n is the length of linked list
def mergeTwoSortedLinkedList(l1, l2):
    if l1 is None:
        return l2
    elif l2 is None:
        return l1
    else:
        pass

    mergedHead = None
    # handle first node
    if l1.data <= l2.data:
        mergedHead = l1
        l1 = l1.next
    else:
        mergedHead = l2
        l2 = l2.next
    currentNode = mergedHead
    # handle rest node
    while l1 is not None and l2 is not None:
        smallerNode = None
        if l1.data <= l2.data:
            smallerNode = l1
            l1 = l1.next
        else:
            smallerNode = l2
            l2 = l2.next
        currentNode.next = smallerNode
    # handle the longer linked list
    if l1 is not None:
        currentNode.next = l1
    elif l2 is not None:
        currentNode.next = l2
    else:
        pass
    return mergedHead

def mergeTwoSortedList(l1, l2):
    m = len(l1)
    n = len(l2)
    l1.extend([-sys.maxsize - 1] * n) # extend the result list so that it can hold all nums, maxsize in python 2 and maxsize in python 3
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
    return l1

def findMedianSortedArrays(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    mergeTwoSortedList(nums1, nums2)
    size = len(nums1)
    if size % 2 != 0:
        return nums1[size // 2]
    else:
        index = size / 2
        return (nums1[index - 1] + nums1[index])/ 2.0

# in python 2.6, use module itertools
def findPermutations(s):
    """:type s: string or list"""
    from itertools import permutations
    cons = [''.join(p) for p in permutations(s)]
    print(cons) # print all permutations
    print(set(cons)) # print permuted list without duplicates

def all_perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                # elements[0:1] works in both string and list contexts
                yield perm[:i] + elements[0:1] + perm[i:]

# function to compare two version
# return 0 if s1 equals s2; -1 if s1 is less than s2; 1 if s1 is greater than s2
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
    if len(l1) < len(l2):
        return -1
    else:
        return 1
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

#
#      1
#  2      3
# 4 5
# tree traversals:
## inorder:   left, root, right: 4 2 5 1 3
## postorder: left, right, root: 4 5 2 3 1
## preorder:  root, left, right: 1 2 4 5 3
## breadth first, level order: 1 2 3 4 5

# given an inorder array (left, root, right) and a postorder array (left, right, root), construct a binary tree
# print as preorder: root, left, right
def create_tree_1(inorder, postorder):
    class TreeNode():
        def __init__(self, data):
            self.data = data
            self.left = None
            self.right = None

    def build_tree_1(in_array, post_array):
        # base case
        if len(in_array) <= 0:
            return None
        # current_root_index is the last index in post array, use it to create a node
        current_root = TreeNode(post_array[-1])
        # return the node if it has no child nodes
        if len(in_array) == 1:
            return current_root
        # find this node's index in inorder array, so that we can recursive
        in_index = in_array.index(current_root.data)
        in_array_left_tree = in_array[0:in_index]
        in_array_right_tree = in_array[in_index + 1:]
        post_array_left_tree = post_array[0:len(in_array_left_tree)]
        post_array_right_tree = post_array[-len(in_array_right_tree) - 1:-1]
        
        # use this index in inorder array to construct left and right subtree
        current_root.right = build_tree_1(in_array_right_tree, post_array_right_tree)
        current_root.left = build_tree_1(in_array_left_tree, post_array_left_tree)
        return current_root

    root = build_tree_1(inorder, postorder)
    return root

# given an inorder array (left, root, right) and a preorder array (root, left, right), construct a binary tree
# input: Inorder sequence: D B E A F C , Preorder sequence: A B D E C F
# output: print as preorder (root, left, right): A B D E C F
def create_tree_2(inorder, preorder):
    class TreeNode():
        def __init__(self, data):
            self.data = data
            self.left = None
            self.right = None
    def build_tree_2(in_array, pre_array):
        # base
        if len(in_array) <= 0 or len(pre_array) <= 0:
            return None
        # current root the first element in preorder array
        current_root = TreeNode(pre_array[0])
        if len(in_array) == 1:
            return current_root
        # find the index of current root in inorder array, so that we can split the inorder array
        in_index = in_array.index(current_root)
        in_array_left_tree = in_array[0:in_index]
        pre_array_left_tree = pre_array[1:len(in_array_left_tree) + 1]
        in_array_right_tree = in_array[in_index + 1:]
        pre_array_right_tree = pre_array[-len(in_array_right_tree):]
        current_root.right = build_tree_2(in_array_right_tree, pre_array_right_tree)
        current_root.left = build_tree_2(in_array_left_tree, pre_array_left_tree)
        return current_root

    root = build_tree_2(inorder, preorder)
    return root

# print a tree by preorder (root, left, right)
def pre_order_print(root):
    if root == None:
        return
    print(root.data, end=" ")
    pre_order_print(root.left)
    pre_order_print(root.right)

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

def mirror_tree(root):
    if root == None:
        return

    # We will do a post-order traversal of the binary tree.
    if root.left != None:
        mirror_tree(root.left)

    if root.right != None:
        mirror_tree(root.right)

    # Let's swap the left and right nodes at current level.
    temp = root.left
    root.left = root.right
    root.right = temp

def level_order_traversal(root):
    if root == None:
        return
    q = deque()
    q.append(root)

    while q:
        temp = q.popleft()
        print(str(temp.data), end = ",")
        if temp.left != None:
            q.append(temp.left)
        if temp.right != None:
            q.append(temp.right)

arr = [100,25,75,15,350,300,10,50,200,400,325,375]
root = create_BST(arr)
#root = create_random_BST(15)
print("\nLevel Order Traversal:", end = "")
level_order_traversal(root)

mirror_tree(root)
print("\nMirrored Level Order Traversal:", end = "")
level_order_traversal(root)

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
    if root.value == low and root.value == high:
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

# LRU
def leastRecentlyUsedDesignNew():
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = set()
            self.cache_vals = LinkedList() # we can use deque as linkedlist, from collections import deque

        def get(self, value):
            if value not in self.cache:
                return None
            else:
                i = self.cache_vals.get_head()
                while i is not None:
                    if i.data == value:
                        return i
                    i = i.next
    
        def set(self, value):
            node = self.get(value)
            if node == None:
                if(self.cache_vals.size >= self.capacity):
                    self.cache_vals.insert_at_tail(value)
                    self.cache.add(value)
                    self.cache.remove(self.cache_vals.get_head().data)
                    self.cache_vals.remove_head()
                else:
                    self.cache_vals.insert_at_tail(value)
                    self.cache.add(value)
            else:
                self.cache_vals.remove(value)
                self.cache_vals.insert_at_tail(value)

        def printcache(self):
            node = self.cache_vals.get_head()
            while node != None:
                print(str(node.data) + ", ")
                node = node.next
      
    cache1 = LRUCache(4)
    cache1.set(10)
    cache1.printcache()
    cache1.set(15)
    cache1.printcache()
    cache1.set(20)
    cache1.printcache()
    cache1.set(25)
    cache1.printcache()
    cache1.set(30)
    cache1.printcache()
    cache1.set(20)
    cache1.printcache()
    pass

def parse_parameters_from_cmd_line(proxyconfig):
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "h",
            ["help", "keypregenerated=", "sslkey=", "sslcertreq=", "sslcert=", "sslchaincert=",
             "appip=", "domainname=", "modjkport=", "usetemplate"
             ]
        )
    except getopt.GetoptError as err:
        print(err)

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
        raise EnvironmentError
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
            self.frequence[key] = (self.frequence[key][0] + 1, datetime.now().timestamp()) # or import time time.time()
        else:
            self.frequence[key] = (1, datetime.now().timestamp())

    def remove_lfu_item(self):
        lfu_value = sys.maxsize
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

#172.16.254.01 is invalid becauase 01 starts with 0
def is_ipv4_2(ip):
    if ip is None:
        return False
    ls = ip.split(".")
    results = []
    for g in ls:
        if len(ls) == 4 and g.isdigit() and str(int(g)) == g and 0<= int(g) <=255:
            continue
        else:
            return False
    return True

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
# calculate the angle between hour and minute
def calculateAngle(h,m):       
  # validate the input
  if (h < 0 or m < 0 or h > 24 or m > 60):
    print('Wrong input')
  if (m == 60):
    h += 1;
  m = m % 60
  h = h % 12
  # Calculate the angles moved by hour and minute hands with reference to 12:00
  hour_angle = 0.5 * (h * 60 + m)
  minute_angle = 6 * m
  # Find the difference between two angles
  angle = abs(hour_angle - minute_angle)
  # Return the smaller angle of two possible angles
  angle = min(360 - angle, angle)
  return angle

# print('9:00 Angle ', calculateAngle(9,0))
# print('9:60 Angle ', calculateAngle(9,60))
# print('2:15 Angle ', calculateAngle(2,15))
# print('2:0 Angle ', calculateAngle(2,0))
# print('2:60 Angle ', calculateAngle(2,60))
# print('13:10 Angle ', calculateAngle(13,10))

# identical tree, not only node data, but also structure
# time complexicity: worst case Linear, O(n), balanced tree O(logn). space complexcity: O(height)
def isTreeIdentical(root1, root2):
    if root1 is None and root2 is None:
        return True
    if root1 is not None and root2 is not None:
        return (root1.data == root2.data and isTreeIdentical(root1.left, root2.left) and isTreeIdentical(root1.right, root2.right))
    return False

# find the maximum sum of any contiguous subarray in an integer array
# The runtime complexity of this solution is linear, O(n).The memory complexity of this solution is constant, O(1).
def findMaxSubArraySum(nums):
  global_max = nums[0]
  current_max = nums[0]
  print(f"current={current_max}, global={global_max}")
  # for i in range(1, len(nums))
  for idx, num in enumerate(nums):
    if idx == 0:
      continue
    if current_max < 0:
      current_max = num
    else:
      current_max += num
    if global_max < current_max:
      global_max = current_max
    print(f"current={current_max}, global={global_max}")
  return global_max
# print(findMaxSubArraySum([-4, 2, -5, 1, 2, 3, 6, -5, 1]))

# Given the root node of a directed graph, clone this graph by creating its deep copy so that the cloned graph has the same vertices and edges as the original graph.
class Node:
  def __init__(self, d):
    self.data = d
    self.neighbors = []

def clone_rec(root, nodes_completed):
  if root == None:
    return None

  pNew = Node(root.data)
  nodes_completed[root] = pNew

  for p in root.neighbors:
    x = nodes_completed.get(p)
    if x == None:
      pNew.neighbors += [clone_rec(p, nodes_completed)]
    else:
      pNew.neighbors += [x]
  return pNew

def clone(root):
  nodes_completed = {}
  return clone_rec(root, nodes_completed)

# python multi thread concurrency, producer and consumer, use queue as lock
""""
Since the Queue has a Condition and that condition has its Lock we don't need to bother about Condition and Lock.
Producer uses Queue.put(item[, block[, timeout]]) to insert data in the queue. It has the logic to acquire the lock 
before inserting data in queue. If optional args block is true and timeout is None (the default), block if necessary 
until a free slot is available. If timeout is a positive number, it blocks at most timeout seconds and raises the Full 
exception if no free slot was available within that time. Otherwise (block is false), put an item on the queue if a 
free slot is immediately available, else raise the Full exception (timeout is ignored in that case).

Also put() checks whether the queue is full, then it calls wait() internally and so producer starts waiting

Consumer uses Queue.get([block[, timeout]]), and it acquires the lock before removing data from queue. If the queue is empty, 
it puts consumer in waiting state. Queue.get() has notify() method
"""

import threading, time, logging, random
from queue import Queue
logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-9s) %(message)s',)
BUF_SIZE = 10
q = Queue(BUF_SIZE)
class ProducerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ProducerThread,self).__init__()
        self.target = target
        self.name = name

    def run(self):
        while True:
            if not q.full():
                item = random.randint(1,10)
                q.put(item)
                logging.debug('Putting ' + str(item) + ' : ' + str(q.qsize()) + ' items in queue')
                time.sleep(random.random())
        return
class ConsumerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name
        return

    def run(self):
        while True:
            if not q.empty():
                item = q.get()
                logging.debug('Getting ' + str(item) + ' : ' + str(q.qsize()) + ' items in queue')
                time.sleep(random.random())
        return
if __name__ == '__main__':
    p = ProducerThread(name='producer')
    c = ConsumerThread(name='consumer')
    p.start()
    time.sleep(2)
    c.start()
    time.sleep(2)

# Use lock object in threading to control access of shared object, in this case it's Counter object
class Counter(object):
    def __init__(self, start = 0):
        self.lock = threading.Lock()
        self.value = start
    def increment(self):
        logging.debug('Waiting for a lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired a lock')
            self.value = self.value + 1
        finally:
            logging.debug('Released a lock')
            self.lock.release()
# function worker is the target of thread, thread will call this function
def worker(c: Counter):
    for i in range(2):
        r = random.random()
        logging.debug('Sleeping %0.02f', r)
        time.sleep(r)
        c.increment()
    logging.debug('Done')

if __name__ == '__main__':
    counter = Counter()
    for i in range(2):
        t = threading.Thread(target=worker, args=(counter,))
        t.start()
    logging.debug('Waiting for worker threads')
    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is not main_thread:
            t.join()
    logging.debug('Counter: %d', counter.value)

"""
Another example:
In this example, worker() tries to acquire the lock three separate times, and counts how many attempts it has to make to do so. 
In the mean time, locker() cycles between holding and releasing the lock, with short sleep in each state used to simulate load.
"""
def locker(lock: threading.Lock()):
    logging.debug('Starting')
    while True:
        lock.acquire()
        try:
            logging.debug('Locking')
            time.sleep(1.0)
        finally:
            logging.debug('Not locking')
            lock.release()
        time.sleep(1.0)
    return

def locker_by_using_with(lock: threading.Lock()):
    logging.debug('Starting')
    while True:
        with lock:
            logging.debug('Locking')
            time.sleep(1.0)
        time.sleep(1.0)
    return

def worker_by_using_with(lock: threading.Lock()):
    logging.debug('Starting')
    num_tries = 0
    num_acquires = 0
    while num_acquires < 3:
        time.sleep(0.5)
        logging.debug('Trying to acquire')
        with lock:
            num_tries += 1
            logging.debug('Try #%d : Acquired',  num_tries)
            num_acquires += 1

    logging.debug('Done after %d tries', num_tries)

def worker(lock: threading.Lock()):
    logging.debug('Starting')
    num_tries = 0
    num_acquires = 0
    while num_acquires < 3:
        time.sleep(0.5)
        logging.debug('Trying to acquire')
        acquired = lock.acquire(0)
        try:
            num_tries += 1
            if acquired:
                logging.debug('Try #%d : Acquired',  num_tries)
                num_acquires += 1
            else:
                logging.debug('Try #%d : Not acquired', num_tries)
        finally:
            if acquired:
                lock.release()
    logging.debug('Done after %d tries', num_tries)

if __name__ == '__main__':
    lock = threading.Lock()

    locker_thread = threading.Thread(target=locker, args=(lock,), name='Locker')
    locker_thread.setDaemon(True)
    locker_thread.start()

    worker_thread = threading.Thread(target=worker, args=(lock,), name='Worker')
    worker_thread.start()

""""
In this chapter, we'll learn another way of synchronizing threads: using a Condition object. Because a condition variable is always 
associated with some kind of lock, it can be tied to a shared resource. A lock can be passed in or one will be created by default. 
Passing one in is useful when several condition variables must share the same lock. The lock is part of the condition object: we don't have to track it separately. 
So, the condition object allows threads to wait for the resource to be updated.
In the following example, the consumer threads wait for the Condition to be set before continuing. The producer thread is responsible for setting the condition 
nd notifying the other threads that they can continue.
"""
def consumer(cv: threading.Condition()):
    logging.debug('Consumer thread started ...')
    with cv:
        logging.debug('Consumer waiting ...')
        cv.wait()
        logging.debug('Consumer consumed the resource')

def producer(cv: threading.Condition()):
    logging.debug('Producer thread started ...')
    with cv:
        logging.debug('Making resource available')
        logging.debug('Notifying to all consumers')
        cv.notifyAll()

if __name__ == '__main__':
    condition = threading.Condition()
    cs1 = threading.Thread(name='consumer1', target=consumer, args=(condition,))
    cs2 = threading.Thread(name='consumer2', target=consumer, args=(condition,))
    pd = threading.Thread(name='producer', target=producer, args=(condition,))

    cs1.start()
    time.sleep(2)
    cs2.start()
    time.sleep(2)
    pd.start()
# if this module is called directly like this: python <file name>
if __name__ == '__main__':
    for part in disk_partitions():
        print(part)
        print("%s\n" % str(disk_usage(part.mountpoint)))

# python multi process: 
# https://www.toptal.com/python/beginners-guide-to-concurrency-and-parallelism-in-python#:~:text=What's%20the%20difference%20between%20Python,child%20processes%20bypassing%20the%20GIL.
# https://timber.io/blog/multiprocessing-vs-multithreading-in-python-what-you-need-to-know/
import json
import logging
import os
from pathlib import Path
from urllib.request import urlopen, Request

logger = logging.getLogger(__name__)

types = {'image/jpeg', 'image/png'}


def get_links(client_id):
    headers = {'Authorization': 'Client-ID {}'.format(client_id)}
    req = Request('https://api.imgur.com/3/gallery/random/random/', headers=headers, method='GET')
    with urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return [item['link'] for item in data['data'] if 'type' in item and item['type'] in types]


def download_link(directory, link):
    download_path = directory / os.path.basename(link)
    with urlopen(link) as image, download_path.open('wb') as f:
        f.write(image.read())
    logger.info('Downloaded %s', link)


def setup_download_dir():
    download_dir = Path('images')
    if not download_dir.exists():
        download_dir.mkdir()
    return download_dir

class DownloadWorker(threading.Thread):

    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            directory, link = self.queue.get()
            try:
                download_link(directory, link)
            finally:
                self.queue.task_done()


def main():
    ts = time()
    client_id = os.getenv('IMGUR_CLIENT_ID')
    if not client_id:
        raise Exception("Couldn't find IMGUR_CLIENT_ID environment variable!")
    download_dir = setup_download_dir()
    links = get_links(client_id)
    # Create a queue to communicate with the worker threads
    queue = Queue()
    # Create 8 worker threads
    for x in range(8):
        worker = DownloadWorker(queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
    # Put the tasks into the queue as a tuple
    for link in links:
        logger.info('Queueing {}'.format(link))
        queue.put((download_dir, link))
    # Causes the main thread to wait for the queue to finish processing all the tasks
    queue.join()
    logging.info('Took %s', time() - ts)

if __name__ == '__main__':
    main()

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import time

from download import setup_download_dir, get_links, download_link

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def main():
    client_id = os.getenv('IMGUR_CLIENT_ID')
    if not client_id:
        raise Exception("Couldn't find IMGUR_CLIENT_ID environment variable!")
    download_dir = setup_download_dir()
    links = get_links(client_id)

    # By placing the executor inside a with block, the executors shutdown method
    # will be called cleaning up threads.
    # 
    # By default, the executor sets number of workers to 5 times the number of
    # CPUs.
    with ThreadPoolExecutor() as executor:

        # Create a new partially applied function that stores the directory
        # argument.
        # 
        # This allows the download_link function that normally takes two
        # arguments to work with the map function that expects a function of a
        # single argument.
        fn = partial(download_link, download_dir)

        # Executes fn concurrently using threads on the links iterable. The
        # timeout is for the entire process, not a single call, so downloading
        # all images must complete within 30 seconds.
        executor.map(fn, links, timeout=30)


if __name__ == '__main__':
    main()

#  This adjacency list could also be represented in code using a dict
# The keys of this dictionary are the source vertices, and the value for each key is a list. This list is usually implemented as a linked list.
graph = {
    1: [2, 3, None], 
    2: [4, None],
    3: [None],
    4: [5, 6, None],
    5: [6, None],
    6: [None]
}

# collections.deque uses an implementation of a linked list in which you can access, insert, or remove elements from the beginning or end of a list with constant O(1) performance.
from collection import deque
llist = deque("abcde")
# deque(['a', 'b', 'c', 'd', 'e'])

llist.append("f")
# deque(['a', 'b', 'c', 'd', 'e', 'f'])

llist.pop()

#solution for solve sudoku
## Given a partially filled 9×9 2D array ‘grid[9][9]’, the goal is to assign digits (from 1 to 9) to the empty cells so that every row, column, 
## and subgrid of size 3×3 contains exactly one instance of the digits from 1 to 9. 
## input:
# grid = { { 3, 1, 6, 5, 7, 8, 4, 9, 2 },
#          { 5, 2, 9, 1, 3, 4, 7, 6, 8 },
#          { 4, 8, 7, 6, 2, 9, 5, 3, 1 },
#          { 2, 6, 3, 0, 1, 5, 9, 8, 7 },
#          { 9, 7, 4, 8, 6, 0, 1, 2, 5 },
#          { 8, 5, 1, 7, 9, 2, 6, 4, 3 },
#          { 1, 3, 8, 0, 4, 7, 2, 0, 6 },
#          { 6, 9, 2, 3, 5, 1, 8, 7, 4 },
#          { 7, 4, 5, 0, 8, 6, 3, 1, 0 } };
## output:
# 3 1 6 5 7 8 4 9 2 
# 5 2 9 1 3 4 7 6 8 
# 4 8 7 6 2 9 5 3 1 
# 2 6 3 4 1 5 9 8 7 
# 9 7 4 8 6 3 1 2 5 
# 8 5 1 7 9 2 6 4 3 
# 1 3 8 9 4 7 2 5 6 
# 6 9 2 3 5 1 8 7 4 
# 7 4 5 2 8 6 3 1 9
def print_grid(arr):
    for row in range(9):
        for col in range(9):
            print(arr[row][col], end = " ")
        print()
# Function to Find the entry in the Grid that is still not used. Searches the grid to find an entry that is still unassigned (spot value is 0).
# If found, the reference parameters row, col will be set the location and true is returned. If no unassigned, false is returned.
# 'l' is a list variable that has been passed from the solve_sudoku function to keep track of incrementation of Rows and Columns
def find_empty_spot(arr,l):
    for row in range(9):
        for col in range(9):
            if(arr[row][col] == 0):
                l[0] = row
                l[1] = col
                return True
    return False
# Returns a boolean which indicates whether the given number has been assigned in the specified row
def used_in_row(arr, row, num):
    for col in range(9):
        if arr[row][col] == num:
            return True
    return False
# Returns a boolean which indicates whether the given number has been assigned entry in the specified column
def used_in_col(arr, col, num):
    for row in range(9):
        if arr[row][col] == num:
            return True
    return False
# Returns a boolean which indicates whether the given number has been assigned within the specified 3x3 box
def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if arr[row + i][col + j] == num:
                return True
    return False
# Checks whether it will be legal to assign num to the given row, col.
# Return True means that the given number is not shown in the row, col and 3*3 box, we can assgin the given number to given row and col
def is_spot_safe(arr, row, col, num):
    return (not used_in_row(arr, row, num)) and (not used_in_col(arr, col, num)) and (not used_in_box(arr, row - row % 3, col - col % 3, num))
# Takes a partially filled-in grid arr (not all 0) and attempts to assign values to all unassigned locations in such a way to meet the requirements
# for Sudoku solution (non-duplication across rows, columns, and boxes)
# solution 1: use recursive, seems wrong
def solve_sudoku_1(arr):
    # l is list, tracks the row and col found in find_empty_spot
    l = [0, 0]
    # base class
    # if not found empty spot, it means all spots have been assgiend non-0 value, sudoku is resolved
    # if found empty spot, l[0] is row, l[1] is col
    if (not find_empty_spot(arr, l)):
        return True
    row, col = l[0], l[1]
    # recursively assign digits 1 to 9 to all spots
    for num in range(1, 10):
        if is_spot_safe(arr, row, col, num):
            arr[row][col] = num
        if solve_sudoku_1(arr):
            return True
        # sudoku is not solved, unmake and try next number
        arr[row][col] = 0
    # this trigger backtracking
    return False
# solution 2: silly way
N = 9
def solve_sudoku_2(arr, row, col):
    # Check if we have reached the 8th row and 9th column (0 indexed matrix) , we are returning true to avoid further backtracking
    if (row == N - 1) and (col == N):
        return True
    # Check if column value  becomes 9, we move to next row and column start from 0
    if (col == N):
        row += 1
        col = 0
    # Check if the current position of the grid already contains value >0, we iterate for next column
    if arr[row][col] != 0:
        return solve_sudoku_2(arr, row, col + 1)
    for num in range(1, N + 1):
        if is_spot_safe(arr, row, col, num):
            arr[row][col] = num
            # Checking for next possibility with next column
            if solve_sudoku_2(arr, row, col + 1):
                return True
        arr[row][col] = 0
    return False
# driver function
def run_sudoku_program():
    grid = [[x for x in range(9)] for y in range(9)]
    # randomly assigning values to the grid
    grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
            [5, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 8, 7, 0, 0, 0, 0, 3, 1],
            [0, 0, 3, 0, 1, 0, 0, 8, 0],
            [9, 0, 0, 8, 6, 3, 0, 0, 5],
            [0, 5, 0, 0, 9, 0, 6, 0, 0],
            [1, 3, 0, 0, 0, 0, 2, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 7, 4],
            [0, 0, 5, 2, 0, 6, 3, 0, 0]]
    if (solve_sudoku_2(grid)):
        print_grid(grid)
    else:
        print("No solution exists")

# exercise_15 rename pictures based on city and sorted by taken date, output sequence must be same as inout
# input: string ended with \n at each line
# output: string ended with \n at each line
## input:
s_input = """photo.jpg, Warsaw, 2013-09-05 14:08:15\n\
    john.png, London, 2015-06-20 15:13:22\n\
    myFriends.png, Warsaw, 2013-09-05 14:07:13\n\
    Eiffel.jpg, Paris, 2015-07-23 08:03:02\n\
    pisatower.jpg, Paris, 2015-07-22 23:59:59\n\
    BOB.jpg, London, 2015-08-05 00:02:03\n\
    notredame.png, Paris, 2015-09-01 12:00:00\n\
    me.jpg, Warsaw, 2013-09-06 15:40:22\n\
    a.png, Warsaw, 2016-02-13 13:33:50\n\
    b.jpg, Warsaw, 2016-01-02 15:12:22\n\
    c.jpg, Warsaw, 2016-01-02 14:34:30\n\
    d.jpg, Warsaw, 2016-01-02 15:15:01\n\
    e.png, Warsaw, 2016-01-02 09:49:09\n\
    f.png, Warsaw, 2016-01-02 10:55:32\n\
    g.jpg, Warsaw, 2016-02-29 22:13:11\n\
"""
## output:
"""
"""
def rename_pictures(input_s: str) -> str:
    import re
    result = list()
    lines = re.split(r"\n", input_s)
    pictures = dict()
    for index, line in enumerate(lines):
        line = line.strip()
        if line == "":
            continue
        picture_info = re.split(r",", line)
        city = picture_info[1].strip()
        extension = picture_info[0].strip().split(".")[1]
        date_taken = picture_info[2].strip()
        # each item in the array of city is a tuple
        if city in pictures:
            pictures[city].append((extension, date_taken, index))
        else:
            pictures[city] = [(extension, date_taken, index)]
    for city in pictures.keys():
        # pictures[city] is an array, each item in the array is a tuple
        # sort the pictures in a city based on taken date
        pictures[city] = sorted(pictures[city], key = lambda n: n[1]) # or: pictures[city].sort(key = lambda n: n[1])
        count = len(pictures[city])
        zeros = len(str(count))
        num = 1
        for picture in pictures[city]:
            # picture is a tuple, -1 is old index, 
            # item is a new tuple, first data is old index, second data is new name
            item = (picture[-1], city + str(num).zfill(zeros) + "." + picture[0])
            result.append(item)
            num += 1
    # sort the result based on the old index, so that output has the same sequence as input
    result = sorted(result, key = lambda n: n[0]) # or: result.sort(key = lambda n: n[0])
    # return a string ended with new line
    return "\n".join(picture[1] for picture in result)

## Given a list of dates in string format, write a Python program to sort the list of dates in ascending order.
## Input : dates = [“24 Jul 2017”, “25 Jul 2017”, “11 Jun 1996”, “01 Jan 2019”, “12 Aug 2005”, “01 Jan 1997”] 
## Output : 
## 01 Jan 2007
## 10 Jul 2016
## 2 Dec 2017
## 11 Jun 2018
## 23 Jun 2018
## 01 Jan 2019
## By default, built-in sorting functions (sorted and list.sort will sort the list of strings in alphabetical order
## In Python, we have the datetime module which makes date based comparison easier. The datetime.strptime() function 
## is used to convert a given string into datetime object. It accepts two arguments: date (string) and format (used to specify the format, returns a datetime object.
## datetime.strptime(date, format)
## %d ---> for Day, %b ---> for Month, %Y ---> for Year       
def sort_date_strings_by_datetime():  
    from datetime import datetime
    dates =  ["23 Jun 2018", "2 Dec 2017", "11 Jun 2018", 
              "01 Jan 2019", "10 Jul 2016", "01 Jan 2007"]  
    # Sort the list in ascending order of dates 
    dates.sort(key = lambda date: datetime.strptime(date, '%d %b %Y')) # or dates = sorted(dates, key = lambda date: datetime.strptime(date, '%d %b %Y'))
    # Print the dates in a sorted order
    for i in range(len(dates)):  
        print(dates[i])
## Given an array of integers arr, your task is to count the number of contiguous subarrays that represent a sawtooth sequence of at least two elements.
## A sawtooth sequence is a sequence of number that alternate between increasing and decreasing. In other words, each element is either strictly greater 
## than it’s neighboring element or strictly less than it’s neighboring elements.

def solution(orig_arr):
    print(f"orig_arr={orig_arr}")
    def is_sawtooth(arr):
        if len(arr) < 3: # handle the case of 2 elements
            if (arr[1] != arr[0]):
                return True
            else:
                return False
        # observe the pattern by checking the first 3 integers, pattern can be: less, greater, less or greater,less,greater
        if (arr[0] < arr[1]) and (arr[1] > arr[2]):
            for i in range(1, len(arr)-1, 2):
                if (arr[i-1] < arr[i]) and (arr[i] > arr[i+1]):
                    continue
                else:
                    return False
        elif (arr[1] < arr[0]) and (arr[1]<arr[2]):
            for i in range(1, len(arr)-1, 2):
                if (arr[i] < arr[i-1]) and (arr[i] < arr[i+1]):
                    continue
                else:
                    return False
        else:
            return False
        return True
    
    result = 0
    for i in range(len(orig_arr)):
        for j in range(i+2, len(orig_arr)+1):
            sub_arr = orig_arr[i:j]
            #print(f"sub_arr={sub_arr}")
            if is_sawtooth(sub_arr):
                result += 1

    return result

'''
You are given an array of non-negative integers numbers. You are allowed to choose any number from this array and swap any two digits in it. 
If after the swap operation the number contains leading zeros, they can be omitted and not considered (eg: 010 will be considered just 10).
Your task is to check whether it is possible to apply the swap operation at most once, so that the elements of the resulting array are strictly increasing.
'''
from collections import Counter
from itertools import permutations
def is_increasing(numbers):
    if isEquals(numbers):
        return True
    else:
        for i in range(len(numbers)-1):
            if numbers[i] > numbers[i+1]:
                nums = swapList(numbers[i])
                if(isSwapEquals(i, nums, numbers)):
                    return True
                else:
                    nums = swapList(numbers[i+1])
                    for j in nums:
                        if j > numbers[i]:
                            return True

    return isEquals(numbers)

def swapList(num):
    return sorted(set([int(''.join(p)) for p in permutations(str(num))]))

def isSwapEquals(i, nums, numbers):
    for j in nums:
        if(i > 0 and numbers[i-1] < j < numbers[i+1]):
            numbers[i] = j
            return isEquals(numbers)
        elif (i == 0 and j < numbers[i+1]):
            numbers[i] = j
            return isEquals(numbers)
    return False

def isEquals(numbers):
        return numbers == sorted(numbers) and numbers == sorted(list(Counter(numbers)))

'''You are given an array of integers a. A new array b is generated by rearranging the elements of a in the following way:
b[0] is equal to a[0];
b[1] is equal to the last element of a;
b[2] is equal to a[1];
b[3] is equal to the second-last element of a;
b[4] is equal to a[2];
b[5] is equal to the third-last element of a;
and so on.
Your task is to determine whether the new array b is sorted in strictly ascending order or not.
Example
For a = [1, 3, 5, 6, 4, 2], the output should be alternatingSort(a) = true.
The new array b will look like [1, 2, 3, 4, 5, 6], which is in strictly ascending order, so the answer is true.
For a = [1, 4, 5, 6, 3], the output should be alternatingSort(a) = false.
The new array b will look like [1, 3, 4, 6, 5], which is not in strictly ascending order, so the answer is false.'''

def alter_array_sort(numbers):
    def is_is_strict_ascending(num_arr):
        for i in range(len(num_arr)-1):
            if num_arr[i] > num_arr[i+1]:
                return False
        return True
    new_numbers = [0] * len(numbers)
    input_ind, cur_ind, last_ind = 0, 0, len(numbers) - 1
    while(i < last_ind):
        new_numbers[cur_ind] = numbers[input_ind]
        cur_ind += 1
        new_numbers[cur_ind] = numbers[last_ind]
        cur_ind += 1
        input_ind += 1
        last_ind -= 1
    if is_is_strict_ascending(new_numbers):
        return True
    else:
        return False

# check if two integers are anagrams 
# a number is said to be an anagram of some other number if it can be made equal to the other number by just shuffling the digits in it.
def is_anagrams(a, b):
    def update_freq(n, freq):
        while n:
            digit = n / 10
            freq[digit] += 1
            n = n // 10
    freq_a, freq_b = [0] * 10, [0] * 10 # instead of using dict, we use list to store the frequency of each digit, the index is the digit
    update_freq(a, freq_a)
    update_freq(b, freq_b)
    for i in range(10):
        if freq_a[i] != freq_b[i]:
            return False

    return True

def is_anagrams_2(a, b):
    a_str, b_str = str(a), str(b)
    if sorted(a_str) == sorted(b_str):
        return True
    else:
        return False

'''Minesweeper is a popular single-player computer game. The goal is to locate mines within a rectangular grid of cells. At the start of the game, all of the cells are concealed. On each turn, the player clicks on a blank cell to reveal its contents, leading to the following result:
If there's a mine on this cell, the player loses and the game is over;
Otherwise, a number appears on the cell, representing how many mines there are within the 8 neighbouring cells (up, down, left, right, and the 4 diagonal directions);
If the revealed number is 0, each of the 8 neighbouring cells are automatically revealed in the same way.
You are given a boolean matrix field representing the distribution of bombs in the rectangular field. You are also given integers x and y, representing the coordinates of the player's first clicked cell - x represents the row index, and y represents the column index, both of which are 0-based.
Your task is to return an integer matrix of the same dimensions as field, representing the resulting field after applying this click. If a cell remains concealed, the corresponding element should have a value of -1.
It is guaranteed that the clicked cell does not contain a mine.
Example
field = [[false, true, true],
         [true, false, true],
         [false, false, true]]
x = 1, and y = 1, the output should be
solution(field, x, y) = [[-1, -1, -1],
                         [-1, 5, -1],
                         [-1, -1, -1]]
There are 5 neighbors of the cell (1, 1) which contain a mine, so the value in (1, 1) should become 5, and the other elements of the resulting matrix should be -1 since no other cell would be expanded.
field = [[true, false, true, true, false],
         [true, false, false, false, false],
         [false, false, false, false, false],
         [true, false, false, false, false]]
x = 3, and y = 2, the output should be
solution(field, x, y) = [[-1, -1, -1, -1, -1],
                         [-1, 3, 2, 2, 1],
                         [-1, 2, 0, 0, 0],
                         [-1, 1, 0, 0, 0]]
Since the value in the cell (3, 2) is 0, all of its neighboring cells ((2, 1), (2, 2), (2, 3), (3, 1), and (3, 3)) are also revealed. Since the value in the cell (2, 2) is also 0, its neighbouring cells (1, 1), (1, 2) and (1, 3) are revealed, and since the value in cell (2, 3) is 0, its neighbours (1, 4), (2, 4), and (3, 4) are also revealed. The cells (3, 3), (2, 4), and (3, 4) also contain the value 0, but since all of their neighbours have already been revealed, no further action is required.'''

'''
Tetris-inspired question
You are given a matrix of integers field of size height × width representing a game field, and also a matrix of integers figure of size 3 × 3 representing a figure. Both matrices contain only 0s and 1s, where 1 means that the cell is occupied, and 0 means that the cell is free.
You choose a position at the top of the game field where you put the figure and then drop it down. The figure falls down until it either reaches the ground (bottom of the field) or lands on an occupied cell, which blocks it from falling further. After the figure has stopped falling, some of the rows in the field may become fully occupied.
Your task is to find the dropping position such that at least one full row is formed. As a dropping position, you should return the column index of the cell in the game field which matches the top left corner of the figure’s 3 × 3 matrix. If there are multiple dropping positions satisfying the condition, feel free to return any of them. If there are no such dropping positions, return -1.
Note: The figure must be dropped so that its entire 3 × 3 matrix fits inside the field, even if part of the matrix is empty. 
This solution starts by defining some dimensions that will be important for the problem. Then, for every valid dropping position—that is, columns in range(width - figure_size + 1)—the code goes row by row, seeing how far the figure will go until it will “stop.” 
To do this, it peeks at the next row and asks: can the figure fit there? If not, it stops at the previous row. The figure doesn’t fit if there is a 1 in the same place in both the figure and the field (offset by the row). The loop can stop when row == height - figure_size + 1, because then the bottom of the figure will have hit the bottom of the field.
Once the code figures out the last row the figure can drop to before it can’t fit anymore, it’s time to see if there’s a fully-filled row created anywhere. From where the figure “stopped,” the code iterates through the field and figure arrays row by row, checking to see if together, they’ve created a filled row. Cleverly, the code sets row_filled = True and then marks it False if one or both of these conditions are met:
Any of the field cells in the row are empty (not 1)
Any of the figure cells in the row are empty (not 1)
'''

def solution(field, figure):
   height = len(field)
   width = len(field[0])
   figure_size = len(figure)
 
   for column in range(width - figure_size + 1):
       row = 1
       while row < height - figure_size + 1:
           can_fit = True
           for dx in range(figure_size):
               for dy in range(figure_size):
                   if field[row + dx][column + dy] == 1 and figure[dx][dy] == 1:
                       can_fit = False
           if not can_fit:
               break
           row += 1
       row -= 1
 
       for dx in range(figure_size):
           row_filled = True
           for column_index in range(width):
            if not (field[row + dx][column_index] == 1 or
                    (column <= column_index < column + figure_size and\
                  figure[dx][column_index - column] == 1)):
                row_filled = False
           if row_filled:
               return column
   return -1
'''
Given an array of integers, calculate the digits that occur the most number of times in the array. Return the array of these digits in ascending order.
[input] array.integer a
An array of positive integers.
Guaranteed constraints:
1 ≤ a.length ≤ 103,
1 ≤ a[i] < 100.
[output] array.integer
The array of most frequently occurring digits, sorted in ascending order.
Example
For a = [25, 2, 3, 57, 38, 41], the output should be solution(a) = [2, 3, 5].
Here are the number of times each digit appears in the array:
0 -> 0
1 -> 1
2 -> 2
3 -> 2
4 -> 1
5 -> 2
6 -> 0
7 -> 1
8 -> 1
9 -> 0
The most number of times any number occurs in the array is 2, and the digits which appear 2 times are 2, 3 and 5. So the answer is [2, 3, 5]'''
def solution(a):
    freq = [0] * 10
    for num in a:
        for c in str(num):
            freq[int(c)] += 1
    max_freq = max(freq) 
    result = list()
    for i, n in enumerate(freq):
        if n == max_freq:
            result.append(i)
    return result


'''
You are given an array of strings arr. Your task is to construct a string from the words in arr, starting with the 0th character from each word (in the order they appear in arr), followed by the 1st character, then the 2nd character, etc. If one of the words doesn't have an ith character, skip that word.

Return the resulting string.


[input] array.string arr

An array of strings containing alphanumeric characters.

Guaranteed constraints:
1 ≤ arr.length ≤ 100,
1 ≤ arr[i].length ≤ 100.

[output] string

Return the resulting string.


Example

    For arr = ["Daisy", "Rose", "Hyacinth", "Poppy"], the output should be solution(arr) = "DRHPaoyoisapsecpyiynth".
        First, we append all 0th characters and obtain string "DRHP";
        Then we append all 1st characters and obtain string "DRHPaoyo";
        Then we append all 2nd characters and obtain string "DRHPaoyoisap";
        Then we append all 3rd characters and obtain string "DRHPaoyoisapsecp";
        Then we append all 4th characters and obtain string "DRHPaoyoisapsecpyiy";
        Finally, only letters in the arr[2] are left, so we append the rest characters and get "DRHPaoyoisapsecpyiynth";
'''
def solution(arr):
    result = ''
    max_len = len(arr[0])
    for i in range(1, len(arr)):
        if max_len < len(arr[i]):
            max_len = len(arr[i])
    i = 0 # i is index of word, j is in index of array
    while i < max_len:
        for j in range(len(arr)):
            if i >= len(arr[j]):
                continue
            else:
                result += arr[j][i]
        i += 1
    return result

'''Given an array of integers a, your task is to find how many of its contiguous subarrays of length m contain a pair of integers with a sum equal to k.

More formally, given the array a, your task is to count the number of indices 0 ≤ i ≤ a.length - m such that a subarray [a[i], a[i + 1], ..., a[i + m - 1]] contains at least one pair (a[s], a[t]), where:

    s ≠ t
    a[s] + a[t] = k

Example

    For a = [2, 4, 7, 5, 3, 5, 8, 5, 1, 7], m = 4, and k = 10, the output should be solution(a, m, k) = 5.

    Let's consider all subarrays of length m = 4 and see which fit the description conditions:
        Subarray a[0..3] = [2, 4, 7, 5] doesn't contain any pair of integers with a sum of k = 10. Note that although the pair (a[3], a[3]) has the sum 5 + 5 = 10, it doesn't fit the requirement s ≠ t.
        Subarray a[1..4] = [4, 7, 5, 3] contains the pair (a[2], a[4]), where a[2] + a[4] = 7 + 3 = 10.
        Subarray a[2..5] = [7, 5, 3, 5] contains two pairs (a[2], a[4]) and (a[3], a[5]), both with a sum of k = 10.
        Subarray a[3..6] = [5, 3, 5, 8] contains the pair (a[3], a[5]), where a[3] + a[5] = 5 + 5 = 10.
        Subarray a[4..7] = [3, 5, 8, 5] contains the pair (a[5], a[7]), where a[5] + a[7] = 5 + 5 = 10.
        Subarray a[5..8] = [5, 8, 5, 1] contains the pair (a[5], a[7]), where a[5] + a[7] = 5 + 5 = 10.
        Subarray a[6..9] = [8, 5, 1, 7] doesn't contain any pair with a sum of k = 10.

    So the answer is 5, because there are 5 contiguous subarrays that contain a pair with a sum of k = 10.

    For a = [15, 8, 8, 2, 6, 4, 1, 7], m = 2, and k = 8, the output should be solution(a, m, k) = 2.

    There are 2 subarrays satisfying the description conditions:
        a[3..4] = [2, 6], where 2 + 6 = 8
        a[6..7] = [1, 7], where 1 + 7 = 8

Input/Output

    [execution time limit] 4 seconds (py3)

    [input] array.integer a

    The given array of integers.

    Guaranteed constraints:
    2 ≤ a.length ≤ 105,
    0 ≤ a[i] ≤ 109.

    [input] integer m

    An integer representing the length of the contiguous subarrays being considered.

    Guaranteed constraints:
    2 ≤ m ≤ a.length.

    [input] integer k
    An non-negative integer value representing the sum of the pairs we're trying to find within each subarray.

Guaranteed constraints:
0 ≤ k ≤ 109.

[output] integeri
    '''
def solution(a, m, k):
    result = 0
    def count_two_sum(arr, target):
        count = 0
        for i in range(len(arr) - 1):
            for j in range(i+1, len(arr)):
                if arr[i] + arr[j] == target:
                    count += 1
        return count
    def is_two_sum(arr, target):
        pairs = dict() # key is the number, and value is the diff between target and number
        for n in arr:
            if n in pairs.values():
                return True
            else:
                pairs[n] = target - n
        return False
        
    for i in range(0, len(a) - m + 1):
        sub_a = a[i: i + m]
        sub_count = count_two_sum(sub_a, k)
        if sub_count > 0:
            result += 1 
    
    return result

'''You are given an array of integers a. A new array b is generated by rearranging the elements of a in the following way:

    b[0] is equal to a[0];
    b[1] is equal to the last element of a;
    b[2] is equal to a[1];
    b[3] is equal to the second-last element of a;
    b[4] is equal to a[2];
    b[5] is equal to the third-last element of a;
    and so on.

Your task is to determine whether the new array b is sorted in strictly ascending order or not.'''
def solution(a):
    size = len(a)
    if size == 1:
        return True
    b = [0] * size
    pointer_1 = 0
    pointer_2 = -1
    for i in range(size):
        if i % 2 == 0:
            b[i] = a[pointer_1]
            pointer_1 += 1
        else:
            b[i] = a[pointer_2]
            pointer_2 -= 1
    for i in range(0, size - 1):
        if b[i] >= b[i+1]:
            return False
    return True

'''Given an array a, your task is to apply the following mutation to it:

    Array a mutates into a new array b of the same length
    For each i from 0 to a.length - 1 inclusive, b[i] = a[i - 1] + a[i] + a[i + 1]
    If some element in the sum a[i - 1] + a[i] + a[i + 1] does not exist, it is considered to be 0
        For example, b[0] equals 0 + a[0] + a[1]
'''
def solution(a):
    size = len(a)
    if size == 1:
        return a
    b = [0] * size
    for i in range(size):
        if i == size - 1:
            b[i] = a[i-1] + a[i] + 0
        elif i == 0:
            b[i] = 0 + a[i] + a[i+1]
        else:
            b[i] = a[i-1] + a[i] + a[i+1]
    return b

'''You are given an array of arrays a. Your task is to group the arrays a[i] by their mean values, so that arrays with equal mean values are in the same group, and arrays with different mean values are in different groups.

Each group should contain a set of indices (i, j, etc), such that the corresponding arrays (a[i], a[j], etc) all have the same mean. Return the set of groups as an array of arrays, where the indices within each group are sorted in ascending order, and the groups are sorted in ascending order of their minimum element.
For

a = [[3, 3, 4, 2],
     [4, 4],
     [4, 0, 3, 3],
     [2, 3],
     [3, 3, 3]]

the output should be

solution(a) = [[0, 4],
                 [1],
                 [2, 3]]
'''
def solution(a):
    import numpy as np
    size = len(a)
    if size == 1:
        return [[0]]
    means = [0] * size
    for index, arr in enumerate(a):
        means[index] = np.mean(arr)
    print(f"means={means}")
    result = dict()
    for i in range(size):
        if str(means[i]) in result.keys():
            continue
        cur = [i]
        for j in range(i+1, size):
            if means[i] == means[j]:
                cur.append(j)
        result[str(means[i])] = cur
    return list(result.values())

'''Given an array of unique integers numbers, your task is to find the number of pairs of indices (i, j) such that i ≤ j and the sum numbers[i] + numbers[j] is equal to some power of 2.
Note: The numbers 1, 2, 4, 8, etc. are considered to be powers of 2.
Example
For numbers = [1, -1, 2, 3], the output should be solution(numbers) = 5.
There is one pair of indices where the sum of the elements is 20 = 1: (1, 2): numbers[1] + numbers[2] = -1 + 2 = 1
There are two pairs of indices where the sum of the elements is 21 = 2: (0, 0) and (1, 3)
There are two pairs of indices where the sum of the elements is 22 = 4: (0, 3) and (2, 2)
In total, there are 1 + 2 + 2 = 5 pairs summing to powers of 2.

Guaranteed constraints:
1 ≤ numbers.length ≤ 10**5
10**(-6) ≤ numbers[i] ≤ 10**6'''
from collections import defaultdict
def solution(numbers):
    counts = defaultdict(int)
    answer = 0
    for element in numbers:
         counts[element] += 1
         for two_power in range(21): # the loop checks for sums of all of the powers of two that are less than 2**21. Because numbers[i] ≤ 10**6, there is no way that two elements could sum to 221
            # To calculate the powers of two, the code uses a left shift bitwise operator: 1 << two_power, in Python is the same as 2two_power.
            second_element = (1 << two_power) - element
            answer += counts[second_element]
    return answer

def solution(a):
   n = len(a)
   b = [0 for _ in range(n)]
   for i in range(n):
       b[i] = a[i]
       if i > 0:
           b[i] += a[i - 1]
       if i < n - 1:
           b[i] += a[i + 1]
   return b 

'''
The International Science Olympiads are held once a year, every summer. From each participating country, 4 to 6 high-school students are sent to compete. In order to choose these students, each country organizes an internal olympiad.
Given the results of the last stage of an internal olympiad, your task is to select the students to send to the international olympiads. For this challenge, let's assume that we need to select up to 4 students for each subject.
The results are located in the /root/devops/ directory, where each school has its own directory. If a school has candidates for two subjects, there can be two files in that school's directory. For example, /root/devops/physmath_school/maths.txt and /root/devops/physmath_school/physics.txt.
All files have .txt extension and in each file there are one or more lines each representing data about one student in the format Name Surname Score. Score is an integer number. All scores are different.

input:
/root/devops/school1/maths.txt
/root/devops/school1/chemistry.txt
/root/devops/school2/maths.txt
/root/devops/school3/maths.txt
/root/devops/school3/physics.txt
/root/devops/school3/chemistry.txt

output:
chemistry:
Ralph Jordan
Anthony Lee
Diana Wood
Charles Clark

maths:
James Davis
Michael Phillips
Daniel Smith
Kenneth Wilson

physics:
Louis Gonzalez
Harry Nelson
'''

# import requests
# import mysql.connector
import pandas as pd
import os
def solution_1(parent_dir='/root/devops') -> str:
    output_s = ""
    output = dict()
    # construct dictionary
    for cur_dir, sub_dirs, cur_filenames in os.walk(parent_dir):
        for sub_dir in sub_dirs:
            directory = os.path.join(parent_dir, sub_dir)
            for _, _, filenames in os.walk(directory):
                for filename in filenames:
                    if filename not in output.keys():
                        output[filename] = list()
                    file = os.path.join(directory, filename)
                    df = pd.read_csv(file, sep=" ", header=None)
                    for index, row in df.iterrows():
                        firstname, lastname, score = row[0], row[1], row[2]
                        output[filename].append((firstname, lastname, score))
    # sort and parse the dict, put it in output_s
    # sorting by key first (filename). If file is same, then sorting by score
    # output_l is a list of tuple (has 4 items), each item in this list is a tuple. First item of the tuple is subject, second item of this tuple a list which includes all students
    # sample: [('math', [('amy', 'dic', 90), ('dsdf', 'sdfs', 98)]), ('physics', [('sd', 'sdfs', 78), ('ks', 'jjd', 98)])] 
    output_l = sorted(output.items(), key=lambda item: (item[0], item[1][2]))
    # constract the output string
    for item in output_l:
        output_s += item[0] + ":\n"
        for i in range(-1, -5,-1):
            output_s += item[1][i][0] + " " + item[1][i][1] + "\n"
    return output_s

def solution_2(parent_dir='/root/devops') -> str:
    output_s = ""
    output = dict() # key is subject name, value is a list of tuple, each tuple is student name and score
    # construct dictionary
    for cur_dir, sub_dirs, cur_filenames in os.walk(parent_dir):
        for sub_dir in sub_dirs:
            directory = os.path.join(parent_dir, sub_dir)
            for _, _, filenames in os.walk(directory):
                for filename in filenames:
                    if filename not in output.keys():
                        output[filename] = list()
                    file = os.path.join(directory, filename)
                    df = pd.read_csv(file, sep=" ", header=None)
                    for index, row in df.iterrows():
                        firstname, lastname, score = row[0], row[1], row[2]
                        output[filename].append((firstname, lastname, score))
    for sub in sorted(output.keys()):
        output_s += sub + "\n"
        sorted_l = sorted(output[sub], key = lambda item: item[2], reverse=True)
        if len(sorted_l) >=4:
            for i in range(4):
                output_s += sorted_l[i][0] + " " + sorted_l[i][1] + "\n" 
        else:
            for item in sorted_l:
                output_s += item[0] + " " + item[1] + "\n"
    return output_s
