#!/usr/bin/env python
# -*- coding: UTF-8 -*

from distutils import extension
import re, os, sys, string, datetime, getopt, random, operator
from collections import namedtuple

# Here’s a function with four required keyword-only arguments
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # or string.ascii_uppercase
LOWERCASE = UPPERCASE.lower() # or: string.ascii_lowercase
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
from typing import Tuple
def longest_valid_parentheses(s: str) -> Tuple[int, str]:
    stack = []
    max_length = 0
    base_index = -1 # Initialize a base index before the string
    start_index = 0  # To track the start of the longest valid substring

    for i, char in enumerate(s):
        if char == '(':
            # Push the index of '(' onto the stack
            stack.append(i)
        else:
            # Pop the stack for ')'
            if stack:
                stack.pop()
                # Calculate the length of the valid substring
                if stack:
                    current_length = i - stack[-1]
                else:
                    current_length = i - base_index
                # Update max length and start index if needed
                if current_length > max_length:
                    max_length = current_length
                    start_index = i - max_length + 1
            else:
                # Reset the base index if no matching '('
                base_index = i
        # if input string not only has "(" and ")", we need add these codes to ignore all other characters
        # else:
        #     # Reset the base index for invalid characters
        #     base_index = i
        #     stack.clear()

    # Extract the longest valid substring using start_index and max_length
    longest_substring = s[start_index:start_index + max_length]
    return max_length, longest_substring

# Example usage
input_string = "(()))())("
length, substring = longest_valid_parentheses(input_string)
print("Length of longest valid parentheses substring:", length)
print("Longest valid parentheses substring:", substring)

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
  pre_c = ""
  for i in range(len(input)):
    if pre_c == input[i]: # this is a repeated character
      continue
    if input[i] in sorted_s:
      pre_c = input[i]
      index = sorted_s.index(input[i])
      sorted_s = sorted_s[index+1:]
    else:
      return input[i] # this is the first non alphabetic order character in the string
  for c in s:
    if c == pre_c:
        continue
    if c in sorted_s:
      pre_c = c
      index = sorted_s.index(c)
      sorted_s = sorted_s[index+1:]
    else:
      return c
  return "" # means all characters in the string are in alphabetic order

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
  print("input string is %s, we will count %s" % (s, str(count)))
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
  for c in s:
    if (c in r_dict.keys()) and (r_dict[c]): # or: r_dict.get(c, False)
      s = s.replace(c, "")
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
    return t_matrix = zip(*matrix)

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
            num1 = list(diff_dict.keys())[list(diff_dict.values()).index(num)]
            result.append((nums.index(num1), index))
        else:
            diff_dict[num] = target - num
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
    return [(-1,-1,-1)]
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
def find_combinations(target, current_combination=None, start=1):
    if current_combination is None:
        current_combination = []

    # Base case: if target is 0, we found a valid combination
    if target == 0:
        print(current_combination)
        return

    # Explore all numbers from 'start' to 'target'
    for num in range(start, target + 1):
        # Include the current number in the combination
        find_combinations(target - num, current_combination + [num], num)

find_combinations(5)

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
    greater = x if x > y else y
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
    result = ""
    for idy, item in enumerate(s):
        for idx, item in enumerate(s):
            derp = s[idy:idx+1]
            if isPalindrome(derp) and (len(derp) > len(result)):
                result = derp
    return result

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
        curname = s.strip("\t") # or s.lstrip("\t")
        tab_count = len(s) - len(curname) # number of tab
        if '.' in curname:
            total_path_len = 0
            for d in range(tab_count):
                total_path_len += pathlen[d]
            maxlen = max(maxlen, total_path_len + len(curname))
        else:
            pathlen[tab_count] = len(curname) + 1 # +1 for "/"
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
def singleNumber(nums: list[int]):
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
        diff = [nums[i] - nums[i - 1] for i in range(1, len(nums))]
        count = 1
        pre = diff[0]
        for i in range(1, len(diff)):
            if diff[i] == pre:
                count += 1
            else:
                ans += count * (count - 1) / 2
                count = 1
            pre = diff[i]
        ans += count * (count - 1) / 2
    return ans

# function to find an item in an ordered list nums (binary search): update index
def binary_search_1(nums,n):
    found = False
    position = None
    low = 0
    high = len(nums) - 1
    while (not found) and (high >= low):
        mid = (high + low) // 2
        if n == nums[mid]:
            found = True
            position = mid
        elif n < nums[mid]:
            high = mid - 1
        else:
            low = mid + 1
    if found:
        return found, position
    else:
        return False, low if nums[low] == n else None

# method 2: update list
def binary_search_2(nums, n):
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
    input = sorted(intervals, key = lambda item: item[1])
    pre_item_end =  -sys.maxsize - 1 # sys.maxsize is the max int in Python 3
    for interval in input:
        if pre_item_end <= interval[0]: # if the current interval does not overlap with the previous one
            pre_item_end = interval[1]
        else:
            count += 1
            removed_items.append(interval)
    return count, removed_items

# input is a string s, check if it's a valid number
def inNumber(s):
    #define DFA state transition tables
    states = [
              # State (0)
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

# l1 and l2 are linked list, each node is an integer
# add them together and return the new list. New node's value is the reminder: (l1.node + l2.node + carry) % 10
class Node:
  def __init__(self, data=None, next=None):
    self.data = data
    self.next  = next

def addLinkedListInteger(l1, l2):
  head = None
  cur = None
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
      cur.next = newNode
    cur = newNode
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
        currentNode = currentNode.next
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
        max_size = max(len(l1[i]), len(l2[i]))
        first = l1[i].ljust(max_size, '0')  # pad with 0 to the right
        second = l2[i].ljust(max_size, '0')  # pad with 0 to the right
        first, second = int(first), int(second)
        if first == second:
            continue
        elif first < second:
            return -1
        else:
            return 1
    if len(l1) < len(l2):
        return -1
    else:
        return 1

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

# Given a string, sort it in decreasing order based on the frequency of characters.
# if the frequency is same, any order of characters is good
def frequency_sort_1(s):
    if s is None:
        raise ValueError
    if len(s) == 0:
        return ""
    frequency = {}.fromkeys(set(s), 0)
    for c in s:
        frequency[c] += 1

    sorted_tuple = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    return "".join([item[0] * item[1] for item in sorted_tuple])

def frequency_sort_2(s):
    from collections import Counter
    frequency, c2 = Counter(s), {}
    for k, v in frequency.items():
        c2.setdefault(v, []).append(k * v)  # c2'key is the frequency of each character

    return "".join(["".join(c2[i]) for i in range(len(s), -1, -1) if i in c2])

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
        # find this current root node's index in inorder array, so that we can recursive
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

def bst_count(root, low, high):
    # base case
    if root is None:
        return 0
    # special case
    if root.value == low and root.value == high:
        return 1

    if low <= root.value <= high: # low <= root.value and root.value <= high:
        return 1 + bst_count(root.left, low, high) + bst_count(root.right, low, high)
    elif root.value < low:
        return bst_count(root.right, low, high)
    else: # root.value > high
        return bst_count(root.left, low, high)

def bst_insert(root, key):
    if root is None:
        return Node(key)
    else:
        if root.val < key:
            root.right = bst_insert(root.right, key)
        else:
            root.left = bst_insert(root.left, key)
    return root

def bst_search(root, key):
    if root is None or root.val == key:
        return root
    if root.val < key:
        return bst_search(root.right, key)
    return bst_search(root.left, key)

# Given a root node reference of a BST and a key (target value), delete the node with the given key in the BST.
# Return the root node reference (possibly updated) of the BST
def bst_delete(root, key):
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
        root.left = bst_delete(root.left,  key)
    elif key > root.value:
        root.right = bst_delete(root.right, key)
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
            root.right = bst_delete(root.right, minNodeInRightSubTree.value)

    return root

# bst is balanced if the difference of left size and right size is at most 1
def is_bst_balanced(root) -> bool:
    # returns (height, is_balanced)
    def check_balance(node) -> tuple[int, bool]:
        if node is None:
            return 0, True

        left_height, left_balanced = check_balance(node.left)
        if not left_balanced:
            return -1, False

        right_height, right_balanced = check_balance(node.right)
        if not right_balanced:
            return -1, False

        height = max(left_height, right_height) + 1
        is_balanced = abs(left_height - right_height) <= 1
        return height, is_balanced

    height, balanced = check_balance(root)
    return balanced

# find kth smallest node in the BST
def kth_smallest_in_bst(root, k):
    def countNodes(root):
        if not root:
            return 0
        return 1 + countNodes(root.left) + countNodes(root.right)

    # The number of nodes in the left subtree of the root
    left_nodes = countNodes(root.left) if root else 0

    # If k is equal to the number of nodes in the left subtree plus 1,
    # That means we must return the root's value as we've reached the k-th smallest
    if k == left_nodes + 1:
        return root.val
    # If there are more than k nodes in the left subtree,
    # The k-th smallest must be in the left subtree.
    elif k <= left_nodes:
        return kth_smallest_in_bst(root.left, k)
    # If there are less than k nodes in the left subtree,
    # The k-th smallest must be in the right subtree.
    else:
        return kth_smallest_in_bst(root.right, k - 1 - left_nodes)

def kth_largest_in_bst(root, k):
    """
    Find the k-th largest element in a BST.
    """
    def reverse_inorder(node):
        nonlocal count, result
        if not node or count >= k:
            return

        # Traverse the right subtree first (larger elements)
        reverse_inorder(node.right)

        # Increment the count and check if we've reached the k-th largest
        count += 1
        if count == k:
            result = node.value
            return

        # Traverse the left subtree
        reverse_inorder(node.left)

    count = 0
    result = None
    reverse_inorder(root)
    return result

def get_height(node):
    """Recursively calculates the height of a subtree."""
    if not node:
        return 0
    left_height = get_height(node.left)
    right_height = get_height(node.right)
    return max(left_height, right_height) + 1

def get_bst_subtree_heights(root):
    """Gets the height of the left and right subtrees of the BST root."""
    if not root:
        return (0, 0)
    left_tree_height = get_height(root.left)
    right_tree_height = get_height(root.right)
    return (left_tree_height, right_tree_height)

# identical tree, not only node data, but also structure
# time complexicity: worst case Linear, O(n), balanced tree O(logn). space complexcity: O(height)
def isTreeIdentical(root1, root2):
    if root1 is None and root2 is None:
        return True
    if root1 is not None and root2 is not None:
        return (root1.data == root2.data and isTreeIdentical(root1.left, root2.left) and isTreeIdentical(root1.right, root2.right))
    return False

# l is the first node of the linked list, this function can remove multiple matched key
def removeNodeFromLinkedList(l, key):
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
            print("help message")

    if argCounter != requiredArgs and argCounter != 0:
        print('help message')

# LRU
def leastRecentlyUsedDesignNew():
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = set()
            self.cache_vals = deque() #LinkedList() # we can use deque as linkedlist, from collections import deque

        def get(self, value):
            if value not in self.cache:
                return None
            else:
                i = 0
                node = self.cache_vals[i]
                while node is not None:
                    if node.data == value:
                        return node
                    i += 1
                    self.cache_vals[i]

        def set(self, value):
            node = self.get(value)
            if node == None:
                if(len(self.cache_vals) >= self.capacity):
                    #self.cache_vals.insert_at_tail(value)
                    self.cache_vals.append(value)
                    self.cache.add(value)
                    self.cache.remove(self.cache_vals[0].data)
                    #self.cache_vals.remove_head()
                else:
                    # self.cache_vals.insert_at_tail(value)
                    self.cache_vals.append(value)
                    self.cache.add(value)
            else:
                self.cache_vals.remove(value)
                # self.cache_vals.insert_at_tail(value)
                self.cache_vals.append(value)

        def printcache(self):
            # node = self.cache_vals.get_head()
            node = self.cache_vals[0]
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
        if cache_node is None:
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
    if len(ls) == 4:
        for g in ls:
            if g.isdigit() and str(int(g)) == g and 0<= int(g) <= 255:
                continue
            else:
                return False
        return True
    else:
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
    resp = requests.get(base_url)
    soup = BeautifulSoup(resp.text, features="html.parser")
    with open("less.txt", "w") as f:
        for paragraph in soup.findall(dir='ltr'):
            f.write(paragraph.text.replace("<span>", ""))
# calculate the angle between hour and minute
def calculateAngle(h,m):
  # validate the input
  if (h < 0 or m < 0 or h > 24 or m > 60):
    print('Wrong input')
  if (m == 60):
    h += 1
  m = m % 60
  h = h % 12
  # Calculate the angles moved by hour and minute hands with reference to 12:00
  hour_angle = 0.5 * (h * 60 + m) # 30h + 0.5m (30 degree each hour, plug 0.5 degree for each minute between hours)
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
# de = collections.deque(); de.pop(): delete and return the item from the right end of the queue; de.popleft(): delete and retrun the item from the left end of the queue
# de.append(): ; de.appendleft(): ; de.index(ele, beg, end): ; de.insert(): ; de.remove(value): remote the first occurance of the value mentioned in the argument; de.count(value)
from collections import deque
llist = deque("abcde")
# deque(['a', 'b', 'c', 'd', 'e'])

llist.append("f")
# deque(['a', 'b', 'c', 'd', 'e', 'f'])

llist.pop()
llist.popleft()

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
    from collections import defaultdict
    result = list()
    lines = re.split(r"\n", input_s)
    pictures = dict() # or pictures = defaultdict(list)
    for index, line in enumerate(lines):
        line = line.strip()
        if line == "":
            continue
        picture_info = re.split(r",", line)
        city = picture_info[1].strip()
        extension = picture_info[0].strip().split(".")[1]
        date_taken = picture_info[2].strip()
        # each item in the array of city is a tuple
        # or if you use defaultdict: pictures[city].append((extension, date_taken, index))
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

def sawtooth(orig_arr):
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
    def is_strict_ascending(num_arr):
        for i in range(len(num_arr)-1):
            if num_arr[i] > num_arr[i+1]:
                return False
        return True
    new_numbers = [0] * len(numbers)
    input_idx, cur_idx, last_idx = 0, 0, len(numbers) - 1
    while(input_idx < last_idx):
        new_numbers[cur_idx] = numbers[input_idx]
        cur_idx += 1
        new_numbers[cur_idx] = numbers[last_idx]
        cur_idx += 1
        input_idx += 1
        last_idx -= 1
    if is_strict_ascending(new_numbers):
        return True
    else:
        return False

# check if two integers are anagrams
# a number is said to be an anagram of some other number if it can be made equal to the other number by just shuffling the digits in it.
def is_number_anagrams(a, b):
    def update_freq(n, freq):
        while n:
            digit = n % 10
            freq[digit] += 1
            n = n // 10
    freq_a, freq_b = [0] * 10, [0] * 10 # instead of using dict, we use list to store the frequency of each digit, the index is the digit
    update_freq(a, freq_a)
    update_freq(b, freq_b)
    for i in range(10):
        if freq_a[i] != freq_b[i]:
            return False
    return True

def is_number_anagrams_2(a, b):
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
You are given a matrix of integers field of size height x width representing a game field, and also a matrix of integers figure of size 3 x 3 representing a figure. Both matrices contain only 0s and 1s, where 1 means that the cell is occupied, and 0 means that the cell is free.
You choose a position at the top of the game field where you put the figure and then drop it down. The figure falls down until it either reaches the ground (bottom of the field) or lands on an occupied cell, which blocks it from falling further. After the figure has stopped falling, some of the rows in the field may become fully occupied.
Your task is to find the dropping position such that at least one full row is formed. As a dropping position, you should return the column index of the cell in the game field which matches the top left corner of the figure’s 3 × 3 matrix. If there are multiple dropping positions satisfying the condition, feel free to return any of them. If there are no such dropping positions, return -1.
Note: The figure must be dropped so that its entire 3 x 3 matrix fits inside the field, even if part of the matrix is empty.
This solution starts by defining some dimensions that will be important for the problem. Then, for every valid dropping position—that is, columns in range(width - figure_size + 1)—the code goes row by row, seeing how far the figure will go until it will “stop.”
To do this, it peeks at the next row and asks: can the figure fit there? If not, it stops at the previous row. The figure doesnt fit if there is a 1 in the same place in both the figure and the field (offset by the row). The loop can stop when row == height - figure_size + 1, because then the bottom of the figure will have hit the bottom of the field.
Once the code figures out the last row the figure can drop to before it can't fit anymore, it's time to see if there's a fully-filled row created anywhere. From where the figure “stopped,” the code iterates through the field and figure arrays row by row, checking to see if together, they’ve created a filled row. Cleverly, the code sets row_filled = True and then marks it False if one or both of these conditions are met:
Any of the field cells in the row are empty (not 1)
Any of the figure cells in the row are empty (not 1)
'''

def tetris(field, figure):
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
                if not (field[row + dx][column_index] == 1 or (column <= column_index < column + figure_size and figure[dx][column_index - column] == 1)):
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
def find_digit_frequency_from_nums(a):
    freq = [0] * 10 #instead of using dict, we use list, the index is the number, value is frequency
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
You are given an array of strings arr. Your task is to construct a string from the words in arr, starting with the 0th character
from each word (in the order they appear in arr), followed by the 1st character, then the 2nd character, etc. If one of the words doesn't
have an ith character, skip that word.
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
def contruct_string_from_each_word_in_array(arr):
    result = ''
    max_len = len(arr[0])
    for i in range(1, len(arr)):
        if max_len < len(arr[i]):
            max_len = len(arr[i])
    # max length word can also be gotten by this:
    max_len_word = max(arr, key=len)
    max_len = len(max_len_word)
    # or max_len = max(len(word) for word in arr)
    i = 0 # i is index of word, j is in index of array
    while i < max_len:
        for j in range(len(arr)):
            if i >= len(arr[j]):
                continue
            else:
                result += arr[j][i]
        i += 1
    # another way
    while i < max_len:
        for word in arr:
            if i >= len(word):
                continue
            else:
                result += word[i]
        i += 1
    return result

'''Given an array of integers a, your task is to find how many of its contiguous subarrays of length m contain a pair of integers with a sum equal to k.
More formally, given the array a, your task is to count the number of indices 0 ≤ i ≤ a.length - m such that a subarray [a[i], a[i + 1], ..., a[i + m - 1]]
contains at least one pair (a[s], a[t]), where:
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
[output] integer'''
def substring_sum(a, m, k):
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
def construct_new_array(a):
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
def mutation(a):
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

'''You are given an array of arrays a. Your task is to group the arrays a[i] by their mean values, so that arrays with equal
mean values are in the same group, and arrays with different mean values are in different groups.

Each group should contain a set of indices (i, j, etc), such that the corresponding arrays (a[i], a[j], etc) all have the same mean.
Return the set of groups as an array of arrays, where the indices within each group are sorted in ascending order, and the groups are
sorted in ascending order of their minimum element.

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
def mean_values(a):
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

'''Given an array of unique integers numbers, your task is to find the number of pairs of indices (i, j) such that i ≤ j and
the sum numbers[i] + numbers[j] is equal to some power of 2.
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
def power_of_2(numbers):
    counts = defaultdict(int)
    answer = 0
    for element in numbers:
         counts[element] += 1
         for two_power in range(21): # the loop checks for sums of all of the powers of two that are less than 2**21. Because numbers[i] ≤ 10**6, there is no way that two elements could sum to 221
            # To calculate the powers of two, the code uses a left shift bitwise operator: 1 << two_power, in Python is the same as two_power.
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
                    df = pd.read_csv(file, sep=" ", header=None) # none means there is no header in the csv file. If file has header, then header=1 by default
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

def ml():
    import sqlalchemy
    # Create an engine to connect to the database
    engine = sqlalchemy.create_engine('dialect+driver://username:password@host:port/database')
    # Read the data from the SQL database
    df = pd.read_sql('SELECT * FROM your_table', engine)
    print(df)

# A simple text encryption exercise using the Caesar Cipher technique.
# The Caesar Cipher for `shift = 3` cyclically shifts every letter of the word by 3 positions:
# a -> d, b -> e, c -> f, ..., x -> a, y -> b, z -> c

# Implement the encryption logic by shifting each alphabet character
def encrypt_text(text):
    shift = 3
    encrypted = ""
    for char in text:
        if char.isalpha():  # check if the character is an alphabet
            # TODO: Use the correct ASCII values to shift the character and add it to 'encrypted'
            # Hint 1: ord('A') = 65, ord('a') = 97
            # Hint 2: you can use modulo (%) operator to create a cycle
            if char.islower():
                start_code = ord('a')
            else:
                start_code = ord('A')
            encrypted += chr((ord(char) - start_code + shift) % 26 + start_code)
        else:
            encrypted += char  # keep non-alphabet characters unchanged
    return encrypted

def swapPairs(head: Optional[Node]) -> Optional[Node]:
    if not head or not head.next:
        return head

    dummy = Node(0)
    dummy.next = head
    prev, current = dummy, head

    while current and current.next:
        # Store the next two nodes
        first = current
        second = current.next

        # Swap the nodes
        prev.next = second
        first.next = second.next
        second.next = first

        # Move pointers forward
        prev = first
        current = first.next

    return dummy.next

# dividing intervals into the minimum number of groups, such that no intervals in a group overlap
def minGroups(intervals: list[list[int]]) -> int:
    import heapq
    # heapq.heapify(li) to create heapq from list, headq is used for priority queue
    if not intervals:
        return 0

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    # Initialize a min-heap to keep track of end times
    min_heap = []

    # Loop through each interval
    for interval in intervals:
        start, end = interval

        # If the current interval can be added to an existing group (i.e., it doesn't overlap),
        # we remove the group with the earliest end time from the heap.
        # headpop alwasy pop the min number in the queue. When push, it will put the number in the begning (0) if the number is less than others
        if min_heap and min_heap[0] < start:
            heapq.heappop(min_heap)  # The group is free, reuse it.

        # Push the end time of the current interval onto the heap (create or extend a group)
        heapq.heappush(min_heap, end)

    # The size of the heap at the end represents the number of groups needed
    return len(min_heap)

# duplicate each digit in n, input: 1234, output: 11223344. Assume: 0 <= n <= 10**4 so to guarante no integer overflow
def duplicate_each_digit(n):
    result = 0
    power = 0
    while n > 0:
        digit = n % 10
        result = result + (digit + digit * 10) * 10**(power)
        power +=2
        n //= 10
    return result

# The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
#
# P   A   H   N
# A P L S I I G
# Y   I   R
# And then read line by line: "PAHNAPLSIIGYIR"
def convert(s: str, numRows: int) -> str:
    # Special case: if numRows is 1, the zigzag is just the original string
    if numRows == 1 or numRows >= len(s):
        return s

    # Create an array to store the characters for each row
    rows = [''] * numRows
    cur_row = 0  # Start at the first row
    going_down = False  # This will track whether we are going "down" the rows or "up"

    # Traverse each character in the string
    for c in s:
        rows[cur_row] += c  # Append the character to the current row

        # If we are at the top or bottom row, we change direction
        if cur_row == 0 or cur_row == numRows - 1:
            going_down = not going_down

        # Move up or down the rows
        cur_row += 1 if going_down else -1

    # Join all the rows to get the final zigzagged string
    return ''.join(rows)

# Example usage:
s = "PAYPALISHIRING"
numRows = 3
result = convert(s, numRows)
print(result)  # Output: PAHNAPLSIIGYIR

# replace each character in a word with the corresponding character opposite in the English alphabet
# A->Z, B->Y.....
def replace_char_in_word(input_str):
    # find a character that is the corresponding opposite character
    def opposite_char(c):
        if 'a' <= c <= 'z': # or use if c.islower()
            return chr(ord('a') + (ord('z') - ord(c)))
        elif 'A' <= c <= 'Z': # or use if c.isupper()
            return chr(ord('A') + (ord('Z') - ord(c)))
        else:
            return c
    words = input_str.split(" ")
    new_words = []
    for w in words:
        new_w = ''.join(opposite_char(c)for c in w)
        new_words.append(new_w)
    # retur the new words, taking the last word first and appending the remaining words in their original order, each separated by spaces.
    if len(new_words) == 1:
        return new_words[0]
    else:
        return new_words[-1] + ' ' + ' '.join(new_words[0:-1])

def robot_position(commands: str) -> str:
    # Initialize the starting position at 0
    position = 0
    # Iterate through the commands
    for command in commands:
        if command.lower() == 'L':
            position -= 1  # Move one step to the left
        elif command.lower() == 'R':
            position += 1  # Move one step to the right
        else:
            continue
    # Determine the final position
    if position < 0:
        return 'L'  # The robot is to the left of the starting point
    elif position > 0:
        return 'R'  # The robot is to the right of the starting point
    else:
        return ''  # The robot is at the starting point

# given a string message and an integer n, replace every nth consonant with the next consonant from the alphabet while keeping the case consistent.
def replace_nth_consonant(message: str, n: int) -> str:
    # Helper function to check if a character is a consonant
    def is_consonant(char):
        return char.isalpha() and char.lower() not in 'aeiou'

    # Helper function to get the next consonant, keeping the case consistent
    def next_consonant(char):
        if not is_consonant(char):
            return char  # If it's not a consonant, return it as is

        # Define the list of consonants
        consonants_lower = "bcdfghjklmnpqrstvwxyz"
        consonants_upper = consonants_lower.upper()

        # Find the next consonant in the sequence
        if char.islower():
            index = consonants_lower.index(char)
            return consonants_lower[(index + 1) % len(consonants_lower)]
        else:
            index = consonants_upper.index(char)
            return consonants_upper[(index + 1) % len(consonants_upper)]

    consonant_count = 0
    result = []
    # Loop through each character in the message
    for char in message:
        if is_consonant(char):
            consonant_count += 1  # Increment the consonant count
            if consonant_count % n == 0:
                # Replace the nth consonant with the next consonant
                result.append(next_consonant(char))
            else:
                result.append(char)
        else:
            # If not a consonant, append the character as is
            result.append(char)

    # Join the result list back into a string
    return ''.join(result)

# there is a group chat with many users writing messages. the content of message includes text and mentions of other users in the chat.
# Mentions in the group chat are formatted as strings starting with @ character and followed by at least one id separated by commas.
# An id is formatted as a string starging with id and followed by a positive integer from 1 to 999. Now you are given two arrays of strings titled members and messages.
# Your task is to calculate the mention statics for the group chat. in other words, count the number of messages that each chat member is mentioned i.
# Chat members mentioned in multiple times in a message should be counted only once per message. return the mention statics in an array of strings,
# where each string follows this format: [user id] = [mentions count]. the array should be sorted by mention count in descending order , or in case of a tie,
# lexicographically by user id in ascending order
def mention_statistics(members, messages):
    import re
    from collections import defaultdict
    # Initialize a dictionary to hold mention counts
    mention_count = defaultdict(int)
    # Regular expression to capture valid mentions in the format @idX where X is a number
    # mention_pattern = re.compile(r'@id([1-9][0-9]{0,2})(?:,id[1-9][0-9]{0,2})*')
    # mention_pattern = re.compile(r'@id([1-9][0-9]{0,2})[\s,]id([1-9][0-9]{0,2})*') # group 1: id number 1, group 2: id number 2
    mention_pattern = re.complile(r'@id([1-9][0-9]{0,2})(?:,)id([1-9][0-9]{0,2})*') # group 1: id number 1, group 2: id number 2
    # Loop through each message
    for message in messages:
        # Set to track unique mentions in the current message
        mentioned_in_message = set()

        # Find all mention matches in the message
        mentions = mention_pattern.findall(message) # mentions is an array, which includes matched groups, in this case, they are group 1 and group 2
        for mention in mentions:
            # Add the user ID to the set of unique mentions in the message
            mentioned_in_message.add(mention)

        # Increment the mention count for each unique user mentioned in the message
        for user_id in mentioned_in_message:
            mention_count[f'id{user_id}'] += 1
    # Prepare the result list with the required format
    result = [f"{user_id} = {mention_count[user_id]}" for user_id in members]
    # Sort the result first by descending mention count, then lexicographically by user id
    result.sort(key=lambda x: (-int(x.split(' = ')[1]), x.split(' = ')[0]))
    return result

# Given an array of integer latencies where each element represents recorded latencies in millisecond,
# a positive integer threshold. Your task is to determine the max length of a continuous subarray such that
# the difference between the max and min latencies within this subarray dos not exceed thread.
# return the length of the longest such contiguous subarray
# input: latencies = [7, 3, 5, 8, 6, 4, 9, 3], threshold = 4
# output: [5, 8, 6, 4], and its length is 4.
def longest_subarray_threshold_1(latencies, threshold):
    from collections import deque
    if not latencies:
        return 0
    max_deque = deque()  # To keep track of the index of max elements in the window
    min_deque = deque()  # To keep track of the index of min elements in the window
    left = 0  # Left pointer for the sliding window
    max_length = 0
    subs = []
    for right in range(len(latencies)):
        # Maintain decreasing order in max_deque
        while max_deque and latencies[max_deque[-1]] <= latencies[right]:
            max_deque.pop()
        max_deque.append(right)
        # Maintain increasing order in min_deque
        while min_deque and latencies[min_deque[-1]] >= latencies[right]:
            min_deque.pop()
        min_deque.append(right)
        # Ensure the current window meets the threshold condition
        while latencies[max_deque[0]] - latencies[min_deque[0]] > threshold:
            left += 1  # Shrink the window from the left
            # Remove elements from the deques that are out of the window
            if max_deque[0] < left:
                max_deque.popleft()
            if min_deque[0] < left:
                min_deque.popleft()
        # Calculate the length of the current valid subarray
        max_length = max(max_length, right - left + 1)
        subs.append((max_length, left, right))
    return max_length, subs

def longest_subarray_threshold_2(latencies, threshold):
    if not latencies:
        return 0
    max_length = 0
    subs = []
    for x in range(len(latencies) - 1):
        for y in range(x+1, len(latencies)+1):
            cur_sub = latencies[x:y]
            cur_len = len(cur_sub)
            cur_min = min(cur_sub)
            cur_max = max(cur_sub)
            if (cur_max - cur_min) > threshold:
                break
            else:
                if cur_len > max_length:
                    subs.append(cur_sub)
                max_length = max(max_length, cur_len)
    return max_length, subs

# You are provided with a string of alphanumeric characters in which each number, regardless of the number of digits,
# is always followed by at least one alphabetic character before the next number appears. The task requires you to return a
# transformed version of the string wherein the first alphabetic character following each number is moved to a new position
# within the string and characters in between are removed.
# input:  "I have 2 apples and 5! oranges and 3 grapefruits."
# output: "I have a2pples and o5ranges and g3rapefruits."
# Specifically, for each number in the original string, identify the next letter that follows it, and then reposition that character to directly precede the number.
#  All spaces and punctuation marks between the number and the letter are removed.
def replace_digits_in_string(input_string):
    import re
    pattern = re.compile(r'\d+')
    nums = pattern.findall(input_string)
    for num in nums:
        num_index = input_string.index(num)
        new_index = num_index + 1
        while input_string[new_index] == " " or input_string[new_index] == "!":
             new_index += 1
        input_string = input_string[0:num_index] + input_string[new_index] + num + input_string[new_index+1:]

    return input_string
# input: timePoints = ['10:00:00', '23:30:00'] and added_seconds = 3600, the output should be ['11:00:00', '00:30:00'].
def add_seconds_to_times(timePoints, seconds):
    result = []
    for time in timePoints:
        h, m, s = int(time.split(":")[0]), int(time.split(":")[1]), int(time.split(":")[2])
        in_s = (h * 3600 + m * 60 + s + seconds) % (24 * 3600) # make sure total seconds not exceeding one day
        new_h, reminder = divmod(in_s, 3600) # divmod return the integer result and reminder
        new_m, new_s = divmod(reminder, 60)
        new_time = f"{new_h:02d}:{new_m:02d}:{new_s:02d}"
        result.append(new_time)
    return result

# You are given an initial date as a string in the format YYYY-MM-DD, along with an integer n which represents a number of days.
# Your task is to calculate the date after adding the given number of days to the initial date and return the result in the YYYY-MM-DD format
def add_days_1(date, n):
    from datetime import datetime, timedelta
    # Parse the initial date string into a datetime object
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    # Add the number of days using timedelta
    new_date = date_obj + timedelta(days=n)
    # Convert the resulting date back to the string in the desired format
    return new_date.strftime("%Y-%m-%d")

def add_days_2(date, n):
    def is_leaf_year(year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    def days_in_month(year, month):
        month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if is_leaf_year(year) and month == 2:
            return 29
        else:
            return month_days[month]
    year, month, day = map(int, date.split('-'))
    while n > 0:
        days_in_current_month = days_in_month(year,month)
        if (day + n) <=  days_in_current_month:
            day = day + n
            break
        else:
            # Move to the next month, after accounting for the days left in this month
            n -= (days_in_current_month - day + 1)
            # reset to the first day of next month
            day = 1
            # move to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
    return f"{year:04d}-{month:02d}-{day:02d}"

# given two lists, sentences and words, each comprising n strings,
# Your task is to find all instances of each word in the corresponding sentence from the sentences list and replace them with the reverse of the word.
# When finding the matching string, you need to ignore case
# The words and sentences at the same index in their respective lists are deemed to correspond to each other.
# Return a new list comprising n strings, where each string is the sentence from the sentences list at the corresponding index,
# with all instances of the word from the words list at the same index replaced with its reverse.
# If the word is not found in the respective sentence, keep the sentence as it is.
# Remember, while replacing the instances of word in the sentence, you should preserve the case of the initial letter of the word.
# If a word starts with a capital letter in the sentence, its reversed form should also start with a capital letter.
def replace_words_in_sentences(sentences, words):
    new_sentenaces = []
    for sentence, word in zip(sentences, words):
        start_index = sentence.lower().find(word.lower())
        while start_index != -1:
            if sentence[start_index].isupper():
                sentence = sentence.lower().replace(word.lower(), word[::-1].capitalize())
            else:
                sentence = sentence.replace(word, word[::-1])
            start_index = sentence.lower().find(word.lower(), start_index+1)
        new_sentenaces.append(sentence)
    return new_sentenaces
# roadA and roadB are integer array, start jumping from road A.Each element in both the roads dictate exactly where to jump on the other road.
# If Alice is at the i-th position of roadA, where roadA[i] = x, then Alice moves to the x-th position of roadB.
# Likewise, if Alice is at the i-th position of roadB, where roadB[i] = y, then she moves to the y-th position of roadA
# input: roadA = [1, 0, 2], roadB = [2, 0, 1], result = [2, 4, 4]
def read_jump_distance(roadA, roadB):
    def jump(start_index):
        pos = start_index
        visitedA = set()
        visitedB = set()
        is_on_road_a = True
        distance = 0
        while True:
            if is_on_road_a:
                if pos in visitedA:
                    break
                visitedA.add(pos)
                next_pos = roadA[pos]
            else:
                if pos in visitedB:
                    break
                visitedB.add(pos)
                next_pos = roadB[pos]

            is_on_road_a = not is_on_road_a
            pos = next_pos
            distance += 1
        return distance
    n = len(roadA)
    m = len(roadB)
    result = [0] * n
    for i in range(n):
        result[i] = jump(i)
    return result

# find the character that is 3 characters before a given character in the English alphabet
# e.g. the input character is 'a', the character 3 positions before it is 'x'
# e.g. the input character is 'd', the character 3 positions before it is 'a'.
def three_before_character(c):
    if 'a' <= c <= 'z': # or c.alpha() and c.islower()
        return chr((ord(c) - ord('a') - 3) % 26 + ord('a'))
    elif 'A' <= c <= 'Z': # or c.alpha() and c.isupper()
        return chr((ord(c) - ord('A') - 3) % 26 + ord('A'))
    else:
        return c # return if it's not a character

# find the character that is 3 characters after a given character in the English alphabet,
# e.g. input a, the character 3 positions after it is 'd'
# e.g. input x, the character 3 positions after it is 'a'
def three_after_character(c):
    if 'a' <= c <= 'z': # or c.alpha() and c.islower()
        return chr((ord(c) - ord('a') + 3) % 26 + ord('a'))
    elif 'A' <= c <= 'Z': # or c.alpha() and c.isupper()
        return chr((ord(c) - ord('A') + 3) % 26 + ord('A'))
    else:
        return c # return if it's not a character
# find the charater that is the oppsite of input character
def opposite_char(c):
    if 'a' <= c <= 'z': # or use if c.islower()
        return chr(ord('a') + (ord('z') - ord(c)))
    elif 'A' <= c <= 'Z': # or use if c.isupper()
        return chr(ord('A') + (ord('Z') - ord(c)))
    else:
        return c
# given two lists, each list has many strings, for each string in list1, if it shuffle each character then can be same as string in list2, then return the string
# input:
# output:
def find_string_anagram(list_1, list_2):
    from collections import defaultdict
    # Create mapping for `list_1`
    mapping_1 = defaultdict(list)
    # mapping_1 stores (sorted anagram) -> list[anagrams] mapping for `list_1`
    for word in list_1:
        sorted_tuple = tuple(sorted(word)) # unique identifier of the anagram
        mapping_1[sorted_tuple].append(word)
        # `mapping_1[sorted_tuple]` stores all anagrams under the same identifier for `list_1`

    # Create mapping for `list_2`
    mapping_2 = defaultdict(list)
    # mapping_2 stores (sorted anagram) -> list[anagrams] mapping for `list_2`
    for word in list_2:
        sorted_tuple = tuple(sorted(word)) # unique identifier of the anagram
        mapping_2[sorted_tuple].append(word)
        # `mapping_2[sorted_tuple]` stores all anagrams under the same identifier for `list_2`

    # Intersect keys from mapping_1 and mapping_2 to get common sorted tuples
    # Every element in `common_tuples` is an anagram identifier that exists in both lists
    common_tuples = set(mapping_1.keys()) & set(mapping_2.keys())

    output = []
    for anagram_tuple in common_tuples:
        for word1 in mapping_1[anagram_tuple]:
            for word2 in mapping_2[anagram_tuple]:
                # Both word1 and word2 have the same anagram identifier, so are anagrams
                output.append((word1, word2))

    return output

# input: docs = ["Hello world", "world of python", "python is a snake"]
# output: {'Hello': {0: 1}, 'world': {0: 1, 1: 1}, 'of': {1: 1}, 'python': {1: 1, 2: 1}, 'is': {2: 1}, 'a': {2: 1}, 'snake': {2: 1}}
def keyword_index(docs):
    result = {}
    for doc_index, doc in enumerate(docs):
        words = doc.split()
        for w in words:
            if w in result.keys():
                if doc_index in result[w].keys():
                    result[w][doc_index] += 1
                else:
                    result[w][doc_index] = 1
            else:
                result[w] = {}
                result[w][doc_index] = 1
    return result

# Python program to find the root of a given function using Binary Search
import math
import numpy as np

# Define the binary search function
# Define a continuous function 'f' where f(x) = x^4 - x^2 - 10
def f(x):
        return x**4 - x**2 - 10
def binary_search(func, target, left, right, precision):
    while abs(func(left) - target) > precision and abs(func(right) - target) > precision: # you can also use np.abs()
        middle = (left + right) / 2
        if func(middle) < target:
            left = middle
        else:
            right = middle

    return middle

epsilon = 1e-6  # to make sure the solution is within an acceptable range
target = 50  # target value for root of function 'f'
start = -5  # starting point of the interval
end = 5  # ending point of the interval
result = binary_search(f, target, start, end, epsilon)
print("The value of x for which f(x) is approximately 0 within the interval [" + str(start) + ", " + str(end) + "] is: ", result)

# given a peculiar list of unique integers - it's sorted in a decreasing order and then rotated at a random pivot.
# So, while you and I know that it's still sorted, it kicks off from an unpredictable point
# find a specific target number in this array and report its index. If the target turns up missing, return -1.
# input: [4, 3, 2, 1, 8, 7, 6, 5], 1), output: 3
# input: [9, 8, 7, 6, 5, 4, 3, 2, 1], 4), output: 5
# input:[5, 4, 3, 2, 1], 8, output: -1
# naive solution: scan all num one by one
def search_dec_rotated(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        # decide which part is sorted
        if nums[left] > nums[mid]: # left part is sorted in descending order
            if nums[left] >= target > nums[mid]: # target is in the left part
                right = mid - 1
            else: # target is in the right part
                left = mid + 1
        else: # right part is sorted in desending order
            if nums[mid] > target >= nums[right]: # target is int the right park
                left = mid + 1
            else: # target is in the left part
                right = mid - 1
    return -1

# find the first and last index of target in nums
# naive solution: from left to right, scan num one by one
def get_first_last_pos(nums, target):
    def binary_search(left, right, find_first):
        if left <= right:
            mid = (left + right) // 2
            # When searching for the first instance, if our midpoint is the same as or higher than our target, we focus on the array's left half
            if nums[mid] > target or (find_first and target == nums[mid]):
                return binary_search(left, mid - 1, find_first)
            # for the last instance, if the midpoint is lower than our target, our attention turns to the right half.
            else:
                return binary_search(mid + 1, right, find_first)
        return left

    first = binary_search(0, len(nums) - 1, True)
    last = binary_search(0, len(nums) - 1, False) - 1
    if first <= last:
        return [first, last]
    else:
        return [-1, -1]

# naive solution: scan num from left to right, find the num matching target or the num that is larger than target, then nums.insert(index, target)
# nums is sorted by ascendimg, use binary search to get it
def search_insert(nums, target):
    nums.append(float('inf'))  # append an infinite element to handle edge case
    left, right = 0, len(nums)
    while right - left > 1:
        mid = (left + right) // 2
        if nums[mid] < target: # left only moves when nums[mid] is strictly less than the target
            left = mid
        else:
            right = mid
    return right # return right to find the leftmost position

# input: forest is array with 0 and 1, move to small index if direction = -1, move to large index if direction = 1
# output: the jump that you can go through the forest safely (without stepping on 1 in each jump)
def calculate_jump(forest, start, direction):
    jump = 1
    while (direction * jump) + start >= 0 and (direction * jump) + start <= len(forest):
        pos = start
        while 0 <= pos < len(forest):
            if forest[pos] == 1:
                break
            pos += jump * direction
        else:
            return jump

        jump += 1
    return -1
# go to one direction only (either go bigger index or smaller index), decide if all unique items in garden are visited
# input: [3, 1, 2, 1, 3, 2, 1], 0, 1), output: 2
# input: [1, 2, 3, 4, 5, 9, 2, 1, 3, 8, 2, 7, 1, 6], 13, -1), output: 1
# input: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 0, -1), output: -1
# input: [1, 5, 2, 5, 3, 5, 4, 5], 3, -1), output: -1
def largest_step(garden, start, direction):
    k = -1
    cur_k = 1
    while 0 <= start + (direction * cur_k) <= len(garden):
        visited = set()
        pos = start
        while 0 <= pos < len(garden):
            visited.add(garden[pos])
            pos += cur_k * direction
        else:
            if set(garden) == visited:
                k = max(k, cur_k)
        cur_k += 1
    return k

# intput: an array of n integer values, each value stands for a trap. A trap will reduce your health
# move from the start position (index 0) to the end position.you can move by x elements in the right direction only, where x ranges from 1 to n.
# Each time you step on a trap, you lose health points equal to the trap's powe
# output: the x that you must choose such that you lose the least amount of health points upon reaching the end of the array.
# If at any point your health points reach 0 or less, you are considered out of the game.
def calculate_health(dungeon, health):
    result = {} # key is jump, value is left health
    jump = 1
    while jump >= 0 and jump <= len(dungeon):
        pos = 0
        cur_health = health
        while 0 <= pos < len(dungeon):
            cur_health -= dungeon[pos]
            if cur_health <= 0:
                break
            pos += jump
        else:
            result[jump] = cur_health
        jump += 1
    if result:
        return sorted(result.items(), key = lambda x: x[1], reverse = True)[0][0]
    else:
        return -1

# rotate array1, find the minimum manhattan for rotated array and array2
# A rotation of an array refers to taking one or more elements from the end and moving these elements to the beginning, maintaining their original order in the process
# you find multiple rotations of array1 that yield the same smallest Manhattan distance with array2. In this case, you should return the rotated array that,
# when converted into an integer number by concatenating all of its digits (from left to right), would be the smallest.
def get_min_manhattan(array1, array2):
    def rotate(ratation):
        new_arr = [0] * len(array1)
        for i in range(len(array1)):
            new_i = (i + rotation) % len(array1)
            new_arr[new_i] = array1[i]
        return new_arr

    def concatenate_digits(arr):
        r = ''
        for digit in arr:
            r += str(digit)
        return r

    if array1 == array2:
        return (array1, 0)

    rotation = 0
    min_manhattan = sys.maxsize
    result = {}
    while 0 <= rotation < len(array1):
        cur_manhattan = 0
        new_arr1 = rotate(rotation)
        for num1, num2 in zip(new_arr1, array2):
            cur_manhattan += abs(num1 - num2)
        if cur_manhattan <= min_manhattan:
            min_manhattan = cur_manhattan
            if cur_manhattan in result.keys():
                result[cur_manhattan].append(new_arr1)
            else:
                result[cur_manhattan] = [new_arr1]
        rotation += 1

    if len(result[min_manhattan]) == 1:
        return (result[min_manhattan][0], min_manhattan)
    else:
        arrays = result[min_manhattan]
        conca = {}
        for arr in arrays:
            num = int(concatenate_digits(arr))
            conca[num] = arr
        min_arr = sorted(conca.items(), key = lambda x: x[0])[0][1]
        return (min_arr, min_manhattan)

# pair each character sequentially in s, then remove the one that comes earlier in the lexicographical order, put the removed ones in output list
# input s = "BCAAB", the output should be ['B', 'A', 'A', 'B', 'C']
def pair_and_remove(s):
    result = []
    if len(s) == 1:
        result.append(s)
        return result
    while True:
        temp_arr = []
        for i in range(0, len(s), 2):
            if i + 1 < len(s):
                temp_arr.append((s[i], s[i+1]))
            else:
                temp_arr.append((s[i]))
        # construct new string by removing
        s = ''
        for item in temp_arr:
            if len(item) == 2:
                if item[0] <= item[1]:
                    result.append(item[0])
                    s += item[1]
                else:
                    result.append(item[1])
                    s += item[0]
            else:
                s += item[0]
        return result

def house_game(houses):
    # transfer all num in houses to string
    houses_string = []
    for house in houses:
        houses_string.append(str(house))
    n = len(houses_string)
    step = 0
    while True:
        step += 1
        new_houses = houses_string.copy()
        for i in range(n):
            if len(houses_string[i]) < step:
                continue
            new_i = (i + 1) % n
            # get the step digit
            step_digit = new_houses[i][step * -1]
            # remove the step digit from current house
            if step == 1:
                new_houses[i] = new_houses[i][0:step * -1]
            else:
                new_houses[i] = new_houses[i][0:step * -1] + new_houses[i][step * -1 + 1:]
            # add the step digit to the front of next house
            new_houses[new_i] = step_digit + new_houses[new_i]

        if new_houses == houses_string:
            return [int(house) for house in new_houses]
        else:
            houses_string = new_houses

# input: [2, 1, -3, 4]) output: (2, 1)
# an array of integers, Each integer a_i in the array signifies how many steps the player can move and in which initial direction
# A positive integer allows the player to move that many steps to the right.
# A negative integer directs the player to move that many steps to the left.
# Zero signifies a blockade that prevents further movement.
# The game proceeds along the following rules:
# The player starts at the first position of the array (0-indexed) and moves according to the value at the player's current position in the array.
# If the value in the current position is zero, then the game ends. If the player's current position leads them outside of the array's boundaries,
# then their ability to move in the current direction ceases.
# If the latter happens, then the player reverses their direction and continues to move according to similar rules, but now the directions are inverted:
# positive integers lead the player to the left, and negative integers point to the right.
# The game ends when the player encounters a blockade or the array boundaries for the second time and so can no longer move.
def evaluatePath(numbers):
    n = len(numbers)
    pos = 0
    moves = 0
    direction_reversed = False
    while True:
        step = numbers[pos]
        if step == 0:
            return (pos, moves)
        if (step > 0 and not direction_reversed) or (step < 0 and direction_reversed):
            # move right
            new_pos = pos + abs(step)
        else:
            # move left
            new_pos = pos - abs(step)
        if new_pos < 0 or new_pos >= n:
            if direction_reversed:
                return (pos, moves)
            else:
                direction_reversed = True
                continue # IMPORTANT: when changing direction, do not increase moves
        else:
            pos = new_pos
        moves += 1
    return (pos, moves)

# num1 and num2 is bigger numbers, return multiplication of these 2 numbers
def two_big_number_product(num1, num2):
    result = []
    j = len(num2) - 1
    while j >= 0:
        line = []
        cur_line_index = len(num2) - 1 - j
        while cur_line_index > 0:
            line.append('0')
            cur_line_index -= 1
        i, carry = len(num1) - 1, 0
        while i >= 0:
            n = (int(num2[j]) * int(num1[i]) + carry) % 10
            line.append(str(n))
            carry = (int(num2[j]) * int(num1[i]) + carry) // 10
            i -= 1
        if carry > 0:
            line.append(str(carry))
        result.append(line)
        j -= 1
    max_column = len(result[-1])
    line = len(result)
    j, res, carry = 0, [0] * max_column, 0
    while j < max_column:
        i, total = 0, 0
        while i < line:
            cur = int(result[i][j]) if j < len(result[i]) else 0
            total += cur
            i += 1
        total += carry
        res[j] = str(total % 10)
        carry = total // 10
        j += 1
    if carry > 0:
        res.append(str(carry))
    res = res[::-1]
    while res[0] == "0" and len(res) > 1:
        res = res[1:]
    return ''.join(res)

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

random_numbers = [random.randint(1, 50) for _ in range(15)]
print("Unsorted List: ", random_numbers)
random_numbers = quick_sort(random_numbers)
# TODO: Use the Quick Sort function to sort the list and print the sorted list
print("sorted List: ", random_numbers)

def partition(arr, low, high):
    # asconding: this method partitions arr[low..high] to move all elements <= arr[high] to the left
    # and returns the index of `pivot` in the updated array
    # descending: change to arr[j] >= pivot
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort_2(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort_2(arr, low, pi - 1)
        quick_sort_2(arr, pi + 1, high)

# Generate a list of random numbers between 1 and 100
random_list = random.sample(range(1, 101), 20)
print('Unsorted list:', random_list)

quick_sort_2(random_list, 0, len(random_list) - 1)
print('Sorted list with Quick Sort:', random_list)

# merge sort, O(nlogn)
def merge_sort(lst):
    # If it's a single element or an empty list, it's already sorted
    if len(lst) <= 1:
        return lst

    # Find the middle point
    mid = len(lst) // 2

    # Recursively sort both halves
    left_half = merge_sort(lst[:mid])
    right_half = merge_sort(lst[mid:])

    # Merge the two sorted halves
    return merge_sorted_array(left_half, right_half)

def merge_sorted_array(left, right):
    result = []
    i = 0
    j = 0

    # Compare the smallest unused elements in both lists and append the smallest to the result
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Once we've exhausted one list, append all remaining elements from the other list to the result
    # Here, we append both lists, as at least one is an empty list
    result.extend(left[i:]) # or result += left[i:]
    result.extend(right[j:]) # or result += right[j:]
    return result

def count_inversions(arr):
    # The code is very similar to the merge_sort implementation
    # The main difference lies in the merge_count_inversions function
    if len(arr) <= 1:
        return arr, 0
    else:
        middle = int(len(arr) / 2)
        left, a = count_inversions(arr[:middle])
        right, b = count_inversions(arr[middle:])
        result, c = merge_count_inversions(left, right)
        return result, (a + b + c)

def merge_count_inversions(x, y):
    count = 0
    i, j = 0, 0
    merged = []
    while i < len(x) and j < len(y):
        if x[i] <= y[j]:
            merged.append(x[i])
            i += 1
            # for anti-inversion, we should do this:
            # count += len(y) - j # all elements from y[j] onwards form an anti-inversion with x[i] (x[i] < y[j])
        else:
            merged.append(y[j])
            j += 1
            # Here, we update the number of inversions
            # Every element from x[i:] and y[j] forms an inversion
            count += len(x) - i
    merged += x[i:]
    merged += y[j:]
    return merged, count

# test
import random, string
# Generate a list of 100 random numbers between 1 and 1000
random_numbers = [random.randint(1, 1000) for i in range(100)]
print(f"Original List: {random_numbers}")
# Outputs: Original List: [402, 122, 544, 724, 31, 515, 845, 2, 168, 311, 262, 498, 421, 25, 757, 171, 795, 634, 115, 572, 232, 94, 547, 177, 823,
# 607, 571, 403, 274, 527, 951, 971, 161, 771, 877, 969, 650, 37, 723, 497, 520, 571, 948, 886, 542, 795, 580, 933, 155, 692, 559, 259, 907, 516,
# 294, 625, 152, 287, 75, 614, 719, 10, 828, 157, 574, 257, 853, 271, 873, 745, 233, 519, 272, 405, 541, 912, 294, 737, 940, 154, 49, 77, 464, 416,
#  738, 143, 364, 223, 385, 201, 636, 493, 757, 10, 792, 555, 384, 362, 101, 109]
# Sort the list
sorted_numbers = merge_sort(random_numbers)
print(f"\nSorted List: {sorted_numbers}")
# Outputs: Sorted List: [2, 10, 10, 25, 31, 37, 49, 75, 77, 94, 101, 109, 115, 122, 143, 152, 154, 155, 157, 161, 168, 171, 177, 201, 223, 232, 233,
#  257, 259, 262, 271, 272, 274, 287, 294, 294, 311, 362, 364, 384, 385, 402, 403, 405, 416, 421, 464, 493, 497, 498, 515, 516, 519, 520, 527, 541,
# 542, 544, 547, 555, 559, 571, 571, 572, 574, 580, 607, 614, 625, 634, 636, 650, 692, 719, 723, 724, 737, 738, 745, 757, 757, 771, 792, 795, 795, 823,
# 828, 845, 853, 873, 877, 886, 907, 912, 933, 940, 948, 951, 969, 971]

# Generate a list of 20 random alphanumeric characters.
random_alphanumeric = [random.choice(string.ascii_letters + string.digits) for _ in range(20)]
print("Original List of random alphanumeric characters:\n", random_alphanumeric)
# Apply merge sort
sorted_alphanumeric = merge_sort(random_alphanumeric)
print("\nSorted List of alphanumeric characters:\n", sorted_alphanumeric)

# quick sort. divide and conquer, can be used to solve this problem more efficiently. By selecting the right pivot for partitioning,
# the input list is divided into two: a left partition, which contains elements less than the pivot, and a right partition, which contains elements greater than the pivot.
def find_kth_smallest(numbers, k):
    if numbers:
        pos = partition(numbers, 0, len(numbers) - 1)
        if k - 1 == pos:
            # The pivot is the k-th element after partitioning
            return numbers[pos]
        elif k - 1 < pos:
            # The pivot index after partitioning is larger than k
            # We'll keep searching in the left part
            return find_kth_smallest(numbers[:pos], k)
        else:
            # The pivot index after partitioning is smaller than k
            # We'll keep searching in the right part
            return find_kth_smallest(numbers[pos + 1:], k - pos - 1)

def partition(nums, l, r):
    # Choosing a random index to make the algorithm less deterministic
    rand_index = random.randint(l, r)
    nums[l], nums[rand_index] = nums[rand_index], nums[l]
    pivot_index = l
    for i in range(l + 1, r + 1):
        if nums[i] <= nums[l]:
            pivot_index += 1
            nums[i], nums[pivot_index] = nums[pivot_index], nums[i]
    nums[pivot_index], nums[l] = nums[l], nums[pivot_index]
    return pivot_index

# a list of daily temperatures recorded in ascending order of day, But on Earth days! Your job is, for each day, to find out how many days
# you'll have to wait until the next cooler day. Is it tomorrow? Or three days later? Or maybe there's no cooler day in sight and that calls for a -1
# input
def days_until_cooler(temps):
    result = [-1] * len(temps)
    stack = [] # track the index of current temperature

    for i in range(len(temps) - 1, -1, -1):
        # This ensures that the stack only contains indices of temperatures that are cooler than the current temperature, keep descending order
        while stack and temps[i] <= temps[stack[-1]]:
            stack.pop()
        if stack:
            result[i] = stack[-1] - i
        stack.append(i)
    return result

print(days_until_cooler([30, 60, 90, 120, 60, 30]))  # Expected: [-1, 4, 2, 1, 1, -1]
print(days_until_cooler([100, 95, 90, 85, 80, 75]))  # Expected: [1, 1, 1, 1, 1, -1]
print(days_until_cooler([1]))  # Expected: [-1]

# analyzing historical stock prices. For each day, you would like to know the previous day when the price was lower than the current price.
def findSmallerPreceeding(numbers):
    result = [-1]
    stack = []
    for num in numbers:
        while stack and stack[-1] >= num: # keep aseconding order
            stack.pop()
        result.append(stack[-1] if stack else -1)
        stack.append(num)
    return result[1:]

# method to create a new matrix that is reflecting over secondary diagonal
# input:
# [
#  [1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]
# ]
# output:
# [
#  [9, 6, 3],
#  [8, 5, 2],
#  [7, 4, 1]
# ]
def reflectOverSecondaryDiagonal(matrix):
    size = len(matrix)
    new_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
           new_matrix[i][j] = matrix[size-1-j][size-1-i]
    return new_matrix

# output the path that follow the highest elevation
# input: mountain, and 1,1, output: [3,5,6]
def trek_path(elevation_map, start_x, start_y):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # East, South, West, North
    path = [elevation_map[start_x][start_y]]

    while True:
        current_height = path[-1]
        # Pre-completed: Find all possible moves from the current position, moving only to higher and not yet visited elevations.
        possible_moves = [
            (start_x + dx, start_y + dy) for dx, dy in directions
            if (0 <= start_x + dx < len(elevation_map) and
                0 <= start_y + dy < len(elevation_map[0]) and
                elevation_map[start_x + dx][start_y + dy] > current_height)
        ]
        if not possible_moves:
            return path
        # TODO: Implement logic to select the next position based on the highest elevation in the possible moves.
        # Hint: Use a key function with the max() function to find the move leading to the highest elevation.
        next_x, next_y = max(possible_moves, key=lambda pos: elevation_map[pos[0]][pos[1]])
        start_x, start_y = next_x, next_y
        path.append(elevation_map[start_x][start_y])

mountain = [
    [1, 2, 3],
    [2, 3, 4],
    [3, 5, 6]
]
print(trek_path(mountain, 1, 1))

# input: string: "1 create 09:00, 2 create 10:00, 1 delete 12:00, 3 create 13:00, 2 delete 15:00, 3 delete 16:00"
# output: [(2, '05:00')]
def analyze_logs(logs):
    log_list = logs.split(", ")
    time_dict = {}
    life_dict = {}
    format = '%H:%M'

    for log in log_list:
        G_ID, action, time = log.split()
        G_ID = int(G_ID)
        time = datetime.strptime(time, format)

        if action == 'create':
            time_dict[G_ID] = time
        else:
            if G_ID in time_dict:
                life_dict[G_ID] = life_dict.get(G_ID, datetime.strptime('00:00', format)) + (time - time_dict[G_ID])
                del time_dict[G_ID]

    max_life = max(life_dict.values())  #Find the longest lifetime
    #Build the result list where each item is a tuple of group ID and its lifetime, if it has the longest lifetime.
    result = [(ID, str(life.hour).zfill(2) + ':' + str(life.minute).zfill(2)) for ID, life in
              life_dict.items() if life == max_life]

    return sorted(result)  #Return the list sorted in ascending order of the group IDs

# input is text, return top n letters and words in the text. ignore punctuation, case insensitive. sort first by frequency then by alphabet
from collections import Counter
def top_common_letters_words(text, n):
    # Normalize the text: replace punctuation with spaces and lower case the text
    normalized_text = re.sub(r"[^\w\s']", ' ', text.lower())
    # Split words (contractions treated as separate words)
    words = re.findall(r"\b\w+\b", normalized_text)
    # Calculate word frequencies
    word_counts = Counter(words)
    # Join text and filter only letters
    letters = ''.join(words)
    # Calculate letter frequencies
    letter_counts = Counter(letters)
    # Sort words by frequency descending, then alphabetically ascending
    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    # Sort letters by frequency descending, then alphabetically ascending
    sorted_letters = sorted(letter_counts.items(), key=lambda x: (-x[1], x[0]))
    # Get top n letters and words
    top_letters = sorted_letters[:n]
    top_words = sorted_words[:n]
    return top_letters, top_words

def format_output(top_letters, top_words, n):
    # Format the output as specified
    result = []
    result.append(f"Top {n} letters:")
    for letter, freq in top_letters:
        result.append(f"{letter}: {freq}")
    result.append("")
    result.append(f"Top {n} words:")
    for word, freq in top_words:
        result.append(f"{word}: {freq}")
    return "\n".join(result)

# Logic to control direction based on edges:
# if dir == 1:  # Moving down-left
#     if row == rows - 1:
#         col += 1
#         dir = -1
#     elif col == 0:
#         row += 1
#         dir = -1
#     else:
#         row += 1
#         col -= 1
# else:  # Moving up-right
#     if col == cols - 1:
#         row += 1
#         dir = 1
#     elif row == 0:
#         col += 1
#         dir = 1
#     else:
#         row -= 1
#         col += 1

# zigzag to traverse the 2D array
# input:
# matrix = [
# [1, -2, 3, -4],
# [5, -6, 7, 8],
# [-9, 10, -11, 12]
# ]
# output: zigzag pattern: [1, -2, 5, -9, -6, 3, -4, 7, 10, -11, 8, 12]
def diagonal_traverse(matrix):
    n = len(matrix)       # Number of rows
    m = len(matrix[0])    # Number of columns
    result = []           # To store indices of negative values

    row, col = 0, 0       # Start from the top-left corner
    direction = 1         # 1 for up-right, -1 for down-left

    while row < n and col < m:
        # Check if the current cell contains a negative integer
        if matrix[row][col] < 0:
            result.append((row + 1, col + 1))  # Convert to 1-based index

        # Move in the current direction
        if direction == 1:  # Moving up-right
            if col + 1 < m and row - 1 >= 0:  # Can move diagonally up-right
                col += 1
                row -= 1
            elif col + 1 < m:  # Hit the top boundary, move right
                col += 1
                direction = -1
            else:  # Hit the right boundary, move down
                row += 1
                direction = -1
        else:  # Moving down-left
            if row + 1 < n and col - 1 >= 0:  # Can move diagonally down-left
                row += 1
                col -= 1
            elif row + 1 < n:  # Hit the left boundary, move down
                row += 1
                direction = 1
            else:  # Hit the bottom boundary, move right
                col += 1
                direction = 1

    return result
# parse 2D array as clock wise
def spiral_traverse_and_vowels(grid):
    vowels = ['a', 'e', 'i', 'o', 'u']
    traverse = []
    rows = len(grid)
    cols = len(grid[0])
    r, c = 0, 0
    last_left, last_top, last_right, last_bottom = 0, 0, cols - 1, rows - 1
    while last_left <= last_right and last_top <= last_bottom:
        # move right
        for c in range(last_left, last_right+1):
            traverse.append(grid[last_top][c])
        last_top += 1
        # move down
        for r in range(last_top, last_bottom+1):
            traverse.append(grid[r][last_right])
        last_right -= 1
        if last_top <= last_bottom:
            # move left
            for c in range(last_right, last_left-1, -1):
                traverse.append(grid[last_bottom][c])
            last_bottom -= 1
        if last_left <= last_right:
            # move up
            for r in range(last_bottom, last_top-1, -1):
                traverse.append(grid[r][last_left])
            last_left += 1

    result = []
    for i, c in enumerate(traverse):
        if c in vowels:
            result.append(i + 1)
    return result

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n//2 + 1):
        if n % i == 0:
            return False
    return True

# parse 2D array as move right, one down, move left, done down, move right
def zigzag_traverse_and_primes(matrix):
    n = len(matrix)
    r, c, result = 0, 0, []
    direction = 1 # 1: rightward, -1: leftward
    while r < n:
        if direction == 1:
            while c < n:
                result.append(matrix[r][c])
                c += 1
            c -= 1
            r += 1
            direction = -1
        else:
            while c >= 0:
                result.append(matrix[r][c])
                c -= 1
            c += 1
            r += 1
            direction = 1
    result_dict = {}
    for i, num in enumerate(result):
        if is_prime(num):
            result_dict[i+1] = num
    return result_dict
# input: A: [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]], B: [[11, 12, 13],[14, 15, 16], [17, 18, 19]], submatrix_coords[[2,3,2,3], [1,2,1,2]] (1 based index)
# output: combine: [[6, 7, 11, 12], [10, 11, 14, 15]] interwoven: [[6, 11, 7, 12], [10, 14, 11, 15]]
def interwoven(matrix_A, matrix_B, submatrix_coords):
    start_row_a, end_row_a, start_col_a, end_col_a = submatrix_coords[0]
    start_row_b, end_row_b, start_col_b, end_col_b = submatrix_coords[1]
    sub_matrix_a = [row[start_col_a-1:end_col_a] for row in matrix_A[start_row_a-1:end_row_a]]
    sub_matrix_b = [row[start_col_b-1:end_col_b] for row in matrix_B[start_row_b-1:end_row_b]]
    # get the matrix that combine sub a and sub b
    matrix_combine = [row_a + row_b for row_a, row_b in zip(sub_matrix_a, sub_matrix_b)]
    # get the mtraix that interwoven sub a and sub b
    matrix_interwoven = []
    for row_a, row_b in zip(sub_matrix_a, sub_matrix_b):
        new_row = []
        for i in range(len(row_a)):
            new_row.append(row_a[i])
            new_row.append(row_b[i])
        matrix_interwoven.append(new_row)
    return matrix_combine, matrix_interwoven

import itertools
# Define the input array A
A = ["a", "b", "c", "d", "e"]
# Input for the value of k
k = 3  # You can change this value as needed
# Ensure the array has more than k items
if len(A) > k:
    # Generate all combinations of k items from A
    B = list(itertools.combinations(A, k))
    # Print the result
    print(f"Array B (combinations of {k} items):")
    for combo in B:
        print(combo)
else:
    print(f"The array must have more than {k} items.")
# Array B (combinations of 4 items):
# ('a', 'b', 'c', 'd')
# ('a', 'b', 'c', 'e')
# ('a', 'b', 'd', 'e')
# ('a', 'c', 'd', 'e')
# ('b', 'c', 'd', 'e')
# Array B (combinations of 2 items):
# ('a', 'b')
# ('a', 'c')
# ('a', 'd')
# ('a', 'e')
# ('b', 'c')
# ('b', 'd')
# ('b', 'e')
# ('c', 'd')
# ('c', 'e')
# ('d', 'e')

# input: t1, t2 like this: Sat 02 May 2015 19:54:36 +0530, Fri 01 May 2015 13:54:36 -0000
# output: time differnce in seconds
def time_delta(t1, t2):
    from datetime import datetime
    format = "%a %d %b %Y %H:%M:%S %z"
    dt1 = datetime.strptime(t1, format)
    dt2 = datetime.strptime(t2, format)
    delta_in_seconds = int(abs((dt1-dt2).total_seconds()))
    return str(delta_in_seconds)

# input: S = "aaadaa", K = "aa"
# output: Occurrences of K in S: [(0, 1), (1, 2), (4, 5)]
def find_substring_indices(S, K):
    # Initialize an empty list to store the indices
    indices = []
    start = 0
    # Loop to find all occurrences of K in S
    while True:
        # Find the next occurrence of K starting from 'start'
        start = S.find(K, start)
        if start == -1:  # If no more occurrences are found, exit the loop
            break
        # Append the start and end indices as a tuple
        indices.append((start, start + len(K) - 1))
        # Move start to the next character for the next search
        start += 1
    return indices

# Both players are given the same string S. Both players have to make substrings using the letters of the string S,
# Stuart has to make words starting with consonants, Kevin has to make words starting with vowels, The game ends when both
# players have made all possible substrings. A player gets +1 point for each occurrence of the substring in the string S.
# solution 1 is better way, without generating all substrings.
# For a character at position i, the number of substrings starting with that character is n - i,
def minion_game_1(string):
    vowels = 'AEIOU'
    n = len(string)
    kevin_score, stuart_score = 0, 0
    for i in range(n):
        if string[i] in vowels:
            kevin_score += n - i
        else:
            stuart_score += n - i
    if kevin_score > stuart_score:
        print(f"Kevin {kevin_score}")
    elif kevin_score < stuart_score:
        print(f"Stuart {stuart_score}")
    else:
        print("Draw")
# solution 2 is native, it generates all substrings and count substring numbers
def minion_game_2(string):
    vowels = 'AEIOU'
    n = len(string)
    kevin_score, stuart_score = 0, 0
    kevin_subs, stuart_subs = [], []
    for i in range(n):
        for j in range(i+1, n+1):
            sub = string[i:j]
            if sub[0] in vowels:
                kevin_subs.append(sub)
            else:
                stuart_subs.append(sub)
    kevin_set, stuart_set = set(kevin_subs), set(stuart_subs)
    for sub in kevin_set:
        kevin_score += kevin_subs.count(sub)
    for sub in stuart_set:
        stuart_score += stuart_subs.count(sub)
    if kevin_score > stuart_score:
        print(f"Kevin {kevin_score}")
    elif kevin_score < stuart_score:
        print(f"Stuart {stuart_score}")
    else:
        print("Draw")

# lists is 2-D array, each item is an arry. lists have K items, K = len(lists)
def maximize_function(K, M, lists):
    from itertools import product
    # pick one item from each array in the lists, Generate all possible combinations from the K lists
    combinations = product(*lists)

    max_value = 0
    for combination in combinations:
        # Calculate the value of S for the current combination
        S = sum(x**2 for x in combination) % M
        # Update max_value if S is greater
        max_value = max(max_value, S)

    return max_value

# input: 1222311, output: (1, 1) (3, 2) (1, 3) (2, 1)
def calculate_character_repeat(s):
    from itertools import groupby
    result = [(len(list(group)), int(key)) for key, group in itertools.groupby(s)]
    print(" ".join(str(item) for item in result))

# non-binary tree, dfs, depth first search
class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_value):
        self.children.append(Node(child_value))

    def dfs(self, visited=None):
        if visited is None:
            visited = set()
        visited.add(self.value)
        print(self.value, end=' -> ')
        for child in self.children:
            if child.value not in visited:
                child.dfs()

def find_path(tree, start, end, visited, path=[]):
    path = path + [start]
    visited.add(start)
    if start == end:
        return path
    for node in tree[start]:
        if node not in visited:
            new_path = find_path(tree, node, end, visited, path)
            if new_path:
                return new_path
    return None

visited = set()
print(find_path(tree, 'A', 'J', visited))
# Output: ['A', 'B', 'F', 'J']
# breadth first search
def bread_first_search_tree(tree, root):
    from collections import deque
    visited = set() # Set to keep track of visited nodes
    visit_order = [] # List to keep visited nodes in order they are visited
    queue = deque() # A queue to add nodes for visiting
    queue.append(root) # We'll start at the root

    while queue: # While there are nodes to visit.
        node = queue.popleft() # Visit the first node in the queue
        visit_order.append(node) # Add it to the list of visited nodes
        visited.add(node) # And mark the node as visited

        # Now add all unvisited children to the queue
        for child in tree[node]:
            if child not in visited:
                queue.append(child)

    return visit_order # Return the order of visited nodes

def can_pile_up(cubes):
    left = 0
    right = len(cubes) - 1
    last_pile = float('inf')  # Start with an infinitely large cube at the bottom

    while left <= right:
        if cubes[left] <= cubes[right]:
            if cubes[right] > last_pile:
                return "No"  # Cannot stack
            last_pile = cubes[right]
            right -= 1
        else:
            if cubes[left] > last_pile:
                return "No"  # Cannot stack
            last_pile = cubes[left]
            left += 1

    return "Yes"  # Successfully stacked all cubes

# Test cases
print(can_pile_up([1, 2, 3, 4, 5]))  # Output: "Yes"
print(can_pile_up([1, 2, 3, 8, 7]))  # Output: "No"

import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests, time_window):
        """
        Initialize the rate limiter.
        :param max_requests: Maximum allowed requests within the time window.
        :param time_window: Time window in seconds.
        :param requests: dict, key is "key"(property), value is an array, which each item is timestamp of each request; alternatively, you can use a deque for value
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
        self.requests_2 = defaultdict(deque)
        self.lock = threading.Lock()  # To ensure thread-safety if used in a multi-threaded environment

    def is_allowed(self, property):
        """
        Check if a request is allowed for the given property (e.g., IP address or credit card number).
        :param property: Attribute to rate limit (e.g., an IP address or credit card).
        :return: True if allowed, False if rate-limited.
        """
        with self.lock:  # Ensure thread-safety
            current_time = time.time()
            request_times = self.requests[property]

            # Remove expired request timestamps outside the time window
            while request_times and request_times[0] <= current_time - self.time_window:
                request_times.pop(0)

            if len(request_times) < self.max_requests:
                request_times.append(current_time)
                self.requests[property] = request_times
                return True
            else:
                return False


# Example usage
if __name__ == "__main__":
    # Create a rate limiter that allows 5 requests per 10 seconds
    rate_limiter = RateLimiter(max_requests=5, time_window=10)

    test_property= "192.168.0.1"  # Example key (e.g., IP address)

    for i in range(10):
        if rate_limiter.is_allowed(test_property):
            print(f"Request {i + 1} allowed.")
        else:
            print(f"Request {i + 1} rate-limited.")
        time.sleep(1)  # Simulate a 1-second delay between requests


def generate_random_ip():
    """Generate a random IPv4 address."""
    return f"{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def test_rate_limiter():
    # Create a rate limiter that allows 3 requests per 5 seconds
    rate_limiter = RateLimiter(max_requests=3, time_window=5)

    # Generate a list of random IPs
    random_ips = [generate_random_ip() for _ in range(5)]  # 5 random IPs
    print(f"Testing with IPs: {random_ips}")

    # Simulate requests for each IP
    for i in range(10):  # Simulate 10 rounds of requests
        print(f"--- Round {i + 1} ---")
        for ip in random_ips:
            if rate_limiter.is_allowed(ip):
                print(f"Request from {ip} allowed.")
            else:
                print(f"Request from {ip} rate-limited.")
        time.sleep(1)  # Simulate a 1-second delay between rounds

if __name__ == "__main__":
    test_rate_limiter()

import time
from threading import Lock
from flask import Flask, request, jsonify
class TokenBucket:
    def __init__(self, capacity: int, rate: float):
        """
        Initialize the Token Bucket rate limiter.
        :param capacity: Maximum number of tokens in the bucket.
        :param rate: Tokens added per second (refill rate).
        """
        self.capacity = capacity
        self.tokens = capacity
        self.rate = rate
        self.last_refill_time = time.time()
        self.lock = Lock()
    
    def _refill(self):
        """Refill the bucket with tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill_time = now

    def allow_request(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.
        :param tokens: Number of tokens required for the request.
        :return: True if request is allowed, False if rate-limited.
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

# Flask API Integration
app = Flask(__name__)
rate_limiter = TokenBucket(capacity=10, rate=2)  # 10 tokens max, refills at 2 tokens/sec

@app.route('/api', methods=['GET'])
def api_endpoint():
    if rate_limiter.allow_request():
        return jsonify({"message": "Request Allowed"}), 200
    else:
        return jsonify({"error": "Too Many Requests"}), 429

if __name__ == "__main__":
    app.run(debug=True)

# Example usage
if __name__ == "__main__":
    rate_limiter = TokenBucket(capacity=10, rate=2)  # 10 tokens max, refills at 2 tokens/sec
    
    for i in range(15):
        if rate_limiter.allow_request():
            print(f"Request {i + 1}: Allowed")
        else:
            print(f"Request {i + 1}: Rate Limited")
        time.sleep(0.3)  # Simulating request intervals

import heapq
# headpop alwasy pop the min number in the queue. When push, it will put the number in the begning (0) if the number is less than others
class MiddleElementFinder:
    def __init__(self):
        self.heaps = [], []

    def add_num(self, num: int) -> None:
        small, large = self.heaps
        heapq.heappush(small, -1 * heapq.heappushpop(large, num))
        if len(large) < len(small):
            heapq.heappush(large, -1 * heapq.heappop(small))
        print(f"large={large}")
        print(f"small={small}")

    def middle_element(self) -> int:
        small, large = self.heaps
        if len(large) > len(small):
            return large[0]
        if large[0] > -1 * small[0]:
            return large[0]
        else:
            return -1 * small[0]

# adjanceny matrix
users = 4  # 4 users: A: 0, B: 1, C: 2, D: 3
M = [[0] * users for _ in range(users)]

# Add friendships, A-B and B-C
M[0][1] = M[1][0] = 1
M[1][2] = M[2][1] = 1

# Print the adjacency matrix
for row in M:
    print(row)

# Output:
# [0, 1, 0, 0]
# [1, 0, 1, 0]
# [0, 1, 0, 0]
# [0, 0, 0, 0]

# Check for friend suggestions
for i in range(users):
    for j in range(i + 1, users):
        if i != j and M[i][j] == 0 and any((M[i][k] == 1 and M[k][j] == 1) for k in range(users)):
            print(f"User {i} and User {j} may know each other.")

# Output:
# User 0 and User 2 may know each other.

# depth first search in graph (ajancency list), resolve problems: establishment of connections within graphs and the discovery of pathways between two nodes
def dfs_adjacency_graph(graph, start, visited):
    visited.add(start)
    print(start, end=' ')

    for next_node in graph[start]:
        if next_node not in visited:
            dfs_adjacency_graph(graph, next_node, visited)

graph = {
    'A': set(['B', 'C']),
    'B': set(['A', 'D', 'E']),
    'C': set(['A']),
    'D': set(['B']),
    'E': set(['B']),
}

visited = set()
dfs_adjacency_graph(graph, 'A', visited)  # Output: A B D E C

def dfs_graph(vertex, visited, graph, parent):
    visited.add(vertex)
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            if dfs_graph(neighbor, visited, graph, vertex):
                return True
        elif neighbor != parent:
            # The parent is already visited, but the parent -> vertex -> parent cycle is degenerate
            return True
    return False

def has_cycle_connected(graph):
    visited = set()
    # Starting DFS from the first vertex in the graph
    return dfs_graph(next(iter(graph)), visited, graph, None)

def breadth_first_search_graph(graph, start):
    visited = set([start])   # a set to keep track of visited nodes
    queue = deque([start])  # a deque (double-ended queue) to manage BFS operations
    while queue:
        node = queue.popleft()  # dequeue a node
        print(node, end=" ")  # Output the visited node
        for neighbor in graph[node]:  # visit all the neighbors
            if neighbor not in visited:  # enqueue unvisited neighbors
                queue.append(neighbor)
                visited.add(neighbor)  # mark the neighbor as visited

# Use an adjacency list to represent the graph
graph = {'A': ['B', 'D'], 'B': ['A', 'C', 'F'], 'C': ['B'], 'D': ['A', 'E'], 'E': ['D'], 'F': ['B']}
breadth_first_search_graph(graph, 'A')  # Call the BFS function, # Output: A B D C F E

from collections import deque

def shortest_path_graph_to_destination(n, graph, start, end):
    # The queue stores tuples `(distance, path)`
    # where `distance` is the minimal distance to the current vertex
    # and `path` is the shortest path from the starting vertex to the current vertex
    queue = deque([(0, [start])])
    visited = set([start])
    min_distances = {start: 0}
    while queue:
        distance, path = queue.popleft()
        node = path[-1]
        min_distances[node] = distance

        if node == end:
            return distance, path

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((distance + 1, path + [neighbor]))

    return float('inf'), []

def shortest_path_to_all_nodes(n, graph, start):
    queue = deque([[start]])
    visited = set([start])
    shortest_paths = {start: [start]}
    while queue:
        path = queue.popleft()
        node = path[-1]
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(path + [neighbour])
                shortest_paths[neighbour] = path + [neighbour]

    return shortest_paths

from abc import ABC, abstractmethod
class Vehicle(ABC):
    def __init__(self, color, engine_type):
        self.color = color
        self.engine_type = engine_type
        self._engine_running = False

    @abstractmethod
    def start_engine(self):
        pass

    @abstractmethod
    def stop_engine(self):
        pass

    @abstractmethod
    def drive(self):
        pass

class Car(Vehicle):
    def start_engine(self):
        self._engine_running = True
        print("Car engine started!")

    def stop_engine(self):
        self._engine_running = False
        print("Car engine stopped!")

    def drive(self):
        if self._engine_running:
            print("Car is driving!")
        else:
            print("Start the engine first!")

# Example usage
car = Car("red", "gasoline")
car.start_engine()
car.drive()
"""
Output:
Car engine started!
Car is driving!
"""

# use output of vmstat as input, and then sort
import subprocess
import pandas as pd
def get_vmstat_output():
    """Runs vmstat command and returns the output as a list of lists."""
    result = subprocess.run(['vmstat'], capture_output=True, text=True)
    lines = result.stdout.splitlines()

    # Extract headers and data
    headers = lines[1].split()
    data = [line.split() for line in lines[2:]]

    return headers, data

def sort_vmstat(headers, data, sort_by='cpu'):
    """Sorts vmstat output by the specified column."""
    df = pd.DataFrame(data, columns=headers)
    df = df.apply(pd.to_numeric, errors='ignore')

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    else:
        print(f"Warning: Column '{sort_by}' not found. Displaying unsorted data.")

    return df

def main():
    headers, data = get_vmstat_output()
    sort_column = 'cpu'  # Change this to the desired column (e.g., 'r', 'b', 'us', 'sy', 'id')
    sorted_df = sort_vmstat(headers, data, sort_by=sort_column)
    print(sorted_df)

if __name__ == "__main__":
    main()

def get_mock_vmstat_output():
    """Generates mock vmstat-like output for environments that do not support subprocess."""
    headers = ["r", "b", "swpd", "free", "buff", "cache", "si", "so", "bi", "bo", "in", "cs", "us", "sy", "id", "wa", "st"]
    data = [
        [0, 0, 100, 200000, 30000, 400000, 0, 0, 10, 20, 100, 200, 5, 2, 90, 3, 0],
        [1, 0, 150, 180000, 25000, 390000, 0, 0, 15, 25, 110, 220, 6, 3, 88, 3, 0],
        [0, 1, 120, 190000, 26000, 380000, 0, 0, 12, 18, 105, 210, 4, 2, 91, 2, 1]
    ]
    return headers, data

def sort_vmstat(headers, data, sort_by='us'):
    """Sorts vmstat output by the specified column."""
    df = pd.DataFrame(data, columns=headers)
    df = df.apply(pd.to_numeric, errors='ignore')

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    else:
        print(f"Warning: Column '{sort_by}' not found. Displaying unsorted data.")

    return df

def main():
    headers, data = get_mock_vmstat_output()
    sort_column = 'us'  # Change this to the desired column (e.g., 'r', 'b', 'us', 'sy', 'id')
    sorted_df = sort_vmstat(headers, data, sort_by=sort_column)
    print(sorted_df)

if __name__ == "__main__":
    main()

def transfer_arr(arr, start, end):
    if arr is None:
        return None
    n = len(arr)
    # support negative index
    start = start if start >=0 else n + start
    end = end if end >=0 else n + end
    # support reverse order
    if start > end:
        start, end = end, start

'''
generate a mine sweeper board, randomly place m mines in a h * w array
'''
# soution 1: download is that the running time is not predictable
def generate_minesweeper_board_1(h, w, m):
    # create an empty board with all cells initialized to 0
    board = [[0 for _ in range(w)] for _ in range(h)]
    # another way to initialize
    # board = []
    # for i in range(h):
    #     row = []
    #     for j in range(w):
    #         row.append(0)
    #     board.append(row)
    mine_pos = set()
    while len(mine_pos) < m:
        row, col = random.randint(0, h-1), random.randint(0, w-1)
        # pair of row and col is unique
        if (row, col) not in mine_pos:
            mine_pos.add((row, col))
    for row, col in mine_pos:
        board[row][col] = "X"

    # another way to fill the board with mines
    n = 0
    while n < m:
        row, col = random.randint(0, h-1), random.randint(0, w-1)
        if board[row][col] != "X":
            board[row][col] = "X"
            n += 1
        else:
            continue
    return board

# solution 2: use sample function to make sure the pair of row and col is unique
def generate_minesweeper_board_2(h, w, m):
    board = [[0 for _ in range(w)] for _ in range(h)]
    # generate all possible positions in the board
    all_positions = [(row, col) for row in range(h) for col in range(w)]
    # randomly sample m unique positions
    mine_pos = set(random.sample(all_positions, m))
    for row, col in mine_pos:
        board[row][col] = "X"
    return board

# besides putting mines randomly, also update the non-mine tiles to the numbers of mines around them
def generate_minesweeper_board_3(h, w, m):
    board = [[0 for _ in range(w)] for _ in range(h)]
    all_positions = [(row, col) for row in range(h) for col in range(w)]
    mine_positions = set(random.sample(all_positions, m))
    for r, c in mine_positions:
        board[r][c] = "X"
    # define directions of 8 adjacent cells
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r, c in mine_positions:
        for dx, dy in directions:
            nr, nc = r + dx, c + dy
            if 0 <= nr < h and 0 <= nc < w and board[nr][nc] != "X":
                board[nr][nc] += 1
    return board