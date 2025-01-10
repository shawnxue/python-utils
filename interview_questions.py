# Given an array A and index D, rotate the array by the D index, such as 
# input: [3, 6, 1, 9, 8, 2]
# index: 2
# output: [8, 2, 3, 6, 1, 9]
# output: {'t': 3, 'h': 1, 'i': 2, 's': 3, 'e': 1}
# Given a string, find the frequency of each character in this string
# input: thisistest
# output: {'t': 3, 'h': 1, 'i': 2, 's': 3, 'e': 1}

# Given an integer arary, find the maximum difference between two elements such that larger element appears after the smaller number (index matters)
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
# given an integer array and a target number, return indices of the two numbers such that they add up to this target.
# input: [3,2,5,8,-1, 0], 5
# return: [(0,1), (2,5)]
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

# Given a string, calculate the frequency of each character
# input: "thistest"
# output: {"t": 3, "h": 1, "i": 1..... }
def count_frequency(input_s):
    return
# Given a string, sort it in decreasing order based on the frequency of characters.
# if the frequency is same, any order of characters is good
def frequencySort(s):
    import operator
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

# given l1 and l2 as two sorted list, merge them and return it, Requirment: DO NOT use array's sort function
# input: l1=[1,3,5,7,9], l2=[2,6]
# output: l3 = [1,2,3,5,6,7,9]
def mergeTwoSortedList(l1, l2):
    import sys
    m = len(l1)
    n = len(l2)
    l1.extend([-sys.maxsize - 1] * n) # extend the result list so that it can hold all nums
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


# function to transfer string x to int x if x is numeric string
def tryint(x):
    try:
        return int(x)
    except ValueError:
        return x

# function to split a string to characters
# this function handle the case that string like this: T920X23.2d3.23dd98
def splitStringToCharacters(s):
    import re
    return tuple(tryint(x) for x in re.split('([0-9]+)', s))

# What key does is it provides a way to specify a function that returns what you would like your items sorted by.
# The function gets an "invisible" argument passed to it that represents an item in the list,
# and returns a value that you would like to be the item's "key" for sorting.

# function to sort a list of names, use splitStringToCharacters as a key function to sorted
# such as input: strings = ['YT4.11', '4.3', 'YT4.2', '4.10', 'PT2.19', 'PT2.9']
# out: ['4.3', '4.10', 'PT2.9', 'PT2.19', 'YT4.2', 'YT4.11']
def stringListSorted(strings):
    return sorted(strings, key=splitStringToCharacters)

student_tuples = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10),]
sorted(student_tuples, key=lambda student: student[2]) # [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]

from operator import itemgetter, attrgetter, methodcaller
sorted(student_tuples, key=itemgetter(2)) # [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
sorted(student_tuples, key=itemgetter(2), reverse=True) #[('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(student_objs, key=attrgetter('age')) # [('Dave', 'B', 10), ('John', 'B', 15), ('Kevin', 'A', 16)]
sorted(student_objs, key=methodcaller('weighted_grade'))
sorted(student_objs, key=getKey)
sorted(student_objs)