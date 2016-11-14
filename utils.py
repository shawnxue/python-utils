# function to check if a string is dui chen
import re
import getopt
import random

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
    print "input string is {0}".format(input)
    sorted_s = "abcdefghijklmnopqrstuvwxyz" # or import string, sorted_s = string.ascii_lowercase
    pre = ""
    for i in range(len(input)):
        if pre == input[i]: # this is a repeated character
            continue
        if input[i] in sorted_s:
            pre = input[i]
            index = sorted_s.index(input[i])
            sorted_s = sorted_s[index+1:-1]
        else:
            return input[i] # this is the first non alphabetic order character in the string

# function to check if a number is armstrong number, assume input number is valid
def isArmstrong(n):
    # TODO: exception handle
    sum = 0
    input = n
    while n > 0:
        digit = n % 10
        sum = sum + digit ** len(str(input))
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

# remove characters of string r from a string s
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

# function to transpose rows and columns in a matrix
# assume it's n row and m column matrix, each row has same column number
def transposeMatrix(matrix):
    m = len(matrix[0]) # get the column number of this matrix
    transposed = []
    # way 1:
    for i in range(m):
            transposed.append([row[i] for row in matrix])
    return transposed
    # way 2
    for i in range(m):
        transposed_row = []
        for row in matrix:
            transposed_row.append(row[i])
        transposed.append(transposed_row)

    # way 3
    return [[row[i] for row in matrix] for i in range(m)]

def convertDecimalToBinary(n):
    """Function to print binary number for the input decimal using recursion"""
    if n > 1:
        convertDecimalToBinary(n//2)
    #print(n % 2, end = '')


def twoSum(nums, target):
    """
    function to return indices of the two numbers such that they add up to a specific target.
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    if len(nums) < 2:
        return [-1,-1]
    diff_dict = {} # key is num, value is the difference between target and this key
    for index, num in enumerate(nums):
        if num in diff_dict.values():
            # find key by value from the dictionary
            num1_s = diff_dict.keys()[diff_dict.values().index(num)]
            num1 = int(num1_s)
            return [nums.index(num1), index]
        else:
            diff_dict[str(num)] = target - num
    return [-1, -1]


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
        bin_s = str(bin(i)[2:])
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

# return True if string is Palindrome
def isPalindrome(s):
    return s.lower() == "".join(reversed(s.lower())) # or s == s[::-1]

def longestPalindrome(s):
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
        return(recur_fibo(n-1) + recur_fibo(n-2))

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
def decodeString2(self, s):
    found = re.search("(\d+)\[([a-z]+)\]",s)
    return decodeString2(s[:found.start()]+ found.group(2)*int(found.group(1))+s[found.end():]) if found else s

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

# function to find an item in an ordered list nums
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

# Given a string containing just the characters '(' and ')',
# find the length of the longest valid (well-formed) parentheses substring.
# eg. input: ")()())", out put 4  because the string is "()()"
def longestValidParentheses(s):
    my_stack = []
    left = {} # save the ')' matching most left '(' success, both use index
    length = 0
    for i, c in enumerate(s):
        if c == "(":
            my_stack.append(i) # push index
        elif c == ")" and len(my_stack):
            l = my_stack.pop()
            r = i
            if left.has_key(l-1):
                left[r] = left[l-1]
            else:
                left[r] = l
            length = r - left[r] + 1
        else:
            continue
    return length

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

def findMedianSortedArrays(self, nums1, nums2):
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
def findPermutations(s)
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

def parseParametersFromCmdline(proxyConfig, log):
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
            if (a.lower() == "yes" or a.lower() == "no"):
                proxyConfig.keyPreGen = a.lower()
                argCounter = argCounter + 1
        elif o == "--sslkey":
            proxyConfig.sslkey = a
            argCounter = argCounter + 1
        elif o == "--sslcertreq":
            proxyConfig.sslcertreq = a
            argCounter = argCounter + 1
        elif o in ("-h", "--help"):
            print "help message"

    if (argCounter != requiredArgs and argCounter != 0):
        print 'help message'

    pass

# if this module is called directly like this: python <file name>
if __name__ == "__main__":
    pass
