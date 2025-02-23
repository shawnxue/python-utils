"""
A cafeteria table consists of a row of N seats, numbered from 1 to N from left to right. Social distancing guidelines require that every diner be seated such that
K seats to their left and K seats to their right (or all the remaining seats to that side if there are fewer than K) remain empty.
There are currently M diners seated at the table, the ith of whom is in seat S_i
​no two diners are sitting in the same seat, and the social distancing guidelines are satisfied.
Determine the maximum number of additional diners who can potentially sit at the table without social distancing guidelines being violated for any new or existing diners,
assuming that the existing diners cannot move and that the additional diners will cooperate to maximize how many of them can sit down.
Please take care to write a solution which runs within the time limit.
Constraints
1 \le N \le 10^{15}1≤N≤10
15

1 \le K \le N1≤K≤N
1 \le M \le 500{,}0001≤M≤500,000
M \le NM≤N
1 \le S_i \le N1≤S
i≤N

N = 10
K = 1
M = 2
S = [2, 6]
Expected Return Value = 3

N = 15
K = 2
M = 3
S = [11, 6, 14]
Expected Return Value = 1

"""

from typing import List
import math
def getMaxAdditionalDinersCountMine(N: int, K: int, M: int, S: List[int]) -> int:
  # [0,0,1,0,0,0,1,0,0,0]
  capacity = [0] * N
  if len(S) != M:
    print(f"input error {0}, {1}".format(M, S))
  for n in S:
    capacity[n] = 1
  extra_student = 0
  for index, status in enumerate(capacity):
    if index == 0:
      pass
    elif index == N - 1:
      pass
    if status == 0 and (index - K) < 0 and (index + K) > (N - 1):
      extra_student += 1
      capacity[index] = 1
    elif status == 0 and () and (index - K) > 0 and capacity(index - K) == 0 and ():
      pass
    else:
      continue
  return extra_student

def getMaxAdditionDinersCount(N: int, K: int, M: int, S: List[int]) -> int:
  # sort to make sure all the seat assignments go from left to right
  S.sort()
  start_index, result = 1, 0
  # add in N + K + 1 since we want a possibility of placing someone on the last available seat
  S.append(N + K + 1)
  for s in S:
      # delta: gap from current seat to left of seat up to start_index
      # K+1: how many units of single gap of size K+1 (this gap can be thought of as one unit gap)
      delta = s - K - start_index
      if delta > 0:
          result += math.ceil(delta / (K+1))
      # now after we found out stuff for current seat lets go K + 1 seats to the right since that is the minimum distancing needed
      start_index = s + K + 1
  return result

"""
Note: Chapter 2 is a harder version of this puzzle. The only difference is a larger constraint on N.
A photography set consists of N cells in a row, numbered from 1 to N in order, and can be represented by a string C of length N.
Each cell ii is one of the following types (indicated by ith character of C):
If C_i = “P”, it is allowed to contain a photographer
If C_i = “A”, it is allowed to contain an actor
If C_i = “B”, it is allowed to contain a backdrop
If C_i = “.”, it must be left empty
A photograph consists of a photographer, an actor, and a backdrop, such that each of them is placed in a valid cell, and such that the actor is between the
photographer and the backdrop. Such a photograph is considered artistic if the distance between the photographer and the actor is between X and Y cells (inclusive),
and the distance between the actor and the backdrop is also between X and Y cells (inclusive). The distance between cells i and j is |i - j| (the absolute value of
the difference between their indices). Determine the number of different artistic photographs which could potentially be taken at the set.
Two photographs are considered different if they involve a different photographer cell, actor cell, and/or backdrop cell.

Constraints 1 <= N <= 200
1<=X<=Y<=N

N = 5
C = APABA
X = 1
Y = 2
Expected Return Value = 1

N = 5
C = APABA
X = 2
Y = 3
Expected Return Value = 0

N = 8
C = .PBAAP.B
X = 1
Y = 3
Expected Return Value = 3

Sample Explanation
In the first case, the absolute distances between photographer/actor and actor/backdrop must be between 1 and 2. The only possible photograph that can be taken
is with the 3 middle cells, and it happens to be artistic.
In the second case, the only possible photograph is again taken with the 3 middle cells. However, as the distance requirement is between 2 and 3, it is not
possible to take an artistic photograph.
In the third case, there are 4 possible photographs, illustrated as follows:
.P.A...B
.P..A..B
..BA.P..
..B.AP..
All are artistic except the first, where the artist and backdrop exceed the maximum distance of 3.
"""
def getArtisticPhotographCount(N: int, C: str, X: int, Y: int) -> int:
  count, lP, rP, lB, rB = 0, 0, 0, 0, 0
  if C[0]=="B": lB = 1
  if C[0]=="P": lP = 1
  for h in range(2*X, min(N, X+Y+1)):
    if C[h]=="B": rB += 1
    if C[h]=="P": rP += 1
  if X<N and C[X] == "A": count = lB*rP + rB*lP
  for i in range(X+1,N-X):
    if i>Y and C[i-Y-1] == "B": lB -= 1
    if i>Y and C[i-Y-1] == "P": lP -= 1
    if C[i-X] == "B": lB += 1
    if C[i-X] == "P": lP += 1
    if i+Y<N and C[i+Y] == "B": rB += 1
    if i+Y<N and C[i+Y] == "P": rP += 1
    if C[i+X-1] == "B": rB -= 1
    if C[i+X-1] == "P": rP -= 1
    if C[i] == "A": count += lB*rP + rB*lP
  return count


"""
There are N dishes in a row on a kaiten belt, with the ith dish being of type D_i. Some dishes may be of the same type as one another.
You're very hungry, but you'd also like to keep things interesting. The N dishes will arrive in front of you, one after another in order, and for each one
you'll eat it as long as it isn't the same type as any of the previous K dishes you've eaten. You eat very fast, so you can consume a dish
before the next one gets to you. Any dishes you choose not to eat as they pass will be eaten by others.
Determine how many dishes you'll end up eating.

Constraints
1≤N≤500,000
1≤K≤N
1≤D≤1,000,000

N = 6
D = [1, 2, 3, 3, 2, 1]
K = 1
Expected Return Value = 5

N = 6
D = [1, 2, 3, 3, 2, 1]
K = 2
Expected Return Value = 4

N = 7
D = [1, 2, 1, 2, 1, 2, 1]
K = 2
Expected Return Value = 2

Sample Explanation
In the first case, the dishes have types of [1, 2, 3, 3, 2, 1], so you'll eat the first 3 dishes, skip the next one as it's another type-3 dish, and then eat the last 2.
In the second case, you won't eat a dish if it has the same type as either of the previous 2 dishes you've eaten. After eating the first, second, and third dishes,
you'll skip the fourth and fifth dishes as they're the same type as the last 2 dishes that you've eaten. You'll then eat the last dish, consuming 4 dishes total.
In the third case, once you eat the first two dishes you won't eat any of the remaining dishes.
"""
def getMaximumEatenDishCount(N: int, D: List[int], K: int) -> int:
  count = 0
  pre_dish = [0] * K
  for n in D:
    if n not in pre_dish:
      if len(pre_dish) == K:
        pre_dish.pop(0)
      pre_dish.append(n)
      count += 1
  return count

"""
You're trying to open a lock. The lock comes with a wheel which has the integers from 1 to N arranged in a circle in order around it (with integers 1 and N adjacent to one another).
The wheel is initially pointing at 1.For example, the following depicts the lock for N = 10N=10 (as is presented in the second sample case).

It takes 1 second to rotate the wheel by 11 unit to an adjacent integer in either direction, and it takes no time to select an integer once the wheel is pointing at it.
The lock will open if you enter a certain code. The code consists of a sequence of M integers, the ith of which is C_i. Determine the minimum number of seconds required
to select all M of the code's integers in order.

Constraints
3≤N≤50,000,000
1≤M≤1,000
1≤C≤N

N = 3
M = 3
C = [1, 2, 3]
Expected Return Value = 2

N = 10
M = 4
C = [9, 4, 4, 8]
Expected Return Value = 11
"""
def getMinCodeEntryTime(N: int, M: int, C:List[int]) -> int:
  min_entry_time = 0
  if len(C) != M:
    print(f"inputs M {0} and C {1} are not valid".format(M, C))
    return 0
  lock = []
  for i in range(1, N+1):
    lock.append(i)

  # another way to handle index: we don't need array lock, we can get the index by num - 1
  current_index = 0
  for num in C:
    num_index = lock.index(num)
    if current_index == num_index:
      continue
    diff_1 = abs(current_index - num_index)
    diff_2 = N - diff_1
    min_entry_time = min_entry_time + min(diff_1, diff_2)
    current_index = num_index
  return min_entry_time

"""
You are spectating a programming contest with N competitors, each trying to independently solve the same set of programming problems. Each problem has a point value, which is either 1 or 2.
On the scoreboard, you observe that the ith competitor has attained a score of S_i, which is a positive integer equal to the sum of the point values of all the problems they have solved.
The scoreboard does not display the number of problems in the contest, nor their point values. Using the information available, you would like to determine the minimum possible number of
problems in the contest.

Constraints
1≤N≤500,000
1≤S_i≤1,000,000,000

N = 6
S = [1, 2, 3, 4, 5, 6]
Expected Return Value = 4

N = 4
S = [4, 3, 3, 4]
Expected Return Value = 3

N = 4
S = [2, 4, 6, 8]
Expected Return Value = 4

In the first case, it's possible that there are as few as 4 problems in the contest, for example with point values [1,1,2,2]. The 6 competitors could have solved the following
subsets of problems: Problem 1 (1 point), Problem 3 (2 points), Problems 2 and 33 (1 + 2 = 3 points), Problems 1, 2, and 4 (1 + 1 + 2 = 4 points),
Problems 2, 3, and 4 (1 + 2 + 2 = 5points), All 4 problems (1 + 1 + 2 + 2 = 6 points)
It is impossible for all 6 competitors to have achieved their scores if there are fewer than 4 problems.
In the second case, one optimal set of point values is [1, 1, 2]
In the third case, one optimal set of point values is [2, 2, 2, 2]
"""

def getMinProblemCount(N: int, S: List[int]) -> int:
  is_odd, min_problem_count = 0, 0
  for score in S:
    is_odd = max(is_odd, score % 2)
    min_problem_count = max(min_problem_count, score / 2)
  return int(is_odd + min_problem_count)

# Given a string, find the length of the longest substring without repeating characters.
def longest_substring_no_repeat_char(s):
  ans = ""
  length = 0
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
def longestSubstringNoLessThanK(s, k):
  for c in set(s):
    print(f"s={s}, c={c}")
    if s.count(c) < k:
      return max(longestSubstringNoLessThanK(item, k) for item in s.split(c))
  return len(s), s

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
        length = max(length, i - my_stack[len(my_stack)-1]) # or my_stack[-1]
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
  return length, my_stack

"""
Stack Stabilization (Chapter 1)
There's a stack of N inflatable discs, with the ith disc from the top having an initial radius of R_i inches.
The stack is considered unstable if it includes at least one disc whose radius is larger than or equal to that of the disc directly under it. In other words,
for the stack to be stable, each disc must have a strictly smaller radius than that of the disc directly under it.
As long as the stack is unstable, you can repeatedly choose any disc of your choice and deflate it down to have a radius of your choice which is strictly
smaller than the disc’s prior radius. The new radius must be a positive integer number of inches.
Determine the minimum number of discs which need to be deflated in order to make the stack stable, if this is possible at all.
If it is impossible to stabilize the stack, return -1 instead.

Constraints
1≤N≤50
1≤R≤1,000,000,000

N = 5
R = [2, 5, 3, 6, 5]
Expected Return Value = 3

N = 3
R = [100, 100, 100]
Expected Return Value = 2

N = 4
R = [6, 5, 4, 3]
Expected Return Value = -1

In the first case, the discs (from top to bottom) have radii of [2, 5, 3, 6, 5]. One optimal way to stabilize the stack is by deflating disc 1 from 2 to 1,
deflating disc 2 from 5 to 2", and deflating disc 4 from 6 to 4. This yields final radii of [1, 2, 3, 4, 5].
In the second case, one optimal way to stabilize the stack is by deflating disc 1 from 100 to 1 and disc 2 from 100 to 10.
In the third case, it is impossible to make the stack stable after any number of deflations
"""
def getMinimumDeflatedDiscCount(N: int, R: List[int]) -> int:
  return 0

"""
A positive integer is considered uniform if all of its digits are equal. For example, 222 is uniform, while 223 is not.
Given two positive integers AA and BB, determine the number of uniform integers between AA and BB, inclusive.
Please take care to write a solution which runs within the time limit.

Constraints
1≤A≤B≤10**12

A = 75
B = 300
Expected Return Value = 5

A = 1
B = 9
Expected Return Value = 9

A = 999999999999
B = 999999999999
Expected Return Value = 1

In the first case, the uniform integers between 75 and 300 are 77, 88, 99, 111, and 2222.
In the second case, all 99 single-digit integers between 1 and 9 (inclusive) are uniform.
In the third case, the single integer under consideration (999999999999) is uniform
"""
def getUniformIntegerCountInInterval(A: int, B:int) -> int:
  def isUniform(C: int) -> bool:
    digits = list()
    while C > 0:
      digit = C % 10
      digits.append(digit)
      C = int(C / 10) # or C // 10
    current_digit = digits[0]
    for d in digits:
      if d != current_digit:
        return False
    return True
  result = 0
  while (A<=B):
    if(isUniform(A)):
      result += 1
    A = A + 1
  return result

def lastRemaining(n):
  nums = [num for num in range(1, n+1)]
  while len(nums) > 1:
    nums = nums[1::2][::-1] # from left, return num starting from index 1 to the end, and step is 2, then ::-1 reverse the list
  return nums[0]

def eraseOverlapIntervals(intervals: List[List[int]]):
  import sys
  count, removed_items = 0, list()
  input = sorted(intervals, key = lambda item:item[1])
  pre_item_end = -sys.maxsize - 1
  for interval in input:
    if interval[0] >= pre_item_end:
      pre_item_end = interval[1]
    else:
      count += 1
      removed_items.append(interval)
  return count, removed_items