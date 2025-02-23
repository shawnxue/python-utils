"""
Game Scoring
Imagine a game where the player can score 1, 2, or 3 points depending on the move they make. Write a function or functions, 
that for a given final score computes every combination of points that a player could score to achieve the specified score in the game.
Signature
int[][] gameScoring(int score)
Input
Integer score representing the desired score
Output
Array of sorted integer arrays demonstrating the combinations of points that can sum to the target score
Example 1:
Input: 
score = 4
Output: 
[ [ 1, 1, 1, 1 ], [ 1, 1, 2 ], [ 1, 2, 1 ], [ 1, 3 ], [ 2, 1, 1 ], [ 2, 2 ], [ 3, 1 ] ]
Example 2:
Input: 
score = 5
Output:
[ [ 1, 1, 1, 1, 1 ], [1, 1, 1, 2 ], [ 1, 1, 2, 1 ], [ 1, 1, 3 ], [ 1, 2, 1, 1 ], [ 1, 2, 2 ], [ 1, 3, 1 ], [ 2, 1, 1, 1 ], [ 2, 1, 2 ], [ 2, 2, 1 ], [ 2, 3 ], 
[ 3, 1, 1 ], [ 3, 2 ] ]
"""
import math
from typing import List
def game_scoring(point: int, ways_to_score=[1,2,3]) -> List[List[int]]:
  result = []
  def score_finder(point, scores, result):
    if point == 0:
      result.append(scores[:])
    elif point > 0:
      for score in ways_to_score:
        scores.append(score)
        print(f"after append secores={scores}")
        score_finder(point-score, scores, result)
        # pop the last score added, so that we can add new one
        scores.pop()
        print(f"after pop secores={scores}")
    else:
      # point is less than 0 because the new score is too big
      pass

    return result
  
  return score_finder(point, [], [])

def find_scoring(points, ways_to_score=[2, 3, 7]):
  def score_finder(points, scores, result):
    if points == 0:
      result.append(scores[:])
    elif points > 0:
      for val in ways_to_score:
        scores.append(val)
        print(f"after append secores={scores}")
        score_finder(points - val, scores, result)
        scores.pop()
        print(f"after pop secores={scores}")

    return result

  return score_finder(points, [], [])
# print(game_scoring(4))
# print(game_scoring(5))


# These are the tests we use to determine if the solution is correct.
# You can add your own at the bottom.
"""
test_case_number = 1

def check(expected, output):
  global test_case_number
  result = True
  if len(expected) == len(output):
    for score in expected:
      result = result & (score in output)
    for score in output:
      result = result & (score in expected)
  else:
    result = False
  rightTick = '\u2713'
  wrongTick = '\u2717'
  if result:
    print(rightTick, ' Test #', test_case_number, sep='')
  else:
    print(wrongTick, ' Test #', test_case_number, ': Expected ', sep='', end='')
    print(expected)
    print(' Your output: ', end='')
    print(output)
    print()
  test_case_number += 1

if __name__ == "__main__":
  test_1 = 4
  expected_1 = [
  [1, 1, 1, 1],
  [1, 1, 2],
  [1, 2, 1],
  [1, 3],
  [2, 1, 1],
  [2, 2],
  [3, 1]
   ]
  output_1 = game_scoring(test_1)
  check(expected_1, output_1)

  test_2 = 5
  expected_2 = [
  [1, 1, 1, 1, 1],
  [1, 1, 1, 2],
  [1, 1, 2, 1],
  [1, 1, 3],
  [1, 2, 1, 1],
  [1, 2, 2],
  [1, 3, 1],
  [2, 1, 1, 1],
  [2, 1, 2],
  [2, 2, 1],
  [2, 3],
  [3, 1, 1],
  [3, 2],
    ]
  output_2 = game_scoring(test_2)
  check(expected_2, output_2)
"""
  # Add your own test cases here

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
      num1_s = list(diff_dict.keys())[list(diff_dict.values()).index(num)] # in python 3, you have to convert dict.values() to list
      num1 = int(num1_s)
      result.append((nums.index(num1), index))
    else:
      diff_dict[str(num)] = target - num
  if len(result) > 0:
    print(f"target={target}, nums={nums}")
    return result
  else:
    print(f"target={target}, nums={nums}")
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

# print(threeSum([-25, -10, -7, -3, 2, 4, 8, 10], -20))
# print(threeSum([-25, -10, -7, -3, 2, 4, 8, 10], -25))
# print(threeSum([-25, -10, -7, -3, 2, 4, 8, 10], -0))
# print(threeSum([-25, -10, -7, -3, 2, 4, 8, 10], -42))
# print(threeSum([-25, -10, -7, -3, 2, 4, 8, 10], 22))
# print(threeSum([-25, -10, -7, -3, 2, 4, 8, 10], 7))

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

def mergeOverlapIntervals(intervals):
  import sys
  removed_items = list()
  input = sorted(intervals, key = lambda  item: item[1])
  pre_item_end =  -sys.maxsize - 1 # sys.maxsize is the max int in Python 3
  for idx, interval in enumerate(input):
    if interval[0] >= pre_item_end:
      pre_item_end = interval[1]
    else:
      removed_items.append(interval)
  for item in removed_items:
    input.remove(item)
  return [input[0][0], input[-1][1]]

def mergeOverlapIntervals_2(intervals):
  result = list()
  result.append(intervals[0][0])
  result.append(intervals[0][1])
  for idx in range(1, len(intervals)):
    current_x = intervals[idx][0]
    current_y = intervals[idx][1]
    if result[-1] > current_x:
      result[-1] = max(result[-1], current_y)
    else:
      continue
  return result

print(mergeOverlapIntervals([(1, 5), (3, 7), (6, 8), (4, 6)]))
print(mergeOverlapIntervals_2([(1, 5), (3, 7), (6, 8), (4, 6)]))
print(mergeOverlapIntervals([(1, 5), (6, 8), (3, 7), (4, 6)]))
print(mergeOverlapIntervals([(15, 12), (10, 12)]))
print(mergeOverlapIntervals([(12, 15), (10, 12)]))
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

# l1 and l2 are linked list, each node is an integer
# add them together and return the new list. New node's value is the reminder: (l1.node + l2.node + carry) % 10
class Node:
  def __init__(self, data=None, next=None):
    self.data = data
    self.next  = next
    
def addLinkedListInteger(l1, l2):
  result = None
  cur_node = None
  carry = 0 # carry is the integer part of sum devided by 10
  while (l1 is not None or l2 is not None or carry > 0):
    first = (0 if l1 is None else l1.data)
    second = (0 if l2 is None else l2.data)
    sum = first + second + carry
    new_node = Node(sum % 10)
    carry = sum // 10
    if result is None:
      result = new_node
    else:
      cur_node.next = new_node
    cur_node = new_node
    if l1 is not None:
      l1 = l1.next
    if l2 is not None:
      l2 = l2.next

  return result

