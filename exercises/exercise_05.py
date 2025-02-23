from typing import List
def get_min_stickers_1(target: str, stickers: List[str]) -> int:
  stickerCount = []
  for i, s in enumerate(stickers):
    stickerCount.append({})
    for c in s:
      stickerCount[i][c] = 1 + stickerCount[i].get(c, 0)
  print(f"stickerCount={stickerCount}")
  dp = {}
  def dfs(t, stick):
    print(f"target={t}, sticker={stick}")
    if t in dp:
      return dp[t]
    res = 1 if stick else 0
    print(f"sticker={stick}, result={res}, dp={dp}")
    remainT = ""
    for c in t:
      if c in stick and stick[c] > 0:
        stick[c] -= 1
      else:
        remainT += c
    used = float("inf")
    if remainT:
      for s in stickerCount:
        if remainT[0] not in s:
          continue
        used = min(used, dfs(remainT, s.copy()))
      print(f"sticker={s}, used={used}, dp={dp}")
      res += used
      dp[remainT] = used
    return res      

  res = dfs(target, {})
  return res if res != float("inf") else -1

# print(get_min_stickers_1('fnemmdgex', ['facebook', 'expedia', 'google', 'amazon']))

"""
Answer a Query
Imagine a length-N array of booleans, initially all false. Over time, some values are set to true, and at various points in time you would like to find the location of 
the nearest true to the right of given indices.
You will receive Q queries, each of which has a type and a value. SET queries have type = 1 and GET queries have type = 2.
When you receive a SET query, the value of the query denotes an index in the array that is set to true. Note that these indices start at 1. 
When you receive a GET query, you must return the smallest index that contains a true value that is greater than or equal to the given index, or -1 if no such index exists.
Signature
int[] answerQueries(ArrayList<Query> queries, int N)
Input
A list of Q queries, formatted as [type, index] where type is either 1 or 2, and index is <= N
1 <= N <= 1,000,000,000
1 <= Q <= 500,000
Output
Return an array containing the results of all GET queries. The result of queries[i] is the smallest index that contains a true value that is greater than or equal to i, or -1 if no index satisfies those conditions.
Example
N = 5
Q = 5
queries = [[2, 3], [1, 2], [2, 1], [2, 3], [2, 2]]
output = [-1, 2, -1, 2]
The initial state of the array is [false, false, false, false, false].
The first query is GET 3, but no values in the array are true, so the answer is -1.
The second query is SET 2, so the value at index 2 is set to true. The new state of the array is [false, true, false, false, false].
The third query is GET 1, and the index of the true value nearest to 1 (to the right) is 2.
The fourth query is GET 3, but no values to the right of index 3 are true, so answer is -1
The fifth query is GET 2, and the value at index 2 is true.
"""
def answerQueries(N: int, queries: List[List[int]]) -> List[int]:
  if N < 1:
    return [-1]
  answers = list()
  states = [False for i in range(N)]
  for query in queries:
    # set operation
    if query[0] == 1:
      if query[1] > N:
        raise ValueError
      else:
        states[query[1] - 1] = True
    # get operation, set answers array
    elif query[0] == 2:
      if query[1] > N:
        raise ValueError
      else:
        index = query[1] - 1
        while (index < N) and (not states[index]):
          index += 1
        if index < N:
          answers.append(index + 1)
        else:
          answers.append(-1)
    else:
      raise ValueError
  return answers
# print(answerQueries(5, [[2, 3], [1, 2], [2, 1], [2, 3], [2, 2]]))
# print(answerQueries(0, [[2, 3], [1, 2], [2, 1], [2, 3], [2, 2]]))
# print(answerQueries(-1, [[2, 3], [1, 2], [2, 1], [2, 3], [2, 2]]))
# print(answerQueries(100000, [[2, 3], [1, 2], [2, 1], [2, 3], [2, 2]]))
# print(answerQueries(5, [[2, 3]]))
# print(answerQueries(5, [[1, 3]]))
"""
Above-Average Subarrays
You are given an array A containing N integers. Your task is to find all subarrays whose average sum is greater than the average sum of the remaining array elements. 
You must return the start and end index of each subarray in sorted order.
A subarray that starts at position L1 and ends at position R1 comes before a subarray that starts at L2 and ends at R2 if L1 < L2, or if L1 = L2 and R1 ≤ R2.
Note that we'll define the average sum of an empty array to be 0, and we'll define the indicies of the array (for the purpose of output) to be 1 through N. 
A subarray that contains a single element will have L1 = R1.
Signature
Subarray[] aboveAverageSubarrays(int[] A)
Input
1 ≤ N ≤ 2,000
1 ≤ A[i] ≤ 1,000,000
Output
A Subarray is an object with two integer fields, left and right, defining the range that a given subarray covers. Return a list of all above-average subarrays 
sorted as explained above.
Example 1
A = [3, 4, 2]
output = [[1, 2], [1, 3], [2, 2]]
The above-average subarrays are [3, 4], [3, 4, 2], and [4].
"""
def aboveAverageSubarrays(A: List[int]) -> List[List[int]]:
  total = 0
  for n in A:
    total += n
  def get_average(start_index, end_index):
    if start_index > end_index:
      return 0
    sub_total = 0
    for i in range(start_index, end_index+1):
      sub_total = sub_total + A[i]
    remain = total - sub_total
    count = end_index - start_index + 1
    avg_1 = int(sub_total/count)
    avg_2 = 0 if len(A) == count else int(remain/(len(A) - count))
    return avg_1, avg_2

  output = []
  for i in range(len(A)):
    for j in range(i, len(A)):
      sub_avg, remain_avg = get_average(i, j)
      if sub_avg > remain_avg:
        output.append([i+1, j+1])
  return output

print(aboveAverageSubarrays([3, 4, 2]))
