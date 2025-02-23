from typing import List
# Facebook logo stickers cost $3 each from the company store. (They used to be $2, but are now more expensive on account of
# a recent company name change.) I have an idea. I want to cut up the stickers, and use the letters to make other words/phrases.
# A Facebook logo sticker contains only the word 'facebook', in all lower-case letters.

# Write a function that takes a string and returns an integer with the number of Facebook logo stickers I will need to buy to make the string.

# get_num_stickers('book') -> 1
# get_num_stickers('cafe') -> 1
# get_num_stickers('coffee') -> 2
# get_num_stickers('coffee kebab') -> 3

# get_num_stickers('o') -> 1

def get_min_stickers(s, sticker='facebook'):
    import math, re
    s = re.sub(r'\s+', '', s)
    min_sticker_count = 0
    count_in_sticker = {}.fromkeys(set(sticker),0)
    for c in sticker:
        count_in_sticker[c] += 1 
    count_in_s = {}.fromkeys(set(s), 0)
    for c in s:
        count_in_s[c] += 1
    for char, count in count_in_s.items():
        if char not in sticker:
            return -1
        else:
            num = math.ceil(count / count_in_sticker[char])
            if num > min_sticker_count:
                min_sticker_count = num

    return min_sticker_count
# s1, s2, s3, s4, s5, s6, s7 = 'book', 'cafe', 'coffee', 'coffee kebab', 'facebook1', 'google', 'o'
# print(get_min_stickers(s1))
# print(get_min_stickers(s2))
# print(get_min_stickers(s3))
# print(get_min_stickers(s4))
# print(get_min_stickers(s5))
# print(get_min_stickers(s6))
# print(get_min_stickers(s7))

# Facebook logo stickers only let you build so many phrases. I want to
# create much more complicated phrases, and I have the stickers of other
# companies to help me get there!

# Write a function that takes a target string and a list of sticker strings
# and returns a dictionary/map where the key is the sticker and the value is
# the number of that sticker you need to buy to build the target string.

# I’m fine with any solution such that subtracting 1 from any value doesn’t
# also produce a valid solution. In other words, I'm fine with any solution
# such that you can't achieve a different solution by only returning
# stickers (not buying new ones).

# Valid solutions:
# get_num_stickers_with_inventory('coffee kebab', ['facebook']) -> {'facebook': 3}
# get_num_stickers_with_inventory('fedex', ['facebook’, 'expedia']) -> {'facebook': 1, 'expedia': 1}
# get_num_stickers_with_inventory('ozone', ['google', 'amazon']) -> {'google': 1, 'amazon': 1}

# Invalid solutions:
# get_num_stickers_with_inventory('fedex', ['facebook', 'expedia']) -> {'facebook': 2, 'expedia': 1}
# get_num_stickers_with_inventory('ozone', ['google', 'amazon']) -> {'google': 1, 'amazon': 2}

# Unsolvables:
# get_num_stickers_with_inventory('google', ['facebook'])

# {'<some_sticker_name>': 0}

# number of letters in target string
# for each sticker
#   number of letters in sticker
#   max of division per character
#   subtract letters given by sticker
# for each sticker
#   number of leftover letters in target string
#   is you add one sticker, are all numbers still negative


def get_min_stickers(s, stickers):
    import math, re
    s = re.sub(r'\s+', '', s)
    count_in_s = {}.fromkeys(set(s), 0)
    for c in s:
        count_in_s[c] += 1

    min_sticker_count = {}.fromkeys(stickers, -1) # key is sticker, value is the number of stick
    left_over_in_s =count_in_s.copy()
    for sticker in stickers:
        count_in_sticker = {}.fromkeys(set(sticker),0)
        for c in sticker:
            count_in_sticker[c] += 1

        for char, count in count_in_s.items():
            if char not in sticker:
                min_sticker_count[sticker] = -1
            else:
                num = math.ceil(count / count_in_sticker[char])
                if num > min_sticker_count[sticker]:
                   min_sticker_count[sticker] = num


    return min_sticker_count

from collections import Counter
from functools import lru_cache
class Solution_1:
    def minStickers(self, stickers: List[str], target: str) -> int:
        n = len(stickers)
        # Convert words to frequency counter
        stickers = [Counter(sticker) for sticker in stickers]
        @lru_cache(None)
        def dp(s):
            if s=='': return 0
            cs = Counter(s)
            ans = float('inf')
            for sticker in stickers:
                # Skip if the first character of the required string is not present
                if s[0] not in sticker: continue
                # Concatenating the elemented with frequnecy after the 
                # frequency reduction due to stickers[i] in string s
                # and conducting dp for that
                ans = min(ans,1+dp(''.join(ch*ct for ch,ct in sorted((cs-sticker).items()))))
            return ans
            
        val =  dp(''.join(sorted(target)))
        # return -1 if a valid answer does not exist
        return -1 if val==float('inf') else val

class Solution_2:
    def minStickers(self, stickers: List[str], target: str) -> int:
        # idea is to maintain state of remaining target string s
		# value for s is the min number of stickers used to reach s
        # objective is to reduce s to empty
        # choosing every sticker from valid stickers to try and then simply replacing s using the chars from sticker, wil guarantee correctness
		# let t,w be size of target and sticker respectively
		# naive
        # height is t, worst case one sticker for one char in target
        # branches is n, worst case can choose any sticker
        # each node is t*w time
        # O(n^t * tw)
		# memo
        # we can hit cache if target is seen before
        # worst case is one sticker removes one char only
        # worst case TC is number of subsequences from empty to target, worst case 2^n if all chars are unique
        # O(2^t) TC and SC
        
        def addSticker(sticker,w): # O(t*w)
            for c in sticker: # O(w)
                w = w.replace(c,'',sticker[c]) # replcae first sticker[c] counts, O(t)
            return w
        
        @lru_cache(None)
        def dfs(s): # returns min for this string
            if not s: # base case no stickers needed if target is empty
                return 0
            res = float('inf')
            for sticker in stickers: # O(n)
                if s[0] not in sticker: continue # add this sticker only if it contains char,O(1)
                w = addSticker(sticker,s) # O(t*w)
                res = min(res,1 + dfs(w)) # cost to use this sticker is 1
            return res
        
        stickers = [Counter(s) for s in stickers]
        res = dfs(target) 
        return res if res!=float('inf') else -1

class Solution_3:
    def minStickers(self, stickers: List[str], target: str) -> int:
        stickerCount = []
        for i, s in enumerate(stickers):
            stickerCount.append({})
            for c in s:
                stickerCount[i][c] = 1 + stickerCout[i].get(c, 0)
        
        dp = {}
        def dfs(t, stick):
            if t in dp:
                return dp[t]
            res = 1 if stick else 0
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
                res += used
                dp[remainT] = used
            return res      
            
        res = dfs(target, {})
        return res if res != float("inf") else -1

print(get_min_stickers('fedex', ['facebook', 'expedia']))

def rotate(nums: list[int], k: int) -> None:
   # given list, pop last element k times, inserting at front
   while k > 0:
      n = nums.pop(-1)
      nums.insert(0, n)
      k-=1

"""
Minimizing Permutations

In this problem, you are given an integer N, and a permutation, P of the integers from 1 to N, denoted as (a_1, a_2, ..., a_N). You want to rearrange the elements of the permutation into increasing order, repeatedly making the following operation:
Select a sub-portion of the permutation, (a_i, ..., a_j), and reverse its order.
Your goal is to compute the minimum number of such operations required to return the permutation to increasing order.

Input
Array arr is a permutation of all integers from 1 to N, N is between 1 and 8

Output
An integer denoting the minimum number of operations required to arrange the permutation in increasing order
Example
If N = 3, and P = (3, 1, 2), we can do the following operations:
Select (1, 2) and reverse it: P = (3, 2, 1).
Select (3, 2, 1) and reverse it: P = (1, 2, 3).
output = 2

"""
def minOperations(arr):
    import collections
    target = "".join([str(num) for num in sorted(arr)])
    curr = "".join([str(num) for num in arr])
    queue = collections.deque([(0, curr)]) # In the queue we store (<level>, <permutation>)
    visited = set([curr])
  
    while queue:
        level, curr = queue.popleft()
        if curr == target:
            return level # We are done
        for i in range(len(curr)):
            for j in range(i, len(curr)):
                # Reverse elements between i and j (inclusive)
                # Note we are operating on strings here, so we create a new copy
                permutation = curr[:i] + curr[i:j + 1][::-1] + curr[j + 1:]
                
                if permutation not in visited:
                    visited.add(permutation)
                    queue.append((level + 1, permutation))
          
    return -1