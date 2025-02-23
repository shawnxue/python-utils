import sys
def getSum(A: int, B: int, C: int) -> int:
  # sanity check
  is_valid = True
  if A is None or B is None or C is None:
    print(f"input is not valid")
  if sys.maxsize < A or sys.maxsize < B or sys.maxsize < C:
    print(f"input is overflow, too big")
  if A < (-sys.maxsize - 1) or B < (-sys.maxsize - 1) or C < (-sys.maxsize - 1):
    print(f"input is overflow, too small")
  sum = 0
  try:
    sum = A + B + C
  except:
    print(f"sum is overflow")
    raise ValueError

"""
There's a multiple-choice test with N questions, numbered from 1 to N. Each question has 2 answer options, labelled A and B. You know that the correct 
answer for the ith question is the ith character in the string C, which is either "A" or "B", but you want to get a score of 0 on this test by answering
every question incorrectly. Your task is to implement the function getWrongAnswers(N, C) which returns a string with N characters, the iih of which is the 
answer you should give for question i in order to get it wrong (either "A" or "B").
Constraints:
1<=N<=100
String C only includes characters A and B
"""

def getWrongAnswers(N: int, C: str) -> str:
  # sanity check
  if N > 100 or N < 1:
    return f"input value N={0}, not valid".format(N)
  for char in C:
    if char not in ["A", "B"]:
      return f"input string C={0}, not valid".format(C)
  if len(C) != N:
    return f"input value N={0} and C={1}, not valid".format(N, C)
  pick_wrong_answer = {"A": "B", "B": "A"}
  wrongAnswer = [pick_wrong_answer[char] for char in C]
  return "".join(wrongAnswer)

"""
You're playing Battleship on a grid of cells with RR rows and CC columns. There are 00 or more battleships on the grid, each occupying a single distinct cell. 
The cell in the ith row from the top and jth column from the left either contains a battleship G{i,j}=1 G{i,j}=0.
You're going to fire a single shot at a random cell in the grid. You'll choose this cell uniformly at random from the R*CR∗C possible cells. 
You're interested in the probability that the cell hit by your shot contains a battleship. Your task is to implement the function getHitProbability(R, C, G) 
which returns this probability. Note: Your return value must have an absolute or relative error of at most 10^{-6} to be considered correct.

Constraints
1<=R, C<=100
G(i,j) 0, or 1 
"""
from typing import List
def getHitProbability(R: int, C: int, G: List[List[int]]) -> float:
  # sanity check
  if R < 1 or C < 1 or R > 100 or C > 100:
    print(f"input value R={0}, C={1} not valid".format(R, C))
    raise ValueError
  # assume: G(i,j) is either 0 or 1
  count_battle_ship = 0
  total_cell = R * C
  for row in range(R):
    for column in range(C):
      count_battle_ship += G[row][column]
  # give 8 digits after the point
  return format(count_battle_ship/total_cell, '.8f')

"""
or we can go through matrix like this:
for i in range(len(G)):
  for j in range(len(G[i])):
    print(G[i][j])
"""
