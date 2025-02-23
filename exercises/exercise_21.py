from itertools import permutations
def swap_digits(num):
    for p in permutations(str(num)):
        print(f"perm={p}")
    for p in permutations(str(num)):
        p_join = ''.join(p)
        print(f"p_str={p_join}")
    for p in permutations(str(num)):
        p_int = int(''.join(p))
        print(f"p_int={p_int}")
        
def swapList(num):
    set_1 = sorted(set([int(''.join(p)) for p in permutations(str(num))])) 
    list_1 = sorted(list([int(''.join(p)) for p in permutations(str(num))]))
    return set_1, list_1

# print("test case 9")
# print(swapList(9))
# print("test case 300")
# print(swapList(300))
# print("test case 217")
# print(swapList(217))
# print(swap_digits(300))
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
field =  [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 1, 0, 1, 0],
          [1, 0, 1, 0, 1]]
figure = [[1, 1, 1],
          [1, 0, 1],
          [1, 0, 1]]
print(solution(field=field, figure=figure))
field = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 0, 0],
         [1, 1, 0]]
figure = [[0, 0, 1],
         [0, 1, 1],
         [0, 0, 1]]
print(solution(field=field, figure=figure))
field =  [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [1, 0, 0, 1],
          [1, 1, 0, 1]]
figure = [[1, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
print(solution(field=field, figure=figure))