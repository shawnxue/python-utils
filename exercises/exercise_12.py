def print_grid(arr):
    for row in range(9):
        for col in range(9):
            print(arr[row][col], end= " ")
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

def solve_sudoku(arr):
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
        if solve_sudoku(arr):
            return True
        # sudoku is not solved, unmake and try next number
        arr[row][col] = 0
    # this trigger backtracking
    return False
N = 9
def solve_sudoku_2(arr, row, col):
    # Check if we have reached the 8th row and 9th column (0 indexed matrix) , we are returning true to avoid further backtracking
    if (row == N - 1) and (col == N):
        return True
    # Check if column value becomes 9, we move to next row and column start from 0
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
        # Removing the assigned num since our assumption was wrong, and we go for next assumption with diff num value
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
    if (solve_sudoku_2(grid, 0, 0)):
        print_grid(grid)
    else:
        print("No solution exists")

run_sudoku_program()