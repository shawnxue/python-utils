'''
given an array A of N integers, returns the smallest positive integer (greater than 0) that does not occur in A.
For example, given A = [1, 3, 6, 4, 1, 2], the function should return 5.
Given A = [1, 2, 3], the function should return 4.
Given A = [−1, −3], the function should return 1.
Write an efficient algorithm for the following assumptions:
N is an integer within the range [1..100,000];
each element of array A is an integer within the range [−1,000,000..1,000,000].
Copyright 2009–2022 by Codility Limited. All Rights Reserved. Unauthorized copying, publication or disclosure prohibited.
'''
def solution(A):
    # write your code in Python 3.6
    smallest_positive_integer = 1
    sorted_a = sorted(A)
    for idx, num in enumerate(sorted_a):
        if num < smallest_positive_integer:
            continue
        else:
            if num > smallest_positive_integer:
                return smallest_positive_integer
            
            smaller_a = sorted_a[idx:]
            '''
            for num in smaller_a:
              if num <= smallest_positive_integer:
                smallest_positive_integer += 1
              else:
                return smallest_positive_integer
            '''
            found = False
            while (not found):
                if smallest_positive_integer in smaller_a:
                    smallest_positive_integer += 1
                else:
                    found = True
    return smallest_positive_integer
