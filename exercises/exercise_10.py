# program to read output from first program, and feed it to second program
import sys
for line in sys.stdin:
  if 'q' == line.rstrip():
	  break
  # line is output of first program
  sys.stdout.write(line)

for line in sys.stdin:
	if 'q' == line.rstrip():
		break
	print(f'Input : {line}')

# program to handle csv file
import csv
import os
filename = 'mycsvfile.csv'
# initialize the title and row
fields = []
rows = []
# reading the csv file
if os.path.exists(filename):
  with open(filename, 'r') as csvfile:
    # create a csv reader object
    csvreader = csv.reader(csvfile)
    # extract field names from the first row
    fields = next(csvreader)
    # extract data from each row (one by one)
    for row in csvreader:
      rows.append(row)
    # get the total number of rows
    print(f"total number of rows: {csvreader.line_number}")
  # print field names
  field_name = ', '.join(field for field in fields)
  print(f"field names are: ${field_name}")
  print(f"First 5 rows are:\n")
  for row in rows[:5]:
    for column in row:
      # end=" " is for removing newline (\n) after printing.
      print("%10s"%column, end=" ")
    print("\n")

# warm up
## given a integral list, return a list that only includes odd numbers and the number must be less than 100
## requirement: write as less line of code as possible
## input: [2, 3, 5, 90, 87, 299, 183, 56, 388]
## output: [3, 5, 87]

# exercise 1
## given an array of strings, return character's frequecy in each string
## input: ["hello", "world", "Python"]
## output: you can decide the output data structure, but it should includes these info:
## hello: h: 1; e:1; l: 2; o: 1
## world: w: 1; o: 1; r: 1; l: 1; d:1
## Python: P: 1; y: 1: t: 1; h: 1; o: 1; n: 1 

# exercise 2
## given a source string, and a string list, find all the strings in the list that are at most 1 character away from source string
## at most one character away means:
## 1. string in the list is same as the source string
## 2. string in the list has only 1 different character as source string, you can either add, or delete, or update one character to make them same
## e.g. 
## source string: hellio
## string list: ["helli", "ellio", "hllio", "hwellio", "132", "hlli", "world", "hello world", "hellio world"]
## output: ["helli", "ellio", "hllio", "hwollio"]
def get_similar_strings(source, targets):
  result = list()
  for target in targets:
    diff = abs(len(source) - len(target))
    if diff > 1:
      continue
    else:
      shorter = source if len(source) < len(target) else target
      longer = target if len(source) < len(target) else source
      for c in shorter:
        if c not in longer:
          diff += 1
          if diff > 1:
            break
      if diff <= 1:
        result.append(target)
  return result
print(get_similar_strings("hellio", ["lelio", "hello", "hllioe", "llioe", "helli", "ellio", "hllio", "hwellio", "132", "hlli", "world", "hello world", "hellio world"]))