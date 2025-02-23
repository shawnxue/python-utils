"""
 Write a method with the following inputs:
 - a search string
 - a list of results matching that search string

 Sort those results according to the following order:
 1. First exact matches
 2. Results where one of the words is an exact match
 3. Results that start with the search string
 4. The rest, alphabetically sorted

example:
search_string = 'word'
results = ['Wordpress', 'Microsoft Word', 'Google word', 'Google AdWords', '1Password', 'Word']
sorted_results = ['Word', 'Google word', 'Microsoft Word', 'Wordpress', '1Password', 'Google AdWords']
"""
# your code here
def sort_search_result(search_string, results):
  search_string = search_string.lower()
  results = [s.lower() for s in results]
  #rule_mapping = {}.fromkeys(['1','2','3','4'], []) # key is rule number, vaule is an array, which has all matching words
  # rule_mapping = {'1': [], '2': [], '3': [], '4': []}
  rule_mapping = {rule: [] for rule in ['1', '2', '3', '4']}
  print(f"before rule_mapping={rule_mapping}")
  for result in results:
    if result == search_string:
      rule_mapping['1'].append(result)
    elif search_string in result.split():
      rule_mapping['2'].append(result)
    elif (result[0:len(search_string)] == search_string) and (len(search_string) < len(result)):
      rule_mapping['3'].append(result)
    else:
      rule_mapping['4'].append(result)
  print(f"after rule_mapping={rule_mapping}")
  sorted_results = list()
  for key, value in rule_mapping.items():
    if key == "1":
      if len(value) == 1:
        sorted_results.append(value)
      elif len(value) < 1:
        # no exact match
        pass
    else:
      sorted_value = sorted(value)
      sorted_results.append(sorted_value)
    print(f"key={key}, value={value}, sorted_results={sorted_results}")
  return sorted_results

search_string = 'word'
results = ['Wordpress', 'Microsoft Word', 'Google word', 'Google AdWords', '1Password', 'Word']
print(sort_search_result(search_string, results))
