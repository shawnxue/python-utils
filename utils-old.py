import sys
# given two words  start and end, and a dictionary of word list, find all shortest transformation sequences
# from start to end. (only one letter can be changed at a time, ech intermediate word must in the dictionary)
# e.g. start: "hit", end:"cog". dictionary: ["hot", "dot", "dog", "lot", "log"]
# return: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
def findLadders(start, end, dict):
    # word, start are string, res is list
    def backTrace(word, start, res, map):
        if word == start:
            res.append(start)
            return
        res.append(word)
        while map[word] is not None:
            for s in map[word]:
                backTrace(s, start, res)

    result = []
    if len(dict) == 0:
        return result
    queue = list([start]) # store the words from start to end
    ladder = {}.fromkeys(dict, sys.maxsize)
    ladder[start] = 0
    dict.append(end)
    my_map = {} # key is word, value is the list of word path
    import string
    # BFS: Dijisktra search, breath first search
    while len(queue) != 0:
        word = queue.pop(-1) # remove and return the last obj from the list
        step = ladder[word] + 1 # step indicates how many steps are need to travel to word
        if step > sys.maxsize:
            break
        for i in range(len(word)): # parse and replace each character in the current word
            for c in string.ascii_lowercase:
                new_word = word.repalce(word[i], c, 1) # construct a new word by replacing character
                if new_word in dict:
                    if new_word in ladder.keys():
                        if step > ladder[new_word]: # check if it's the shortest path to one word
                            continue
                        elif step < ladder[new_word]:
                            queue.append(new_word)
                            ladder[new_word, step]
                        else: # it is a KEY line. If one word is already in the one ladder, DO NOT insert it into the queue again
                            pass
                    if new_word in my_map.keys(): # build adjacent graph, so that we can back trace
                        my_map[new_word].append(word)
                    else:
                        my_map[new_word] = list([word])
                    if new_word == end:
                        min = step
    # back tracking, to construct the result, result is list, you can pass it as reference like java, but not string
    backTrace(end, start, result, my_map)
    return result

# in python 2.6, use module itertools
def findPermutations(s):
    """:type s: string or list"""
    from itertools import permutations
    cons = [''.join(p) for p in permutations(s)]
    print(cons) # print all permutations
    print(set(cons)) # print permuted list without duplicates

def all_perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                # elements[0:1] works in both string and list contexts
                yield perm[:i] + elements[0:1] + perm[i:]