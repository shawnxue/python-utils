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

# Design and implement a data structure for Least Frequently Used (LFU) cache.
# It should support the following operations: get and set. Capacity is applied
# get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
# set(key, value) - Set or insert the value if the key is not already present. When the cache reaches its capacity,
# it should invalidate the least frequently used item before inserting a new item. For the purpose of this problem,
# when there is a tie (i.e., two or more keys that have the same frequency), the least recently used key would be evicted.
def leastFrequencyUsedDesignOld():
    class LFUCacheOld(object):
        def __init__(self, capacity):
            self.capacity = capacity
            self.content = dict()  # store the pair of key, value
            self.lfu_stack = list()  # the least frequently used key is in the beginning, and latest used key is at the end

        def get(self, key):
            if key in self.content.keys():
                self.updateFrequency(key)
                return self.content(key)
            else:
                return -1


        def set(self, key, value):
            if len(self.content) >= self.capacity:
                self.evictLFUItem()

            self.content[key] = value
            self.updateFrequency()

        # update frequency, put latest used key (get/set) at the end
        def updateFrequency(self, key):
            if key in self.lfu_stack:
                self.lfu_stack.remove(key)

            self.lfu_stack.append(key)

        # sort the cache by value, and return sorted dict
        def sortContentByValue(self):
            sorted_tuple = sorted(self.content.items(), key = operator.itemgetter(1))
            sorted_content = {}
            for item in sorted_tuple:
                sorted_content[item(0)] = item[1]
            return sorted_content

        # sort the cache by value, split it into two lists: keys and values
        def convertDictToLists(self):
            return [], []

        def evictLFUItem(self):
            removed_key = self.lfu_stack.pop(0)
            del self.content[removed_key]
    # end of class LFUCache
    # run some tests here
    pass


class LFUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {} # {key: cache_node}, key here is same as the key used in cache_node
        self.freq_link_head = None # how many times a node has been accessed (get)

    # get the node value if key exists, and modify this key's frequency (move it forward along the freq_node list)
    def get(self, key):
        if key in self.cache:
            cache_node = self.cache[key]
            freq_node = cache_node.freq_node
            self.move_forward(cache_node,  freq_node)
            return cache_node.value
        else:
            return -1

    def set(self, key, value, freq_node, pre, nxt):
        pass

    # move the cache node along the freq_link (double link), means: increase access time by 1
    def move_forward(self, cache_node, cur_freq_node):
        if cur_freq_node.nxt is not None: # current freq node is not the last node
            if cur_freq_node.freq == cur_freq_node.nxt.freq: # append the cache node to next freq node
                pass
            else: # create a new freq node and insert after cur freq_node
                pass
        else: # create a new freq node and insert after cur freq_node
            pass

def get_mount_point(pathname):
    """Get the mount point of the filesystem containing pathname"""
    pathname= os.path.normcase(os.path.realpath(pathname))
    parent_device= path_device= os.stat(pathname).st_dev
    while parent_device == path_device:
        mount_point= pathname
        pathname= os.path.dirname(pathname)
        if pathname == mount_point: break
        parent_device= os.stat(pathname).st_dev
    return mount_point

def get_mounted_device(pathname):
    """Use /proc/mounts to get the device mounted at pathname"""
    pathname= os.path.normcase(pathname) # might be unnecessary here
    try:
        with open("/proc/mounts", "r") as ifp:
            for line in ifp:
                fields= line.rstrip('\n').split()
                # note that line above assumes that
                # no mount points contain whitespace
                if fields[1] == pathname:
                    return fields[0]
    except EnvironmentError:
        raise EnvironmentError
    return None # explicit

def get_fs_freespace(pathname):
    """Get the free space of the filesystem containing pathname"""
    stat= os.statvfs(pathname)
    # use f_bfree for superuser, or f_bavail if filesystem has reserved space for superuser
    return stat.f_bfree*stat.f_bsize

def disk_usage(path):
    """Return disk usage associated with path."""
    usage_ntuple = namedtuple('usage',  'total used free percent')
    st = os.statvfs(path)
    free = (st.f_bavail * st.f_frsize)
    total = (st.f_blocks * st.f_frsize)
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    try:
        percent = ret = (float(used) / total) * 100
    except ZeroDivisionError:
        percent = 0
    # the percentage is -5% than what shown by df due to reserved blocks that we are currently not considering:
    # http://goo.gl/sWGbH
    return usage_ntuple(total, used, free, round(percent, 1))