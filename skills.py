#!/usr/bin/env python
'''
Created on Oct 3, 2016

@author: sxue
'''
import sys, os, ConfigParser, getopt, json
import requests
from requests.auth import HTTPBasicAuth
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from urllib import quote_plus
from influxdb import InfluxDBClient
from datetime import datetime
from dateutil import parser
import consul
import mysql.connector
import pymysql
from mercurial.templatefilters import age

config = ConfigParser.SafeConfigParser()
config.optionxform = str
config.read('/path/to/conf')
config.has_option('mysqld', 'binlog')
value = config.get('mysqld', 'binlog')

# do a API call, and get response, convert response to json
requests.packages.urllib3.disable_warnings ( InsecureRequestWarning )
cred = requests.HTTPBasicAuth("username", "password/token")
headers = {"": "", "": ""}
api_resp = requests.get(url='host + request_url + API_endpoint' , auth=cred, headers=headers, verify=False)
if api_resp.status_code == requests.codes.ok:
    for content in api_resp.json:
        # do something here, such as put them into dict, or list
        continue
    api_resp.json()

ci_build_queue = []
ci_build_queue[ 0 ][ "fields" ][ "QueueToBuildStart" ] = (
                    datetime.strptime ( json_output[ "startDate" ] , "%Y%m%dT%H%M%S+0000" ) - datetime.strptime (
                        json_output[ "queuedDate" ] , "%Y%m%dT%H%M%S+0000" )).seconds

time = parser.parse ( commit[ "created_at" ] ).strftime ( "%Y%m%dT%H%M%S+0000" )

# create json body
json_body = [{'measurement': '' , 'tags': {} , 'time': '' , 'fields': {}}]
headers = {'Accept': 'application/json' , 'Content-type': 'application/json'}
keys = json_body[0].keys()

# team city api authentication
auth = HTTPBasicAuth('username', 'user token')

try:
  cnx = mysql.connector.connect(user='scott', password='tiger',host='127.0.0.1',database='employees')
  cursor = cnx.cursor()
except mysql.connector.Error as err:
  if err.errno == mysql.errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == mysql.errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
  cnx.close()

config = {
  'user': 'scott',
  'password': 'tiger',
  'host': '127.0.0.1',
  'database': 'employees',
  'raise_on_warnings': True,
}

try:
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
except mysql.connector.Error as err:
    if err.errno == mysql.errorcode.ER_ACCESS_DENIED_ERROR:
        print "Credential is wrong"
    elif err.errno == mysql.errorcode.ER_BAD_DB_ERROR:
        print "DB does not exist"
    else:
        print err
else:
    cnx.close()

env_vars = {"util_env_vars": {"MYSQL_SOCK": "", "APPADMINUSER": "", "APPADMINPASSWORD": ""}}
try:
    connection = pymysql.connect(
        unix_socket = env_vars['util_env_vars']['MYSQL_SOCK'],
        user = env_vars['util_env_vars']['APPADMINUSER'],
        password = env_vars['util_env_vars']['APPADMINPASSWORD'],
        db = 'mysql',
        charset = 'utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    with connection.cursor() as cursor:
        # Create a new record
        sql = "show slave status"
        cursor.execute(sql)
        result = cursor.fetchone()
        if result is None:
            env_vars['is_slave'] = False
        else:
            env_vars['is_slave'] = True
finally:
    connection.close()

# data structure for creating table
DB_NAME = 'employees'

TABLES = dict() # or TABLES = {}
TABLES['employees'] = (
    "CREATE TABLE `employees` ("
    "  `emp_no` int(11) NOT NULL AUTO_INCREMENT,"
    "  `birth_date` date NOT NULL,"
    "  `first_name` varchar(14) NOT NULL,"
    "  `last_name` varchar(16) NOT NULL,"
    "  `gender` enum('M','F') NOT NULL,"
    "  `hire_date` date NOT NULL,"
    "  PRIMARY KEY (`emp_no`)"
    ") ENGINE=InnoDB")

TABLES['departments'] = (
    "CREATE TABLE `departments` ("
    "  `dept_no` char(4) NOT NULL,"
    "  `dept_name` varchar(40) NOT NULL,"
    "  PRIMARY KEY (`dept_no`), UNIQUE KEY `dept_name` (`dept_name`)"
    ") ENGINE=InnoDB")
try:
    cursor.execute("USE {}".format(DB_NAME))
    if err.errno == errorcode.ER_BAD_DB_ERROR:
        cursor.execute("CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
        print("Database {} created successfully.".format(DB_NAME))
        cnx.database = DB_NAME
        for table_name in TABLES:
            table_description = TABLES[table_name]
            cursor.execute(table_description)
    else:
        print(err)
        exit(1)
except mysql.connector.Error as err:
    print (err)
finally:
    cursor.close()
    cnx.close()

'''
some sql JOIN statements https://www.techonthenet.com/mysql/joins.php

INNER JOIN returns only intersection of two tables

SELECT suppliers.supplier_id, suppliers.supplier_name, orders.order_date
FROM suppliers
INNER JOIN orders
ON suppliers.supplier_id = orders.supplier_id;

old syntax:
SELECT suppliers.supplier_id, suppliers.supplier_name, orders.order_date
FROM suppliers,orders
WHERE suppliers.supplier_id = orders.supplier_id;

LEFT JOIN returns all rows from table1 (suppliers) and intersection of table1 and table2(orders), table1 is on the left

SELECT suppliers.supplier_id, suppliers.supplier_name, orders.order_date
FROM suppliers
LEFT JOIN orders
ON suppliers.supplier_id = orders.supplier_id

RIGHT JOIN returns all rows from table2 (orders) and intersection of table1(suppliers) and table2, table 2 is on the right

SELECT orders.order_id, orders.order_date, suppliers.supplier_name
FROM suppliers
RIGHT JOIN orders
ON suppliers.supplier_id = orders.supplier_id;
'''

# reverse a string
'hello world'[::-1] # output is 'dlrow olleh'
# or you can use this
''.join(reversed('hello world'))
a = [1, 2, 3, 4, 5, 6, 7, 8]
print a[::-1] # [8, 7, 6, 5, 4, 3, 2, 1]

datetime.now().strftime("%A, %d. %B %Y %I:%M:%S %p: ") # Wednesday, 21. November 2012 03:06:05 PM


def splitStringToNumber(s):
    return tuple(int(x) for x in s.split('.'))

# function to transfer string x to int x if x is numeric string
def tryint(x):
    try:
        return int(x)
    except ValueError:
        return x

# function to split a string to characters
# this function handle the case that string like this: T920X23.2d3.23dd98
def splitStringToCharacters(s):
    import re
    return tuple(tryint(x) for x in re.split('([0-9]+)', s))

# What key does is it provides a way to specify a function that returns what you would like your items sorted by.
# The function gets an "invisible" argument passed to it that represents an item in the list,
# and returns a value that you would like to be the item's "key" for sorting.

# function to sort a list of names, use splitStringToCharacters as a key function to sorted
# such as input: strings = ['YT4.11', '4.3', 'YT4.2', '4.10', 'PT2.19', 'PT2.9']
# out: ['4.3', '4.10', 'PT2.9', 'PT2.19', 'YT4.2', 'YT4.11']
def stringListSorted(strings):
    return sorted(strings, key=splitStringToCharacters)

def getFileChecksum(filePath):
    try: 
        from md5 import md5
    except ImportError:
        from hashlib import md5
    return md5(open(filePath,"r").read()).hexdigest()

# function to get immediate subdirectories of a directory, return a list of immediate subdirectory's names, just name, not full path
def getImmediateSubdirectories(directory):
    return os.walk(directory).next()[1]

# function to get files (full path) from a target directory and its sub dir.
# parameters: Use *arg to pass non-keyworded variable-length argument list, which is used to filter unwanted files
# return: a list of files 
def getFilesFromDir(targetDir, *args):
    files = []
    for dirname, dirnames, filenames in os.walk(targetDir): # current dir (string), direct dirs (lists), files (list) in current dir
        for filename in filenames:
            fi = os.path.join(dirname, filename)
            filtered = False
            for ar in args:
                if ar in fi:
                    filtered = True
                    break
            if not filtered:
                files.append(f)
    return files

def get_files_from_dir2(target_dir):
    files = list()
    for dirpath, sub_dirs, filenames in os.walk(target_dir):
        files += [os.path.join(dirpath, filename) for filename in filenames] # files.extend(another_list)
    return files

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# replace some values
letters[2:5] = ['C', 'D', 'E']
# clear the list by replacing all the elements with an empty list
letters[:] = []
# nest lists (create lists containing other lists)
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n] # [['a', 'b', 'c'], [1, 2, 3]]
x[0] # ['a', 'b', 'c']
x[0][1] # 'b'
# loop list
questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
# combined loop
# after zip two lists, the result is still a same-size list, but each member in the new list is a tuple
for q, a in zip(questions, answers):
    print "What is your {0}? It is {1}".format(q, a)
# sequence loop
for i, v in enumerate(["sd", "adf", "23423"]): # i is index, v is value
    print "index is %s, value is %s" % (i, v)
# no sequence loop
for q in questions:
    print q
# sorting
sorted([5, 2, 3, 1, 4]) # [1, 2, 3, 4, 5]

sorted("This is a test string from Andrew".split(), key=str.lower) # ['a', 'Andrew', 'from', 'is', 'string', 'test', 'This']

student_tuples = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10),]
sorted(student_tuples, key=lambda student: student[2]) # [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]

# __repr__ function tells Python how we want the object to be represented as.
# In more complex terms, it tells the interpreter how to display the object when it is printed to the screen.
class Student:
    def __init__(self, name, grade, age):
        self.name, self.grade, self.age = name, grade, age
    # return a printable object
    def __repr__(self):
        return repr((self.name, self.grade, self.age))
    # return a string of this object when calling str()
    def __str__(self):
        return "{0} is {1} years old and in grade {2}".format(self.name, self.age, self.grade)
    # override it for obj compare, so you can call sorted without specifying key
    def __cmp__(self, other):
        if hasattr(other, 'age'):
            return self.age.__cmp__(other.age)
    # another way to override this function, to handle the case of different obj into one list
    def __cmp__(self, other):
        if hasattr(other, 'getKey')
            return self.getKey().__cmp__(other.getKey())
    def getKey(self):
        return self.age
    def weighted_grade(self):
        return 'CBA'.index(self.grade) / float(self.age)

    pass

def getKey(student_obj):
    if hasattr(student_obj.name):
        return student_obj.name

student_objs = [Student('John', 'B', 15), Student('Kevin', 'A', 16), Student('Dave', 'B', 10)]
sorted(student_objs, key=lambda student:student.age) # [('Dave', 'B', 10), ('John', 'B', 15), ('Kevin', 'A', 16)]

from operator import itemgetter, attrgetter, methodcaller
sorted(student_tuples, key=itemgetter(2)) # [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
sorted(student_tuples, key=itemgetter(2), reverse=True) #[('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(student_objs, key=attrgetter('age')) # [('Dave', 'B', 10), ('John', 'B', 15), ('Kevin', 'A', 16)]
sorted(student_objs, key=methodcaller('weighted_grade'))
sorted(student_objs, key=getKey)
sorted(student_objs)

# python json encode (dump): convert python data structure, such as dict, to json file or string
import json, io
# serialize python data structure (object) to string
var_s = json.dumps({"c": 0, "b": 0, "a": 0}, sort_keys=True) # {"a": 0, "b": 0, "c": 0}
# serialize python data structure(object) to file-like stream
from StringIO import StringIO
io = StringIO()
json.dump(['Streaming API'],io)
io.getvalue() # '["streaming API"]'

data = {'key': 'value', 'whatever': [1, 42, 3.141, 1337]}
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

# python json decode (load), convert json file to python data structure: dict or list (depends on the content)
# deserialize file like obj to python obj
# the contents of data.json 
{
    "maps": [
        {
            "id": "blabla",
            "iscategorical": "0"
        },
        {
            "id": "blabla",
            "iscategorical": "0"
        }
    ],
    "masks": {
        "id": "valore"
    },
    "om_points": "value",
    "parameters": {
        "id": "valore"
    }
}
import pprint
pp = pprint.PrettyPrinter(indent=4)
with open('data.json', 'r') as f:
    data_json = json.load(f) # after load, it returns dict
pprint.pformat(data_json)
data_json['maps'][0]['id'] # blabla
data_json['masks']['id']

# deserialize str to python obj
var_l = json.loads('["foo", {"bar":["baz", null, 1.0, 2]}]') # return a list: [u'foo', {u'bar': [u'baz', None, 1.0, 2]}]

# dict
dict([('sape', 4139), ('guido', 4127), ('jack', 4098)]) # convert list (each item is a tuple)to dict, {'sape': 4139, 'jack': 4098, 'guido': 4127}
{x: x**2 for x in (2, 4, 6)} #{2: 4, 4: 16, 6: 36}
d = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
d.keys() # get the list of keys
d.values() # get the list of values
del d['Name']; # remove entry with key 'Name'
d.clear();     # remove all entries in dict
del d ;        # delete entire dictionary
print [k for k in d.iterkeys()]
print [v for v in d.itervalues()]
print [x for x in d.iteritems()] # [(k1,v1), (k2,v2), (k3,v3)]
for k, v in d.items():
    print (k,v)
print "key is {0}, value is {1}".format(k, v) for k, v in d.iteritems()
print 'key is %s, value is %s' %  (k, v for k, v in d.iteritems())
var = "abcde"
my_dict = {}.fromkeys(var, 0) # {'a': 0, 'c': 0, 'b': 0, 'd': 0, 'e': 0}
GIT_COMMIT_FIELDS = ['id', 'author_name', 'author_email', 'date', 'message']
# after zip two list, it will become a new list, each member in the list is a tuple
my_dict = [dict(zip(GIT_COMMIT_FIELDS, row)) for row in logfile]

# sort a dict
import operator
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_x_list = sorted(x.items(), key=operator.itemgetter(1)) # sorted_x is a list, each item is tuple

mydict = {'carl':40,
          'alan':2,
          'bob':1,
          'danny':3}
# sort a dict by value (Python 2.4 or greater):
for key, value in sorted(mydict.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)
# sort a dict by keys (Python 2.4 or greater):
for key in sorted(mydict.keys()) # or sorted(mydict.iterkeys)
    print "%s: %s" % (key, mydict[key])
# sort a dict by keys (Python older than 2.4)
keylist = my_dict.keys()
sorted(keylist)
for key in keylist:
    print "%s: %s" % (key, my_dict[key])

# ways for dict copy
python -m timeit -s "d={1:1, 2:2, 3:3}" "new = d.copy()" # call dict function copy()
python -m timeit -s "d={1:1, 2:2, 3:3}" "new = dict(d)" # call built-in function dict()
python -m timeit -s "from copy import copy; d={1:1, 2:2, 3:3}" "new = copy.copy(d)" # use module copy

# set (no ordering), immutable, an unordered collection with no duplicate elements
# use {} to define set
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'} # {'orange', 'banana', 'pear', 'apple'}
a = {x for x in 'abracadabra' if x not in 'abc'} # {'r', 'd'}
# use built-in function set to define it
a = set([1, 2, 3, 4])
b = set([3, 4, 5, 6])
a | b # Union {1, 2, 3, 4, 5, 6}
a & b # Intersection, {3, 4}
c = a.intersection(b) # equivalent to c = a & b
a < b # Subset, False
c.issubset(a) # True
c <= a # True
c.issuperset(a) # False
c >= a # False
a - b # Difference {1, 2}
a.difference(b)
a ^ b # Symmetric Difference {1, 2, 5, 6}
a.symmetric_difference(b)

# tuple: a number of values separated by commas, immutable
t = 12345, 54321, 'hello!' # (12345, 54321, 'hello!')
u = t, (1, 2, 3, 4, 5) # nested tuple, ((12345, 54321, 'hello!'), (1, 2, 3, 4, 5))

# regex in python https://docs.python.org/2/library/re.html
''' 
*: the previous character can be matched zero or more times. e.g. a[bcd]*b, matches the letter 'a', zero or more letters from the class [bcd], ends with a 'b'.
+: the previous character can be matched one or more times.
?:the previous character can be matched either once or zero times, e.g. home-?brew matches either homebrew or home-brew
matched string   python string literals for this match     python raw string for this match
"\section"       "\\\\section"                            r"\\section"
"ab*"            "ab*"                                    r"ab*"
"\\w+\\s+\\1"    "\\w+\\s+\\1"                            r"\w+\s+\1"

<character>$: match the end of a string or line (ending with \n), same as \Z
^<character>: match the beginning of a string, same as \A
\b<string>\b: matches only at the beginning or end of a word. if \b is put both at beginning and ending, it means string should be a complete word,
not inside other strings
\B: opposite of \b
[]: character class, meta characters inside it are considered as common string, characters inside it won't be kept in the result list, they will be replaced by ''
(): grouping, characters inside it are considered as whole part, and they will be kept in the result list as element
\d: match decimal digits, same as class [0-9]
\D: match non decimal digits, same as class [^0-9]
\s: match any whitespace character, same as class [ \t\n\r\f\v]
\S: matches any non-whitespace character; this is equivalent to the class [^ \t\n\r\f\v]
\w: Matches any alphanumeric character; this is equivalent to the class [a-zA-Z0-9]
\W: Matches any non-alphanumeric character; this is equivalent to the class [^a-zA-Z0-9]
'.': often used where you want to match “any character”
[\s,.] is a character class that will match any whitespace character, or ',' or '.'
'''
import re
p = re.compile('[a-z]+') # metacharacter [ and ] create a character class, inside character class, other metacharacters are considered as string
m = p.match("") # return m is None
m = p.match("sdfalkjsd") # return an object m
p = re.compile('goes')
m = p.match( 'string goes here' )
if m:
    print 'Match found: ', m.group()
else:
    print 'No match'
p = re.compile('\d+')
p.findall('12 drummers drumming, 11 pipers piping, 10 lords a-leaping') # ['12', '11', '10']
itr = p.finditer('12 drummers drumming, 11 pipers piping, 10 lords a-leaping')
for match in itr:
    print m.span()
m = re.match(r'From\s+', 'Fromage amk') # m is None

p = re.compile(r'\W+')
p2 = re.compile(r'(\W+)')
p.split('This... is a test.') # only return the list of text ['This', 'is', 'a', 'test', '']
p2.split('This... is a test.') # return list of text and delimiter ['This', '... ', 'is', ' ', 'a', ' ', 'test', '.', '']
p.split('This is a test, short and sweet, of split().', 3) # split at most 3 times ['This', 'is', 'a', 'test, short and sweet, of split().']
p.split('This is a test, short and sweet, of split().') # ['This', 'is', 'a', 'test', 'short', 'and', 'sweet', 'of', 'split', '']

re.sub(pattern, repl, string, max=0)

# replacement is a string "colour"
p = re.compile('(blue|white|red)')
p.sub('colour', 'blue socks and red shoes') # return new string: 'colour socks and colour shoes'
re.sub('(blue|white|red)', 'colour', 'blue socks and red shoes') # return new string: 'colour socks and colour shoes'
p.sub('colour', 'blue socks and red shoes', count=1) # return new string 'colour socks and red shoes'
p.subn('colour', 'blue socks and red shoes') # return a tuple containing new string and replacement number ('colour socks and colour shoes', 2)
p.subn('colour', 'no colours at all') # return a tuple containing new string and replacement number: ('no colours at all', 0)

# replacement is function,
def hexrepl(match):
    '''Return the hex string for a decimal number'''
    value = int(match.group())
    return hex(value)
p = re.compile(r'\d+')
p.sub(hexrepl, 'Call 65490 for printing, 49152 for user code.') # 'Call 0xffd2 for printing, 0xc000 for user code.'
# or
re.sub(r'\d+', hexrepl, 'Call 65490 for printing, 49152 for user code.')

ord(<char>): # convert a character to an integer (ASCII value)
chr(<integer>) # convert an integer (ASCII value to character)

bin(<integer>) [2:] # convert integer to binary
bin(6)[2:].zfill(8) # convert integer 6 to binary and make the binary string length 8
'{0:08b}'.format(6) # same
'''
{} places a variable into a string
0 takes the variable at argument position 0
: adds formatting options for this variable (otherwise it would represent decimal 6)
08 formats the number to eight digits zero-padded on the left
b converts the number to its binary representation
d converts the number to decimal
'''
"{0:08b},{1:08d}".format(6,6) # '00000110,00000006'

class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

    def __repr__(self):
        print "Person's name " + self.name + ", age is " + self.age

    def __str__(self):
        return str(self.name + self.age)

    def __cmp__(self, other):
        if hasattr(other, age):
            return self.age.__cmp__(other.age)

jack1 = Person('Jack', 23)
jack2 = Person('Jack', 23)

jack1 == jack2 #True
jack1 is jack2 #False

class Node:
    def __init__(self, contents=None, next=None):
        self.contents = contents
        self.next  = next

    def getContents(self):
        return self.contents

    def __str__(self):
        return str(self.contents)

def print_list(node):
    while node:
        print(node.getContents())
        node = node.next
    print()

def testList():
    node1 = Node("car")
    node2 = Node("bus")
    node3 = Node("lorry")
    node1.next = node2
    node2.next = node3
    print_list(node1)

# ways to copy a list, performance is slower with the sequence http://stackoverflow.com/questions/2612802/how-to-clone-or-copy-a-list
new_list = old_list[:] # use list slice
new_list = list(old_list) # use built-in list() function
import copy
new_list = copy.copy(old_list)
new_list = copy.deepcopy(old_list) # deepcopy will copy the obj inside the list as well, so it's not a reference

# urllib and urlparse for handling url
# commands to run external commands
# os, system, os.path

if __name__ == '__main__':
    for index, arg in enumerate(sys.argv):
        print str(sys.argv[index]) + "is" + arg

# You can create a function that accepts any number of positional arguments as well as some keyword-only
# arguments by using the * operator to capture all the positional arguments and then specify optional
# keyword-only arguments after the * capture.
# this is for python 3, python 2 doesn't support
def product(*numbers, initial=1):
    total = initial
    for n in numbers:
        total *= n
    return total
'''
#>>> product(4, 4)
16
>>> product(4, 4, initial=1)
16
>>> product(4, 5, 2, initial=3)
120
'''

# If you want to accept keyword-only arguments and you’re not using a * to accept any number of positional arguments,
# you can use a * without anything after it.
# here’s a modified version of Django’s django.shortcuts.render function.
# This version disallows calling render by specifying every argument positionally.
# The context_type, status, and using arguments must be specified by their name.
def render(request, template_name, context=None, *, content_type=None, status=None, using=None):
    content = loader.render_to_string(template_name, context, request, using=using)
    return HttpResponse(content, content_type, status)
'''
good call: render(request, '500.html', {'error': error}, status=500)
bad call:  render(request, '500.html', {'error': error}, 500)
TypeError: render() takes from 2 to 3 positional arguments but 4 were given
'''

# Here’s a function with four required keyword-only arguments
from random import choice, shuffle
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE = UPPERCASE.lower()
DIGITS = "0123456789"
ALL = UPPERCASE + LOWERCASE + DIGITS
def random_password(*, upper, lower, digits, length):
    import random
    chars = [
        *(random.choice(UPPERCASE) for _ in range(upper)),
        *(random.choice(LOWERCASE) for _ in range(lower)),
        *(random.choice(DIGITS) for _ in range(digits)),
        *(random.choice(ALL) for _ in range(length-upper-lower-digits)),
    ]
    random.shuffle(chars)
    return "".join(chars)

# Python allows functions to capture any keyword arguments provided to them using the ** operator
# when defining the function:
def format_attributes(**attributes):
    """Return a string of comma-separated key-value pairs."""
    return ", ".join(
        f"{param}: {value}" for param, value in attributes.items()
    )
"""
example 1:
call: format_attributes(name="Trey", website="http://treyhunner.com", color="purple")
result: name: Trey, website: http://treyhunner.com, color: purple

example 2: taking every key/value pair from a dictionary and passing them in as keyword arguments

items = {'name': "Trey", 'website': "http://treyhunner.com", 'color': "purple"}
format_attributes(name=items['name'], website=items['website'], color=items['color'])

items = {'name': "Trey", 'website': "http://treyhunner.com", 'color': "purple"}
format_attributes(**items)

result: name: Trey, website: http://treyhunner.com, color: purple
"""

# example of using PyYAML
import yaml

# Define data
data = {'a list': [1, 42, 3.141, 1337, 'help', u'€'],
        'a string': 'bla',
        'another dict': {'foo': 'bar',
                         'key': 'value',
                         'the answer': 42
                         }
        }
# write YAML file: yuml dump converts python data structure(dict) to yaml file
with open('data.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

# read YAML file: yum load coverts yaml file to python data structure (such as dict, depends on file content)
with open('data.yaml', 'r') as f:
    data_loaded = yaml.load(f)
print(data == data_loaded)

# The yaml documents are separated by ---, and if any stream (e.g. a file) contains more than one document
# then you should use the yaml.load_all function rather than yaml.load
f = open("test.yaml", "r")
docs = yaml.load_all(f)
for doc in docs:
    for k,v in doc.items():
        print k, "->", v
    print "\n",

with open('example.yaml', 'r') as stream:
    try:
        print(yaml.load(stream))
    except yaml.YAMLError as exc:
        print(exc)

# list/dict operation del, pop and remove. Index will be key if it's dict, and dict has no remove function
# del alist[index]: del removes element by index, no return value, which support index range, like start:end
# alist.pop[index]: pop removes element by index and return the value. if no index specified, last element will be popped
# alist.remove[element]: remove removes element by value, no return value, and raise ValueError if no value found in list

#
[(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]
# output: [(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]
# same as
combs = []
for x in [1,2,3]:
    for y in [3,1,4]:
        if x != y:
            combs.append((x, y))

# transpose a matrix
matrix = [
    [1,2, 3, 4],
    [5,6, 7, 9],
    [9,10,11,12]
]
# best solution: use zip, haha
zip(*matrix)
# option 1
[[row[i] for row in matrix]for i in range(4)] # 4 is column
# option 2
transposed = []
for i in range(4): # 4 is column
    transposed_row = []
    for row in matrix:
        transposed_row.append(row[i])
    transposed.append(transposed_row)