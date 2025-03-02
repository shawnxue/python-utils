# solution 1:
def solution(S: str) -> str:
    # write your code in Python 3.6
    lines = S.split('\n')
    rv = []
    photos = {}
    for i, line in enumerate(lines):
        fn, city, dt = [s.strip() for s in line.split(',')]
        name, extension = fn.split('.')
        if city in photos:
            photos[city].append((name, extension, dt, i))
        else:
            photos[city] = [(name, extension, dt, i)]
    for city in photos:
        count = len(photos[city])
        zeros = len(str(count))
        c = 1
        photos[city].sort(key=lambda n: n[-2])
        for name in photos[city]:
            rv.append((name[-1], city+str(c).zfill(zeros)+'.'+name[1]))
            c += 1
    rv.sort(key=lambda n: n[0])
    rv = [i[1] for i in rv]
    return '\n'.join(rv)

'''
photo.jpg, Warsaw, 2013-09-05 14:08:15
john.png, London, 2015-06-20 15:13:22
myFriends.png, Warsaw, 2013-09-05 14:07:13
Eiffel.jpg, Paris, 2015-07-23 08:03:02
pisatower.jpg, Paris, 2015-07-22 23:59:59
BOB.jpg, London, 2015-08-05 00:02:03
notredame.png, Paris, 2015-09-01 12:00:00
me.jpg, Warsaw, 2013-09-06 15:40:22
a.png, Warsaw, 2016-02-13 13:33:50
b.jpg, Warsaw, 2016-01-02 15:12:22
c.jpg, Warsaw, 2016-01-02 14:34:30
d.jpg, Warsaw, 2016-01-02 15:15:01
e.png, Warsaw, 2016-01-02 09:49:09
f.png, Warsaw, 2016-01-02 10:55:32
g.jpg, Warsaw, 2016-02-29 22:13:11
'''

# solution 2:
string = """
photo.jpg, Warsaw, 2013-09-05 14:08:15
john.png, London, 2015-06-20 15:13:22
myFriends.png, Warsaw, 2013-09-05 14:07:13
Eiffel.jpg, Paris, 2015-07-23 08:03:02
pisatower.jpg, Paris, 2015-07-22 23:59:59
BOB.jpg, London, 2015-08-05 00:02:03
notredame.png, Paris, 2015-09-01 12:00:00
me.jpg, Warsaw, 2013-09-06 15:40:22
a.png, Warsaw, 2016-02-13 13:33:50
b.jpg, Warsaw, 2016-01-02 15:12:22
c.jpg, Warsaw, 2016-01-02 14:34:30
d.jpg, Warsaw, 2016-01-02 15:15:01
e.png, Warsaw, 2016-01-02 09:49:09
f.png, Warsaw, 2016-01-02 10:55:32
g.jpg, Warsaw, 2016-02-29 22:13:11
"""


def fetch_date_time(photo):
    return photo.split(', ')[2]


def prefixed_number(n, max_n):
    len_n = len(str(n))
    len_max_n = len(str(max_n))
    prefix = ''.join(['0' for i in range(len_max_n-len_n)]) + str(n)
    return prefix


def solution(S):

    list_of_pics = S.split('\n')

    city_dict = {}

    for pic in list_of_pics:
        city = pic.split(', ')[1]
        if city in city_dict:
            city_dict[city].append(pic)
        else:
            city_dict[city] = [pic]

    final_string = ""

    for city_group in city_dict:
        city_dict[city_group].sort(key=fetch_date_time)
        for ind, photo in enumerate(city_dict[city_group]):
            city = photo.split(',')[1]
            ext = photo.split(', ')[0].split('.')[-1]
            max_len = len(city_dict[city_group])
            number = prefixed_number(ind+1, max_len)
            city_dict[city_group][ind] = city + number + '.' + ext + '\n'
        final_string += ''.join(city_dict[city_group])

    return final_string

# python renaming
import os
import glob
from PIL import Image
from PIL.ExifTags import TAGS
import time

def get_exif(fn):
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
    return ret

os.chdir("/path/to/images")

files = glob.glob("*.JPG")

for file in files:
    print(file)
    time = get_exif(file)["DateTimeOriginal"]

    time = time.replace(":", "")
    time = time.replace(" ", "_")
    number = 0
    new_name = time+"_additional_information.jpg"
    if new_name == file:
        print(new_name, "already ok")
        continue
    while os.path.exists(new_name):
        number += 1
        new_name = time+"_"+str(number)+"_additional_information.jpg"
    os.rename(file, new_name)
