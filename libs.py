#!/usr/bin/env python
# -*- coding: UTF-8 -*
'''
https://www.oreilly.com/learning/20-python-libraries-you-arent-using-but-should
ptpython: alternative interpreter interface
flit: tool to help you create python package and submit to PyPI (python package index)
'''

import logging
import psutil  # psutil is to get system information
# watchdog is to receive notifications of changes in the file system, not pulling, but pushing
from watchdog.observers import Observer
from watchdog.events import (PatternMatchingEventHandler, FileModifiedEvent, FileCreatedEvent)
import hug  # library to create  API for web service
import webcolors  #  library to convert web-safe color between different formats like name, hex, rgb and so on
import inflect  # library to handle words
import datetime
import arrow  # library to handle date and time (aware), better than datetime module (native and aware)
import parsedatetime as pdt  # library to parse text into date and time
import collections
import string

logger = logging.getLogger("useful libraries")


class MyHandler(PatternMatchingEventHandler):  # you have to subclass one of handler class, and override the event you want to handle
    def on_created(self, event=FileCreatedEvent):
        logger.info("File Created: " + event.src_path)

    def on_modified(self, event=FileModifiedEvent):
        print("File Modified: %s [%s]" % (event.src_path, event.event_type))


def test_watchdog():
    observer = Observer()  # create an Observer instance
    observer.schedule(event_handler=MyHandler("*"), path=".")  # watch all files under current directory
    observer.daemon = False
    observer.start()
    try:
        observer.join()  # watchdog run in a separate thread, force the program to block at this point
    except KeyboardInterrupt:
        logger.info("Program stopped by ctrl + c")
        observer.stop()
        observer.join()


def test_hug():
    @hug_get
    def hex_to_name(h=hug.types.text):
        return webcolors.hex_to_name("#" + h)

    @hug_get
    def name_to_hex(name=hug.types.text):
        return webcolors.name_to_hex(name)

    @hug_get(version=1)
    def singular(word=hug.types.text):
        return inflect.engine.singular_noun(word).lower()

    @hug_get(version=2)
    def singular(word=hug.types.text):
        return inflect.engine.singular_noun(word)
'''
    1. use cmd hug to start your python file
    2. curl http://localhost:8000/hex_to_name?hex=ff0000, it returns red
    3. curl http://localhost:8000/name_to_hex?name=lightskyblue, it returns #87cefa
    4. curl http://localhost:8000/v1/singular?word=Silly%20Walks, it returns silly walk
    5. curl http://localhost:8000/v2/singular?word=Silly%20Walks, it returns Silly Walk
    6. curl http://localhost:8000/v2, it returns the help in JSON
'''


def test_psutil():
    cpu = psutil.cpu_percent(interval=5, percpu=True)
    logger.info(cpu)


def test_arrow():
    t0 = arrow.now()
    t1 = arrow.utcnow()
    diff1 = (t1 - t0).total_seconds()
    print('difference should be zero: %.2f seconds' % diff1)
    print t0.humanize(locale='ja')
    t3 = datetime.datetime.now()
    t4 = datetime.datetime.utcnow()
    diff2 = (t4 - t3).total_seconds()
    print('difference should be 28800: %.2f seconds ' % diff2)


def test_parsedatetime():
    cal = pdt.Calendar()
    examples = [
        "19 November 1975",
        "19 November 75",
        "19 Nov 75",
        "tomorrow",
        "yesterday",
        "10 minutes from now",
        "the first of January, 2001",
        "3 days ago",
        "in four days' time",
        "two weeks from now",
        "three months ago",
        "2 weeks and 3 days in the future",
    ]
    print("Now: {}".format(datetime.datetime.now().ctime(), end="\n\n"))
    print("{:40s}{:>30s}".format("Input", "Result"))
    print("=" * 70)
    for e in examples:
        dt, result = cal.parseDT(e)
        print("{:<40s}{:>30s}".format('"' + e + '"', dt.ctime()))


def test_ordereddict():
    collections.OrderedDict(zip(string.ascii_lowercasae, range(5)))  # [('a', 0), ('b', 1), ('c', 2), ('d', 3),('e', 4)]
    collections.OrderedDict(zip(string.ascii_lowercase, range(6)))  # [('a', 0), ('b', 1), ('c', 2), ('d', 3),('e', 4), ('f', 5)]
    collections.OrderedDict(a=1, b=2, c=3)  # [('b', 2), ('a', 1), ('c', 3)] keyword arguments not working, because the keyword arguments are first processed as a normal dict before they are passed on to the OrderedDict

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="./tmp.log")
    test_watchdog()
