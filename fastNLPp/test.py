# -*- coding:utf-8 -*-

import sys
# sys.path.insert(0, '..')
from pyltpp.pyltp_segment import PyltpSegment


if __name__ == '__main__':
    fenci = PyltpSegment()
    print(fenci.cut('我吃饭了'))
    print(123)
