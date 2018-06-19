#!/usr/bin/env python3
from __future__ import print_function
# import numpy as np
from pygt3file import GT3File
import argparse
import sys


class A:
    """ base class for argparse,"""
    pass


parser = argparse.ArgumentParser(description='Show contents of gt3file.')

parser.add_argument(
    '-H', '--header', help='Show header only.',
    action='store_true')
parser.add_argument(
    '-T', '--table', help='Show data table only.',
    action='store_true')
parser.add_argument(
    'file', help='file name.')

parser.add_argument(
    '-n', '--numbers', help='Space separated data number(s) to be shown.',
    type=int, nargs='*')

parser.add_argument(
    '-d', '--debug', help='debug output',
    action='store_true')

a = A()
parser.parse_args(namespace=a)

file = a.file

if (a.numbers is None):
    opt_numbers = ()
    opt_all = True
else:
    opt_numbers = tuple(a.numbers)
    opt_all = False

opt_header_only = a.header
opt_show_table = a.table
opt_debug = a.debug

if (opt_debug):
    print("dbg:opt_header_only:", opt_header_only)
    print("dbg:opt_show_table:", opt_show_table)
    print("dbg:opt_numbers:", opt_numbers)
    print("dbg:file:", file)

f = GT3File(file)
f.opt_debug = opt_debug
f.scan()

if (opt_show_table):
    f.show_table()
    sys.exit(0)

while True:
    f.read_one_header()
    if (f.is_eof):
        break
    if (opt_all or f.current_header.number in opt_numbers):
        f.dump_current_header()

    if (opt_header_only):
        f.skip_one_data()
    else:
        if (opt_all or f.current_header.number in opt_numbers):
            f.read_one_data()
            f.dump_current_data()
        else:
            f.skip_one_data()

f.close()
