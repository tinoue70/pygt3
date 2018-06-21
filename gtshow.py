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

parser.add_argument(
    '-x', '--xidx', help='slice at X-direction.',
    type=int, nargs='+', default=None)
parser.add_argument(
    '-y', '--yidx', nargs='+', help='slice at Y-direction.',
    type=int, default=None)
parser.add_argument(
    '-z', '--zidx', nargs='+', help='slice at Z-direction.',
    type=int, default=None)

    
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

if (a.xidx is None):
    opt_xidx = None
else:
    if (len(a.xidx) == 1):
        opt_xidx = a.xidx[0]
    else:
        opt_xidx = tuple(a.xidx[:2])  # should be sorted after sliced?

if (a.yidx is None):
    opt_yidx = None
else:
    if (len(a.yidx) == 1):
        opt_yidx = a.yidx[0]
    else:
        opt_yidx = tuple(a.yidx[:2])  # should be sorted after sliced?
if (a.zidx is None):
    opt_zidx = None
else:
    if (len(a.zidx) == 1):
        opt_zidx = a.zidx[0]
    else:
        opt_zidx = tuple(a.zidx[:2])  # should be sorted after sliced?

if (opt_debug):
    print("dbg:opt_header_only:", opt_header_only)
    print("dbg:opt_show_table:", opt_show_table)
    print("dbg:opt_numbers:", opt_numbers)
    print("dbg:opt_xidx:", opt_xidx)
    print("dbg:opt_yidx:", opt_yidx)
    print("dbg:opt_zidx:", opt_zidx)
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
            f.dump_current_data(xidx=opt_xidx, yidx=opt_yidx, zidx=opt_zidx)
        else:
            f.skip_one_data()

f.close()
