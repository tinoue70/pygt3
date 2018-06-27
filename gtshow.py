#!/usr/bin/env python3
from __future__ import print_function
# import numpy as np
from pygt3file import GT3File
import argparse
import sys


class A:
    """ base class for argparse,"""
    pass


def set_idx_range(idx):
    if (idx is None):
        idx = ()
    if (len(idx) == 0):
        pass
    elif (len(idx) == 1):
        idx.append(idx[0]+1)
    else:
        idx = idx[:2]
        idx.sort()
    return idx


description = 'Show contents of gt3file.'

epilog = """
Index range must be single integer to be sliced at, or two integers to
be a range.  Note that these are treated as a slice object, that is,
`-x 1` shows data[:,:,1], but `-x 0 1` shows data[:,:,0:1] so data[:,:,1]
is excluded. But NEVER use negative index.
"""

###############################################################################
# Here we go
###############################################################################
parser = argparse.ArgumentParser(
    description=description,
    epilog=epilog)

parser.add_argument(
    '-d', '--debug', help='debug output',
    action='store_true')
parser.add_argument(
    '-v', '--verbose', help='verbose output',
    action='store_true')

parser.add_argument(
    'file', help='gt3 file name.')

# How to show
parser.add_argument(
    '-H', '--header_only', help='Show header only.',
    action='store_true')
parser.add_argument(
    '-T', '--show_table', help='Show data table only.',
    action='store_true')
parser.add_argument(
    '-i', '--indexed', help='Show data with grid indices',
    action='store_true', default=False)

# What to show
parser.add_argument(
    '-n', '--numbers', help='Space separated data number(s) to be shown.',
    type=int, nargs='+', default=())
parser.add_argument(
    '-x', '--xidx', help='index range X-direction.',
    type=int, nargs='+', default=())
parser.add_argument(
    '-y', '--yidx', help='index range Y-direction.',
    type=int, nargs='+', default=())
parser.add_argument(
    '-z', '--zidx', help='index range Z-direction.',
    type=int, nargs='+', default=())


opt = A()
parser.parse_args(namespace=opt)

opt.all = (len(opt.numbers) == 0)

opt.xidx = set_idx_range(opt.xidx)
opt.yidx = set_idx_range(opt.yidx)
opt.zidx = set_idx_range(opt.zidx)


if (opt.debug):
    print("====== Options ======")
    print("  opt.header_only:", opt.header_only)
    print("  opt.show_table:", opt.show_table)
    print("  opt.indexed", opt.indexed)
    print("  opt.numbers:", opt.numbers)
    print("  opt.xidx:", opt.xidx)
    print("  opt.yidx:", opt.yidx)
    print("  opt.zidx:", opt.zidx)
    print("  file:", opt.file)


with GT3File(opt.file) as f:
    f.opt_debug = opt.debug
    f.opt_verbose = opt.verbose
    f.scan()

    if (opt.show_table):
        f.show_table()
        sys.exit(0)

    while True:
        f.read_one_header()
        if (f.is_eof):
            break
        if (opt.all or f.current_header.number in opt.numbers):
            f.dump_current_header()

        if (opt.header_only):
            f.skip_one_data()
        else:
            if (opt.all or f.current_header.number in opt.numbers):
                f.read_one_data()
                f.dump_current_data(xidx=opt.xidx,
                                    yidx=opt.yidx,
                                    zidx=opt.zidx,
                                    indexed=opt.indexed)
            else:
                f.skip_one_data()

sys.exit(0)
