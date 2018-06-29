#!/usr/bin/env python
import numpy as np
# import pandas as pd
from pygt3file import GT3File, GT3Header
# import sys
import argparse

class A:
    """ general purpose bare class."""
    pass

###############################################################################
# Here We Go
###############################################################################

item = 'u-velocity'
title = 'eastward velocity'
unit = 'm/s'

init_date = "20180619 000000"
date = init_date
deltaT = 3*3600  # 3hour
utim = 'sec'


parser = argparse.ArgumentParser(description='Plot one data in gt3file')

parser.add_argument(
    '-d', '--debug', help='debug output',
    action='store_true')
parser.add_argument(
    '-v', '--verbose', help='verbose output',
    action='store_true')

parser.add_argument(
    '-o', '--ofile', help='output netcdf file name.',
    default='new.gt3')
parser.add_argument(
    '-n', '--num_times', help='number of time steps.',
    type=int, default=5)
parser.add_argument(
    '--item', help='item name.',
    default=item)
parser.add_argument(
    '--title', help='title of variable.',
    default=title)
parser.add_argument(
    '--unit', help='unit of item.',
    default=unit)

opt = A()
parser.parse_args(namespace=opt)

if (opt.debug):
    opt.verbose = True

if (opt.debug):
    print("dbg:ofile:", opt.ofile)
    print("dbg:num_times",opt.num_times)



aitm1, astr1, aend1 = ('GLON64', 1, 64)
aitm2, astr2, aend2 = ('GGRA32', 1, 32)
aitm3, astr3, aend3 = ('STDPL17', 1, 17)

d = np.empty(shape=(17,32,64), dtype='>f4')

with GT3File(opt.ofile, mode='wb') as f:
    f.current_header = GT3Header(
        dset='test',
        item=opt.item, unit=opt.unit, title=opt.title,
        date=date, utim=utim,
        aitm1=aitm1, astr1=1, aend1=64,
        aitm2=aitm2, astr2=1, aend2=32,
        aitm3=aitm3, astr3=1, aend3=17,
    )

    f.set_current_data(d)

    f.write_one_header()
    f.write_one_data()

    for i in range(opt.num_times-1):
        d[:,:,:] = d[:,:,:] + 1.0
        time = f.current_header.time + deltaT

        f.current_header.set_time_date(time=time)

        f.write_one_header()
        f.write_one_data()


