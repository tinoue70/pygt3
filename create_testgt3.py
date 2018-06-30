#!/usr/bin/env python
import numpy as np
# import pandas as pd
from pygt3file import GT3File, GT3Header, GT3Axis
import sys
import argparse

class A:
    """ general purpose bare class."""
    pass

###############################################################################
# Here We Go
###############################################################################

dset = 'test'
item = 'u-velocity'
title = 'eastward velocity'
unit = 'm/s'

date = "20180619 000000"
utim = 'sec'
dt = 3*3600  # 3hour
xax = 'GLON64'
yax = 'GGLA32'
zax = 'SFC1'
dfmt = 'UR4'

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
    '--dset', help='dset name.',
    default=dset)
parser.add_argument(
    '--item', help='item name.',
    default=item)
parser.add_argument(
    '--title', help='title of variable.',
    default=title)
parser.add_argument(
    '--unit', help='unit of item.',
    default=unit)
parser.add_argument(
    '--xax', help='X axis name.',
    default=xax)
parser.add_argument(
    '--yax', help='Y axis name.',
    default=yax)
parser.add_argument(
    '--zax', help='Z axis name.',
    default=zax)
parser.add_argument(
    '--axdir', help='axis file directory.',
    default=None)
parser.add_argument(
    '--date', help='initial date (YYYYMMDD HHMMSS)',
    default=date)
parser.add_argument(
    '--utim', help='unit of time (sec, day, etc.)',
    default=utim)
parser.add_argument(
    '-n', '--num_times', help='number of time steps.',
    type=int, default=5)
parser.add_argument(
    '--dt', help='delta t (in unit `utim`)',
    default=dt)
parser.add_argument(
    '--dfmt', help='dfmt for GTOOL3 format.',
    default=dfmt)

opt = A()
parser.parse_args(namespace=opt)

if (opt.debug):
    opt.verbose = True

if (opt.debug):
    print("dbg:ofile:", opt.ofile)
    print("dbg:num_times:", opt.num_times)
    print("dbg:item:", opt.item)
    print("dbg:title:", opt.title)
    print("dbg:unit:", opt.unit)
    print("dbg:xax:", opt.xax)
    print("dbg:yax:", opt.yax)
    print("dbg:zax:", opt.zax)
    print("dbg:axdir:", opt.axdir)
    print("dbg:date:", opt.date)
    print("dbg:utim:", opt.utim)
    print("dbg:dfmt:", opt.dfmt)

if (opt.axdir is not None):
    opt.axdir = opt.axdir.split(':')

xax = GT3Axis(opt.xax, opt.axdir)
if (xax.file is None):
    sys.exit(1)
if (opt.debug):
    xax.dump()
yax = GT3Axis(opt.yax, opt.axdir)
if (yax.file is None):
    sys.exit(1)
if (opt.debug):
    yax.dump()
zax = GT3Axis(opt.zax, opt.axdir)
if (zax.file is None):
    sys.exit(1)
if (opt.debug):
    zax.dump()

aitm1, astr1, aend1 = (xax.name, 1, xax.size)
aitm2, astr2, aend2 = (yax.name, 1, yax.size)
aitm3, astr3, aend3 = (zax.name, 1, zax.size)

if (opt.dfmt == 'UR4'):
    dtype = '>f4'
elif (opt.dfmt == 'UR8'):
    dtype = '>f8'
elif (opt.dfmt == 'URC'):
    dtype = '>f8'
elif (opt.dfmt == 'URY'):
    dtype = '>f8'
else:
    print('Error: Invalid dfmt: %s' % opt.dfmt)
    sys.exit(1)


x = np.sin(np.linspace(-np.pi*2., np.pi*2., xax.size), dtype=dtype)
y = np.cos(np.linspace(-np.pi, np.pi, yax.size), dtype=dtype)
z = np.exp(np.linspace(np.pi, 0., zax.size), dtype=dtype)
d = np.outer(z,np.outer(y,x)).reshape(zax.size,yax.size,xax.size)

with GT3File(opt.ofile, mode='wb') as f:
    f.current_header = GT3Header(
        dset=opt.dset,
        item=opt.item, unit=opt.unit, title=opt.title,
        dfmt=opt.dfmt,
        date=opt.date, utim=opt.utim,
        aitm1=aitm1, astr1=astr1, aend1=aend1,
        aitm2=aitm2, astr2=astr2, aend2=aend2,
        aitm3=aitm3, astr3=astr3, aend3=aend3)

    f.set_current_data(d)

    f.write_one_header()
    f.write_one_data()

    for i in range(opt.num_times-1):
        time = f.current_header.time\
               + np.timedelta64(opt.dt, unit=opt.utim)

        f.current_header.set_time_date(time=time)

        f.write_one_header()
        f.write_one_data()
