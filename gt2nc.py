#!/usr/bin/env python3
from __future__ import print_function
import netCDF4
import numpy as np
# import pandas as pd
from pygt3file import GT3File, GT3Axis
import argparse
import sys
import os
import datetime


class A:
    """ general purpose bare class."""
    pass


###############################################################################
# Here We Go
###############################################################################
parser = argparse.ArgumentParser(description='Plot one data in gt3file')

parser.add_argument(
    '-d', '--debug', help='debug output',
    action='store_true')
parser.add_argument(
    '-v', '--verbose', help='verbose output',
    action='store_true')
parser.add_argument(
    '-T', '--table', help='Show data table only.',
    action='store_true', dest='show_table')
parser.add_argument(
    'ifile', help='input gt3 file name.')
parser.add_argument(
    '-o', '--ofile', help='output netcdf file name.',
    default='new.nc')
parser.add_argument(
    '-n', '--number', help='data number to plot.',
    type=int, default=0, dest='data_number')
parser.add_argument(
    '--axdir', help='axis file directory.',
    default=None)
parser.add_argument(
    '--zlib', help='use zlib compression.',
    action='store_true', dest='zlib')
parser.add_argument(
    '-c', '--create_ctl', help='create ctl file for GrADS.',
    action='store_true', dest='create_ctl')

opt = A()
parser.parse_args(namespace=opt)

if (opt.axdir is not None):
    opt.axdir = opt.axdir.split(':')

if (opt.debug):
    opt.verbose = True

if (opt.debug):
    print("dbg:show_table:", opt.show_table)
    print("dbg:data_number:", opt.data_number)
    print("dbg:axdir:", opt.axdir)
    print("dbg:create_ctl:", opt.create_ctl)
    print("dbg:zlib:", opt.zlib)
    print("dbg:ifile:", opt.ifile)
    print("dbg:ofile:", opt.ofile)

gf = GT3File(opt.ifile)
gf.opt_debug = opt.debug
gf.opt_verbose = opt.verbose

print('Read in %s.' % opt.ifile)
gf.scan()
if (opt.show_table):
    gf.show_table()
    sys.exit(0)

if (gf.num_of_items > 1):
    print('Multi items is not implemented yet, sorry')
    raise NotImplementedError

# for safety
if (any([gf.table['aitm1'].nunique() > 1,
         gf.table['aitm2'].nunique() > 1,
         gf.table['aitm3'].nunique() > 1])):
    print('Multi axis is not supported.')
    raise NotImplementedError

# freezed below for a while.
# if (opt.data_number not in range(gf.num_of_data)):
#     print('Error: data number out of range: %d is not in range(%d)'
#           % (opt.data_number, gf.num_of_data))
#     sys.exit(1)

gf.read_one_header()
gf.read_one_data()

if (gf.current_header.dfmt == 'UR4'):
    dtype = 'f4'
elif (gf.current_header.dfmt == 'UR8'):
    dtype = 'f8'
elif (gf.current_header.dfmt == 'URC'):
    dtype = 'f4'
elif (gf.current_header.dfmt[:3] == 'URY'):
    dtype = 'f4'
else:
    print('Error: Unknown dfmt: %s' % gf.current_header.dfmt)
    sys.exit(1)

xax = GT3Axis(gf.current_header.aitm1, opt.axdir)
if (xax is None):
    sys.exit(1)
if (opt.debug):
    xax.dump()
if (xax.unit == 'deg'):
    xax.unit = 'degrees_east'

yax = GT3Axis(gf.current_header.aitm2, opt.axdir)
if (yax is None):
    sys.exit(1)
if (opt.debug):
    yax.dump()
if (yax.unit == 'deg'):
    yax.unit = 'degrees_north'

zax = GT3Axis(gf.current_header.aitm3, opt.axdir)
if (zax is None):
    sys.exit(1)
if (opt.debug):
    zax.dump()

###############################################################################
# netCDF4
###############################################################################
nf = netCDF4.Dataset(opt.ofile, mode='w')
nf.title = gf.current_header.titl
# nf.Conventions = 'CF-1.4'
nf.history = 'Converted at  %s by pygt3file library.' \
             % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

xdim = nf.createDimension(xax.title, xax.size)
ydim = nf.createDimension(yax.title, yax.size)
zdim = nf.createDimension(zax.title, zax.size)
tdim = nf.createDimension('time', None)  # unlimited axis

xvar = nf.createVariable(xdim.name, np.float32, (xdim.name,))
xvar.long_name = xdim.name
xvar.units = xax.unit

yvar = nf.createVariable(ydim.name, np.float32, (ydim.name,))
yvar.long_name = ydim.name
yvar.units = yax.unit

zvar = nf.createVariable(zdim.name, np.float32, (zdim.name,))
zvar.long_name = zdim.name
zvar.units = zax.unit

tvar = nf.createVariable('time', np.float64, ('time',))
tvar.long_name = 'time'
tvar.units = 'seconds since 1970-01-01 00:00:00'

ncvar = nf.createVariable(
    gf.current_header.item,
    dtype,
    (tdim.name, zdim.name, ydim.name, xdim.name),
    chunksizes=(1, zax.size, yax.size, xax.size),
    zlib=opt.zlib)
ncvar.long_name = gf.current_header.titl
ncvar.units = gf.current_header.unit

xvar[:] = xax.data
yvar[:] = yax.data
zvar[:] = zax.data

tidx = 0
ncvar[tidx, :, :, :] = gf.current_data
ts = gf.current_header.date.astype(datetime.datetime)
tvar[tidx] = netCDF4.date2num(ts, units=tvar.units)
if (opt.verbose):
    print('Item: %s' % gf.current_header.item)
    print('Converted tidx=%d' % tidx)

while True:
    gf.read_one_header()
    if (gf.is_eof):
        break
    gf.read_one_data()
    tidx += 1
    ncvar[tidx, :, :, :] = gf.current_data
    ts = gf.current_header.date.astype(datetime.datetime)
    tvar[tidx] = netCDF4.date2num(ts, units=tvar.units)
    if (opt.verbose):
        print('Converted tidx=%d' % tidx)

gf.close()

print('Created %s.' % opt.ofile)

if (opt.debug):
    print('==== netCDF file: ====')
    print(nf)
    print('==== netCDF var: ====')
    print(ncvar)
    print("-- Some pre-defined attributes for variable :")
    print("dimensions:", ncvar.dimensions)
    print("shape:", ncvar.shape)
    print("dtype:", ncvar.dtype)
    print("ndim:", ncvar.ndim)
    print('==== netCDF axis: ====')
    for dim in nf.dimensions.items():
        print(dim)

if (opt.create_ctl):
    ctlfile = os.path.splitext(opt.ofile)[0]+'.ctl'
    with open(ctlfile, 'w') as cf:
        cf.write(u'DSET ^%s\n' % os.path.basename(opt.ofile))
        cf.write(u'DTYPE netcdf\n')
        cf.write(u'XDEF %s\n' % xdim.name)
        cf.write(u'YDEF %s\n' % ydim.name)
        cf.write(u'ZDEF %s\n' % zdim.name)
        cf.write(u'TDEF %s\n' % tdim.name)
        cf.write(u'VARS 1\n')
        cf.write(u'%s\n' % ncvar.name)
        cf.write(u'ENDVARS\n')
    print('Created %s.' % ctlfile)
nf.close()

sys.exit(0)
