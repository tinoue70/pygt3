#!/usr/bin/env python3
from __future__ import print_function
import netCDF4
import numpy as np
import pandas as pd
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
    '-c', '--create_ctl', help='create ctl file for GrADS.',
    action='store_true', dest='create_ctl')

opt = A()
parser.parse_args(namespace=opt)

ifile = opt.ifile
ofile = opt.ofile

if (opt.axdir is not None):
    opt.axdir = opt.axdir.split(':')

if (opt.debug):
    opt.verbose = True

if (opt.debug):
    print("dbg:opt.show_table:", opt.show_table)
    print("dbg:opt.data_number:", opt.data_number)
    print("dbg:opt.axdir:", opt.axdir)
    print("dbg:opt.create_ctl:", opt.create_ctl)
    print("dbg:ifile:", ifile)
    print("dbg:ofile:", ofile)

gf = GT3File(ifile)
gf.opt_debug = opt.debug
gf.opt_verbose = opt.verbose

gf.scan()
if (opt.show_table):
    gf.show_table()
    sys.exit(0)

if (gf.num_of_items > 1):
    print('Multi items is not implemented yet, sorry')
    raise NotImplementedError

# for safety
if (gf.table['aitm1'].nunique() >1
    or gf.table['aitm2'].nunique() >1
    or gf.table['aitm3'].nunique() >1):
    print('Multi axis is not supported.')
    raise NotImplementedError

# freezed below for a while.
# if (opt.data_number not in range(gf.num_of_data)):
#     print('Error: data number out of range: %d is not in range(%d)'
#           % (opt.data_number, gf.num_of_data))
#     sys.exit(1)

gtvar = A()
gtvar.header = None
gtvar.data = None

gf.read_one_header()
gf.read_one_data()

gtvar.header = gf.current_header
gtvar.data = gf.current_data

###############################################################################
# Prepare axis data
###############################################################################

xax = GT3Axis(gtvar.header.aitm1, opt.axdir)
if (xax.file is None):
    sys.exit(1)
if (opt.debug):
    xax.dump()

yax = GT3Axis(gtvar.header.aitm2, opt.axdir)
if (yax.file is None):
    sys.exit(1)
if (opt.debug):
    yax.dump()

zax = GT3Axis(gtvar.header.aitm3, opt.axdir)
if (zax.file is None):
    sys.exit(1)
if (opt.debug):
    zax.dump()

###############################################################################
# netCDF4
###############################################################################
nf = netCDF4.Dataset(ofile, mode='w')
nf.title = gtvar.header.titl

xdim = nf.createDimension(xax.title, xax.size)
ydim = nf.createDimension(yax.title, yax.size)
zdim = nf.createDimension(zax.title, zax.size)
tdim = nf.createDimension('time', None)  # unlimited axis

xvar = nf.createVariable(xdim.name, np.float32, (xdim.name,))
xvar.long_name = xdim.name
xvar.units = xax.header.unit
yvar = nf.createVariable(ydim.name, np.float32, (ydim.name,))
yvar.long_name = ydim.name
yvar.units = yax.header.unit
zvar = nf.createVariable(zdim.name, np.float32, (zdim.name,))
zvar.long_name = zdim.name
xvar.units = xax.header.unit
tvar = nf.createVariable('time', np.float64, ('time',))
tvar.long_name = 'time'
tvar.units = 'seconds since 1970-01-01 00:00:00'

chunksize = xax.size
ncvar = nf.createVariable(
    gtvar.header.item, np.float64, (tdim.name, zdim.name, ydim.name, xdim.name),
    chunksizes=(1,zax.size, yax.size, xax.size),
    zlib=True)
ncvar.long_name = gtvar.header.titl
ncvar.units = gtvar.header.unit

xvar[:] = xax.data
yvar[:] = yax.data
zvar[:] = zax.data

tidx = 0
ncvar[tidx,:,:,:] = gtvar.data
ts = gf.current_header.date.astype(datetime.datetime)
tvar[tidx] = netCDF4.date2num(ts,units=tvar.units)

while True:
    gf.read_one_header()
    if (gf.is_eof):
        break
    gf.read_one_data()
    tidx += 1
    ncvar[tidx,:,:,:] = gf.current_data
    ts = gf.current_header.date.astype(datetime.datetime)
    tvar[tidx] = netCDF4.date2num(ts,units=tvar.units)

gf.close()

if (opt.verbose):
    print('==== netCDF file: ====')
    print(nf)  #dbg
    print('==== netCDF var: ====')
    print(ncvar)
    print("-- Some pre-defined attributes for variable :")
    print("dimensions:", ncvar.dimensions)
    print("shape:", ncvar.shape)
    print("dtype:", ncvar.dtype)
    print("ndim:", ncvar.ndim)
    print('==== netCDF axis: ====')
    for dim in nf.dimensions.items(): #dbg
        print(dim)

ctlfile=os.path.splitext(ofile)[0]+'.ctl'

if (opt.create_ctl):
    with open(ctlfile,'w') as cf:
        cf.write('DSET ^%s\n' % os.path.basename(ofile))
        cf.write('DTYPE netcdf\n')
        cf.write('XDEF %s\n' % xdim.name)
        cf.write('YDEF %s\n' % ydim.name)
        cf.write('ZDEF %s\n' % zdim.name)
        cf.write('TDEF %s\n' % tdim.name)
        cf.write('VARS 1\n')
        cf.write('%s\n' % ncvar.name)
        cf.write('ENDVARS\n')

nf.close()

sys.exit(0)
