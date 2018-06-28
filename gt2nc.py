#!/usr/bin/env python3
from __future__ import print_function
import netCDF4
import numpy as np
from pygt3file import GT3File, GT3Axis
import argparse
import sys


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
    print("dbg:ifile:", ifile)
    print("dbg:ofile:", ofile)

###############################################################################
# Extract target data
###############################################################################
gf = GT3File(ifile)
gf.opt_debug = opt.debug
gf.opt_verbose = opt.verbose

gf.scan()
if (opt.show_table):
    gf.show_table()
    sys.exit(0)

if (opt.data_number not in range(gf.num_of_data)):
    print('Error: data number out of range: %d is not in range(%d)'
          % (opt.data_number, gf.num_of_data))
    sys.exit(1)

gtvar = A()
gtvar.header = None
gtvar.data = None

gtvar.header, gtvar.data = gf.read_nth_data(opt.data_number)
if (opt.verbose):
    gtvar.header.dump()
gf.close()
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

xdim = nf.createDimension(xax.title, xax.size)
ydim = nf.createDimension(yax.title, yax.size)
zdim = nf.createDimension(zax.title, zax.size)

# todo: tdim

for dim in nf.dimensions.items():
    print(dim)


nf.title = gtvar.header.titl
print(nf.title)

xvar = nf.createVariable(xdim.name, np.float32, (xdim.name,))
xvar.long_name = xdim.name
yvar = nf.createVariable(ydim.name, np.float32, (ydim.name,))
yvar.long_name = ydim.name
zvar = nf.createVariable(zdim.name, np.float32, (zdim.name,))
zvar.long_name = zdim.name

ncvar = nf.createVariable(
    nf.title, np.float64, (zdim.name, ydim.name, xdim.name))

print('==== var: ====')
print(ncvar)
print("-- Some pre-defined attributes for variable :")
print("dimensions:", ncvar.dimensions)
print("shape:", ncvar.shape)
print("dtype:", ncvar.dtype)
print("ndim:", ncvar.ndim)

xvar[:] = xax.data
yvar[:] = yax.data
zvar[:] = zax.data
ncvar[:] = gtvar.data

print(ncvar[:, :, :].shape, ncvar[:, :, :].min(), ncvar[:, :, :].max())
print(nf)
nf.close()

sys.exit(0)
