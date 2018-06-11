#!/usr/bin/env python3
from __future__ import print_function
import netCDF4
import numpy as np
from pygt3file import GT3File
import argparse
import sys
import os

# opt_debug = True

class A:
    """ general purpose bare class."""
    pass


def find_axfile(name, search_path=[u".", u"$GT3AXISDIR", u"$GTOOLDIR/gt3"]):
    """
    Find gtool3 axis file with given axis name `name` from path listed as `search_path`.

    Return path of the found axis file or `None` unless found.
    """
    axis_path = map(os.path.expandvars, search_path)
    axis_path = [a for a in axis_path if os.path.exists(a)]

    if (opt_debug):
        print('dbg:axis_path:',axis_path)

    found = False
    for axdir in axis_path:
        # print(axdir)
        axfile = os.path.join(axdir, 'GTAXLOC.'+name)
        # print("dbg:axfile:",axfile)
        # print("dbg:os.path.exists(axfile):",os.path.exists(axfile))
        if (os.path.exists(axfile)):
            found = True
            break

    if (found):
        if (opt_verbose):
            print("dbg:found axfile:",axfile)
        return axfile
    else:
        print('Axis "%s" Not found in path(s): %s' % (name, ":".join(search_path)))
        return None

    return axfile


def read_axis(name):
    ax = A()
    ax.name = name   # "GLONxx" etc.
    ax.header = None
    ax.data = None

    if (opt_verbose):
        print("dbg:ax.name:", ax.name)
    fname = find_axfile(ax.name)
    if (fname is not None):
        f = GT3File(fname)
    else:
        sys.exit(1)
    if (f is None):
        return None
    f.scan()
    ax.header, ax.data = f.read_nth_data(0)
    if (opt_verbose):
        f.dump_current_header()
    if (opt_debug):
        f.dump_current_data()
    ax.title = f.current_header.titl  # "longitude" etc.
    ax.data = f.current_data.flatten()
    if (f.current_header.cyclic):
        ax.data = ax.data[:-1]
    f.close()
    ax.size = len(ax.data)
    return ax


parser = argparse.ArgumentParser(description='Plot one data in gt3file')

parser.add_argument(
    '-d', '--debug', help='debug output',
    action='store_true')
parser.add_argument(
    '-v', '--verbose', help='verbose output',
    action='store_true')
parser.add_argument(
    '-H', '--header', help='Show header only.',
    action='store_true')
parser.add_argument(
    '-T', '--table', help='Show data table only.',
    action='store_true')
parser.add_argument(
    'ifile', help='input gt3 file name.')
parser.add_argument(
    '-o', '--ofile', help='output netcdf file name.',
     default='new.nc')

parser.add_argument(
    '-n', '--number', help='data number to plot.',
    type=int, default=0)


a = A()
parser.parse_args(namespace=a)

ifile = a.ifile
ofile = a.ofile

opt_data_number = a.number

opt_header_only = a.header
opt_show_table = a.table
opt_debug = a.debug
opt_verbose = a.verbose

if (opt_debug):
    opt_verbose = True

if (opt_debug):
    print("dbg:opt_header_only:", opt_header_only)
    print("dbg:opt_show_table:", opt_show_table)
    print("dbg:opt_data_number:", opt_data_number)
    print("dbg:ifile:", ifile)
    print("dbg:ofile:", ofile)

################################################################################
# Extract target data
################################################################################
f = GT3File(ifile)
f.opt_debug = opt_debug
f.opt_verbose = opt_verbose

f.scan()
if (opt_show_table):
    f.show_table()
    sys.exit(0)

if (opt_data_number not in range(f.num_of_data)):
    print('Error: data number out of range: %d is not in range(%d)' 
          % (opt_data_number, f.num_of_data))
    sys.exit(1)

gtvar = A()
gtvar.header = None
gtvar.data = None

gtvar.header, gtvar.data = f.read_nth_data(opt_data_number)
if (opt_verbose):
    gtvar.header.dump()
f.close()
################################################################################
# Prepare axis data
################################################################################

xax = read_axis(gtvar.header.aitm1)
print(xax.title, xax.size)

yax = read_axis(gtvar.header.aitm2)
print(yax.title, yax.size)

zax = read_axis(gtvar.header.aitm3)
print(zax.title, zax.size)

if (zax is None):
    print("zaxis is None:", zax.title)
    sys.exit(1)

################################################################################
# netCDF4
################################################################################
nf = netCDF4.Dataset(ofile, mode='w')

xdim = nf.createDimension(xax.title, xax.size)
ydim = nf.createDimension(yax.title, yax.size)
zdim = nf.createDimension(zax.title, zax.size)

# todo: tdim

for dim in nf.dimensions.items():
    print(dim)


nf.title=gtvar.header.titl
print(nf.title)

xvar = nf.createVariable(xdim.name, np.float32, (xdim.name,))
xvar.long_name = xdim.name
yvar = nf.createVariable(ydim.name, np.float32, (ydim.name,))
yvar.long_name = ydim.name
zvar = nf.createVariable(zdim.name, np.float32, (zdim.name,))
zvar.long_name = zdim.name

ncvar = nf.createVariable(nf.title, np.float64, (zdim.name, ydim.name, xdim.name))

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

print(ncvar[:,:,:].shape, ncvar[:,:,:].min(), ncvar[:,:,:].max())
print(nf)
nf.close()

