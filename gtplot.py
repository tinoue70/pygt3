#!/usr/bin/env python3
from __future__ import print_function
# import numpy as np
from pygt3file import GT3File, GT3Axis
import argparse
import sys
import matplotlib.pyplot as plt
# import mpl_toolkits.basemap as basemap
# from cartopy import config
import cartopy.crs as ccrs


class A:
    """ base class for argparse,"""
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
# parser.add_argument(
#     '-H', '--header', help='Show header only.',
#     action='store_true')
parser.add_argument(
    '-c', '--contour', help='contour plot',
    action='store_true')
parser.add_argument(
    '-T', '--table', help='Show data table only.',
    action='store_true')
parser.add_argument(
    'file', help='file name.')

parser.add_argument(
    '-l', '--level', help='vertical level to plot.',
    type=int, default=0)

parser.add_argument(
    '-n', '--number', help='data number to plot.',
    type=int, default=0)

parser.add_argument(
    '--axdir', help='axis file directory.',
    default=None)

opt = A()
parser.parse_args(namespace=opt)

file = opt.file

opt.data_number = opt.number
opt.vert_level = opt.level
if (opt.axdir is not None):
    opt.axdir = opt.axdir.split(':')
else:
    opt.axdir = None

opt.show_table = opt.table
opt.debug = opt.debug
opt.verbose = opt.verbose

opt.contour = opt.contour

if (opt.debug):
    opt.verbose = True

if (opt.debug):
    print("dbg:opt.contour:", opt.contour)
    print("dbg:opt.show_table:", opt.show_table)
    print("dbg:opt.data_number:", opt.data_number)
    print("dbg:opt.vert_level:", opt.vert_level)
    print("dbg:opt.axdir:", opt.axdir)
    print("dbg:file:", file)

f = GT3File(file)
f.opt_debug = opt.debug
f.opt_verbose = opt.verbose

f.scan()

if (opt.show_table):
    f.show_table()
    sys.exit(0)


if (opt.data_number not in range(f.num_of_data)):
    print('Error: data number out of range: %d is not in range(0:%d)'
          % (opt.data_number, f.num_of_data))
    sys.exit(1)

###############################################################################
# Extract target data
###############################################################################

target_header = None
target_data = None

while True:
    f.read_one_header()
    if (f.is_eof):
        break
    if (opt.data_number == f.current_header.number):
        if (opt.debug):
            print("dbg: data #%d found." % opt.data_number)
        f.read_one_data()
        break
    else:
        f.skip_one_data()

target_header = f.current_header
target_data = f.current_data
if (opt.verbose):
    target_header.dump()

if (opt.debug):
    print("dbg: target_data:")
    print("* frags:")
    print(target_data.flags)
    print("* dtype:", target_data.dtype)
    print("* size:", target_data.size)
    print("* itemsize:", target_data.itemsize)
    print("* ndim:", target_data.ndim)
    print("* shape:", target_data.shape)

if (opt.vert_level not in range(target_data.shape[0])):
    print('Error: vertical level out of range: %d is not in range(:%d)'
          % (opt.vert_level, target_data.shape[0]))
    sys.exit(1)

###############################################################################
# Prepare axis data
###############################################################################

xax = GT3Axis(target_header.aitm1, opt.axdir)
if (xax.file is None):
    sys.exit(1)
if (opt.debug):
    xax.dump()

yax = GT3Axis(target_header.aitm2, opt.axdir)
if (yax.file is None):
    sys.exit(1)
if (opt.debug):
    yax.dump()

zax = GT3Axis(target_header.aitm3, opt.axdir)
if (zax.file is None):
    sys.exit(1)
if (opt.debug):
    zax.dump()

zlevel = zax.data[opt.vert_level]

###############################################################################
# Start plotting
###############################################################################

# if (opt.debug):
#     print("dbg: target_data.shape:", target_data.shape)
#     print("dbg: xax.data.shape:", xax.data.shape)
#     print("dbg: yax.data.shape:", yax.data.shape)

# 1D plot
# fig = plt.figure
# ax = plt.axes()
# plt.plot(yax.data,target_data[opt.vert_level, :, 1])
# plt.show()

# 2D plot
# ax = plt.axes(projection=ccrs.Mollweide())
# ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
if (opt.contour):
    img = ax.contour(xax.data, yax.data, target_data[opt.vert_level, :, :],
                     10, transform=ccrs.PlateCarree())
else:
    img = ax.contourf(xax.data, yax.data, target_data[opt.vert_level, :, :],
                      60, transform=ccrs.PlateCarree())
ax.axis("image")
title = "%s:%s[%s] lvl=%g[%s]" % (
    target_header.item, target_header.titl, target_header.unit,
    zax.data[opt.vert_level], zax.header.unit)
ax.set(title=title)
ax.coastlines()
plt.colorbar(img, ax=ax, orientation="horizontal", extend="both")
plt.show()
# done
f.close()
sys.exit(0)
