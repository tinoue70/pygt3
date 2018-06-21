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

a = A()
parser.parse_args(namespace=a)

file = a.file

opt_data_number = a.number
opt_vert_level = a.level
if (a.axdir is not None):
    opt_axdir = a.axdir.split(':')
else:
    opt_axdir = None

opt_show_table = a.table
opt_debug = a.debug
opt_verbose = a.verbose

opt_contour = a.contour

if (opt_debug):
    opt_verbose = True

if (opt_debug):
    print("dbg:opt_contour:", opt_contour)
    print("dbg:opt_show_table:", opt_show_table)
    print("dbg:opt_data_number:", opt_data_number)
    print("dbg:opt_vert_level:", opt_vert_level)
    print("dbg:opt_axdir:", opt_axdir)
    print("dbg:file:", file)

f = GT3File(file)
f.opt_debug = opt_debug
f.opt_verbose = opt_verbose

f.scan()

if (opt_show_table):
    f.show_table()
    sys.exit(0)


if (opt_data_number not in range(f.num_of_data)):
    print('Error: data number out of range: %d is not in range(0:%d)'
          % (opt_data_number, f.num_of_data))
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
    if (opt_data_number == f.current_header.number):
        if (opt_debug):
            print("dbg: data #%d found." % opt_data_number)
        f.read_one_data()
        break
    else:
        f.skip_one_data()

target_header = f.current_header
target_data = f.current_data
if (opt_verbose):
    target_header.dump()

if (opt_debug):
    print("dbg: target_data:")
    print("* frags:")
    print(target_data.flags)
    print("* dtype:", target_data.dtype)
    print("* size:", target_data.size)
    print("* itemsize:", target_data.itemsize)
    print("* ndim:", target_data.ndim)
    print("* shape:", target_data.shape)

if (opt_vert_level not in range(target_data.shape[0])):
    print('Error: vertical level out of range: %d is not in range(:%d)'
          % (opt_vert_level, target_data.shape[0]))
    sys.exit(1)

###############################################################################
# Prepare axis data
###############################################################################

xax = GT3Axis(target_header.aitm1, opt_axdir)
if (xax.file is None):
    sys.exit(1)
if (opt_debug):
    xax.dump()

yax = GT3Axis(target_header.aitm2, opt_axdir)
if (yax.file is None):
    sys.exit(1)
if (opt_debug):
    yax.dump()

zax = GT3Axis(target_header.aitm3, opt_axdir)
if (zax.file is None):
    sys.exit(1)
if (opt_debug):
    zax.dump()

zlevel = zax.data[opt_vert_level]

###############################################################################
# Start plotting
###############################################################################

# if (opt_debug):
#     print("dbg: target_data.shape:", target_data.shape)
#     print("dbg: xax.data.shape:", xax.data.shape)
#     print("dbg: yax.data.shape:", yax.data.shape)

# 1D plot
# fig = plt.figure
# ax = plt.axes()
# plt.plot(yax.data,target_data[opt_vert_level, :, 1])
# plt.show()

# 2D plot
# ax = plt.axes(projection=ccrs.Mollweide())
# ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
if (opt_contour):
    img = ax.contour(xax.data, yax.data, target_data[opt_vert_level, :, :],
                     10, transform=ccrs.PlateCarree())
else:
    img = ax.contourf(xax.data, yax.data, target_data[opt_vert_level, :, :],
                      60, transform=ccrs.PlateCarree())
ax.axis("image")
title = "%s:%s[%s] lvl=%g[%s]" % (
    target_header.item, target_header.titl, target_header.unit,
    zax.data[opt_vert_level], zax.header.unit)
ax.set(title=title)
ax.coastlines()
plt.colorbar(img, ax=ax, orientation="horizontal", extend="both")
plt.show()
# done
f.close()
sys.exit(0)
