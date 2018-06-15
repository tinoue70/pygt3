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

################################################################################
# Here We Go
################################################################################


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
    'file', help='file name.')

parser.add_argument(
    '-l', '--level', help='vertical level to plot.',
    type=int, default=0)

parser.add_argument(
    '-n', '--number', help='data number to plot.',
    type=int, default=0)


a = A()
parser.parse_args(namespace=a)

file = a.file

opt_data_number = a.number
opt_vert_level = a.level

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
    print("dbg:opt_vert_level:", opt_vert_level)
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

################################################################################
# Extract target data
################################################################################

target_header = None
target_data = None

while True:
    f.read_one_header()
    if (f.is_eof):
        break
    # if (opt_debug):
    #     print('dbg: current data:', f.current_header.number)
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
    print(target_data.flags)
    print(target_data.dtype)
    print(target_data.size)
    print(target_data.itemsize)
    print(target_data.ndim)
    print(target_data.shape)
    print(target_data.size)


if (opt_vert_level not in range(target_data.shape[0])):
    print('Error: vertical level out of range: %d is not in range(%d)'
          % (opt_vert_level, target_data.shape[0]))
    sys.exit(1)

################################################################################
# Prepare axis data
################################################################################

xax = GT3Axis(target_header.aitm1)
if (xax.file is None):
    sys.exit(1)
if (opt_debug):
    xax.dump()

yax = GT3Axis(target_header.aitm2)
if (yax.file is None):
    sys.exit(1)
if (opt_debug):
    yax.dump()

################################################################################
# Start plotting
################################################################################
if (opt_debug):
    print("dbg: target_data.shape:", target_data.shape)
    print("dbg: xax.data.shape:", xax.data.shape)
    print("dbg: yax.data.shape:", yax.data.shape)

# ax = plt.axes(projection=ccrs.Mollweide())
# ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
img = ax.contourf(xax.data, yax.data, target_data[opt_vert_level, :, :], 60,
                  transform=ccrs.PlateCarree())

ax.axis("image")
title = "%s:%s[%s]" % (target_header.item, target_header.titl, target_header.unit)
ax.set(title=title)
ax.coastlines()

plt.colorbar(img, ax=ax, orientation="horizontal", extend="both")
plt.show()

sys.exit(0)
