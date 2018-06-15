#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
import math
import os

import unittest


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InvalidArgumentError(Error):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message='Invalid Argument', argument=''):
        self.message = message
        self.argument = argument


class BitPacker:
    base_bit_width = 32

    def calc_packed_length(olen, nbit):

        if (nbit > BitPacker.base_bit_width):
            raise InvalidArgumentError(nbit)

        return math.ceil(nbit*olen/BitPacker.base_bit_width)

    def unpack(packed, pack_bit_width, len_unpacked):
        """ From ndarray `packed` with packed width `pack_bit_width` extract values
        return "int32" ndarray whose length is `len_unpacked`,"""

        mask = (1 << pack_bit_width) - 1
        unpacked = np.empty(len_unpacked, dtype='int32')
        for i in range(0, len_unpacked):
            i2 = i // BitPacker.base_bit_width
            i3 = i % BitPacker.base_bit_width
            pos = pack_bit_width*i2 + ((pack_bit_width*i3)//BitPacker.base_bit_width)
            off = pack_bit_width + ((pack_bit_width*i3) % BitPacker.base_bit_width) - BitPacker.base_bit_width
            off2 = off-BitPacker.base_bit_width

            # if ( off > 0 ):
            #     print(i,i2,i3,pos,off)
            # else:
            #     print(i,i2,i3,pos,off,off2)

            if (off <= 0):
                val = (packed[pos] >> -off) & mask
            else:
                val = (packed[pos] << off) & mask
                if (off2 >= 0):
                    val = val | ((packed[pos+1] << off2) & mask)
                else:
                    val = val | ((packed[pos+1] >> -off2) & mask)
            unpacked[i] = val

        return unpacked


################################################################################
# Tests for BitPacker
################################################################################
class TestBitPacker(unittest.TestCase):
    def test_calc_packed_length_00(self):
        """ Test calc_packed_length(20,12) """
        self.assertEqual(BitPacker.calc_packed_length(20, 12), 8)

    def test_calc_packed_length_01(self):
        """ Test calc_packed_length(300,9) """
        self.assertEqual(BitPacker.calc_packed_length(300, 9), 85)

    def test_calc_packed_length_02(self):
        """ Test calc_packed_length(300, 32) """
        self.assertEqual(BitPacker.calc_packed_length(300, 32), 300)

    def test_calc_packed_length_03(self):
        """ Test calc_packed_length(300, 33), except InvalidArgumentError."""
        with self.assertRaises(InvalidArgumentError):
            print(BitPacker.calc_packed_length(256, 33), 256)

    def test_unpack_01(self):
        """ Unpack 12bits packed """
        packed = np.array([0x11122233, 0x34445556, 0x66777888, 0x77700000], 'int32')
        reference = np.array([0x111, 0x222, 0x333, 0x444, 0x555, 0x666, 0x777, 0x888, 0x777], 'int32')
        # for val in packed:
        #     print(format(val, '12d'), format(val, '#034b'), format(val, '#08x'))

        unpacked = BitPacker.unpack(packed, 12, 9)
        # for val in unpacked:
        #     print(format(val, '12d'), format(val, '#014b'), format(val, '#05x'))
        self.assertEqual(tuple(unpacked), tuple(reference))

    def test_unpack_02(self):
        """ Unpack 12bits packed """
        packed = np.array([0xfffeeedd, 0xdcccbbba, 0xaa999888, 0xfff00000], 'uint32')
        reference = np.array([0xfff, 0xeee, 0xddd, 0xccc, 0xbbb, 0xaaa, 0x999, 0x888, 0xfff], 'uint32')
        # for val in packed:
        #     print(format(val, '12d'), format(val, '#034b'), format(val, '#08x'))
        unpacked = BitPacker.unpack(packed, 12, 9)
        # for val in unpacked:
        #     print(format(val, '12d'), format(val, '#014b'), format(val, '#05x'))
        self.assertEqual(tuple(unpacked), tuple(reference))

    def test_unpack_11(self):
        """ Unpack 11bits packed """
        packed = np.array([-285631830, -1002019192, -2004352786, -536870912], 'uint32')
        reference = np.array([0x777, 0x666, 0x555, 0x444, 0x333, 0x222, 0x111, 0x000, 0x777])
        # for val in packed:
        #     print(format(val, '12d'), format(val, '#034b'), format(val, '#08x'))
        unpacked = BitPacker.unpack(packed, 11, 9)
        # for val in unpacked:
        #     print(format(val, '12d'), format(val, '#013b'), format(val, '#05x'))
        self.assertEqual(tuple(unpacked), tuple(reference))

    def test_unpack_21(self):
        """ Unpack 1bits packed """
        packed = np.array([1979711488], 'uint32')
        reference = np.array([0x000, 0x001, 0x001, 0x001, 0x000, 0x001, 0x001, 0x000, 0x000])
        # for val in packed:
        #     print(format(val, '12d'), format(val, '#034b'), format(val, '#08x'))
        unpacked = BitPacker.unpack(packed, 1, 9)
        # for val in unpacked:
        #     print(format(val, '12d'), format(val, '#013b'), format(val, '#05x'))
        self.assertEqual(tuple(unpacked), tuple(reference))


################################################################################
class GT3Header:
    """gtool3 format header."""

    def __init__(self,
                 dset='', item='', title='', unit='', date='', utim='',
                 aitm1='', astr1=0, aend1=0,
                 aitm2='', astr2=0, aend2=0,
                 aitm3='', astr3=0, aend3=0,
                 dfmt='UR4',
                 time=0, tdur=0):
        self.dset = dset
        self.item = item
        self.titl = title
        self.unit = unit
        self.time = int(time)
        self.date = date
        self.utim = utim
        self.tdur = int(tdur)
        self.aitm1 = aitm1
        self.astr1 = int(astr1)
        self.aend1 = int(aend1)
        self.aitm2 = aitm2
        self.astr2 = int(astr2)
        self.aend2 = int(aend2)
        self.aitm3 = aitm3
        self.astr3 = int(astr3)
        self.aend3 = int(aend3)
        self.dfmt = dfmt
        self.number = -1
        pass

    def set(self, hd, fname=None):
        self.dset = hd[1].strip().decode('UTF-8')
        self.cyclic = (self.dset[0] == 'C')
        self.item = hd[2].strip().decode('UTF-8')
        self.titl = (hd[13]+hd[14]).strip().decode('UTF-8')
        self.unit = hd[15].strip().decode('UTF-8')
        self.time = int(hd[24])
        self.date = hd[26].decode('UTF-8')
        self.tdur = int(hd[27])
        self.utim = hd[25].strip().decode('UTF-8')
        self.aitm1 = hd[28].strip().decode('UTF-8')
        self.astr1 = int(hd[29])
        self.aend1 = int(hd[30])
        self.aitm2 = hd[31].strip().decode('UTF-8')
        self.astr2 = int(hd[32])
        self.aend2 = int(hd[33])
        self.aitm3 = hd[34].strip().decode('UTF-8')
        self.astr3 = int(hd[35])
        self.aend3 = int(hd[36])
        self.dfmt = hd[37].strip().decode('UTF-8')
        self.miss = float(hd[38])
        self.dmin = float(hd[39])
        self.dmax = float(hd[40])
        self.divs = float(hd[41])
        self.divl = float(hd[42])
        self.styp = int(hd[43])
        self.coptn = hd[44].strip().decode('UTF-8')
        if (hd[45].strip().decode('UTF-8') != ''):
            self.ioptn = int(hd[45])
        else:
            self.ioptn = None
        if (hd[46].strip().decode('UTF-8') != ''):
            self.roptn = float(hd[45])
        else:
            self.roptn = None
        self.cdate = hd[59].strip().decode('UTF-8')
        self.csign = hd[60].strip().decode('UTF-8')
        self.mdate = hd[61].strip().decode('UTF-8')
        self.msign = hd[62].strip().decode('UTF-8')
        self.size = int(hd[63])

        self.isize = self.aend1 - self.astr1+1
        self.jsize = self.aend2 - self.astr2+1
        self.ksize = self.aend3 - self.astr3+1
        self.shape = (self.ksize, self.jsize, self.isize)

        if (self.dfmt[:3] == 'UR8'):
            self.data_bits = self.size*8+8
        elif (self.dfmt[:3] == 'UR4'):
            self.data_bits = self.size*4+8
        elif (self.dfmt[:3] == 'URC'):
            raise NotImplementedError
        elif (self.dfmt[:3] == 'URY'):
            knum = self.ksize
            ijnum = self.isize*self.jsize
            self.packed_bit_width = int(self.dfmt[3:])
            self.ijnum_packed = BitPacker.calc_packed_length(ijnum, self.packed_bit_width)
            self.data_bits = knum*2*8+8 + self.ijnum_packed*knum*8+8

        if (fname is not None):
            self.fname = fname
        pass

    def dump(self):
        if (self is not None):
            liner = '====== %s: header #%d ' % (self.fname, self.number)
            liner += "="*(80-len(liner))
            print(liner)
            print("dset : %s" % str(self.dset))
            print("item : %s[%s]: %s" % (self.item, self.unit, self.titl))
            print("date : %s(%d) with %d[%s]" % (self.date, self.time, self.tdur, self.utim))
            if (self.aitm3 != ''):
                print("axis : %s[%d:%d] x %s[%d:%d] x %s[%d:%d]"
                      % (self.aitm1, self.astr1, self.aend1,
                         self.aitm2, self.astr2, self.aend2,
                         self.aitm3, self.astr3, self.aend3))
            elif (self.aitm2 != ''):
                print("axis : %s[%d:%d] x %s[%d:%d]"
                      % (self.aitm1, self.astr1, self.aend1,
                         self.aitm2, self.astr2, self.aend2))
            else:
                print("axis : %s[%d:%d]"
                      % (self.aitm1, self.astr1, self.aend1))
            print("cycl :", self.cyclic)
            print("dfmt :", self.dfmt)
            print("miss :", self.miss)
            print("size :", self.size)
            print("cdate: %s by %s" % (self.cdate, self.csign))
            print("mdate: %s by %s" % (self.mdate, self.msign))
            print('=' * len(liner))
        pass


################################################################################
# Tests for GT3Header
################################################################################
class TestGT3Header(unittest.TestCase):
    # def test_hoge_01(self):
    #     pass

    pass


######################################################################
class GT3File:
    """ Class for abstracting GTOOL3 file."""
    def __init__(self, name, mode='rb'):
        if (mode == 'rb' or mode == 'wb'):
            pass
        else:
            raise InvalidArgumentError(mode)
        self.f = open(name, mode)
        self.name = name
        self.mode = mode
        self.current_header = GT3Header()
        self.current_data = None
        self.is_after_header = False
        self.is_eof = False
        self.num_of_data = -1
        self.num_of_times = -1
        self.num_of_items = -1
        self.table = None
        self.opt_debug = False
        self.opt_verbose = False

        # self.packer = BitPacker()
        return None

    def open(self, name, mode='rb'):
        """ re-open other file within the same instance. """
        self.close()
        self.f = open(name, mode)
        self.name = name
        self.mode = mode
        self.is_after_header = False
        self.is_eof = False
        return None

    def close(self):
        self.f.close()
        self.name = ''
        return None

    def rewind(self):
        self.f.seek(0)
        self.current_header = GT3Header()
        self.current_data = None
        self.is_after_header = False
        self.is_eof = False
        self.current_header.number = -1
        return None

    def scan(self):
        """
        Scan whole file and create data table, which is pandas.DataFrame instance.
        Note that file position is on the top after this method.
        """
        tbl = []
        self.num_of_data = 0
        self.rewind()
        while True:
            self.read_one_header()
            if (self.is_eof):
                break
            self.skip_one_data()
            self.num_of_data += 1
            tbl.append([self.current_header.item, self.current_header.time, self.current_header.dfmt])
        self.rewind()
        self.table = pd.DataFrame(tbl)
        self.table.columns = ['item', 'time', 'dfmt']
        self.num_of_times = self.table.pivot_table(index='time', aggfunc=[len]).shape[0]
        self.num_of_items = self.table.pivot_table(index='item', aggfunc=[len]).shape[0]

        if (self.opt_verbose):
            liner = "="*5 + " %s: Scan result: " % self.name
            liner += "="*(80-len(liner))
            print(liner)
            print("* num_of_data :", self.num_of_data)
            print("* num_of_times:", self.num_of_times)
            print("* num_of_items:", self.num_of_items)
            print("="*len(liner))

        return None

    def show_table(self):
        """
        Show data table, created by scan().
        """

        if (self.table is not None):
            liner = "="*5 + " Data table: "
            liner += "="*(80-len(liner))
            print(liner)
            print(self.table.to_string())
            print("="*len(liner))
        else:
            print("Data table is not created, use .scan() first.")

    def read_one_header(self):
        """
        Read one header and it as a `current_header`.
        """

        dt = np.dtype([("head", ">i4"), ("header", "a16", 64), ("tail", ">i4")])

        chunk = np.fromfile(self.f, dtype=dt, count=1)
        if (len(chunk)):
            self.current_header.set(chunk["header"][0], self.name)
            self.current_header.number += 1
            self.is_eof = False
            self.is_after_header = True
        else:
            self.current_header = None
            self.is_eof = True

        return None

    def read_one_data(self):
        if (self.current_header.dfmt[:3] == 'UR8'):
            dt = np.dtype([("head", ">i4"), ("data", ">f8", self.current_header.size), ("tail", ">i4")])
            chunk = np.fromfile(self.f, dtype=dt, count=1)
            if (chunk["head"] != chunk["tail"]):
                raise IOError
            self.current_data = np.array(chunk["data"][0]).reshape(self.current_header.shape)
        elif (self.current_header.dfmt[:3] == 'UR4'):
            dt = np.dtype([("head", ">i4"), ("data", ">f4", self.current_header.size), ("tail", ">i4")])
            chunk = np.fromfile(self.f, dtype=dt, count=1)
            if (chunk["head"] != chunk["tail"]):
                raise IOError
            self.current_data = np.array(chunk["data"][0]).reshape(self.current_header.shape)
        elif (self.current_header.dfmt[:3] == 'URC'):
            raise NotImplementedError
        elif (self.current_header.dfmt[:3] == 'URY'):
            packed_bit_width = int(self.current_header.dfmt[3:])
            # print("dbg:packed_bit_width=%d" % packed_bit_width)

            ijnum = self.current_header.isize * self.current_header.jsize
            knum = self.current_header.ksize
            # print("dbg:ijnum, knum=%d, %d" % (ijnum,knum))

            # coeffs[*,0] is the offset values, coeffs[*,1] is the scale values.
            dt = np.dtype([("head", ">i4"), ("data", ">f8", knum*2), ("tail", ">i4")])
            chunk = np.fromfile(self.f, dtype=dt, count=1)
            if (chunk["head"] != chunk["tail"]):
                raise IOError
            coeffs = chunk["data"][0].reshape(knum, 2)
            # print('dbg:type(coeffs):',type(coeffs),coeffs.shape, coeffs.dtype)
            # print("dbg:coeffs:\n",coeffs)

            ijnum_packed = BitPacker.calc_packed_length(ijnum, packed_bit_width)
            # print("dbg:ijnum_packed=%d" % ijnum_packed)
            dt = np.dtype([("head", ">i4"), ("data", ">i4", ijnum_packed*knum), ("tail", ">i4")])
            chunk = np.fromfile(self.f, dtype=dt, count=1)
            if (chunk["head"] != chunk["tail"]):
                raise IOError
            packed = chunk["data"][0].reshape(knum, ijnum_packed)
            # print("dbg:packed:",packed.dtype,packed.shape)

            for k in range(knum):
                unpacked = BitPacker.unpack(packed[k, :], packed_bit_width, ijnum)
                # print("dbg:unpacked:",unpacked)
                self.current_data = np.ndarray(shape=(knum, ijnum), dtype="float64")
                # print('dbg',self.current_data.shape)
                self.current_data[k, :] = coeffs[k, 0] + unpacked[:] * coeffs[k, 1]
            self.current_data = self.current_data.reshape(self.current_header.shape)
        elif (self.current_header.dfmt[:3] == 'MRY'):
            raise NotImplementedError

        self.is_after_header = False
        return None

    def skip_one_data(self):
        if (not self.is_after_header):
            self.read_one_header()
        if (self.current_header is None):
            return None

        size = self.current_header.size
        if (self.current_header.dfmt[:3] == 'UR8'):
            size = size*8+8
        elif (self.current_header.dfmt[:3] == 'UR4'):
            size = size*4+8
        elif (self.current_header.dfmt[:3] == 'URC'):
            raise NotImplementedError
        elif (self.current_header.dfmt[:3] == 'URY'):
            packed_bit_width = int(self.current_header.dfmt[3:])
            knum = self.current_header.ksize
            ijnum = self.current_header.isize*self.current_header.jsize
            ijnum_packed = BitPacker.calc_packed_length(ijnum, packed_bit_width)
            size = knum*2*8+8 + ijnum_packed*knum*8+8
        else:
            raise NotImplementedError

        self.f.seek(size, 1)
        self.is_after_header = False
        return None

    def dump_current_header(self):
        self.current_header.dump()
        return None

    def dump_current_data(self, **kwargs):
        np.set_printoptions(threshold=np.inf, linewidth=110, suppress=True)

        liner = '====== %s: data #%d ' % (self.name, self.current_header.number)
        liner += "="*(80-len(liner))
        if (self.opt_debug):
            print("dbg:current_data:")
            print("  flags:")
            print(self.current_data.flags)
            print("  dtype:", self.current_data.dtype)
            print("  size,itemsize:", self.current_data.size, self.current_data.itemsize)
            print("  ndim, shape, strides:", self.current_data.ndim, self.current_data.shape, self.current_data.strides)
        print(liner)
        if (len(kwargs) > 0):
            np.set_printoptions(**kwargs)
        print(self.current_data)
        print('='*len(liner))

        return None

    def read_nth_data(self, num):
        """
        Read and return `num`-th header and data as a tuple.

        If not found, return (None, None).
        """

        self.rewind()
        while True:
            self.read_one_header()
            if (self.opt_debug):
                print("dbg: reading data #%d." % self.current_header.number)
            if (self.is_eof):
                h = None
                d = None
                return(None, None)
                break
            if (num == self.current_header.number):
                self.read_one_data()
                if (self.opt_debug):
                    print("dbg: data #%d found." % num)
                h = self.current_header
                d = self.current_data
                break
            else:
                self.skip_one_data()

        return(h, d)


################################################################################
# Tests for GT3File
################################################################################
class TestGT3File(unittest.TestCase):
    def test_read_00(self):
        """ Test for opening not exist file. """
        with self.assertRaises(OSError):
            filename = 'prcpr'  # not exist
            mode = 'rb'
            f1 = GT3File(filename, mode)

    def test_read_01(self):
        """ Test for invalid mode."""
        with self.assertRaises(InvalidArgumentError):
            filename = 'prcpx'
            mode = 'w'  # error
            f1 = GT3File(filename, mode)


################################################################################
# Axis for GTOOL3
################################################################################
class GT3Axis():
    """
    gtool3 axis.
    """
    
    default_search_paths = [u".", u"$GT3AXISDIR", u"$GTOOLDIR/gt3"]

    def __init__(self, name, search_paths=None):
        self.name = name
        if ( search_paths is None):
            self.search_paths = self.default_search_paths
        else:
            self.search_paths = search_paths
        self.find_axfile()
        if (self.file is not None):
            f = GT3File(self.file)
        else:
            self.file = None
            return
        if (f is None):
            self = None
        f.scan()

        self.name = name   # "GLONxx" etc.
        self.header, self.data = f.read_nth_data(0)
        self.title = f.current_header.titl  # "longitude" etc.
        self.data = f.current_data.flatten()
        if (f.current_header.cyclic):
            self.data = self.data[:-1]
        f.close()
        self.size = len(self.data)
        pass

    def find_axfile(self):
        """
        Find gtool3 axis file with given axis name `name` from path listed as `self.search_paths`.

        Return path of the found axis file or `None` unless found.
        """
        axis_path = map(os.path.expandvars, self.search_paths)
        axis_path = [a for a in axis_path if os.path.exists(a)]

        found = False
        for axdir in axis_path:
            axfile = os.path.join(axdir, 'GTAXLOC.'+self.name)
            if (os.path.exists(axfile)):
                found = True
                break

        if (found):
            self.file = axfile
        else:
            print('Axis "%s" Not found in path(s): %s' % (self.name, ":".join(self.search_paths)))
            self.file = None

        pass

    def dump(self):
        liner = '='*6 + ' Axis: %s ' % self.name
        liner += '='*(80-len(liner))
        print(liner)
        print("path:", self.file)
        print("title:", self.title)
        print("size:", self.size)
        print("data:")
        print(self.data)
        print('='*len(liner))
        pass

################################################################################
# Here we go.
################################################################################


if __name__ == '__main__':

    unittest.main()
