#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
import math
import os
import sys
import unittest
import tempfile
from collections import deque
# from datetime import datetime


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
            pos = pack_bit_width*i2 \
                + ((pack_bit_width*i3)//BitPacker.base_bit_width)
            off = pack_bit_width \
                + ((pack_bit_width*i3) % BitPacker.base_bit_width)\
                - BitPacker.base_bit_width
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


###############################################################################
# Tests for BitPacker
###############################################################################
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
        packed = np.array(
            [0x11122233, 0x34445556, 0x66777888, 0x77700000],
            'int32')
        result = BitPacker.unpack(packed, 12, 9)
        expected = np.array(
            [0x111, 0x222, 0x333, 0x444, 0x555, 0x666, 0x777, 0x888, 0x777],
            'int32')
        self.assertEqual(tuple(result), tuple(expected))

    def test_unpack_02(self):
        """ Unpack 12bits packed """
        packed = np.array(
            [0xfffeeedd, 0xdcccbbba, 0xaa999888, 0xfff00000],
            'uint32')
        result = BitPacker.unpack(packed, 12, 9)
        expected = np.array(
            [0xfff, 0xeee, 0xddd, 0xccc, 0xbbb, 0xaaa, 0x999, 0x888, 0xfff],
            'uint32')
        self.assertEqual(tuple(result), tuple(expected))

    def test_unpack_11(self):
        """ Unpack 11bits packed """
        packed = np.array(
            [-285631830, -1002019192, -2004352786, -536870912],
            'uint32')
        result = BitPacker.unpack(packed, 11, 9)
        expected = np.array(
            [0x777, 0x666, 0x555, 0x444, 0x333, 0x222, 0x111, 0x000, 0x777])
        self.assertEqual(tuple(result), tuple(expected))

    def test_unpack_21(self):
        """ Unpack 1bits packed """
        packed = np.array([1979711488], 'uint32')
        result = BitPacker.unpack(packed, 1, 9)
        expected = np.array(
            [0x000, 0x001, 0x001, 0x001, 0x000, 0x001, 0x001, 0x000, 0x000])
        self.assertEqual(tuple(result), tuple(expected))


###############################################################################
class GT3Header:
    """GTOOL3 format header.

    GTOOL3 format header is consist of
    `character(len=16,dimension=64)` array in Fortran, which I call
    `hdarray` here. See 'XXX' for a complete list of it.

    This class is an abstraction of this information. All of elements
    of `hdarray` are assigned as attributes of this class.
    Some of them are casted to integer and/or float type.

    Note that three attributes, `ettl`, `edit` and `memo`, are
    considered as a `queue` in original GTOOL3 format, and these are
    implemented as `deque` class, with limited `maxlen`, of
    `collections` package in this class.
    """

    def __init__(self,
                 dset='', item='', title='', unit='', date=None, utim='sec',
                 time=None, tdur=0,
                 aitm1='', astr1=0, aend1=0,
                 aitm2='', astr2=0, aend2=0,
                 aitm3='', astr3=0, aend3=0,
                 dfmt='UR4',
                 miss=-999., dmin=-999., dmax=-999., divs=-999., divl=-999.,
                 edit=None,
                 fnum=0, dnum=0,
                 ettl=None,
                 memo=None,
                 fname=None):

        self.dset = dset
        self.item = item

        self.edit = deque("", 8)
        self.ettl = deque("", 8)
        self.memo = deque("", 10)
        self.add_attribs(edit=edit, ettl=ettl, memo=memo)

        self.fnum = fnum
        self.dnum = dnum
        self.titl = title
        self.unit = unit
        self.utim = utim
        self.date = None
        self.time = None
        self.set_time_date(date, time)
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
        self.miss = miss
        self.dmin = dmin
        self.dmax = dmax
        self.divs = divs
        self.divl = divl
        self.styp = 1
        self.coptn = ''
        self.ioptn = 0
        self.roptn = 0.
        # self.time2 = time
        # self.utim2 = date
        self.cdate = "{0:%Y%m%d %H%M%S}".format(pd.Timestamp.now())
        self.csign = 'pygt3 library'
        self.mdate = self.cdate
        self.msign = 'pygt3 library'

        self.fname = fname
        self.number = -1
        self.set_hidden_attribs()

        pass

    def set_time_date(self, date=None, time=None, utim=None):
        """
        Set and keep `date`,`time` and `utim` attributes consistently.

        `date` must be:
        - 'compact' : 'YYYYMMDD HHMMSS' that is used in GTOOL3 header array,
        - 'standard': 'YYYY-MM-DD HH-MM-SS' or acceptable by np.datetime64(),
        - np.datetime64 instance.

        According to the GTOOL3 specification, `time` field is counted
        from '0000/01/01 00:00:00', so we use np.datetime64 here.


        If date is given, convert it to time, and vice versa, then set to self.
        If both is given, time is discarded even if not None.
        """

        if (date is None and time is None):
            return None
        if (date is not None and time is not None):
            raise InvalidArgumentError()

        if (utim is None):
            utim = self.utim
        if (utim is None):
            utim = 'sec'
        self.utim = utim

        if (utim[0].upper() == 'Y'):
            unit = 'Y'
        elif (utim[:2].upper() == 'MO'):
            unit = 'M'
        elif (utim[0].upper() == 'D'):
            unit = 'D'
        elif (utim[0].lower() == 'h'):
            unit = 'h'
        elif (utim[0].lower() == 'm'):  # 'minutes' or 'mn'
            unit = 'm'
        elif (utim[0].lower() == 's'):
            unit = 's'
        else:
            raise InvalidArgumentError(utim)

        orig = np.datetime64('0000-01-01', unit)

        if (date is not None):  # date -> time
            if (isinstance(date, str)):
                if (len(date.strip()) == 15 and date[8] == ' '):
                    # seems compact form
                    date = date[0:4] + '-' + date[4:6] + '-' \
                           + date[6:8] + ' ' + date[9:11] + ':' \
                           + date[11:13] + ':' + date[13:15]
                else:
                    pass
            self.date = np.datetime64(date, unit)
            self.time = self.date - orig
        else:  # time -> date
            self.time = np.timedelta64(time, unit)
            self.date = time + orig

        return None

    def set_from_hdarray(self, hdarray, fname=None):
        """
        Set GT3Header from hdarray, which is an 'a16'*64 ndarray.

        In some of gtool3 datefile, such as axis, input data, etc.,
        year in `date` field is set as 0000, which cannot be handled
        by standard datetime module. So these are set as `None` here
        tentatively.
        """
        self.dset = hdarray[1].strip().decode('UTF-8')
        self.item = hdarray[2].strip().decode('UTF-8')
        fnum = hdarray[11].strip().decode('UTF-8')
        if (fnum == ''):
            self.fnum = None
        else:
            self.fnum = int(fnum)
        dnum = hdarray[12].strip().decode('UTF-8')
        if (dnum == ''):
            self.dnum = None
        else:
            self.dnum = int(dnum)
        self.titl = (hdarray[13]+hdarray[14]).strip().decode('UTF-8')
        self.unit = hdarray[15].strip().decode('UTF-8')
        self.utim = hdarray[25].strip().decode('UTF-8')
        self.set_time_date(hdarray[26].strip().decode('UTF-8'))
        self.tdur = int(hdarray[27])
        self.aitm1 = hdarray[28].strip().decode('UTF-8')
        self.astr1 = int(hdarray[29])
        self.aend1 = int(hdarray[30])
        self.aitm2 = hdarray[31].strip().decode('UTF-8')
        self.astr2 = int(hdarray[32])
        self.aend2 = int(hdarray[33])
        self.aitm3 = hdarray[34].strip().decode('UTF-8')
        self.astr3 = int(hdarray[35])
        self.aend3 = int(hdarray[36])
        self.dfmt = hdarray[37].strip().decode('UTF-8')
        self.miss = float(hdarray[38])
        self.dmin = float(hdarray[39])
        self.dmax = float(hdarray[40])
        self.divs = float(hdarray[41])
        self.divl = float(hdarray[42])
        self.styp = int(hdarray[43])
        self.coptn = hdarray[44].strip().decode('UTF-8')
        if (hdarray[45].strip().decode('UTF-8') != ''):
            self.ioptn = int(hdarray[45])
        else:
            self.ioptn = None
        if (hdarray[46].strip().decode('UTF-8') != ''):
            self.roptn = float(hdarray[46])
        else:
            self.roptn = None
        # self.time2 = hdarray[47].strip().decode('UTF-8')
        # self.utim2 = hdarray[48].strip().decode('UTF-8')

        x1 = list(hdarray[i+3].strip().decode('UTF-8') for i in range(8))
        y1 = None
        y1 = list(x1[i] for i in range(8) if x1[i] != u'')

        x2 = list(hdarray[i+16].strip().decode('UTF-8') for i in range(8))
        y2 = None
        y2 = list(x2[i] for i in range(8) if x2[i] != u'')

        x3 = list(hdarray[i+49].strip().decode('UTF-8') for i in range(10))
        y3 = None
        y3 = list(x3[i] for i in range(10) if x3[i] != u'')
        self.add_attribs(edit=y1, ettl=y2, memo=y3)

        self.cdate = hdarray[59].strip().decode('UTF-8')
        self.csign = hdarray[60].strip().decode('UTF-8')
        self.mdate = hdarray[61].strip().decode('UTF-8')
        self.msign = hdarray[62].strip().decode('UTF-8')
        self.size = int(hdarray[63])

        self.set_hidden_attribs()

        if (fname is not None):
            self.fname = fname
        pass

    def set_hidden_attribs(self):
        """
        Set "hidden" attributes.
        """
        self.cyclic = ((len(self.dset) > 0) and self.dset[0] == 'C')
        self.isize = self.aend1 - self.astr1+1
        self.jsize = self.aend2 - self.astr2+1
        self.ksize = self.aend3 - self.astr3+1
        self.shape = (self.ksize, self.jsize, self.isize)
        self.size = self.isize * self.jsize * self.ksize
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
            self.ijnum_packed = BitPacker.calc_packed_length(
                ijnum, self.packed_bit_width)
            self.data_bits = knum*2*8+8 + self.ijnum_packed*knum*8+8

    def add_attribs(self, ettl=None, edit=None, memo=None):
        """
        Add queue type attributes,  `ettl`, `edit`, `memo`.

        These three attributes are implemented as __queue__, whose length
        is 8, 8, 10, respectively.  If length of resulting list are
        more than that, FIFO manner is applied.  Given argument must be
        single string or a list, each element must be 16 characters or
        less.

        """
        if (ettl is not None):
            if (isinstance(ettl, list)):
                self.ettl.extend(ettl)
            elif (isinstance(ettl, tuple)):
                self.ettl.extend(list(ettl))
            else:
                self.ettl.append(ettl)
        if (edit is not None):
            if (isinstance(edit, list)):
                self.edit.extend(edit)
            elif (isinstance(edit, tuple)):
                self.edit.extend(list(edit))
            else:
                self.edit.append(edit)
        if (memo is not None):
            if (isinstance(memo, list)):
                self.memo.extend(memo)
            elif (isinstance(memo, tuple)):
                self.memo.extend(list(memo))
            else:
                self.memo.append(memo)
        pass

    def dump(self, file=None):
        """
        Output summarize of this instance.

        If `file` is given, this is given to print() method.
        """
        if (self is not None):
            liner = '====== %s: header #%d ' % (self.fname, self.number)
            liner += "="*(80-len(liner))
            print(liner, file=file)
            print("dset : %s" % str(self.dset), file=file)
            print("item : %s[%s]: %s"
                  % (self.item, self.unit, self.titl), file=file)
            print("date : %s(%s) with %d[%s]" %
                  (self.date, str(self.time).split()[0],
                   self.tdur, self.utim), file=file)
            if (self.aitm3 != ''):
                print("axis : %s[%d:%d] x %s[%d:%d] x %s[%d:%d]"
                      % (self.aitm1, self.astr1, self.aend1,
                         self.aitm2, self.astr2, self.aend2,
                         self.aitm3, self.astr3, self.aend3), file=file)
            elif (self.aitm2 != ''):
                print("axis : %s[%d:%d] x %s[%d:%d]"
                      % (self.aitm1, self.astr1, self.aend1,
                         self.aitm2, self.astr2, self.aend2), file=file)
            else:
                print("axis : %s[%d:%d]"
                      % (self.aitm1, self.astr1, self.aend1), file=file)
            print("cycl :", self.cyclic, file=file)
            print("dfmt :", self.dfmt, file=file)
            print("miss :", self.miss, file=file)
            print("size :", self.size, file=file)
            print("edit :", list(self.edit), file=file)
            print("ettl :", list(self.ettl), file=file)
            print("memo :", list(self.memo), file=file)
            print("cdate: %s by %s" % (self.cdate, self.csign), file=file)
            print("mdate: %s by %s" % (self.mdate, self.msign), file=file)
            print("isize,jsize,ksize: %d, %d, %d"
                  % (self.isize, self.jsize, self.ksize), file=file)
            print('=' * len(liner), file=file)
        pass

    def pack(self):
        """
        Pack attribs of this instance to the array suitable for gtool3
        header and return it.
        """
        hdarray = np.zeros((64,), dtype='a16')
        hdarray[0] = "%16d" % 9010
        hdarray[1] = "%-16s" % self.dset
        hdarray[2] = "%-16s" % self.item
        hdarray[3:11] = ["%-16s" % "" for i in range(8)]
        i = 0
        for v in self.edit:
            hdarray[3+i] = "%-16s" % v
            i += 1
        if (self.fnum is None):
            hdarray[11] = "%16d" % ''
        else:
            hdarray[11] = "%16d" % self.fnum
        if (self.dnum is None):
            hdarray[12] = "%16d" % ''
        else:
            hdarray[12] = "%16d" % self.dnum
        hdarray[13] = "%-16s" % self.titl[:16]
        hdarray[14] = "%-16s" % self.titl[16:]
        hdarray[15] = "%-16s" % self.unit
        hdarray[16:24] = ["%-16s" % "" for i in range(8)]
        i = 0
        for v in self.ettl:
            hdarray[16+i] = "%-16s" % v
            i += 1
        hdarray[24] = "%16d" % int(str(self.time).split()[0])
        hdarray[25] = "%-16s" % self.utim
        # np.datetime64 doesn't have strftime, use pd.Timestamp instead.
        hdarray[26] = "%-16s" \
                      % pd.to_datetime(self.date).strftime("%Y%m%d %H%M%S")
        hdarray[27] = "%16d" % self.tdur
        hdarray[28] = "%-16s" % self.aitm1
        hdarray[29] = "%16d" % self.astr1
        hdarray[30] = "%16d" % self.aend1
        hdarray[31] = "%-16s" % self.aitm2
        hdarray[32] = "%16d" % self.astr2
        hdarray[33] = "%16d" % self.aend2
        hdarray[34] = "%-16s" % self.aitm3
        hdarray[35] = "%16d" % self.astr3
        hdarray[36] = "%16d" % self.aend3
        hdarray[37] = "%-16s" % self.dfmt
        hdarray[38] = "%16.7e" % self.miss
        hdarray[39] = "%16.7e" % self.dmin
        hdarray[40] = "%16.7e" % self.dmax
        hdarray[41] = "%16.7e" % self.divs
        hdarray[42] = "%16.7e" % self.divl
        hdarray[43] = "%16d" % self.styp
        hdarray[44] = "%-16s" % self.coptn
        hdarray[45] = "%16d" % self.ioptn
        hdarray[46] = "%16.7e" % self.roptn
        # hdarray[47] = "%-16s" % self.time2
        # hdarray[48] = "%-16s" % self.utim2
        hdarray[47] = "%-16s" % ""
        hdarray[48] = "%-16s" % ""
        hdarray[49:59] = ["%-16s" % "" for i in range(10)]
        i = 0
        for v in self.memo:
            hdarray[49+i] = "%-16s" % v
            i += 1
        hdarray[59] = "%-16s" % self.cdate
        hdarray[60] = "%-16s" % self.csign
        hdarray[61] = "%-16s" % self.mdate
        hdarray[62] = "%-16s" % self.msign
        hdarray[63] = "%16d" % self.size

        return hdarray


###############################################################################
# Tests for GT3Header
###############################################################################
class TestGT3Header(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        np.set_printoptions(threshold=np.inf, linewidth=90, suppress=True)
        self.orig_hdarray = np.array(
            [b'            9010', b'test            ', b'hoge            ', b'edit0           ',
             b'edit1           ', b'edit2           ', b'edit3           ', b'edit4           ',
             b'edit5           ', b'edit6           ', b'edit7           ', b'               1',
             b'               1', b'testdata for Tes', b'tGT3Header.pack(', b'-               ',
             b'etitle0         ', b'etitle1         ', b'etitle2         ', b'etitle3         ',
             b'etitle4         ', b'etitle5         ', b'etitle6         ', b'etitle7         ',
             b'          737226', b'DAYS            ', b'20180616 000000 ', b'               0',
             b'GLON64          ', b'               1', b'              64', b'GGLA32          ',
             b'               1', b'              32', b'SFC1            ', b'               1',
             b'               1', b'UR4             ', b'  -9.9900000e+02', b'  -9.9900000e+02',
             b'  -9.9900000e+02', b'  -9.9900000e+02', b'  -9.9900000e+02', b'               1',
             b'                ', b'               0', b'   0.0000000e+00', b'                ',
             b'                ', b'memo0           ', b'memo1           ', b'memo2           ',
             b'memo3           ', b'memo4           ', b'memo5           ', b'memo6           ',
             b'memo7           ', b'memo8           ', b'memo9           ', b'20180615 162709 ',
             b'pygt3 library   ', b'20180615 162709 ', b'pygt3 library   ', b'            2048'])
        pass

    def test_init_dump_00(self):
        """ __init__() and dump() """
        with tempfile.TemporaryFile('w+') as f:
            header = GT3Header(fname='test')
            header.dump(file=f)
            f.seek(0)
            result = f.read()
        expected = (
            "====== test: header #-1 ========================================================\n"
            "dset : \n"
            "item : []: \n"
            "date : None(None) with 0[sec]\n"
            "axis : [0:0]\n"
            "cycl : False\n"
            "dfmt : UR4\n"
            "miss : -999.0\n"
            "size : 1\n"
            "edit : []\n"
            "ettl : []\n"
            "memo : []\n"
            "cdate: %s by pygt3 library\n"
            "mdate: %s by pygt3 library\n"
            "isize,jsize,ksize: 1, 1, 1\n"
            "================================================================================\n"
            % (header.cdate, header.mdate))

        self.assertMultiLineEqual(result, expected)

    def test_set_from_hdarray_01(self):
        """ set_from_hdarray() and dump() """
        header = GT3Header(fname='test')
        header.set_from_hdarray(self.orig_hdarray, fname='test')
        with tempfile.TemporaryFile('w+') as f:
            header.dump(file=f)
            f.seek(0)
            result = f.read()
        expected = (
            "====== test: header #-1 ========================================================\n"
            "dset : test\n"
            "item : hoge[-]: testdata for TestGT3Header.pack(\n"
            "date : 2018-06-16(737226) with 0[DAYS]\n"
            "axis : GLON64[1:64] x GGLA32[1:32] x SFC1[1:1]\n"
            "cycl : False\n"
            "dfmt : UR4\n"
            "miss : -999.0\n"
            "size : 2048\n"
            "edit : ['edit0', 'edit1', 'edit2', 'edit3', 'edit4', 'edit5', 'edit6', 'edit7']\n"
            "ettl : ['etitle0', 'etitle1', 'etitle2', 'etitle3', 'etitle4', 'etitle5', 'etitle6', 'etitle7']\n"
            "memo : ['memo0', 'memo1', 'memo2', 'memo3', 'memo4', 'memo5', 'memo6', 'memo7', 'memo8', 'memo9']\n"
            "cdate: %s by pygt3 library\n"
            "mdate: %s by pygt3 library\n"
            "isize,jsize,ksize: 64, 32, 1\n"
            "================================================================================\n"
            % (header.cdate, header.mdate))
        self.assertMultiLineEqual(result, expected)

    def test_init_pack_02(self):
        """ __init__() and pack() GT3Header and hdarray """
        header = GT3Header(dset='test', item='hoge',
                           title='testdata for TestGT3Header.pack()', unit='-',
                           edit=["edit%d" % n for n in range(3)],
                           fnum=1, dnum=1,
                           ettl=["etitle%d" % n for n in range(3)],
                           date='20180616 151200', utim='DAYS',
                           aitm1='GLON64', astr1=1, aend1=64,
                           aitm2='GGLA32', astr2=1, aend2=32,
                           aitm3='SFC1', astr3=1, aend3=1,
                           memo=["memo%d" % n for n in range(10)])
        expected = np.array(
            [b'            9010', b'test            ', b'hoge            ', b'edit0           ',
             b'edit1           ', b'edit2           ', b'                ', b'                ',
             b'                ', b'                ', b'                ', b'               1',
             b'               1', b'testdata for Tes', b'tGT3Header.pack(', b'-               ',
             b'etitle0         ', b'etitle1         ', b'etitle2         ', b'                ',
             b'                ', b'                ', b'                ', b'                ',
             b'          737226', b'DAYS            ', b'20180616 000000 ', b'               0',
             b'GLON64          ', b'               1', b'              64', b'GGLA32          ',
             b'               1', b'              32', b'SFC1            ', b'               1',
             b'               1', b'UR4             ', b'  -9.9900000e+02', b'  -9.9900000e+02',
             b'  -9.9900000e+02', b'  -9.9900000e+02', b'  -9.9900000e+02', b'               1',
             b'                ', b'               0', b'   0.0000000e+00', b'                ',
             b'                ', b'memo0           ', b'memo1           ', b'memo2           ',
             b'memo3           ', b'memo4           ', b'memo5           ', b'memo6           ',
             b'memo7           ', b'memo8           ', b'memo9           ', b'20180616 150542 ',
             b'pygt3 library   ', b'20180616 150542 ', b'pygt3 library   ', b'          2048  '])

        hdarray = header.pack()

        # Skip check of hdarray[59:],these are changed by time to time.
        for n in range(59):
            self.assertEqual(expected[n], hdarray[n],
                             msg="n=%d is mismatch !!" % n)

    def test_set_from_hdarray_pack_03(self):
        """ set_from_hdarray() and pack() GT3Header and hdarray """
        header = GT3Header()
        header.set_from_hdarray(self.orig_hdarray, fname='test')
        hdarray = header.pack()
        # np.testing.assert_array_equal(self.orig_hdarray, hdarray)
        self.assertListEqual(list(hdarray), list(self.orig_hdarray))

    def test_add_attribs_00(self):
        """ add_attribs() for `ettl` """
        header = GT3Header()
        qq = deque('', maxlen=8)
        x = "x1"  # single string
        header.add_attribs(ettl=x)
        qq.append(x)
        self.assertSequenceEqual(header.ettl, qq)
        x = ("x2", "x3", "x4", "x5")  # tuple
        header.add_attribs(ettl=x)
        qq.extend(x)
        self.assertSequenceEqual(header.ettl, qq)
        x = ["x6", "x7", "x8", "x9"]  # list, over queue size
        header.add_attribs(ettl=x)
        qq.extend(x)
        self.assertSequenceEqual(header.ettl, qq)

    def test_add_attribs_01(self):
        """ add_attribs() for `edit` """
        header = GT3Header()
        qq = deque('', maxlen=8)
        x = "x1"  # one string
        header.add_attribs(edit=x)
        qq.append(x)
        self.assertSequenceEqual(header.edit, qq)
        x = ("x2", "x3", "x4", "x5")  # tuple
        header.add_attribs(edit=x)
        qq.extend(x)
        self.assertSequenceEqual(header.edit, qq)
        x = ["x6", "x7", "x8", "x9"]  # list, over queue size
        header.add_attribs(edit=x)
        qq.extend(x)
        self.assertSequenceEqual(header.edit, qq)
        # print(list(header.edit))

    def test_add_attribs_02(self):
        """ add_attribs() for `memo` """
        header = GT3Header()
        qq = deque('', maxlen=10)
        x = "x1"  # one string
        header.add_attribs(memo=x)
        qq.append(x)
        self.assertSequenceEqual(header.memo, qq)
        x = ("x2", "x3", "x4", "x5")  # tuple
        header.add_attribs(memo=x)
        qq.extend(x)
        self.assertSequenceEqual(header.memo, qq)
        x = ["x6", "x7", "x8", "x9", "xA", "xB", "xC"]  # list, over queue size
        header.add_attribs(memo=x)
        qq.extend(x)
        self.assertSequenceEqual(header.memo, qq)
        # print(list(header.memo))

    def test_set_time_date_00(self):
        """ test set_time_date. """
        t = (2018, 6, 19, 10, 16, 23)
        expect = {'date': np.datetime64('2018-06-19'),
                  'time': np.timedelta64(737229, 'D')}

        h = GT3Header(utim='days')
        h.set_time_date(date="%04d%02d%02d %02d%02d%02d" % t)
        self.assertEqual(h.date, expect['date'])
        self.assertEqual(h.time, expect['time'])

    def test_set_time_date_01(self):
        """ test set_time_date. """
        t = (2018, 6, 19, 10, 16, 23)
        expect = {'date': np.datetime64('2018-06-19 10:16:23'),
                  'time': np.timedelta64(63696622583, 's')}

        h = GT3Header(utim='sec')
        h.set_time_date(date="%04d%02d%02d %02d%02d%02d" % t)
        self.assertEqual(h.date, expect['date'])
        self.assertEqual(h.time, expect['time'])

    def test_set_time_date_10(self):
        """ test set_time_date. """
        t = (2018, 6, 19, 10, 16, 23)
        expect = {'date': np.datetime64('2018-06-19'),
                  'time': np.timedelta64(737229, 'D')}

        h = GT3Header(utim='days')
        h.set_time_date(date="%04d-%02d-%02d %02d:%02d:%02d" % t)
        self.assertEqual(h.date, expect['date'])
        self.assertEqual(h.time, expect['time'])

    def test_set_time_date_11(self):
        """ test set_time_date. """
        t = (2018, 6, 19, 10, 16, 23)
        expect = {'date': np.datetime64('2018-06-19 10:16:23'),
                  'time': np.timedelta64(63696622583, 's')}

        h = GT3Header(utim='sec')
        h.set_time_date(date="%04d-%02d-%02d %02d:%02d:%02d" % t)
        self.assertEqual(h.date, expect['date'])
        self.assertEqual(h.time, expect['time'])

    def test_set_time_date_20(self):
        """ test set_time_date. """
        t = 63696622583  # sec
        expect = {'date': np.datetime64('2018-06-19 10:16:23'),
                  'time': np.timedelta64(63696622583, 's')}

        h = GT3Header(utim='sec')
        h.set_time_date(time=t)
        self.assertEqual(h.date, expect['date'])
        self.assertEqual(h.time, expect['time'])


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

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

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

    def dump(self):
        raise NotImplementedError

    def scan(self):
        """
        Scan whole file and create data table.

        Data table is a pandas.DataFrame instance.
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
            tbl.append([self.current_header.item,
                        int(str(self.current_header.time).split()[0]),
                        self.current_header.tdur,
                        self.current_header.utim,
                        self.current_header.dfmt,
                        self.current_header.date,
                        self.current_header.aitm1,
                        self.current_header.aitm2,
                        self.current_header.aitm3])
        self.rewind()
        self.table = pd.DataFrame(tbl)
        self.table.columns = ['item',
                              'time',
                              'tdur',
                              'utim',
                              'dfmt',
                              'date',
                              'aitm1',
                              'aitm2',
                              'aitm3']
        # self.num_of_times = self.table.pivot_table(
        #     index='time', aggfunc=[len]).shape[0]
        # self.num_of_items = self.table.pivot_table(
        #     index='item', aggfunc=[len]).shape[0]
        self.num_of_times = self.table['time'].nunique()
        self.num_of_items = self.table['item'].nunique()

        if (self.opt_verbose):
            liner = "="*5 + " %s: Scan result: " % self.name
            liner += "="*(80-len(liner))
            print(liner)
            print("* num_of_data :", self.num_of_data)
            print("* num_of_times:", self.num_of_times)
            print("* num_of_items:", self.num_of_items)
            print("="*len(liner))

        return None

    def show_table(self, file=None):
        """
        Show data table, created by scan().
        """

        if (self.table is None):
            self.scan()
        liner = "="*5 + " Data table: "
        liner += "="*(80-len(liner))
        print(liner, file=file)
        print(self.table.to_string(), file=file)
        print("="*len(liner), file=file)

        return None

    def read_one_header(self):
        """
        Read one header and it as a `current_header`.
        """

        dt = np.dtype(
            [("h", ">i4"), ("b", "a16", 64), ("t", ">i4")])

        chunk = np.fromfile(self.f, dtype=dt, count=1)
        if (len(chunk)):
            if (chunk["h"] != chunk["t"]):
                raise IOError
            self.current_header.set_from_hdarray(
                chunk["b"][0], self.name)
            self.current_header.number += 1
            self.is_eof = False
            self.is_after_header = True
        else:
            self.current_header = None
            self.is_eof = True

        return None

    def read_one_data(self):
        if (self.current_header.dfmt[:3] == 'UR8'):
            dt = np.dtype([
                ("h", ">i4"),
                ("b", ">f8", self.current_header.size),
                ("t", ">i4")])
            chunk = np.fromfile(self.f, dtype=dt, count=1)
            if (chunk["h"] != chunk["t"]):
                raise IOError
            self.current_data = np.array(
                chunk["b"][0]).reshape(self.current_header.shape)
        elif (self.current_header.dfmt[:3] == 'UR4'):
            dt = np.dtype([
                ("h", ">i4"),
                ("b", ">f4", self.current_header.size),
                ("t", ">i4")])
            chunk = np.fromfile(self.f, dtype=dt, count=1)
            if (chunk["h"] != chunk["t"]):
                raise IOError
            self.current_data = np.array(
                chunk["b"][0]).reshape(self.current_header.shape)
        elif (self.current_header.dfmt[:3] == 'URC'):
            raise NotImplementedError
        elif (self.current_header.dfmt[:3] == 'URY'):
            packed_bit_width = int(self.current_header.dfmt[3:])
            ijnum = self.current_header.isize * self.current_header.jsize
            knum = self.current_header.ksize
            imiss = (1 << packed_bit_width) - 1

            # coeffs[*,0] is the offset values,
            # coeffs[*,1] is the scale values.
            dt = np.dtype([
                ("h", ">i4"),
                ("b", ">f8", knum*2),
                ("t", ">i4")])
            chunk = np.fromfile(self.f, dtype=dt, count=1)
            if (chunk["h"] != chunk["t"]):
                raise IOError
            coeffs = chunk["b"][0].reshape(knum, 2)

            ijnum_packed = BitPacker.calc_packed_length(
                ijnum, packed_bit_width)
            dt = np.dtype([
                ("h", ">i4"),
                ("b", ">i4", ijnum_packed*knum),
                ("t", ">i4")])
            chunk = np.fromfile(self.f, dtype=dt, count=1)
            if (chunk["h"] != chunk["t"]):
                raise IOError
            packed = chunk["b"][0].reshape(knum, ijnum_packed)
            self.current_data = np.ndarray(shape=(knum, ijnum), dtype="f8")
            for k in range(knum):
                unpacked = BitPacker.unpack(
                    packed[k, :], packed_bit_width, ijnum)
                for i in range(ijnum):
                    if (unpacked[i] == imiss):
                        # self.current_data[k, i] = self.current_header.miss
                        self.current_data[k, i] = np.nan
                    else:
                        self.current_data[k, i] = (
                            coeffs[k, 0] + unpacked[i] * coeffs[k, 1])
            self.current_data = self.current_data.reshape(
                self.current_header.shape)
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
            ijnum_packed = BitPacker.calc_packed_length(
                ijnum, packed_bit_width)
            size = knum*2*8+8 + ijnum_packed*knum*8+8
        else:
            raise NotImplementedError

        self.f.seek(size, 1)
        self.is_after_header = False
        return None

    def select_data_range(self, xidx=(), yidx=(), zidx=()):
        """
        Select data index range from current_data and return it.

        Index range is specified by a single integer to slice at it,
        or two element tuple or list to specify range.
        """
        # print('dbg:', xidx, yidx, zidx)
        d = self.current_data
        if (len(xidx) == 0):
            xidx = [0, d.shape[2]]
        if (len(yidx) == 0):
            yidx = [0, d.shape[1]]
        if (len(zidx) == 0):
            zidx = [0, d.shape[0]]

        d = d[zidx[0]:zidx[1], yidx[0]:yidx[1], xidx[0]:xidx[1]]
        return xidx, yidx, zidx, d

    def dump_current_header(self):
        self.current_header.dump()
        return None

    def dump_current_data(self, file=None,
                          xidx=(), yidx=(), zidx=(),
                          indexed=False, **kwargs):
        np.set_printoptions(threshold=np.inf, linewidth=100, suppress=True)

        xidx, yidx, zidx, d = self.select_data_range(xidx, yidx, zidx)

        liner = '====== %s: data #%d ' \
                % (self.name, self.current_header.number)
        liner += "="*(80-len(liner))
        if (self.opt_debug):
            print("dbg:current_data:", file=file)
            print("  flags:", file=file)
            print(d.flags, file=file)
            print("  dtype:", d.dtype, file=file)
            print("  size,itemsize:",
                  d.size, d.itemsize, file=file)
            print("  xrange:", xidx, file=file)
            print("  yrange:", yidx, file=file)
            print("  zrange:", zidx, file=file)
            print("  ndim, shape, strides:",
                  d.ndim, d.shape,
                  d.strides, file=file)
        print(liner)
        if (len(kwargs) > 0):
            np.set_printoptions(**kwargs)

        if (indexed):
            print("#%8s %8s %8s %20s" % ("xindex", "yindex", "zindex", "data"),
                  file=file)
            for k in range(zidx[1]-zidx[0]):
                for j in range(yidx[1]-yidx[0]):
                    for i in range(xidx[1]-xidx[0]):
                        print(" %8d %8d %8d %20f"
                              % (i+xidx[0], j+yidx[0], k+zidx[0], d[k, j, i]),
                              file=file)
        else:
            print(d, file=file)
        print('='*len(liner), file=file)

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

    def write_one_header(self):
        """
        Write `self.current_header` as one header.
        """

        dt = np.dtype([
            ("h", ">i4"),
            ("b", "a16", 64),
            ("t", ">i4")])
        chunk = np.empty((1,), dtype=dt)
        chunk["h"] = 64*16
        chunk["t"] = chunk["h"]
        chunk["b"] = self.current_header.pack()
        chunk.tofile(self.f)
        self.if_after_header = True

    def set_current_data(self, d):
        if (d.shape != self.current_header.shape):
            print("Error: shape mismatch!")
            print("in header:", self.current_header.shape)
            print("in data  :", d.shape)
            sys.exit(1)
        self.current_data = d
        pass

    def write_one_data(self):
        if (self.current_header.dfmt[:3] == 'UR4'):
            dt = np.dtype([
                ("h", ">i4"),
                ("b", ">f4", self.current_header.size),
                ("t", ">i4")])
            bytes = self.current_header.size*4
            pass
        elif (self.current_header.dfmt[:3] == 'UR8'):
            bytes = self.current_header.size*8
            raise NotImplementedError
        elif (self.current_header.dfmt[:3] == 'URC'):
            raise NotImplementedError
        elif (self.current_header.dfmt[:3] == 'URY'):
            raise NotImplementedError
        else:
            raise NotImplementedError('Unknown dfmt: %s'
                                      % self.current_header.dfmt)
        chunk = np.empty((1,), dtype=dt)
        chunk["h"] = bytes
        chunk["t"] = chunk["h"]
        chunk["b"] = self.current_data.flatten()
        chunk.tofile(self.f)
        self.if_after_header = False

    def extract_t_axis(self):
        """
        Extract axis for time series.

        self must
        - have data tabel created by scan(),
        - have num_of_times > 0 and num_of_items == 1, ie, one
          variable in the file,

        If num_of_items > 1, assumes all times have same and time
        value(s).

        Returns dict{"values","unit'}, and values is an ndarray of
        `time`.
        """
        if (self.table is None):
            self.scan()

        if (self.num_of_times <= 0):
            print('Warn: num_of_times is not positive: %d' % self.num_of_times)
            result = None
        elif (self.num_of_items > 1):
            result =  {"values" : self.table["time"].unique(),
                       "unit" : self.table["utim"].values[0]}
        else:
            result =  {"values" : self.table["time"].values,
                       "unit" : self.table["utim"].values[0]}

        return result


###############################################################################
# Tests for GT3File
###############################################################################
class TestGT3File(unittest.TestCase):
    def setUp(self):
        """ write one header and data """
        date = '20180616 150000'
        utim = u'hours'
        deltaT = 3

        with GT3File("test00", 'wb') as f1:
            f1.current_header = GT3Header(
                dset='test', item='hoge',
                title='testdata for TestGT3File', unit='-',
                date=date, utim=utim, tdur=deltaT,
                aitm1='GLON64', astr1=1, aend1=64,
                aitm2='GGLA32', astr2=1, aend2=32,
                aitm3='SFC1', astr3=1, aend3=1)
            f1.set_current_data(np.zeros(shape=(1, 32, 64), dtype='f4'))
            f1.write_one_header()
            f1.write_one_data()

            for n in range(11):
                f1.current_header.set_time_date(time=f1.current_header.time+deltaT)
                f1.set_current_data(np.zeros(shape=(1, 32, 64), dtype='f4'))
                f1.write_one_header()
                f1.write_one_data()

        with GT3File("test01", 'wb') as f2:
            f2.current_header = GT3Header(
                dset='test',
                title='testdata for TestGT3File', unit='-',
                date=date, utim=utim, tdur=deltaT,
                aitm1='GLON64', astr1=1, aend1=64,
                aitm2='GGLA32', astr2=1, aend2=32,
                aitm3='SFC1', astr3=1, aend3=1)
            f2.set_current_data(np.zeros(shape=(1, 32, 64), dtype='f4'))
            for n in range(8):
                f2.current_header.item = 'hoge%02d' % n
                f2.write_one_header()
                f2.write_one_data()


    def test_write_00(self):
        pass

    def test_read_00(self):
        """ Test for opening not exist file. """
        with self.assertRaises(OSError):
            filename = 'prcpr'  # not exist
            mode = 'rb'
            with GT3File(filename, mode) as f:
                pass

    def test_read_01(self):
        """ Test for invalid mode."""
        with self.assertRaises(InvalidArgumentError):
            filename = 'prcpx'
            mode = 'w'  # error, must be 'wb'
            with GT3File(filename, mode) as f:
                pass

    def test_read_02(self):
        """ Test for writing and reading """
        expect = (
            "===== Data table: ==============================================================\n"
            "    item      time  tdur   utim dfmt                date   aitm1   aitm2 aitm3\n"
            "0   hoge  17693439     3  hours  UR4 2018-06-16 15:00:00  GLON64  GGLA32  SFC1\n"
            "1   hoge  17693442     3  hours  UR4 2018-06-16 18:00:00  GLON64  GGLA32  SFC1\n"
            "2   hoge  17693445     3  hours  UR4 2018-06-16 21:00:00  GLON64  GGLA32  SFC1\n"
            "3   hoge  17693448     3  hours  UR4 2018-06-17 00:00:00  GLON64  GGLA32  SFC1\n"
            "4   hoge  17693451     3  hours  UR4 2018-06-17 03:00:00  GLON64  GGLA32  SFC1\n"
            "5   hoge  17693454     3  hours  UR4 2018-06-17 06:00:00  GLON64  GGLA32  SFC1\n"
            "6   hoge  17693457     3  hours  UR4 2018-06-17 09:00:00  GLON64  GGLA32  SFC1\n"
            "7   hoge  17693460     3  hours  UR4 2018-06-17 12:00:00  GLON64  GGLA32  SFC1\n"
            "8   hoge  17693463     3  hours  UR4 2018-06-17 15:00:00  GLON64  GGLA32  SFC1\n"
            "9   hoge  17693466     3  hours  UR4 2018-06-17 18:00:00  GLON64  GGLA32  SFC1\n"
            "10  hoge  17693469     3  hours  UR4 2018-06-17 21:00:00  GLON64  GGLA32  SFC1\n"
            "11  hoge  17693472     3  hours  UR4 2018-06-18 00:00:00  GLON64  GGLA32  SFC1\n"
            "================================================================================\n")

        with GT3File('test00') as f:
            f.scan()
            with tempfile.TemporaryFile('w+') as ff:
                f.show_table(file=ff)
                ff.seek(0)
                result = ff.read()

        self.assertMultiLineEqual(result, expect)

    def test_read_03(self):
        """ Test for read_current_header """
        with GT3File('test00') as f:
            f.scan()
            f.read_one_header()
            f.read_one_data()

        self.assertTrue(f.current_data.dtype == np.dtype('>f4'))
        self.assertTupleEqual(f.current_data.shape, (1, 32, 64))
        self.assertAlmostEqual(f.current_data.mean(), 0.)

    def test_extract_t_axis_00(self):
        """ Test for extract_t_axis() with multi-items file. """
        ts = np.array([17693439])
        expect = {'values': ts, 'unit': 'hours'}

        with GT3File('test01') as f:
            result = f.extract_t_axis()

        self.assertEqual(tuple(result['values']), tuple(expect['values']))
        self.assertEqual(result['unit'], expect['unit'])

    def test_extract_t_axis_01(self):
        """ Test for extract_t_axis(). """
        ts = np.array(
            [17693439, 17693442, 17693445, 17693448,
             17693451, 17693454, 17693457, 17693460,
             17693463, 17693466, 17693469, 17693472])
        expect = {'values': ts,'unit': 'hours'}

        with GT3File('test00') as f:
            result = f.extract_t_axis()

        self.assertEqual(tuple(result['values']), tuple(expect['values']))
        self.assertEqual(result['unit'], expect['unit'])

###############################################################################
# Axis for GTOOL3
###############################################################################
class GT3Axis():
    """
    gtool3 axis.
    """

    default_search_paths = [u".", u"$GT3AXISDIR", u"$GTOOLDIR/gt3"]

    def __init__(self, name, search_paths=None):
        self.name = name
        if (search_paths is None):
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
        Find gtool3 axis file.

        With given axis name `name` from path listed as
        `self.search_paths`.

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
            print('Axis "%s" Not found in path(s): %s'
                  % (self.name, ":".join(self.search_paths)))
            self.file = None

        pass

    def dump(self, file=None):
        liner = '='*6 + ' Axis: %s ' % self.name
        liner += '='*(80-len(liner))
        print(liner, file=file)
        print("path:", self.file, file=file)
        print("title:", self.title, file=file)
        print("size:", self.size, file=file)
        print("data:", file=file)
        print(self.data, file=file)
        print('='*len(liner), file=file)
        pass

###############################################################################
# Here we go.
###############################################################################


if __name__ == '__main__':

    unittest.main()
