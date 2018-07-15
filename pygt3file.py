#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini library to read/write GTOOL3 format file.
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import math
import os
import sys
from collections import deque
from contextlib import redirect_stdout
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
    """Class for Bit Packing used in gt3.

    This class contains several methods used to pack/unpack data for
    gt3 datafile.

    Packing method used in gt3 as dfmt 'URYxx' is to convert original
    data to 'xx'-bit length integer data with scale/offset value, then
    store them as a `base_bit_width` width integer data array.

    For example, dfmt 'URY20' with `base_bit_width=32` means that each
    element of original data array are converted 20-bit integer, and
    stored as an 32-bit integer array. In this case, if original data
    array has 8 elements, resulting converted data is 160 bits in
    total, so at least 5 elements of 32-bit integer array is necessary
    to store it.

    Note
    ----
    Offset/scale coefficients are calculated by each k-index, so they
    are stored as a integer array with shape of (2,KDIM), and written
    prior to converted data in one entry for gt3 data.

    """

    base_bit_width = 32
    """
    Base bit width of packed array, ie., packed data are stored as an
    array of this bit length integer data.
    """

    def calc_packed_length(olen, nbit):
        """Calculate necessary length of packed array, for given original
        array length and given converted bit width for each data.

        Parameters
        ----------
        olen : int
          Length of original data
        nbit : int
          Bit width for packed data

        Returns
        -------
        int
          Necessary length of conveted(packed) array.

        Warnings
        --------
        Don't know why but first argument of this method is diminished
        in the document created by sphinx(-_-"
        """

        if (nbit > BitPacker.base_bit_width):
            raise InvalidArgumentError(nbit)

        return math.ceil(nbit*olen/BitPacker.base_bit_width)

    def pack():
        """Pack data.

        TODO
        ----
          Not implemented.

        """
        pass

    def unpack(packed, pack_bit_width, len_unpacked):
        """Unpack data.

        `pack_bit_width` is known from dfmt of gt3 header attribute,
        and `len_unpacked` is calculated by `calc_packed_length()` of
        this class.

        Parameters
        ----------
        packed : numpy.ndarray
          Packed data to be converted back.
        pack_bit_width : int
          Packing bit length
        len_unpacked : int
          Length of unpacked data array.

        Warnings
        --------
        Don't know why but first argument of this method is diminished
        in the document created by sphinx(-_-"

        """

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
class GT3Header:
    """GTOOL3 format header.

    GTOOL3 format header is consist of
    `character(len=16,dimension=64)` array in Fortran, which I call
    `hdarray` and implemented as `numpy.ndarray` with `dtype='a16'*16`
    here.

    This class is an abstraction of these information. All of elements
    of `hdarray` are assigned as attributes of this class.
    Some of them are casted to integer and/or float type.

    Three attributes, `ettl`, `edit` and `memo`, are considered as a
    `queue` in original GTOOL3 format specification, so these are
    implemented as `deque` class (with limited `maxlen`) of
    `collections` package in this class.

    Parameters
    ----------
        dset : str
        item : str
        title : str
        unit : str
        date : str or datetime.datetime
        utim : str
        time : int
        tdur : int
        aitm1 : str
        astr1 : int
        aend1 : int
        aitm2 : str
        astr2 : int
        aend2 : int
        aitm3 : str
        astr3 : int
        aend3 : int
        dfmt : str
        miss : float
        dmin : float
        dmax : float
        divs : float
        divl : float
        edit : list or tuple of str
        fnum : int
        dnum : int
        ettl : list or tuple of str
        memo : list or tuple of str
        fname : str

    See Also
    --------
        add_attribs : add list type attributes, `edit`, `ettl`, `memo`.
        set_time_date : Set and keep `date`,`time` and `utim` attributes consistently.
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
        self._set_hidden_attribs()

        pass

    def set_time_date(self, date=None, time=None, utim=None):
        """
        Set and keep `date`, `time` and `utim` attributes consistently.

        If `date` is given, convert it to `time` (and vice versa),
        then set to self.  If both are given, `time` is discarded even
        if not `None`.

        According to the GTOOL3 specification, `time` field is counted
        from '0000/01/01 00:00:00', so we use np.datetime64 here.

        Parameters
        ----------
        date : str or `numpy.datetime64` instance
          `Date` attribute for gt3, must be;

          - 'compact' : 'YYYYMMDD HHMMSS' that is used in GTOOL3 header array,
          - 'standard': 'YYYY-MM-DD HH-MM-SS' or acceptable by `numpy.datetime64(),`
          - `numpy.datetime64` instance.

        time : int
          `time` attribute for gt3, must be acceptable by `numpy.timedelta64()`

        utim : str
          `utim` attribute for gt3, unit of `time`.
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
        Set attributes from 'hdarray' at once.

        Parameters
        ----------
        hdarray : numpy.ndarray(`dtype='a16'*64`)
          gt3 header array

        Todo
        ----
        Remove an argument `fname`.
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

        self._set_hidden_attribs()

        if (fname is not None):
            self.fname = fname
        pass

    def _set_hidden_attribs(self):
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

        Parameters
        ----------
        ettl : str or list or tuple
        edit : str or list or tuple
        memo : str or list or tuple
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

    def dump(self):
        """
        Output summarize of this instance.
        """
        if (self is not None):
            liner = '====== %s: header #%d ' % (self.fname, self.number)
            liner += "="*(80-len(liner))
            print(liner)
            print("dset : %s" % str(self.dset))
            print("item : %s[%s]: %s"
                  % (self.item, self.unit, self.titl))
            print("date : %s(%s) with %d[%s]" %
                  (self.date, str(self.time).split()[0],
                   self.tdur, self.utim))
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
            print("edit :", list(self.edit))
            print("ettl :", list(self.ettl))
            print("memo :", list(self.memo))
            print("cdate: %s by %s" % (self.cdate, self.csign))
            print("mdate: %s by %s" % (self.mdate, self.msign))
            print("isize,jsize,ksize: %d, %d, %d"
                  % (self.isize, self.jsize, self.ksize))
            print('=' * len(liner))
        pass

    def pack(self):
        """
        Pack attribs of this instance to the array suitable for gtool3
        header and return it.

        Return
        ------
        numpy.ndarray(dtype='a16'*64):
           hdarray packs current attribues of this instance.


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


class GT3Data(np.ndarray):
    """Class for data part of gt3 file format.

    Implemented as a subclass of numpy.ndarray, adding some attributes
    and methods.

    Attributes
    ----------
    name : str
      name of this data
    number : int
      number in the gt3 file of this data
    """

    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, name=None, number=None):
        # Borrowed from
        # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.  It
        # also triggers a call to GT3Data.__array_finalize__
        obj = super(GT3Data, subtype).__new__(subtype, shape, dtype,
                                              buffer, offset, strides,
                                              order)
        # set the new 'name' attribute to the value passed
        obj.name = name
        obj.number = number
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(GT3Data, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. GT3Data():
        #    obj is None
        #    (we're in the middle of the GT3Data.__new__
        #    constructor, and self.info will be set when we return to
        #    GT3Data.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(GT3Data):
        #    obj is arr
        #    (type(obj) can be GT3Data)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is GT3Data
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # GT3Data.__new__ constructor, but also with
        # arr.view(GT3Data).
        self.name = getattr(obj, 'name', None)
        self.number = getattr(obj, 'number', None)
        # We do not need to return anything

    def select_data_range(self, xidx=(), yidx=(), zidx=()):
        """
        Select data index range and return it.

        Index range is specified by a single integer to slice at it,
        or two element tuple or list to specify range.


        Parameters
        ---------
        xidx : int or tuple
        yidx : int or tuple
        zidx : int or tuple

        """
        # print('dbg:', xidx, yidx, zidx)
        if (len(xidx) == 0):
            xidx = [0, self.shape[2]]
        if (len(yidx) == 0):
            yidx = [0, self.shape[1]]
        if (len(zidx) == 0):
            zidx = [0, self.shape[0]]

        d = self[zidx[0]:zidx[1], yidx[0]:yidx[1], xidx[0]:xidx[1]]
        return xidx, yidx, zidx, d

    def dump(self,
             xidx=(), yidx=(), zidx=(),
             indexed=False, opt_debug=False, **kwargs):
        """
        Output values.

        If `xidx`, `yidx`, `zidx` are given, slice or subpart of the
        array is output.

        If `indexed` is  given, each value is output with i,j,k-indices.

        Parameters
        ----------
        xidx : int or tuple
        yidx : int or tuple
        zidx : int or tuple
        indexed : bool
        opt_debug : bool

        Todo
        ----
        Use `kwargs` to control print format

        See Also
        --------
        select_data_range : Select data index range and return it.

        """
        np.set_printoptions(threshold=np.inf, linewidth=100, suppress=True)

        xidx, yidx, zidx, d = self.select_data_range(xidx, yidx, zidx)
        liner = '======'
        if (self.name):
            liner += '%s:' % self.name
        if (self.number):
            liner += '#%d' % self.number
        liner += "="*(80-len(liner))
        if (opt_debug):
            print("dbg:current_data:")
            print("  flags:")
            print(d.flags)
            print("  dtype:", d.dtype)
            print("  size,itemsize:",
                  d.size, d.itemsize)
            print("  xrange:", xidx)
            print("  yrange:", yidx)
            print("  zrange:", zidx)
            print("  ndim, shape, strides:",
                  d.ndim, d.shape,
                  d.strides)
        print(liner)
        if (len(kwargs) > 0):
            np.set_printoptions(**kwargs)

        if (indexed):
            print("#%8s %8s %8s %20s" % ("xindex", "yindex", "zindex", "data"))
            for k in range(zidx[1]-zidx[0]):
                for j in range(yidx[1]-yidx[0]):
                    for i in range(xidx[1]-xidx[0]):
                        print(" %8d %8d %8d %20f"
                              % (i+xidx[0], j+yidx[0], k+zidx[0], d[k, j, i]))
        else:
            print(d)
        print('='*len(liner))

        return None


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
        self.reset_attribs()

        # self.packer = BitPacker()
        return None

    def reset_attribs(self):
        self.is_after_header = False
        self.is_eof = False
        self.num_of_data = -1
        self.num_of_times = -1
        self.num_of_items = -1
        self.table = None
        self.opt_debug = False
        self.opt_verbose = False

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # print('GT3File.__exit__() called')
        self.close()

    def open(self, name, mode='rb'):
        """ re-open other file within the same instance. """
        self.f.close()
        self.f = open(name, mode)
        self.name = name
        self.mode = mode
        return None

    def close(self):
        self.f.close()
        self.reset_attribs()
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
        Scan whole file and create data table.

        Data table is a pandas.DataFrame instance.
        Note that file position is on the top after this method.
        """
        tbl = []
        self.num_of_data = 0
        self.rewind()
        for h, d in self.read(skip_data=True):
            # self.num_of_data += 1
            tbl.append([h.item,
                        int(str(h.time).split()[0]),
                        h.tdur,
                        h.utim,
                        h.dfmt,
                        h.date,
                        h.aitm1,
                        h.aitm2,
                        h.aitm3])
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
        self.num_of_data = len(self.table.index)
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

    def show_table(self):
        """
        Show data table, created by scan().
        """

        if (self.table is None):
            print("Error: call scan() before show_table()")
            sys.exit(1)
        liner = "="*5 + " Data table: "
        liner += "="*(80-len(liner))
        print(liner)
        print(self.table.to_string())
        print("="*len(liner))

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
                chunk["b"][0]).reshape(self.current_header.shape).view(GT3Data)
        elif (self.current_header.dfmt[:3] == 'UR4'):
            dt = np.dtype([
                ("h", ">i4"),
                ("b", ">f4", self.current_header.size),
                ("t", ">i4")])
            chunk = np.fromfile(self.f, dtype=dt, count=1)
            if (chunk["h"] != chunk["t"]):
                raise IOError
            self.current_data = np.array(
                chunk["b"][0]).reshape(self.current_header.shape).view(GT3Data)
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
                self.current_header.shape).view(GT3Data)
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

    def dump_current_header(self):
        self.current_header.dump()
        return None

    def dump_current_data(self, **kwargs):
        self.current_data.dump(**kwargs)
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
                d = self.current_data.view(GT3Data)
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
        self.current_data = d.view(GT3Data)
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
            result = {"values": self.table["time"].unique(),
                      "unit": self.table["utim"].values[0]}
        else:
            result = {"values": self.table["time"].values,
                      "unit": self.table["utim"].values[0]}

        return result

    def read(self,skip_data=False):
        """
        Iterator Generator
        """
        while True:
            self.read_one_header()
            if (self.is_eof):
                return
            if (skip_data):
                self.skip_one_data()
                self.current_data = None
            else:
                self.read_one_data()
            yield self.current_header, self.current_data


###############################################################################
# Axis for GTOOL3
###############################################################################
class GT3Axis():
    """
    Special purpose class for gt3 axis file.

    This class is to get and gt3 axis data.
    Find axis file with given axis  `name` from `search_path`.

    Parameters
    ----------
    name : str
      Axis name, such as `GLON64` etc.
    search_path : list of str
      Seach path for axis file. It can contain environment variables.

    Attributes
    ----------
    name : str
      Axis name, such as `GLON64` etc.
    search_path : list of str
      Seach path for axis file. It can contain environment variables.
      Current default path is specified by `default_search_paths`.
    file : str
      Filename of axis file.
    title : str
      title of axis, such as `longitude` etc.
    unit : str
      unit of axis data
    size : int
      length of axis
    header : GT3Header.
      Gt3 header part of axis file.
    data : GT3Data.
      Gt3 data part of axis file.
    """

    default_search_paths = [u".", u"$GT3AXISDIR", u"$GTOOLDIR/gt3"]
    """
    Default list of `search_path`
    """
    
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
            self = None
            return
        if (f is None):
            self = None
        f.scan()

        self.name = name   # "GLONxx" etc.
        self.header, self.data = f.read_nth_data(0)
        self.title = f.current_header.titl  # "longitude" etc.
        self.unit = f.current_header.unit
        self.data = f.current_data.flatten()
        if (f.current_header.cyclic):
            self.data = self.data[:-1]
        f.close()
        self.size = len(self.data)
        pass

    def find_axfile(self):
        """
        Find gtool3 axis file with given axis `name` from path listed
        as `self.search_paths`.

        Return
        ------
          str or None : path of the found axis file or `None` unless found.
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
        """
        Output summarize of this instance.
        """
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
