#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import unittest

from collections import deque
from io import StringIO
from contextlib import redirect_stdout

from pygt3file import GT3Header


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
        header = GT3Header(fname='test')
        with StringIO() as o:
            with redirect_stdout(o):
                header.dump()
            result = o.getvalue()
        #  Do not remove trailing whitespace from this!!
        expect = ("""\
====== test: header #-1 ========================================================
dset : 
item : []: 
date : None(None) with 0[sec]
axis : [0:0]
cycl : False
dfmt : UR4
miss : -999.0
size : 1
edit : []
ettl : []
memo : []
cdate: %s by pygt3 library
mdate: %s by pygt3 library
isize,jsize,ksize: 1, 1, 1
================================================================================
""" % (header.cdate, header.mdate))

        self.assertMultiLineEqual(result, expect)

    def test_set_from_hdarray_01(self):
        """ set_from_hdarray() and dump() """
        header = GT3Header(fname='test')
        header.set_from_hdarray(self.orig_hdarray, fname='test')
        with StringIO() as o:
            with redirect_stdout(o):
                header.dump()
            result = o.getvalue()
        expect = ("""\
====== test: header #-1 ========================================================
dset : test
item : hoge[-]: testdata for TestGT3Header.pack(
date : 2018-06-16(737226) with 0[DAYS]
axis : GLON64[1:64] x GGLA32[1:32] x SFC1[1:1]
cycl : False
dfmt : UR4
miss : -999.0
size : 2048
edit : ['edit0', 'edit1', 'edit2', 'edit3', 'edit4', 'edit5', 'edit6', 'edit7']
ettl : ['etitle0', 'etitle1', 'etitle2', 'etitle3', 'etitle4', 'etitle5', 'etitle6', 'etitle7']
memo : ['memo0', 'memo1', 'memo2', 'memo3', 'memo4', 'memo5', 'memo6', 'memo7', 'memo8', 'memo9']
cdate: %s by pygt3 library
mdate: %s by pygt3 library
isize,jsize,ksize: 64, 32, 1
================================================================================
""" % (header.cdate, header.mdate))
        self.assertMultiLineEqual(result, expect)

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
        expect = np.array(
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
            self.assertEqual(expect[n], hdarray[n],
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


###############################################################################
# Here we go.
###############################################################################
if __name__ == '__main__':
    unittest.main()
