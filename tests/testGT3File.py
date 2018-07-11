#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
import unittest
from io import StringIO
from contextlib import redirect_stdout

from pygt3file import GT3File, GT3Header, InvalidArgumentError


###############################################################################
# Tests for GT3File
###############################################################################
class TestGT3File(unittest.TestCase):
    def setUp(self):
        """ write one header and data """
        self.maxDiff = None
        date = '20180616 150000'
        utim = u'hours'
        deltaT = 3
        dtype='f4'
        aitm1, astr1, aend1 = ('GLON64', 1, 64)
        aitm2, astr2, aend2 = ('GGLA32', 1, 32)
        aitm3, astr3, aend3 = ('SFC1', 1, 1)
        x = np.sin(np.linspace(-np.pi*2., np.pi*2., aend1), dtype=dtype)
        y = np.cos(np.linspace(-np.pi, np.pi, aend2), dtype=dtype)
        z = np.exp(np.linspace(np.pi, 0., aend3), dtype=dtype)
        d = np.outer(z,np.outer(y,x)).reshape(aend3, aend2, aend1)

        # single variable multi time file
        with GT3File("test00", 'wb') as f1:
            f1.current_header = GT3Header(
                dset='test', item='hoge',
                title='testdata for TestGT3File', unit='-',
                date=date, utim=utim, tdur=deltaT,
                aitm1=aitm1, astr1=astr1, aend1=aend1,
                aitm2=aitm2, astr2=astr1, aend2=aend2,
                aitm3=aitm3, astr3=astr3, aend3=aend3)
            f1.set_current_data(d)
            f1.write_one_header()
            f1.write_one_data()

            for n in range(11):
                f1.current_header.set_time_date(
                    time=f1.current_header.time+deltaT)
                f1.set_current_data(d)
                f1.write_one_header()
                f1.write_one_data()

        # multi variable single time file
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

    def test_open_00(self):
        """ Test for opening not exist file. """
        with self.assertRaises(OSError):
            filename = 'prcpr'  # not exist
            mode = 'rb'
            with GT3File(filename, mode) as f:
                pass

    def test_open_01(self):
        """ Test for invalid mode."""
        with self.assertRaises(InvalidArgumentError):
            filename = 'prcpx'
            mode = 'w'  # error, must be 'wb'
            with GT3File(filename, mode) as f:
                pass

    def test_scan_00(self):
        """Test scan() with single variable file."""
        expect1 = (12, 12, 1)
        expect2 = pd.Series([
            17693439, 17693442, 17693445, 17693448,
            17693451, 17693454, 17693457, 17693460,
            17693463, 17693466, 17693469, 17693472])

        with GT3File('test00') as f:
            f.scan()
            result1 = (f.num_of_data, f.num_of_times, f.num_of_items)
            result2 = f.table.iloc[:, 1]  # date

        self.assertEqual(result1, expect1)
        self.assertEqual((result2-expect2).mean(), 0.0)

    def test_scan_01(self):
        """Test scan() with multi variable file."""
        expect1 = (8, 1, 8)
        expect2 = ["hoge00", "hoge01", "hoge02", "hoge03",
                   "hoge04", "hoge05", "hoge06", "hoge07"]

        with GT3File('test01') as f:
            f.scan()
            result1 = (f.num_of_data, f.num_of_times, f.num_of_items)
            result2 = f.table.iloc[:, 0].values.tolist()

        self.assertEqual(result1, expect1)
        self.assertEqual(result2, expect2)

    def test_show_table_00(self):
        """ Test show_table() to a stdout. """
        expect = ("""\
===== Data table: ==============================================================
    item      time  tdur   utim dfmt                date   aitm1   aitm2 aitm3
0   hoge  17693439     3  hours  UR4 2018-06-16 15:00:00  GLON64  GGLA32  SFC1
1   hoge  17693442     3  hours  UR4 2018-06-16 18:00:00  GLON64  GGLA32  SFC1
2   hoge  17693445     3  hours  UR4 2018-06-16 21:00:00  GLON64  GGLA32  SFC1
3   hoge  17693448     3  hours  UR4 2018-06-17 00:00:00  GLON64  GGLA32  SFC1
4   hoge  17693451     3  hours  UR4 2018-06-17 03:00:00  GLON64  GGLA32  SFC1
5   hoge  17693454     3  hours  UR4 2018-06-17 06:00:00  GLON64  GGLA32  SFC1
6   hoge  17693457     3  hours  UR4 2018-06-17 09:00:00  GLON64  GGLA32  SFC1
7   hoge  17693460     3  hours  UR4 2018-06-17 12:00:00  GLON64  GGLA32  SFC1
8   hoge  17693463     3  hours  UR4 2018-06-17 15:00:00  GLON64  GGLA32  SFC1
9   hoge  17693466     3  hours  UR4 2018-06-17 18:00:00  GLON64  GGLA32  SFC1
10  hoge  17693469     3  hours  UR4 2018-06-17 21:00:00  GLON64  GGLA32  SFC1
11  hoge  17693472     3  hours  UR4 2018-06-18 00:00:00  GLON64  GGLA32  SFC1
================================================================================
""")
        with GT3File('test00') as f:
            f.scan()
            with StringIO() as o:
                with redirect_stdout(o):
                    f.show_table()
                    result = o.getvalue()
        self.assertMultiLineEqual(result, expect)

    def test_read_one_header_00(self):
        """Test for read_one_header() and skip_one_data()."""
        expect = u"""\
2018-06-16T15 17693439 hours
2018-06-16T18 17693442 hours
"""
        with GT3File('test00') as f:
            f.scan()
            with StringIO() as o:
                f.read_one_header()
                f.skip_one_data()
                print(f.current_header.date,
                      f.current_header.time,
                      file=o)

                f.read_one_header()
                print(f.current_header.date,
                      f.current_header.time,
                      file=o)

                result = o.getvalue()
        self.assertMultiLineEqual(result, expect)

    def test_read_one_data_00(self):
        """Test for read_one_data()."""
        expect = """
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : False
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
>f4
2048 4
3 (1, 32, 64)
"""
        with GT3File('test00') as f:
            f.scan()
            with StringIO() as o:
                f.read_one_header()
                f.read_one_data()
                f.read_one_header()
                f.read_one_data()
                with redirect_stdout(o):
                    print()
                    print(f.current_data.flags)
                    print(f.current_data.dtype)
                    print(f.current_data.size, f.current_data.itemsize)
                    print(f.current_data.ndim, f.current_data.shape)

                result = o.getvalue()
        self.assertMultiLineEqual(result, expect)

    def test_read_00(self):
        """ Test for read(). """
        expect = """\
2018-06-16T15 4.0460477e-06
2018-06-16T18 4.0460477e-06
2018-06-16T21 4.0460477e-06
2018-06-17T00 4.0460477e-06
2018-06-17T03 4.0460477e-06
2018-06-17T06 4.0460477e-06
2018-06-17T09 4.0460477e-06
2018-06-17T12 4.0460477e-06
2018-06-17T15 4.0460477e-06
2018-06-17T18 4.0460477e-06
2018-06-17T21 4.0460477e-06
2018-06-18T00 4.0460477e-06
"""

        with GT3File('test00') as f:
            with StringIO() as o:
                for h, d in f.read():
                    print(h.date, d[0, 0, 0], file=o)
                result = o.getvalue()
        self.assertMultiLineEqual(result, expect)

    def test_extract_t_axis_00(self):
        """Test for extract_t_axis() with multi-items file."""
        ts = np.array([17693439])
        expect = {'values': ts, 'unit': 'hours'}

        with GT3File('test01') as f:
            result = f.extract_t_axis()

        self.assertEqual(tuple(result['values']), tuple(expect['values']))
        self.assertEqual(result['unit'], expect['unit'])

    def test_extract_t_axis_01(self):
        """Test for extract_t_axis() with single-item file."""
        ts = np.array(
            [17693439, 17693442, 17693445, 17693448,
             17693451, 17693454, 17693457, 17693460,
             17693463, 17693466, 17693469, 17693472])
        expect = {'values': ts, 'unit': 'hours'}

        with GT3File('test00') as f:
            result = f.extract_t_axis()

        self.assertEqual(tuple(result['values']), tuple(expect['values']))
        self.assertEqual(result['unit'], expect['unit'])


###############################################################################
# Here we go.
###############################################################################
if __name__ == '__main__':
    unittest.main()
