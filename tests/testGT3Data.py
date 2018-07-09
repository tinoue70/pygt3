#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from io import StringIO
import unittest
from contextlib import redirect_stdout
from pygt3file import GT3Data


class TestGT3Data(unittest.TestCase):
    def setUp(self):
        pass

    def test_init_00(self):
        """Test of explicit constructor."""
        d = GT3Data((3,))
        self.assertTrue(isinstance(d,GT3Data))
        self.assertIsNone(d.name)

    def test_init_01(self):
        """Test of explicit constructor with attributes"""
        d = GT3Data((3,), name='test', number=1)
        self.assertTrue(isinstance(d,GT3Data))
        self.assertEqual(d.name, 'test')
        self.assertEqual(d.number, 1)

    def test_init_02(self):
        """Test of new-from-template(slicing)."""
        d = GT3Data((3,), name='test', number=1)
        v = d[1:]
        self.assertTrue(isinstance(v,GT3Data))
        self.assertEqual(v.name, 'test')
        self.assertEqual(v.number, 1)

    def test_init_03(self):
        """Test of view casting."""
        d = np.arange(10).view(GT3Data)
        self.assertTrue(isinstance(d,GT3Data))
        self.assertIsNone(d.name)



    def test_dump_00(self):
        """Test dump without attributes, not indexed."""
        expect=("""\
================================================================================
[[[ 0.          0.58823529  1.17647059]
  [ 1.76470588  2.35294118  2.94117647]
  [ 3.52941176  4.11764706  4.70588235]]

 [[ 5.29411765  5.88235294  6.47058824]
  [ 7.05882353  7.64705882  8.23529412]
  [ 8.82352941  9.41176471 10.        ]]]
================================================================================
""")
        shape = (2,3,3,)
        d = np.linspace(0,10,
                        shape[0]*shape[1]*shape[2]
        ).reshape(shape).view(GT3Data)
        # d.name = 'test'
        # d.number = 2
        indexed = False
        f = StringIO()
        with redirect_stdout(f):
            d.dump(indexed=indexed)
        result = f.getvalue()
        self.assertMultiLineEqual(expect,result)

    def test_dump_01(self):
        """Test dump with attributes, not indexed."""
        expect=("""\
======test:#2===================================================================
[[[ 0.          0.58823529  1.17647059]
  [ 1.76470588  2.35294118  2.94117647]
  [ 3.52941176  4.11764706  4.70588235]]

 [[ 5.29411765  5.88235294  6.47058824]
  [ 7.05882353  7.64705882  8.23529412]
  [ 8.82352941  9.41176471 10.        ]]]
================================================================================
""")
        shape = (2,3,3,)
        d = np.linspace(0,10,
                        shape[0]*shape[1]*shape[2]
        ).reshape(shape).view(GT3Data)
        d.name = 'test'
        d.number = 2
        indexed = False
        f = StringIO()
        with redirect_stdout(f):
            d.dump(indexed=indexed)
        result = f.getvalue()
        self.assertMultiLineEqual(expect,result)

    def test_dump_02(self):
        """Test dump without attributes, indexed."""
        expect=("""\
================================================================================
#  xindex   yindex   zindex                 data
        0        0        0             0.000000
        1        0        0             0.588235
        2        0        0             1.176471
        0        1        0             1.764706
        1        1        0             2.352941
        2        1        0             2.941176
        0        2        0             3.529412
        1        2        0             4.117647
        2        2        0             4.705882
        0        0        1             5.294118
        1        0        1             5.882353
        2        0        1             6.470588
        0        1        1             7.058824
        1        1        1             7.647059
        2        1        1             8.235294
        0        2        1             8.823529
        1        2        1             9.411765
        2        2        1            10.000000
================================================================================
""")
        shape = (2,3,3,)
        d = np.linspace(0,10,
                        shape[0]*shape[1]*shape[2]
        ).reshape(shape).view(GT3Data)
        # d.name = 'test'
        # d.number = 2
        indexed = True
        f = StringIO()
        with redirect_stdout(f):
            d.dump(indexed=indexed)
        result = f.getvalue()
        self.assertMultiLineEqual(expect,result)

    def test_dump_03(self):
        """Test dump with attributes, indexed."""
        expect=("""\
======test:#2===================================================================
#  xindex   yindex   zindex                 data
        0        0        0             0.000000
        1        0        0             0.588235
        2        0        0             1.176471
        0        1        0             1.764706
        1        1        0             2.352941
        2        1        0             2.941176
        0        2        0             3.529412
        1        2        0             4.117647
        2        2        0             4.705882
        0        0        1             5.294118
        1        0        1             5.882353
        2        0        1             6.470588
        0        1        1             7.058824
        1        1        1             7.647059
        2        1        1             8.235294
        0        2        1             8.823529
        1        2        1             9.411765
        2        2        1            10.000000
================================================================================
""")
        shape = (2,3,3,)
        d = np.linspace(0,10,
                        shape[0]*shape[1]*shape[2]
        ).reshape(shape).view(GT3Data)
        d.name = 'test'
        d.number = 2
        indexed = True
        f = StringIO()
        with redirect_stdout(f):
            d.dump(indexed=indexed)
        result = f.getvalue()
        self.assertMultiLineEqual(expect,result)



###############################################################################
# Here we go.
###############################################################################
if __name__ == '__main__':
    unittest.main()
