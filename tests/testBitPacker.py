#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import unittest

from pygt3file import BitPacker, InvalidArgumentError




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
# Here we go.
###############################################################################
if __name__ == '__main__':
    unittest.main()
