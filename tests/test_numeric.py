"""
Tests for core numeric routines.
"""

import pytest

from cmfpy.common import s_dot, s_T_dot

import numpy as np
from numpy.testing import assert_allclose

TOL = 1e-10


def test_sdot():

    ov = np.array([[1., 1., 1.]])
    B = np.array([[0.,  1.,  2.,  3.],
                  [4.,  5.,  6.,  7.],
                  [8.,  9., 10., 11.]])

    shifts = [2, 1, 0, -1, -2]
    expects = [
        np.array([[0., 0., 12., 15.]]),
        np.array([[0., 12., 15., 18.]]),
        np.array([[12., 15., 18., 21.]]),
        np.array([[15., 18., 21., 0.]]),
        np.array([[18., 21., 0., 0.]]),
    ]

    for s, expected in zip(shifts, expects):
        found = s_dot(ov, B, s)
        assert_allclose(found, expected)


def test_sTdot():

    ov = np.array([[1., 1., 1.]])
    B = np.array([[0.,  1.,  2.],
                  [3.,  4.,  5.],
                  [6.,  7.,  8.],
                  [9., 10., 11.]])

    shifts = [2, 1, 0, -1, -2]
    expects = [
        np.array([[0.,  3., 6.,   9.]]),
        np.array([[1.,  7., 13., 19.]]),
        np.array([[3., 12., 21., 30.]]),
        np.array([[3.,  9., 15., 21.]]),
        np.array([[2.,  5., 8.,  11.]]),
    ]

    for s, expected in zip(shifts, expects):
        found = s_T_dot(ov, B, s)
        assert_allclose(found, expected)
