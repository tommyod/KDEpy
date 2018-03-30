#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests.
"""
import numpy as np
from KDEpy import KDE


def test_defaults():
    """
    Test closest pair of points in a line.
    """
    kde = KDE('box')
    kde.fit(np.array([0]))
    x = np.array([0])
    y = kde.evaluate_naive(x)
    assert np.allclose(y, np.array([1.]))

 
if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])