#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests.
"""

try:
    from kde import (KDE)
except ValueError:
    pass


def test_closest_pair_line():
    """
    Test closest pair of points in a line.
    """
    assert 1 + 1 == 2
    


    
if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])