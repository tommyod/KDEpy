#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import numpy as np


def autogrid(data, kernel_support, num_points=1024, percentile=0.05):
    """
    Automatically select a grid if the user did not supply one.
    Input is (obs, dims), and so is ouput.
    
    number of grid : should be a power of two
    percentile : is how far out we go out
    """
    obs, dims = data.shape
    minimums, maximums = data.min(axis=0), data.max(axis=0)
    ranges = maximums - minimums
    
    grid_points = np.empty(shape=(num_points // 2**(dims - 1), dims))

    generator = enumerate(zip(minimums, maximums, ranges))
    for i, (minimum, maximum, rang) in generator:
        outside_borders = max(percentile * rang, kernel_support)
        grid_points[:, i] = np.linspace(minimum - outside_borders,
                                        maximum + outside_borders,
                                        num=num_points // 2**(dims - 1))

    return grid_points
    

if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v', '--capture=sys'])