#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for binning functions.
"""
import numpy as np

from KDEpy.binning import (binning_numpy, binning_numba)
import pytest


def naivebinning(data, num_points, weights=None):
    """
    DO NOT USE.
    
    Only for testing purposes. Very slow.
    
    Time on 1 million data points: 2.16 s ± 45.8 ms
    """
    
    # Convert the data to numpy Arrays
    data = np.asarray_chkfinite(data, dtype=np.float)
    
    if weights is None:
        weights = np.ones_like(data)
        
    weights = np.asarray_chkfinite(weights, dtype=np.float)
    weights = weights / np.sum(weights)

    # Prepare to transform data
    n = num_points - 1  # Number of intervals
    min_grid = np.min(data)
    max_grid = np.max(data)
    transformed_data = (data - min_grid) / (max_grid - min_grid) * n
    
    result = np.zeros(num_points)
    
    # Go through data points and weights, use O(1) lookups and weight the
    # data point linearily by distance and the perscribed weights
    for data_point, w in zip(transformed_data, weights):
        
        # Retrieve the integral and fractional parts quickly
        integral, fractional = int(data_point), (data_point) % 1
        
        # Add to the leftmost grid point, and the rightmost if possible
        result[int(integral)] += (1 - fractional) * w
        if (integral + 1) < len(result):
            result[int(integral) + 1] += fractional * w

    grid = np.linspace(min_grid, max_grid, num_points)
    return grid, result


class TestBinningFunctions():
    
    @pytest.mark.parametrize("data", [[1, 2, 3, 4, 5, 6],
                                      [0.04, 0.54, 0.33, 0.85, 0.16],
                                      [-4.12, 0.98, -4.3, -1.85],
                                      [0, 0, 1]])
    def test_invariance_under_permutations_numpy_binning(self, data):
        """
       
        """
        data = np.array(data)
        x1, y1 = binning_numpy(data, 5, weights=None)
        np.random.seed(123)
        p = np.random.permutation(len(data))
        x2, y2 = binning_numpy(data[p], 5, weights=None)
        
        assert np.allclose(y1, y2)
        assert np.allclose(x1, x2)
        
    @pytest.mark.parametrize("data", [[1, 2, 3, 4, 5, 6],
                                      [0.04, 0.54, 0.33, 0.85, 0.16],
                                      [-4.12, 0.98, -4.3, -1.85],
                                      [0, 0, 1]])
    def test_invariance_under_permutations_numba_binning(self, data):
        """
       
        """
        data = np.array(data)
        x1, y1 = binning_numba(data, 5, weights=None)
        np.random.seed(123)
        p = np.random.permutation(len(data))
        x2, y2 = binning_numba(data[p], 5, weights=None)
        
        assert np.allclose(y1, y2)
        assert np.allclose(x1, x2)
        
    @pytest.mark.parametrize("data, weights, ans", 
                             [([1, 2, 2.5, 3], None, 
                               np.array([1., 1.5, 1.5]) / 4),
                              ([1, 2, 2.5, 3], [2, 1, 3, 2], 
                               np.array([2, 2.5, 3.5]) / 8)])
    def test_binning_simple_examples(self, data, weights, ans):
        
        for func in [naivebinning, binning_numpy, binning_numba]:
            
            grid, y = func(data, 3, weights=weights)
            assert np.allclose(y, ans)
 
    
if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v', 
                      '--durations=15'
                      ])