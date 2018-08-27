#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for binning functions.
"""
import pytest
import numpy as np
from KDEpy.utils import autogrid
from KDEpy.binning import (linbin_numpy, linear_binning, linbin_Ndim_python)
import itertools
import random


def naivebinning(data, grid_points, weights=None):
    """
    DO NOT USE.
    
    Only for testing purposes. Very slow.
    
    Time on 1 million data points: 2.16 s ± 45.8 ms
    
    Examples
    --------
    >>> data = np.array([2, 2.5, 3, 4])
    >>> ans = naivebinning(data, np.arange(6), weights=None)
    >>> assert np.allclose(ans, np.array([0, 0, 0.375, 0.375, 0.25, 0]))
    >>> ans = naivebinning(data, np.arange(6), weights=np.arange(1, 5))
    >>> assert np.allclose(ans, np.array([0, 0, 0.2, 0.4, 0.4, 0]))
    >>> data = np.array([2, 2.5, 3, 4])
    >>> ans = naivebinning(data, np.arange(1, 7), weights=None)
    >>> assert np.allclose(ans, np.array([0, 0.375, 0.375, 0.25, 0, 0]))
    """
    
    # Convert the data to numpy Arrays
    data = np.asarray_chkfinite(data, dtype=np.float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=np.float)
    
    if weights is None:
        weights = np.ones_like(data)
        
    weights = np.asarray_chkfinite(weights, dtype=np.float)
    weights = weights / np.sum(weights)

    # Prepare to transform data
    n = len(grid_points) - 1  # Number of intervals
    min_grid = np.min(grid_points)
    max_grid = np.max(grid_points)
    transformed_data = (data - min_grid) / (max_grid - min_grid) * n
    
    result = np.zeros_like(grid_points, dtype=np.float)
    
    # Go through data points and weights, use O(1) lookups and weight the
    # data point linearily by distance and the perscribed weights
    for data_point, w in zip(transformed_data, weights):

        # Retrieve the integral and fractional parts quickly
        integral, fractional = int(data_point), (data_point) % 1
        
        # Add to the leftmost grid point, and the rightmost if possible
        result[int(integral)] += (1 - fractional) * w
        if (integral + 1) < len(result):
            result[int(integral) + 1] += fractional * w

    return result


class TestBinningFunctions():
    
    @pytest.mark.parametrize("data", [[1, 2, 3, 4, 5, 6],
                                      [0.04, 0.54, 0.33, 0.85, 0.16],
                                      [-4.12, 0.98, -4.3, -1.85],
                                      [0, 0, 1]])
    def test_invariance_under_permutations_numpy_binning(self, data):
        """
       
        """
        data = np.array(data)
        grid = np.linspace(np.min(data) - 1, np.max(data), num=5)
        y1 = linbin_numpy(data, grid, weights=None)
        np.random.seed(123)
        p = np.random.permutation(len(data))
        y2 = linbin_numpy(data[p], grid, weights=None)
        
        assert np.allclose(y1, y2)
        
    @pytest.mark.parametrize("data, weights, ans", 
                             [([1, 2, 2.5, 3], None, 
                               np.array([1., 1.5, 1.5]) / 4),
                              ([1, 2, 2.5, 3], [2, 1, 3, 2], 
                               np.array([2, 2.5, 3.5]) / 8)])
    def test_binning_simple_examples(self, data, weights, ans):
        
        grid = np.array([1, 2, 3])
        for func in [naivebinning, linbin_numpy]:
            
            y = func(data, grid, weights=weights)
            assert np.allclose(y, ans)
            
    @pytest.mark.parametrize("dims, use_weights, eq_grid", 
                             itertools.product([1, 2, 3, 4], 
                                               [True, False],
                                               [True, False]))  
    def test_cython_binning(self, dims, use_weights, eq_grid):
        """
        Test the fast N-dimensional binning up against the naive
        Python implementation - using weights, no weights, equal grid values
        in each direction and unequal ones.
        """
        
        num_points = 1000
        data = np.random.randn(dims * num_points).reshape(num_points, dims) / 7
        
        if use_weights:
            weights = np.random.randn(num_points)
        else:
            weights = None
            
        if eq_grid:
            num_points = (16,) * dims
        else:
            num_points = tuple([random.randint(8, 16) for i in range(dims)])
            
        grid_points = autogrid(np.array([[0] * dims]), num_points=num_points)
        result = linear_binning(data, grid_points, weights=weights)
        result_slow = linbin_Ndim_python(data, grid_points, weights=weights)
        
        assert np.allclose(result, result_slow)
        
    @pytest.mark.parametrize("dims", [1, 2, 3, 4])
    def test_binning_correctness_single_point(self, dims):
        """
        Permute a single grid poind make sure that same point is weighted
        highly.
        """
        
        eps = 10e-6
        for subtest in range(25):
            data = np.random.randint(-2, 2, size=(1, dims)) - eps
            grid_points = autogrid(np.array([[0] * dims]), 
                                   num_points=(7,) * dims)
            answer = linear_binning(data, grid_points)
            
            for grid_point, a in zip(grid_points, answer):
                diff = np.sum((grid_point - data.ravel())**2)
                if diff < eps:
                    assert np.allclose(a, (1 - eps)**dims)
 
    
if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v', '--durations=15'])