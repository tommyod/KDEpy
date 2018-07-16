#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import pytest
import numpy as np
import numba

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
    n = num_points - 1 # Number of intervals
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


def binning_numpy(data, num_points, weights=None):
    """
    Binning using NumPy only.
    
    Time on 1 million data points: 156 ms ± 1.74 ms
    """
    # Convert the data to numpy Arrays
    data = np.asarray_chkfinite(data, dtype=np.float)
    
    if weights is None:
        weights = np.ones_like(data)
        
    weights = np.asarray_chkfinite(weights, dtype=np.float)
    weights = weights / np.sum(weights)
    
    if not len(data) == len(weights):
        raise ValueError('Length of data must match length of weights.')

    # Transform the data
    min_grid = np.min(data)
    max_grid = np.max(data)
    transformed_data = (data - min_grid) / (max_grid - min_grid)
    
    # Compute the integral and fractional part of the data
    # The integral part is used for lookups, the fractional part is used
    # to weight the data
    num_intervals = num_points - 1 # Number of intervals
    fractional, integral = np.modf(transformed_data * num_intervals)
    integral = integral.astype(np.int)

    # Sort the integral values, and the fractional data and weights by
    # the same key. This lets us use binary search, which is faster
    # than using a mask in the the loop below
    indices_sorted = np.argsort(integral)
    integral = integral[indices_sorted]
    fractional = fractional[indices_sorted]
    weights = weights[indices_sorted]
    
    # Pre-compute these products, as they are used in the loop many times
    frac_weights = fractional * weights
    neg_frac_weights = weights - frac_weights # (1 - fractional) * weights
    
    result = np.zeros(num_points + 1)
    for grid_point in np.unique(integral):
        
        # Use binary search to find indices for the grid point
        # Then sum the data assigned to that grid point
        low_index = np.searchsorted(integral, grid_point, side='left')
        high_index = np.searchsorted(integral, grid_point, side='right')
        result[grid_point] += neg_frac_weights[low_index:high_index].sum()
        result[grid_point + 1] += frac_weights[low_index:high_index].sum()
       
    grid = np.linspace(min_grid, max_grid, num_points)
    return grid, result[:-1]


@numba.jit
def binning_weighted(transformed_data, weights, result):
    """
    Fast binning using Numba.
    """
    for i in range(len(transformed_data)):
        data_point, weight = transformed_data[i], weights[i]
        integral, fractional = int(data_point), (data_point) % 1
        result[integral] += (1 - fractional) * weight
        
        if (integral + 1) < len(result):
            result[integral + 1] +=  fractional * weight
            
    return result

def binning_numba(data, num_points, weights=None):
    """
    First attempt at fast binning.
    
    Time on 1 million data points: 43.1 ms ± 1.57 ms
    """
    
    data = np.asarray_chkfinite(data, dtype=np.float)
    
    if weights is None:
        weights = np.ones_like(data)
        
    # This is important, since weights [1, 2, 3, ..] might overflow on sum
    weights = np.asarray_chkfinite(weights, dtype=np.float)
    weights = weights / np.sum(weights)
    
    if not len(data) == len(weights):
        raise ValueError('Length of data must match length of weights.')

    result = np.zeros(num_points, dtype=np.float64)
    
    # Normalize the data, and multiply by the number of intervals
    min_obs = np.min(data)
    max_obs = np.max(data)
    data = (data - min_obs) / (max_obs - min_obs) * (num_points - 1)
    
    # Compute the grid, and call the optimized Numba function
    grid = np.linspace(min_obs, max_obs, num_points)
    return grid, binning_weighted(data, weights, result)


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])