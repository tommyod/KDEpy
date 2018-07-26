#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import pytest
import numpy as np

_numba_installed = True
try:
    import numba
except ImportError:
    _numba_installed = False
    
    
def linbin_numpy(data, grid_points, weights=None):
    """
    Linear binning using NumPy. Assigns weights to grid points from data.
    
    This function is fast for data sets upto approximately 1-10 million,
    it uses vectorized NumPy functions to perform linear binning.
    
    Time on 1 million data points: 79.6 ms ± 1.01 ms
    Time on 10 million data points: 879 ms ± 4.55 ms
    Time on 100 million data points: 10.3 s ± 663 ms
    
    
    Examples
    --------
    >>> data = np.array([2, 2.5, 3, 4])
    >>> linbin_numpy(data, np.arange(6), weights=None)
    array([0.   , 0.   , 0.375, 0.375, 0.25 , 0.   ])
    >>> linbin_numpy(data, np.arange(6), weights=np.arange(1, 5))
    array([0. , 0. , 0.2, 0.4, 0.4, 0. ])
    >>> data = np.array([2, 2.5, 3, 4])
    >>> linbin_numpy(data, np.arange(1, 7), weights=None)
    array([0.   , 0.375, 0.375, 0.25 , 0.   , 0.   ])
    
    """
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=np.float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=np.float)
    
    # Verify that the grid is equidistant
    diffs = np.diff(grid_points)
    assert np.allclose(np.ones_like(diffs) * diffs[0], diffs)
    
    if weights is None:
        weights = np.ones_like(data)
        
    weights = np.asarray_chkfinite(weights, dtype=np.float)
    weights = weights / np.sum(weights)
    
    if not len(data) == len(weights):
        raise ValueError('Length of data must match length of weights.')
        
    # Transform the data
    min_grid = np.min(grid_points)
    max_grid = np.max(grid_points)
    num_intervals = len(grid_points) - 1  # Number of intervals
    dx = (max_grid - min_grid) / num_intervals
    transformed_data = (data - min_grid) / dx
    
    # Compute the integral and fractional part of the data
    # The integral part is used for lookups, the fractional part is used
    # to weight the data
    num_intervals = len(grid_points) - 1  # Number of intervals
    fractional, integral = np.modf(transformed_data)
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
    neg_frac_weights = weights - frac_weights
    
    # If the data is not a subset of the grid, the integral values will be
    # outside of the grid. To solve the problem, we filter these values away
    unique_integrals = np.unique(integral)
    unique_integrals = unique_integrals[(unique_integrals >= 0) & 
                                        (unique_integrals <= len(grid_points))]
    
    result = np.asfarray(np.zeros(len(grid_points) + 1))
    for grid_point in unique_integrals:
        
        # Use binary search to find indices for the grid point
        # Then sum the data assigned to that grid point
        low_index = np.searchsorted(integral, grid_point, side='left')
        high_index = np.searchsorted(integral, grid_point, side='right')
        result[grid_point] += neg_frac_weights[low_index:high_index].sum()
        result[grid_point + 1] += frac_weights[low_index:high_index].sum()
       
    return result[:-1]
    

def linear_binning(data, num_points=2**10, weights=None):
    """
    Compute binning by setting a linear grid and weighting points linearily
    by their distance to the grid points. In addition, weight asssociated with
    data points may be passed.
    
    Parameters
    ----------
    data
        The data points.
    num_points
        The number of points in the grid.
    weights
        The weights.
        
    Returns
    -------
    (grid, data)
        Data weighted at each grid point.
        
    Examples
    --------
    >>> data = [1, 1.5, 1.5, 2, 2.8, 3]
    >>> grid, data = linear_binning(data, num_points=3)
    >>> np.allclose(data, np.array([0.33333, 0.36667, 0.3]))
    True
    """
    _numba_installed = False
    if _numba_installed:
        return binning_numba(data, num_points, weights=None)
    else:
        return binning_numpy(data, num_points, weights=None)


def binning_numpy(data, num_points, weights=None):
    """
    DEPRECATE IN FAVOR OF linbin_numpy
    
    
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
    num_intervals = num_points - 1  # Number of intervals
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
    neg_frac_weights = weights - frac_weights  # (1 - fractional) * weights
    
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
            result[integral + 1] += fractional * weight
            
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
    
    data = np.array([2, 2.5, 3, 4])
    grid_points = np.array([0, 1, 2, 3, 4, 5])
    result = linbin_numpy(data, grid_points, weights=None)
    print(result)