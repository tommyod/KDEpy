#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for functions related to linear binning. These functions employ
linear binning to weighted data. This is typically a preprocessing step
before convolving with a kernel in the FFTKDE, but may also be used to
reduce the effective number of data points in any algorithm.

The idea behind linear binning is the following: (1) go through every
data point and (2) assign weight to the 2^dims nearest grid points.
In `dims` dimensions, there are 2 points on the grid to consider in
each direction, so a total of 2^dims grid points to assign weights to
for every data point. The weights are determined by the proportion of
the volume of this hypercube that is enclosed by the data point.

A ------------------------------------ B
|                          |           |
|                          |           |
|                          X-----------|
|                                      |
|                                      |
|                                      |
|                                      |
C ------------------------------------ C
"""
import numpy as np
import itertools
import functools
import operator

try:
    import cutils
    _use_Cython = True
except ModuleNotFoundError:
    _use_Cython = False


def linbin_cython(data, grid_points, weights=None):
    """
    1D Linear binning using Cython. Assigns weights to grid points from data.
    
    from KDEpy.binning import linbin_cython
    import numpy as np
    data = np.random.randn(10**7)
    %timeit linbin_cython(data, np.linspace(-8,8, num=2**10))
    -> 547 ms ± 8.32 ms

    Time on 1 million data points: 30 ms
    Time on 10 million data points: 290 ms
    Time on 100 million data points: 2.86 s
    """
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=np.float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=np.float)

    # Verify that the grid is equidistant
    diffs = np.diff(grid_points)
    assert np.allclose(np.ones_like(diffs) * diffs[0], diffs)

    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=np.float)
        weights = weights / np.sum(weights)

    if (weights is not None) and (len(data) != len(weights)):
        raise ValueError('Length of data must match length of weights.')

    # Transform the data
    min_grid = np.min(grid_points)
    max_grid = np.max(grid_points)
    num_intervals = len(grid_points) - 1  # Number of intervals
    dx = (max_grid - min_grid) / num_intervals
    transformed_data = (data - min_grid) / dx

    result = np.asfarray(np.zeros(len(grid_points) + 1))

    if weights is None:
        result = cutils.iterate_data(transformed_data, result)
        return np.asfarray(result[:-1]) / len(transformed_data)
    else:
        res = cutils.iterate_data_weighted(transformed_data, weights, result)
        return np.asfarray(res[:-1])


def linbin_numpy(data, grid_points, weights=None):
    """
    1D Linear binning using NumPy. Assigns weights to grid points from data.

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
    num_intervals = len(grid_points) - 1  
    dx = (max_grid - min_grid) / num_intervals
    transformed_data = (data - min_grid) / dx

    # Compute the integral and fractional part of the data
    # The integral part is used for lookups, the fractional part is used
    # to weight the data
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


def linbin_Ndim(data, grid_points, weights=None):
    """
    N-dimensional linear binning. This is a slow, pure-Python function.
    Although it is slow, it works and may be used as a starting point for
    developing faster Cython implementations.
    
    With :math:`N` data points, and :math:`n` grid points in each dimension
    :math:`d`, the running time is :math:`O(N2^d)`. For each point the
    algorithm finds the nearest points, of which there are two in each
    dimension.
   
    Parameters
    ----------
    data : array-like
        The data must be of shape (obs, dims).
    grid_points : array-like
        Grid, where cartesian product is already performed.
    weights : array-like
        Must have shape (obs,).
        
    Examples
    --------
    >>> 1 + 1
    2
    """
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=np.float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=np.float)
    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=np.float)
    else:
        # This is not efficient, but this function should just be correct
        weights = np.ones_like(data.shape[0])
    weights = weights / np.sum(weights)

    if (weights is not None) and (data.shape[0] != len(weights)):
        raise ValueError('Length of data must match length of weights.')
    
    obs_tot, dims = grid_points.shape
    
    # Compute the number of grid points for each dimension in the grid
    grid_num = (grid_points[:, i] for i in range(dims))
    grid_num = np.array(list(len(np.unique(g)) for g in grid_num))
    
    # Scale the data to the grid
    min_grid = np.min(grid_points, axis=0)
    max_grid = np.max(grid_points, axis=0)
    num_intervals = (grid_num - 1)  # Number of intervals
    dx = (max_grid - min_grid) / num_intervals
    data = (data - min_grid) / dx

    # Create results
    result = np.zeros(grid_points.shape[0], dtype=np.float)
        
    # Go through every data point
    for observation, weight in zip(data, weights):
        
        # Compute integer part and fractional part for every x_i
        # Compute relation to previous grid point, and next grid point
        int_frac = (((int(coordinate), 1 - (coordinate % 1)), 
                     (int(coordinate) + 1, (coordinate % 1)))
                    for coordinate in observation)

        # Go through every cartesian product, i.e. every corner in the
        # hypercube grid points surrounding the observation
        for cart_prod in itertools.product(*int_frac):
            
            fractions = (frac for (integral, frac) in cart_prod)
            integrals_rev = list(integral for (integral, frac) in 
                                 reversed(cart_prod))
            
            # Find the index in the resulting array, compured by
            # sum integra_valuel * grid_num**i 
            index = sum((i * g**c) for ((c, i), g) in 
                        zip(enumerate(integrals_rev), grid_num))
            value = functools.reduce(operator.mul, fractions)
            result[index % obs_tot] += value * weight
        
    assert np.allclose(np.sum(result), 1)
    return result


def linbin_2dim(data, grid_points, weights=None):
    """
    2-dimensional linear binning.
    
    With :math:`N` data points, and :math:`n` grid points in each dimension
    :math:`d`, the running time is :math:`O(N2^d)`. For each point the
    algorithm finds the nearest points, of which there are two in each
    dimension.
    
    Approximately 200 times faster than pure python implementation.
    
    Parameters
    ----------
    data : array-like
        The data must be of shape (obs, dims).
    grid_points : array-like
        Grid, where cartesian product is already performed.
    weights : array-like
        Must have shape (obs,).
        
    Examples
    --------
    >>> 1 + 1
    2
    """
    data_obs, data_dims = data.shape
    assert len(grid_points.shape) == 2
    assert data_dims == 2
    
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=np.float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=np.float)
    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=np.float)
        weights = weights / np.sum(weights)

    if (weights is not None) and (data.shape[0] != len(weights)):
        raise ValueError('Length of data must match length of weights.')
    
    obs_tot, dims = grid_points.shape
    
    # Compute the number of grid points for each dimension in the grid
    grid_num = (grid_points[:, i] for i in range(dims))
    grid_num = np.array(list(len(np.unique(g)) for g in grid_num))
    
    # Scale the data to the grid
    min_grid = np.min(grid_points, axis=0)
    max_grid = np.max(grid_points, axis=0)
    num_intervals = (grid_num - 1)
    dx = (max_grid - min_grid) / num_intervals
    data = (data - min_grid) / dx

    # Create results
    result = np.zeros(grid_points.shape[0], dtype=np.float)
    
    # Call the Cython implementation
    if weights is not None:
        result = cutils.iterate_data_weighted_2D(data, weights, result, 
                                                 grid_num, obs_tot)
        result = np.asarray_chkfinite(result, dtype=np.float)
    else:
        result = cutils.iterate_data_2D(data, result, grid_num, obs_tot)
        result = np.asarray_chkfinite(result, dtype=np.float)
        result = result / data_obs

    assert np.allclose(np.sum(result), 1)
    return result


def linear_binning(data, grid_points, weights=None):
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
    >>> grid_points = [1, 2, 3]
    >>> data = linear_binning(data, grid_points)
    >>> np.allclose(data, np.array([0.33333, 0.36667, 0.3]))
    True
    """
    data = np.asarray_chkfinite(data, dtype=np.float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=np.float)
    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=np.float)
        
    try:
        obs, dims = data.shape
    except ValueError:
        dims = 1
        
    if dims == 1:
        if _use_Cython:
            return linbin_cython(data.ravel(), grid_points, weights=weights)   
        else:
            return linbin_numpy(data.ravel(), grid_points, weights=weights)
    elif dims == 2:
        return linbin_2dim(data, grid_points, weights=weights)
    else:
        assert False


if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v', '--capture=sys'])
