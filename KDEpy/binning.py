#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for functions related to linear binning. These functions perform
linear binning on weighted data. This is typically a preprocessing step
before convolving with a kernel in the FFTKDE, but may also be used to
reduce the effective number of data points in any algorithm.

The idea behind linear binning is the following: (1) go through every
data point and (2) assign a weight to the 2^dims nearest grid points.
In `dims` dimensions, there are 2 points on the grid to consider in
each dimension, so a total of 2^dims grid points must be assigned weights to
for every data point. The weights are determined by the proportion of
the volume of this hypercube that is enclosed by the data point.

A ------------------------------ B
|                    |           |
|                    |           |
|                    X-----------|
|                                |
|                                |
|                                |
C ------------------------------ C

References
----------
- Fan, Jianqing, and James S. Marron.
  “Fast Implementations of Nonparametric Curve Estimators.”
  Journal of Computational and Graphical Statistics 3, no. 1 (March 1, 1994).
  https://doi.org/10.1080/10618600.1994.10474629.
"""
import numpy as np
import itertools
import functools
import operator
from KDEpy.utils import cartesian
import cutils


grid_is_sorted = cutils.grid_is_sorted

# This parameter was in use when a NumPy implementation and a Cython
# implementation were both used. Now only Cython is used.
_use_Cython = True


def linbin_cython(data, grid_points, weights=None):
    """
    1D Linear binning using Cython. Assigns weights to grid points from data.

    Runs in approx 10 ms on 1 million data points.

    Parameters
    ----------
    data : array-like
        The data to bin. Must be of shape (obs,).
    grid_points : array-like
        Equidistant grid points to assign weights to.
        Must be of shape (points,).
    weights : array-like
        The weights of the data points.
        Must be of shape (obs,).

    Examples
    --------
    >>> data = np.array([2, 2.5, 3, 4])
    >>> ans = linbin_cython(data, np.arange(6), weights=None)
    >>> np.allclose(ans, np.array([0, 0, 0.375, 0.375, 0.25, 0]))
    True
    >>> ans = linbin_cython(data, np.arange(6), weights=np.arange(1, 5))
    >>> np.allclose(ans, np.array([0, 0, 0.2, 0.4, 0.4, 0]))
    True
    >>> data = np.array([2, 2.5, 3, 4])
    >>> ans = linbin_cython(data, np.arange(1, 7), weights=None)
    >>> np.allclose(ans, np.array([0, 0.375, 0.375, 0.25, 0, 0]))
    True
    """
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=float)

    assert len(data.shape) == 1
    assert len(grid_points.shape) == 1

    # Verify that the grid is equidistant
    diffs = np.diff(grid_points)
    assert np.allclose(np.ones_like(diffs) * diffs[0], diffs)

    if weights is not None:
        assert len(weights.shape) == 1
        weights = np.asarray_chkfinite(weights, dtype=float)
        weights = weights / np.sum(weights)

    if (weights is not None) and (len(data) != len(weights)):
        raise ValueError("Length of data must match length of weights.")

    # Transform the data
    min_grid = np.min(grid_points)
    max_grid = np.max(grid_points)
    num_intervals = len(grid_points) - 1  # Number of intervals
    dx = (max_grid - min_grid) / num_intervals
    transformed_data = (data - min_grid) / dx

    result = np.asfarray(np.zeros(num_intervals + 2))

    # Two Cython functions are implemented, one for weighted data and one
    # for unweighted data, since creating equal weights is costly w.r.t time
    if weights is None:
        result = cutils.iterate_data_1D(transformed_data, result)
        return np.asfarray(result[:-1]) / transformed_data.shape[0]
    else:
        res = cutils.iterate_data_1D_weighted(transformed_data, weights, result)
        return np.asfarray(res[:-1])  # Remove last, outside of grid


def linbin_numpy(data, grid_points, weights=None):
    """
    1D Linear binning using NumPy. Assigns weights to grid points from data.

    This function is fast for data sets upto approximately 1-10 million,
    it uses vectorized NumPy functions to perform linear binning. Takes around
    100 ms on 1 million data points, so not nearly as fast as the Cython
    implementation (10 ms).

    Parameters
    ----------
    data : array-like
        Must be of shape (obs,).
    grid_points : array-like
        Must be of shape (points,).
    weights : array-like
        Must be of shape (obs,).

    Examples
    --------
    >>> data = np.array([2, 2.5, 3, 4])
    >>> ans = linbin_numpy(data, np.arange(6), weights=None)
    >>> np.allclose(ans, np.array([0, 0, 0.375, 0.375, 0.25, 0]))
    True
    >>> ans = linbin_numpy(data, np.arange(6), weights=np.arange(1, 5))
    >>> np.allclose(ans, np.array([0, 0, 0.2, 0.4, 0.4, 0]))
    True
    >>> data = np.array([2, 2.5, 3, 4])
    >>> ans = linbin_numpy(data, np.arange(1, 7), weights=None)
    >>> np.allclose(ans, np.array([0, 0.375, 0.375, 0.25, 0, 0]))
    True
    """
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=float)
    assert len(data.shape) == 1
    assert len(grid_points.shape) == 1

    # Verify that the grid is equidistant
    diffs = np.diff(grid_points)
    assert np.allclose(np.ones_like(diffs) * diffs[0], diffs)

    if weights is None:
        weights = np.ones_like(data)

    weights = np.asarray_chkfinite(weights, dtype=float)
    weights = weights / np.sum(weights)

    if not len(data) == len(weights):
        raise ValueError("Length of data must match length of weights.")

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
    integral = integral.astype(int)

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
    unique_integrals = unique_integrals[(unique_integrals >= 0) & (unique_integrals <= len(grid_points))]

    result = np.asfarray(np.zeros(len(grid_points) + 1))
    for grid_point in unique_integrals:

        # Use binary search to find indices for the grid point
        # Then sum the data assigned to that grid point
        low_index = np.searchsorted(integral, grid_point, side="left")
        high_index = np.searchsorted(integral, grid_point, side="right")
        result[grid_point] += neg_frac_weights[low_index:high_index].sum()
        result[grid_point + 1] += frac_weights[low_index:high_index].sum()

    return result[:-1]


def linbin_Ndim_python(data, grid_points, weights=None):
    """
    d-dimensional linear binning. This is a slow, pure-Python function.
    Mainly used for testing purposes.

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
    >>> from KDEpy.utils import autogrid
    >>> grid_points = autogrid(np.array([[0, 0, 0]]), num_points=(3, 3, 3))
    >>> d = linbin_Ndim_python(np.array([[1.0, 0, 0]]), grid_points, None)
    """
    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=float)
    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=float)
    else:
        # This is not efficient, but this function should just be correct
        # The faster algorithm is implemented in Cython
        weights = np.ones(data.shape[0])
    weights = weights / np.sum(weights)

    if (weights is not None) and (data.shape[0] != len(weights)):
        raise ValueError("Length of data must match length of weights.")

    obs_tot, dims = grid_points.shape

    # Compute the number of grid points for each dimension in the grid
    grid_num = (grid_points[:, i] for i in range(dims))
    grid_num = np.array(list(len(np.unique(g)) for g in grid_num))

    # Scale the data to the grid
    min_grid = np.min(grid_points, axis=0)
    max_grid = np.max(grid_points, axis=0)
    num_intervals = grid_num - 1  # Number of intervals
    dx = (max_grid - min_grid) / num_intervals
    data = (data - min_grid) / dx

    # Create results
    result = np.zeros(grid_points.shape[0], dtype=float)

    # Go through every data point
    for observation, weight in zip(data, weights):

        # Compute integer part and fractional part for every x_i
        # Compute relation to previous grid point, and next grid point
        int_frac = (
            (
                (int(coordinate), 1 - (coordinate % 1)),
                (int(coordinate) + 1, (coordinate % 1)),
            )
            for coordinate in observation
        )

        # Go through every cartesian product, i.e. every corner in the
        # hypercube grid points surrounding the observation
        for cart_prod in itertools.product(*int_frac):

            fractions = (frac for (integral, frac) in cart_prod)
            integrals = list(integral for (integral, frac) in cart_prod)
            # Find the index in the resulting array, compured by
            # x_1 * (g_2 * g_3 * g_4) + x_2 * (g_3 * g_4) + x_3 * (g_4) + x_4

            index = integrals[0]
            for j in range(1, dims):
                index = grid_num[j] * index + integrals[j]

            value = functools.reduce(operator.mul, fractions)
            result[index % obs_tot] += value * weight

    assert np.allclose(np.sum(result), 1)
    return result


def linbin_Ndim(data, grid_points, weights=None):
    """
    d-dimensional linear binning, when d >= 2.

    With :math:`N` data points, and :math:`n` grid points in each dimension
    :math:`d`, the running time is :math:`O(N2^d)`. For each point the
    algorithm finds the nearest points, of which there are two in each
    dimension. Approximately 200 times faster than pure Python implementation.

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
    >>> from KDEpy.utils import autogrid
    >>> grid_points = autogrid(np.array([[0, 0, 0]]), num_points=(3, 3, 3))
    >>> d = linbin_Ndim(np.array([[1.0, 0, 0]]), grid_points, None)
    """
    data_obs, data_dims = data.shape
    assert len(grid_points.shape) == 2
    assert data_dims >= 2

    # Convert the data and grid points
    data = np.asarray_chkfinite(data, dtype=float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=float)
    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=float)
        weights = weights / np.sum(weights)

    if (weights is not None) and (data.shape[0] != len(weights)):
        raise ValueError("Length of data must match length of weights.")

    obs_tot, dims = grid_points.shape

    # Compute the number of grid points for each dimension in the grid
    grid_num = (grid_points[:, i] for i in range(dims))
    grid_num = np.array(list(len(np.unique(g)) for g in grid_num))

    # Scale the data to the grid
    min_grid = np.min(grid_points, axis=0)
    max_grid = np.max(grid_points, axis=0)
    num_intervals = grid_num - 1
    dx = (max_grid - min_grid) / num_intervals
    data = (data - min_grid) / dx

    # Create results
    result = np.zeros(grid_points.shape[0], dtype=float)

    # Call the Cython implementation. Loops are unrolled if d=1 or d=2,
    # and if d >= 3 a more general routine is called. It's a bit slower since
    # the loops are not unrolled.

    # Weighted data has two specific routines
    if weights is not None:
        if data_dims >= 3:
            binary_flgs = cartesian(([0, 1],) * dims)
            result = cutils.iterate_data_ND_weighted(data, weights, result, grid_num, obs_tot, binary_flgs)
        else:
            result = cutils.iterate_data_2D_weighted(data, weights, result, grid_num, obs_tot)
        result = np.asarray_chkfinite(result, dtype=float)

    # Unweighted data has two specific routines too. This is because creating
    # uniform weights takes relatively long time. It's faster to have a
    # specialize routine for this case.
    else:
        if data_dims >= 3:
            binary_flgs = cartesian(([0, 1],) * dims)
            result = cutils.iterate_data_ND(data, result, grid_num, obs_tot, binary_flgs)
        else:
            result = cutils.iterate_data_2D(data, result, grid_num, obs_tot)
        result = np.asarray_chkfinite(result, dtype=float)
        result = result / data_obs

    assert np.allclose(np.sum(result), 1)
    return result


def linear_binning(data, grid_points, weights=None):
    """
    This wrapper function computes d-dimensional binning, very quickly.

    Computes binning by setting a linear grid and weighting points linearily
    by their distance to the grid points. In addition, weight asssociated with
    data points may be passed. Depending on whether or not weights are passed
    and the dimensionality of the data, specific sub-routines are called for
    fast evaluation.

    Parameters
    ----------
    data
        The data points.
    grid_points
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
    data = np.asarray_chkfinite(data, dtype=float)
    grid_points = np.asarray_chkfinite(grid_points, dtype=float)
    if weights is not None:
        weights = np.asarray_chkfinite(weights, dtype=float)

    # Make sure the dimensionality makes sense
    try:
        data_obs, data_dims = data.shape
    except ValueError:
        data_dims = 1

    try:
        grid_obs, grid_dims = grid_points.shape
    except ValueError:
        grid_dims = 1

    if not data_dims == grid_dims:
        raise ValueError("Shape of data and grid points must be the same.")

    if data_dims == 1:
        if _use_Cython:
            return linbin_cython(data.ravel(), grid_points.ravel(), weights=weights)
        else:
            return linbin_numpy(data.ravel(), grid_points.ravel(), weights=weights)
    else:
        return linbin_Ndim(data, grid_points, weights=weights)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[".", "--doctest-modules", "-v", "--capture=sys"])
