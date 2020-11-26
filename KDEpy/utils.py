#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for utility functions.
"""
import numpy as np
import numbers


def cartesian(arrays):
    """
    Generate a cartesian product of input arrays.
    Adapted from:
        https://github.com/scikit-learn/scikit-learn/blob/
        master/sklearn/utils/extmath.py#L489

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def autogrid(data, boundary_abs=3, num_points=None, boundary_rel=0.05):
    """
    Automatically select a grid if the user did not supply one.
    Input is (obs, dims), and so is ouput.

    number of grid : should be a power of two
    percentile : is how far out we go out

    Parameters
    ----------
    data : array-like
        Data with shape (obs, dims).
    boundary_abs: float
        How far out from boundary observations the grid goes in each dimension.
    num_points: int
        The number of points in the resulting grid (after cartesian product).
        Should be a number such that k**dims = `num_points`.
    boundary_rel: float
        How far out to go, relatively to max - min.

    Returns
    -------
    grid : array-like
        A grid of shape (obs, dims).

    Examples
    --------
    >>> autogrid(np.array([[0, 0]]), boundary_abs=1, num_points=3)
    array([[-1., -1.],
           [-1.,  0.],
           [-1.,  1.],
           [ 0., -1.],
           [ 0.,  0.],
           [ 0.,  1.],
           [ 1., -1.],
           [ 1.,  0.],
           [ 1.,  1.]])
    >>> autogrid(np.array([[0, 0]]), boundary_abs=0.5, num_points=(2, 3))
    array([[-0.5, -0.5],
           [-0.5,  0. ],
           [-0.5,  0.5],
           [ 0.5, -0.5],
           [ 0.5,  0. ],
           [ 0.5,  0.5]])
    """
    obs, dims = data.shape
    minimums, maximums = data.min(axis=0), data.max(axis=0)
    ranges = maximums - minimums

    if num_points is None:
        num_points = [int(np.power(1024, 1 / dims))] * dims
    elif isinstance(num_points, (numbers.Number,)):
        num_points = [num_points] * dims
    elif isinstance(num_points, (list, tuple)):
        pass
    else:
        msg = "`num_points` must be None, a number, or list/tuple for dims"
        raise TypeError(msg)

    if not len(num_points) == dims:
        raise ValueError("Number of points must be sequence matching dims.")

    list_of_grids = []

    generator = enumerate(zip(minimums, maximums, ranges, num_points))
    for i, (minimum, maximum, rang, points) in generator:
        assert points >= 2
        outside_borders = max(boundary_rel * rang, boundary_abs)
        list_of_grids.append(np.linspace(minimum - outside_borders, maximum + outside_borders, num=points))

    return cartesian(list_of_grids)


def weighted_std(values, weights, ddof=0):
    """Return the weighted standard deviation.

    Examples
    --------
    >>> weighted_std([1, 2, 2], weights=[1, 1, 1])
    0.4714045207910317
    >>> weighted_std([1, 2], weights=[1, 2])
    0.4714045207910317
    >>> weighted_std([1, 2, 2], weights=[1, 1, 1], ddof=1)
    0.5773502691896257
    >>> weighted_std([1, 2], weights=[1, 2], ddof=1)
    0.5773502691896257

    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    assert np.all(weights > 0), "All weights must be >= 0"
    assert isinstance(ddof, numbers.Integral), "ddof must be int"

    # If the degrees of freedom is greater than zero, we need to scale results
    if ddof > 0:
        smallest_weight = np.min(weights)
        weights_summed = np.sum(weights)
        factor = weights_summed / (weights_summed - ddof * smallest_weight)
    else:
        factor = 1

    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(factor * variance)


def weighted_percentile(values, perc, weights=None):
    """Compue the weighted percentile.

    Based on: https://stackoverflow.com/a/61343915

    Examples
    --------
    >>> weighted_percentile([1, 2, 2], 0)
    1.0
    >>> weighted_percentile([1, 2, 2], 1.)
    2.0
    >>> # These computations differ, but it does not matter
    >>> weighted_percentile([1, 2], 0.5, [1, 2])
    1.66666...
    >>> weighted_percentile([1, 2, 2], 0.5)
    2.0

    """
    values = np.asarray(values)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights)
        assert np.all(weights > 0), "All weights must be >= 0"

    ix = np.argsort(values)
    values = values[ix]  # sort data
    weights = weights[ix]  # sort weights
    weights_cumsum = np.cumsum(weights)
    cdf = (weights_cumsum - 0.5 * weights) / weights_cumsum[-1]
    return np.interp(perc, cdf, values)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
