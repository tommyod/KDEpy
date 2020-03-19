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


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[".", "--doctest-modules", "-v", "--capture=sys"])
