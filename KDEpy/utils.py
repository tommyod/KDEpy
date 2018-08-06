#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import numpy as np


def cartesian(arrays):
    """
    Generate a cartesian product of input arrays.
    Adapted from from: 
        https://github.com/scikit-learn/scikit-learn/blob/
        master/sklearn/utils/extmath.py#L489
        
    Parameters
    ----------
    arrays : ndarry
        An array of shape (obs, dims).
    out : ndarray
        Array to place the cartesian product in.
    
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    
    Examples
    --------
    >>> data = np.array([[0, 0, 0], [1, 1, 1]])
    >>> cartesian(data)
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [1, 1, 1]])
    """
    obs, dims = arrays.shape
    arrays = list(arrays[:, i] for i in range(dims))
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
        
    Examples
    --------
    >>> 2 + 1
    3
    """
    obs, dims = data.shape
    minimums, maximums = data.min(axis=0), data.max(axis=0)
    ranges = maximums - minimums
    
    if num_points is None:
        num_points = int(np.power(1024, 1 / dims))
        
    grid_points = np.empty(shape=(num_points, dims))

    generator = enumerate(zip(minimums, maximums, ranges))
    for i, (minimum, maximum, rang) in generator:
        outside_borders = max(boundary_rel * rang, boundary_abs)
        grid_points[:, i] = np.linspace(minimum - outside_borders,
                                        maximum + outside_borders,
                                        num=num_points)  

    return grid_points
    

if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v', '--capture=sys'])