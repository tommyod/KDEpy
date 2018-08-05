#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests.
"""
import numpy as np
from KDEpy.FFTKDE import FFTKDE
from KDEpy.NaiveKDE import NaiveKDE
from KDEpy.TreeKDE import TreeKDE
import itertools
import pytest

kernels = list(NaiveKDE._available_kernels.keys())
kdes = [NaiveKDE, TreeKDE, FFTKDE]
kde_pairs = list(itertools.combinations(kdes, 2))


@pytest.mark.parametrize("kde1, kde2, bw, kernel", 
                         [(k[0], k[1], bw, ker) for (k, bw, ker) in 
                          itertools.product(kde_pairs,
                                            [0.1, 'silverman', 1],
                                            kernels)])
def test_api_models_kernels_bandwidths(kde1, kde2, bw, kernel):
    """
    Test the API.
    """
    
    # TODO: Put weights into BaseKDE? If it is applicable in every sub-class.
    data = np.array([-1, 0, 0.1, 3, 10])
    weights = [1, 2, 1, 0.8, 2]
    
    # Chained expression
    x1, y1 = kde1(kernel=kernel, bw=bw).fit(data, weights).evaluate()
    
    # Step by step, with previous grid
    model = kde2(kernel=kernel, bw=bw)
    model.fit(data, weights)
    y2 = model.evaluate(x1)
    
    # Mean error
    err = np.sqrt(np.mean((y1 - y2) ** 2))
    if kernel == 'box':
        assert err < 0.025
    else:
        assert err < 0.002
        

type_functions = [tuple, 
                  np.array, 
                  np.asfarray, 
                  lambda x:np.asfarray(x).reshape(-1, 1)]        


@pytest.mark.parametrize("kde, bw, kernel, type_func", 
                         itertools.product(kdes,
                                           ['silverman', 'scott', 'ISJ', 0.5],
                                           ['epa', 'gaussian'],
                                           type_functions))   
def test_api_types(kde, bw, kernel, type_func):
    """
    Test the api.
    """
    # Test various input types
    data = [1, 2, 3]
    weights = [4, 5, 6]
    data = np.random.randn(64)
    weights = np.random.randn(64) + 10
    model = kde(kernel=kernel, bw=bw)
    x, y = model.fit(data, weights).evaluate()
    
    data = type_func(data)
    weights = type_func(weights)
    y1 = model.fit(data, weights).evaluate(x)
    assert np.allclose(y, y1)
    
    
def test_api_2D_data():
    """
    Test working on 2D data.
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create 2D data of shape (obs, dims)
    np.random.seed(123)
    n = 16
    data = np.concatenate((np.random.randn(n).reshape(-1, 1), 
                           np.random.randn(n).reshape(-1, 1)), axis=1)

    from KDEpy.NaiveKDE import NaiveKDE
    
    grid_points = 2**5  # Grid points in each dimension
    N = 16  # Number of contours

    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    
    for ax, norm in zip(axes, [1, 2, np.inf]):
        
        ax.set_title(f'Norm $p={norm}$')
        
        # Compute
        kde = NaiveKDE(kernel='gaussian', norm=norm)
        grid, points = kde.fit(data).evaluate(grid_points)
    
        # The grid is of shape (obs, dims), points are of shape (obs, 1)
        x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        z = points.reshape(grid_points, grid_points).T
        
        # Plot the kernel density estimate
        ax.contour(x, y, z, N, linewidths=0.8, colors='k')
        ax.contourf(x, y, z, N, cmap="RdBu_r")
        ax.plot(data[:, 0], data[:, 1], 'ok', ms=3)

    plt.tight_layout()
    plt.close()


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v', '--capture=sys',
                      '-k test_api_types'
                      ])