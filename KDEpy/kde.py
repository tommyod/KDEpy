#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import numpy as np

if False:
    # --durations=10  <- May be used to show potentially slow tests
    # pytest.main(args=['.', '--doctest-modules', '-v'])

    from KDEpy import NaiveKDE, TreeKDE
    from KDEpy.binning import linbin_numpy
    import matplotlib.pyplot as plt
    from time import perf_counter

    np.random.seed(123)
    data = np.random.lognormal(3, 0.2, 2**12)

    st = perf_counter()
    x, y = NaiveKDE('epa').fit(data).evaluate()
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(x, y, label='NaiveKDE')

    st = perf_counter()
    x, y = TreeKDE('epa').fit(data).evaluate()
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(x, y, label='TreeKDE')

    st = perf_counter()
    grid = np.linspace(np.min(data) - 5, np.max(data) + 5, 2**6)
    weights = linbin_numpy(data, grid)
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(grid, weights, '-o', label='linbin_numpy')

    st = perf_counter()
    x, y = TreeKDE('epa').fit(grid, weights=weights).evaluate()
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(x, y, label='TreeKDE on binned')

    plt.scatter(data, np.zeros_like(data))
    plt.legend(loc='best')
    
    
if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v', '--capture=sys'])
    
    
def main():
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create 2D data of shape (obs, dims)
    np.random.seed(123)
    n = 200
    data = np.concatenate((np.random.randn(n).reshape(-1, 1) * 5, 
                           np.random.randn(n).reshape(-1, 1) * 5), axis=1)

    from KDEpy.TreeKDE import TreeKDE
    
    grid_points = 2**6  # Grid points in each dimension
    N = 8  # Number of contours

    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    
    for ax, norm in zip(axes, [1, 2, np.inf]):
        
        ax.set_title(f'Norm $p={norm}$')
        
        # Compute the kernel density estimate
        st = time.perf_counter()
        kde = TreeKDE(kernel='gaussian', norm=norm, bw=5)
        grid, points = kde.fit(data).evaluate(grid_points)
        print(time.perf_counter() - st)
    
        # The grid is of shape (obs, dims), points are of shape (obs, 1)
        x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        z = points.reshape(grid_points, grid_points).T
        
        # Plot the kernel density estimate
        ax.contour(x, y, z, N, linewidths=0.8, colors='k')
        ax.contourf(x, y, z, N, cmap="RdBu_r")

    plt.tight_layout()
    plt.show()
    
    print()
    
    
if __name__ == "__main__":
    main()
