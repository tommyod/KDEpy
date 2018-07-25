#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""

if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    # pytest.main(args=['.', '--doctest-modules', '-v'])
    
    import numpy as np
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