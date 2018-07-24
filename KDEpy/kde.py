#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:52:17 2018

@author: tommy
"""
import pytest

if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    # pytest.main(args=['.', '--doctest-modules', '-v'])
    
    import numpy as np
    from KDEpy import NaiveKDE, TreeKDE
    from KDEpy.binning import binning
    import matplotlib.pyplot as plt
    from time import perf_counter
    
    np.random.seed(123)
    data = np.random.lognormal(3, 0.2, 2**10)
    
    st = perf_counter()
    x, y = TreeKDE('epa').fit(data).evaluate()
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(x, y)
    
    st = perf_counter()
    points, weights = binning(data, int(np.sqrt(len(data))))
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(points, weights, '-o')
    
    st = perf_counter()
    x, y = TreeKDE('epa').fit(points, weights=weights).evaluate()
    print(f'Ran in {round(perf_counter() - st, 4)}')
    plt.plot(x, y)
    
    
    plt.scatter(data, np.zeros_like(data))
    
    
    
