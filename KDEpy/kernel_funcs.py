#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:52:43 2018

@author: tommy
"""

import numpy as np
import collections.abc
import numbers

# In R, the following are implemented:
# "gaussian", "rectangular", "triangular", "epanechnikov", 
# "biweight", "cosine" or "optcosine"

# Wikipedia
# uniform, trinagular, epanechnikov, quartic,triweight, tricube,
# gaussian, cosine, logistic, sigmoid, silverman


def epanechnikov(x):
    out = np.zeros_like(x)
    mask = np.logical_and((x < 1), (x > -1))
    out[mask] = 0.75 * (1 - x * x)[mask]
    return out


def gaussian(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


def box(x):
    out = np.zeros_like(x)
    mask = np.logical_and((x < 1), (x > -1))
    out[mask] = 0.5
    return out


def tri(x):
    out = np.zeros_like(x)
    out[x >= 0] = np.maximum(0, 1 - x)[x >= 0]
    out[x < 0] = np.maximum(0, 1 + x)[x < 0]
    return out


def biweight(x):
    out = np.zeros_like(x)
    mask = np.logical_and((x < 1), (x > -1))
    out[mask] = ((15 / 16) * (1 - x**2)**2)[mask]
    return out


def triweight(x):
    out = np.zeros_like(x)
    mask = np.logical_and((x < 1), (x > -1))
    out[mask] = ((35 / 32) * (1 - x**2)**3)[mask]
    return out


def tricube(x):
    out = np.zeros_like(x)
    mask = np.logical_and((x < 1), (x > -1))
    out[mask] = ((70 / 81) * (1 - np.abs(x)**3)**3)[mask]
    return out


def cosine(x):
    out = np.zeros_like(x)
    mask = np.logical_and((x < 1), (x > -1))
    out[mask] = ((np.pi / 4) * np.cos((np.pi * x) / 2))[mask]
    return out


def logistic(x):
    return 1 / (2 + 2 * np.cosh(x))


def sigmoid(x):
    return (1 / (np.pi * np.cosh(x)))


class Kernel(collections.abc.Callable):
    
    def __init__(self, function, var=1, support=(-3, 3)):
        """
        Initialize a new kernel function.
        
        function: callable, numpy.arr -> numpy.arr
        expected_value : peak, typically 0
        left_bw: support to the left
        left_bw: support to the right
        """
        self.function = function
        self.var = var
        self.finite_support = np.all(np.isfinite(np.array(support)))
        
        # If the function has finite support, scale the support so that it
        # corresponds to the support of the function when it is scaled to have
        # unit variance.
        self.support = tuple(supp / np.sqrt(self.var) for supp in support)
            
        assert self.support[0] < self.support[1]
    
    def evaluate(self, x, bw=1):
        """
        Evaluate the kernel.
        """
        
        # If x is a number, convert it to a length-1 NumPy vector
        if isinstance(x, numbers.Number):
            x = np.asarray_chkfinite([x])
        else:
            x = np.asarray_chkfinite(x)
            
        # Scale the function, such that bw=1 corresponds to the function having
        # a standard deviation (or variance) equal to 1
        real_bw = bw / np.sqrt(self.var)
        return self.function(x / real_bw) / real_bw
    
    __call__ = evaluate
    
    
gaussian = Kernel(gaussian, var=1, support=(-np.inf, np.inf))
box = Kernel(box, var=1 / 3, support=(-1, 1))
tri = Kernel(tri, var=1 / 6, support=(-1, 1))
epa = Kernel(epanechnikov, var=1 / 5, support=(-1, 1))
biweight = Kernel(biweight, var=1 / 7, support=(-1, 1))
triweight = Kernel(triweight, var=1 / 9, support=(-1, 1))
tricube = Kernel(tricube, var=35 / 243, support=(-1, 1))
cosine = Kernel(cosine, var=(1 - (8 / np.pi**2)), support=(-1, 1))
logistic = Kernel(logistic, var=(np.pi**2 / 3), support=(-np.inf, np.inf))
sigmoid = Kernel(sigmoid, var=(np.pi**2 / 4), support=(-np.inf, np.inf))

_kernel_functions = {'gaussian': gaussian,
                     'box': box,
                     'tri': tri,
                     'epa': epa,
                     'biweight': biweight,
                     'triweight': triweight,
                     'tricube': tricube,
                     'cosine': cosine,
                     'logistic': logistic,
                     'sigmoid': sigmoid}


if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])
    
    print(box.support)