#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:52:43 2018

@author: tommy
"""

import numpy as np
import collections.abc
import numbers
from scipy.spatial import distance
from scipy.special import gamma

# In R, the following are implemented:
# "gaussian", "rectangular", "triangular", "epanechnikov", 
# "biweight", "cosine" or "optcosine"

# Wikipedia
# uniform, trinagular, epanechnikov, quartic,triweight, tricube,
# gaussian, cosine, logistic, sigmoid, silverman

# All kernel functions take x of shape (obs, dims) and returns (obs, 1)
# All kernel functions integrate to unity

# TODO: Make sure kernels integrate to unity in 2D too

def euclidean_norm(x):
    """
    The 2 norm of an array of shape (obs, dims)
    """
    return np.sqrt((x * x).sum(axis=1)).reshape(-1, 1)

def euclidean_norm_sq(x):
    """
    The squared 2 norm of an array of shape (obs, dims)
    """
    return (x * x).sum(axis=1).reshape(-1, 1)

def infinity_norm(x):
    """
    The infinity norm of an array of shape (obs, dims)
    """
    return np.abs(x).max(axis=1).reshape(-1, 1)


def volume_hypershpere(d):
    """
    The volume of a d-dimensional hypersphere of radius 1.
    """
    return (np.pi ** (d / 2.)) / gamma((d / 2.) + 1)


def volume_hypercube(d):
    """
    The volume of a d-dimensional hypercube of radius 1.
    """
    return np.power(2., d)


def epanechnikov(x):
    obs, dims = x.shape
    normalization = volume_hypershpere(dims) * (2 / (dims + 2))
    dist_sq = euclidean_norm_sq(x)
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] =  (1 - dist_sq)[mask] / normalization
    return out


def gaussian(x):
    obs, dims = x.shape
    normalization = (2 * np.pi)**(dims / 2)
    dist_sq = euclidean_norm_sq(x)
    return np.exp(-dist_sq / 2) / normalization


def box(x):
    obs, dims = x.shape
    normalization = volume_hypershpere(dims) #* (2 / (dims + 2))
    dist = euclidean_norm(x)
    out = np.zeros_like(dist)
    mask = dist < 1
    out[mask] = 1 / normalization #2**dims
    return out

def tri(x):
    obs, dims = x.shape
    normalization = volume_hypershpere(dims) * (1 / (dims + 1))
    dist = euclidean_norm(x)
    out = np.zeros_like(dist)
    mask = dist < 1
    out[mask] = np.maximum(0, 1 - dist)[mask] / normalization #2**dims
    return out


def biweight(x):
    obs, dims = x.shape
    normalization = volume_hypershpere(dims) * (8 / ((dims + 2) * (dims + 4)))
    dist_sq = euclidean_norm_sq(x)
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = np.maximum(0, (1 - dist_sq)**2)[mask] / normalization #2**dims
    return out


    dist = (x*x).sum(axis = 1).reshape(-1, 1)
    out = np.zeros_like(dist)
    mask = np.logical_and((dist < 1), (dist > -1))
    out[mask] = ((15 / 16) * (1 - dist)**2)[mask]
    return out


def triweight(x):
    dist = (x * x).sum(axis = 1).reshape(-1, 1)
    out = np.zeros_like(dist)
    mask = np.logical_and((dist < 1), (dist > -1))
    out[mask] = ((35 / 32) * (1 - dist)**3)[mask]
    return out


def tricube(x):
    dist = (x * x).sum(axis = 1).reshape(-1, 1)
    out = np.zeros_like(dist)
    mask = np.logical_and((dist < 1), (dist > -1))
    out[mask] = ((70 / 81) * (1 - dist**(3/2))**3)[mask]
    return out


def cosine(x):
    dist = (x * x).sum(axis = 1).reshape(-1, 1)
    out = np.zeros_like(dist)
    mask = np.logical_and((dist < 1), (dist > -1))
    out[mask] = ((np.pi / 4) * np.cos((np.pi * np.sqrt(dist)) / 2))[mask]
    return out


def logistic(x):
    dist = (x * x).sum(axis = 1).reshape(-1, 1)
    dist = np.sqrt(dist)
    return 1 / (2 + 2 * np.cosh(dist))


def sigmoid(x):
    dist = (x * x).sum(axis = 1).reshape(-1, 1)
    dist = np.sqrt(dist)
    return (1 / (np.pi * np.cosh(dist)))


class Kernel(collections.abc.Callable):
    
    def __init__(self, function, var=1, support=(-3, 3), norm=2):
        """
        Initialize a new kernel function.
        
        function: callable, numpy.arr -> numpy.arr, should integrate to 1
        expected_value : peak, typically 0
        left_bw: support to the left
        left_bw: support to the right
        """
        self.function = function
        self.var = var
        self.finite_support = np.all(np.isfinite(np.array(support)))
        self.norm = norm
        
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
            
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
            
        # Scale the function, such that bw=1 corresponds to the function having
        # a standard deviation (or variance) equal to 1
        real_bw = bw / np.sqrt(self.var)
        obs, dims = x.shape
        return self.function(x / real_bw) / (real_bw**dims)
    
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
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import scipy
    for name, func in _kernel_functions.items():

        print('-'*2**7)
        print(name)
        print(func([-1, 0, 1]))
        print(func(np.array([[0, -1], [0, 0], [0, 1]])))
        print(func(np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])))
        
        # Plot in 1D
        n = 50
        x = np.linspace(-3, 3, num=n*3)
        plt.plot(x, func(x))
        plt.show()
        
        # Plot in 2D
        n = 50
        linspace = np.linspace(-3, 3, num=n)

        x, y = linspace, linspace
        k = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
        z = func(k).reshape((n, n))
        
        x, y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                               linewidth=1, antialiased=True, shade=True)
        
        angle = 90
        ax.view_init(30, angle)

        
        plt.show()
        
        # Perform integration 1D
        def int1D(x1):
            return func(x1)
        
        ans, err = scipy.integrate.nquad(int1D, [[-5, 5]])
        print(f'1D integration result: {ans}')
        assert np.allclose(ans, 1, rtol=10e-2, atol=10e-2)
        
        # Perform integration 2D
        def int2D(x1, x2):
            return func([[x1, x2]])
        
        ans, err = scipy.integrate.nquad(int2D, [[-5, 5], [-5, 5]],
                                         opts={'epsabs':10e-2, 'epsrel':10e-2})
        print(f'2D integration result: {ans}')
        assert np.allclose(ans, 1, rtol=10e-2, atol=10e-2)
        
        # Perform integration 3D
        def int3D(x1, x2, x3):
            return func([[x1, x2, x3]])
        
        ans, err = scipy.integrate.nquad(int3D, [[-5, 5], [-5, 5], [-5, 5]],
                                         opts={'epsabs':10e-1, 'epsrel':10e-1})
        print(f'3D integration result: {ans}')
        #assert np.allclose(ans, 1)
        

    
    
    
    