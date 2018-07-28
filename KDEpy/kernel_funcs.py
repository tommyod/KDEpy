#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:52:43 2018

@author: tommy
"""

import numpy as np
import collections.abc
import numbers
from scipy.special import gamma, factorial, factorial2
from scipy.stats import norm
from scipy.optimize import brentq

# In R, the following are implemented:
# "gaussian", "rectangular", "triangular", "epanechnikov", 
# "biweight", "cosine" or "optcosine"

# Wikipedia
# uniform, trinagular, epanechnikov, quartic,triweight, tricube,
# gaussian, cosine, logistic, sigmoid, silverman

# All kernel functions take x of shape (obs, dims) and returns (obs, 1)
# All kernel functions integrate to unity


def gauss_integral(n):
    """
    Solve the integral
    \int_0^1 exp(-0.5 * x * x) x^n dx
    
    See
    https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions
    
    Examples
    --------
    >>> ans = gauss_integral(3)
    >>> np.allclose(ans, 2)
    True
    >>> ans = gauss_integral(4)
    >>> np.allclose(ans, 3.75994)
    True
    """
    factor = np.sqrt(np.pi * 2)
    if n % 2 == 0:
        return factor * factorial2(n - 1) / 2
    elif n % 2 == 1:
        return factor * norm.pdf(0) * factorial2(n - 1)
    else:
        raise ValueError('n must be odd or even.')


def trig_integral(k):
    """
    Returns the solutions to
    Is(k) = int_0^1 sin(pi * x / 2) x^k dx
    Ic(k) = int_0^1 cos(pi * x / 2) x^k dx
    using a recursive formula. Returns a tuple (Is, Ic).
    
    Examples
    --------
    >>> import numpy as np
    >>> Is, Ic = trig_integral(2) # Verify with solution from WolframAlpha
    >>> np.allclose([Is, Ic], [0.29454, 0.12060], rtol=10e-5)
    True
    """

    Ic = 2 / np.pi
    Is = 2 / np.pi
    
    if k <= 0:
        return Is, Ic

    for i in range(1, k + 1):
        Ic, Is = (2 / np.pi) * (1 - i * Is), (2 / np.pi) * i * Ic
        
    return Is, Ic


def p_norm(x, p):
    """
    The 2 norm of an array of shape (obs, dims)
    
    Examples
    --------
    >>> x = np.arange(9).reshape((3, 3))
    >>> p = 2
    >>> np.allclose(p_norm(x, p), euclidean_norm(x))
    True
    """
    if np.isinf(p):
        return infinity_norm(x)
    return np.power(np.power(np.abs(x), p).sum(axis=1), 1 / p).reshape(-1, 1)


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


def taxicab_norm(x):
    """
    The taxicab norm of an array of shape (obs, dims)
    """
    return np.abs(x).sum(axis=1).reshape(-1, 1)


def volume_hypershpere(d):
    """
    The volume of a d-dimensional hypersphere of radius 1.
    """
    return (np.pi**(d / 2.)) / gamma((d / 2.) + 1)


def volume_hypercube(d):
    """
    The volume of a d-dimensional hypercube of radius 1.
    """
    return np.power(2., d)


def volume_dual_hypercube(d):
    """
    The volume of a d-dimensional cross-polytope of radius 1.
    """
    return np.power(2., d) / factorial(d)


def epanechnikov(x, dims=1, volume_func=volume_hypershpere):
    normalization = volume_func(dims) * (2 / (dims + 2))
    dist_sq = x**2 
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = (1 - dist_sq)[mask] / normalization
    return out


def gaussian(x, dims=1, volume_func=volume_hypershpere):
    normalization = volume_func(dims) * dims * gauss_integral(dims - 1)
    dist_sq = x**2
    return np.exp(-dist_sq / 2) / normalization


def box(x, dims=1, volume_func=volume_hypershpere):
    normalization = volume_func(dims)
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = 1 / normalization
    return out


def exponential(x, dims=1, volume_func=volume_hypershpere):
    normalization = volume_func(dims) * gamma(dims) * dims
    return np.exp(-x) / normalization


def tri(x, dims=1, volume_func=volume_hypershpere):
    normalization = volume_func(dims) * (1 / (dims + 1))
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = np.maximum(0, 1 - x)[mask] / normalization 
    return out


def biweight(x, dims=1, volume_func=volume_hypershpere):
    normalization = (volume_func(dims) * 
                     (8 / ((dims + 2) * (dims + 4))))
    dist_sq = x**2
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = np.maximum(0, (1 - dist_sq)**2)[mask] / normalization
    return out


def triweight(x, dims=1, volume_func=volume_hypershpere):
    normalization = (volume_func(dims) * 
                     (48 / ((dims + 2) * (dims + 4) * (dims + 6))))
    dist_sq = x**2
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = np.maximum(0, (1 - dist_sq)**3)[mask] / normalization
    return out


def tricube(x, dims=1, volume_func=volume_hypershpere):
    normalization = (volume_func(dims) * 
                     (162 / ((dims + 3) * (dims + 6) * (dims + 9))))
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = np.maximum(0, (1 - x**3)**3)[mask] / normalization
    return out


def cosine(x, dims=1, volume_func=volume_hypershpere):
    Is, Ic = trig_integral(dims - 1)
    normalization = volume_func(dims) * Ic
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = np.cos((np.pi * x) / 2)[mask] / (normalization * dims)
    return out


def logistic(x, dims=1, volume_func=volume_hypershpere):
    dist = (x * x).sum(axis=1).reshape(-1, 1)
    dist = np.sqrt(dist)
    return 1 / (2 + 2 * np.cosh(dist))


def sigmoid(x, dims=1, volume_func=volume_hypershpere):
    dist = (x * x).sum(axis=1).reshape(-1, 1)
    dist = np.sqrt(dist)
    return (1 / (np.pi * np.cosh(dist)))


class Kernel(collections.abc.Callable):
    
    def __init__(self, function, var=1, support=3):
        """
        Initialize a new kernel function.
        
        function: callable, numpy.arr -> numpy.arr, should integrate to 1
        expected_value : peak, typically 0
        support: support of the function.
        """
        self.function = function
        self.var = var
        self.finite_support = np.isfinite(support)
        
        # If the function has finite support, scale the support so that it
        # corresponds to the support of the function when it is scaled to have
        # unit variance.
        self.support = support / np.sqrt(self.var)
        
    def practical_support(self, bw, atol=10e-5):
        """
        Return the support for practical purposes.
        """
        # If the kernel has finite support, return the support accounting for
        # the bw
        if self.finite_support:
            return self.support * bw
        
        # If the function does not have finite support, find a practical value
        else:
            def f(x):
                return self.evaluate(x, bw=bw) - atol
            return brentq(f, a=0, b=10 * bw, full_output=False)
    
    def evaluate(self, x, bw=1, norm=2):
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
        
        if norm == np.infty:
            volume_func = volume_hypercube
        elif norm == 1:
            volume_func = volume_dual_hypercube
        else:
            volume_func = volume_hypershpere
            
        distances = p_norm(x, norm)
            
        return (self.function(distances / real_bw, dims, volume_func) / 
                (real_bw**dims))
            
    __call__ = evaluate
    
    
gaussian = Kernel(gaussian, var=1, support=np.inf)
exp = Kernel(exponential, var=4, support=np.inf)
box = Kernel(box, var=1 / 3, support=1)
tri = Kernel(tri, var=1 / 6, support=1)
epa = Kernel(epanechnikov, var=1 / 5, support=1)
biweight = Kernel(biweight, var=1 / 7, support=1)
triweight = Kernel(triweight, var=1 / 9, support=1)
tricube = Kernel(tricube, var=35 / 243, support=1)
cosine = Kernel(cosine, var=(1 - (8 / np.pi**2)), support=1)
logistic = Kernel(logistic, var=(np.pi**2 / 3), support=np.inf)
sigmoid = Kernel(sigmoid, var=(np.pi**2 / 4), support=np.inf)

_kernel_functions = {'gaussian': gaussian,
                     'exponential': exp,
                     'box': box,
                     'tri': tri,
                     'epa': epa,
                     'biweight': biweight,
                     'triweight': triweight,
                     'tricube': tricube,
                     'cosine': cosine,
                     'logistic': logistic,
                     'sigmoid': sigmoid
                     }


if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])
    
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        a = Axes3D
        import scipy
        for name, func in _kernel_functions.items():
    
            print('-' * 2**7)
            print(name)
            print(func([-1, 0, 1]))
            print(func(np.array([[0, -1], [0, 0], [0, 1]])))
            print(func(np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])))
            
            # Plot in 1D
            n = 50
            x = np.linspace(-3, 3, num=n * 3)
            plt.plot(x, func(x))
            plt.show()
            
            # Plot in 2D
            n = 50
            linspace = np.linspace(-3, 3, num=n)
    
            x, y = linspace, linspace
            k = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
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
            
            ans, err = scipy.integrate.nquad(int1D, [[-4, 4]])
            print(f'1D integration result: {ans}')
            assert np.allclose(ans, 1, rtol=10e-3, atol=10e-3)
            
            # Perform integration 2D
            def int2D(x1, x2):
                return func([[x1, x2]])
            
            ans, err = scipy.integrate.nquad(int2D, [[-4, 4], [-4, 4]],
                                             opts={'epsabs': 10e-3, 
                                                   'epsrel': 10e-3})
            print(f'2D integration result: {ans}')
            assert np.allclose(ans, 1, rtol=10e-3, atol=10e-3)
        
if __name__ == "__main__":
    
    bw = 2
    print(gaussian.practical_support(bw))
    print(epa.practical_support(bw))
    import matplotlib.pyplot as plt
    x = np.linspace(-10, 10, num=2**8)
    y = gaussian(x, bw=bw)
    plt.plot(x, y)
    plt.scatter([-gaussian.practical_support(bw), 
                 gaussian.practical_support(bw)], [0, 0])
    
    y = epa(x, bw=bw)
    plt.plot(x, y)
    plt.scatter([-epa.practical_support(bw), 
                 epa.practical_support(bw)], [0, 0])
    
    