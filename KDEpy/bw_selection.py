#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for bandwidth selection.
"""
import numpy as np
from KDEpy.binning import linbin_numpy
from KDEpy.utils import autogrid
from scipy import fftpack
from scipy.optimize import brentq

# This notice is included since some of the functions are based on the
# MATLAB code by Zdravko Botev
_botev_notice = """Copyright (c) 2015, Zdravko Botev
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution
* Neither the name of The University of New South Wales nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""


def _fixed_point(t, N, I_sq, a2):
    r"""
    Compute the fixed point as described in the paper by Botev et al.
    
    .. math:
        
        t = \xi \gamma^{5}(t)
    
    Parameters
    ----------
    t : float
        Initial guess.
    N : int
        Number of data points.
    I_sq : array-like
        The numbers [1, 2, 9, 16, ...]
    a2 : array-like
        The DCT of the original data, divided by 2 and squared.
    
    Examples
    --------
    >>> # From the matlab code
    >>> ans = _fixed_point(0.01,50,np.arange(1, 51),np.arange(1, 51))
    >>> assert np.allclose(ans, 0.009947962622371)
    >>> # another
    >>> ans = _fixed_point(0.07,25,np.arange(1, 11),np.arange(1, 11))
    >>> assert np.allclose(ans, 0.069100181315957)
    """
    
    # This is important, as the powers might overflow if not done
    I_sq = np.asfarray(I_sq)
    a2 = np.asfarray(a2)
    
    # ell = 7 corresponds to the 5 steps recommended in the paper
    ell = 7
    
    # Fast evaluation of |f^l|^2 using the DCT, see Plancherel theorem
    f = 2 * np.pi**(2 * ell) * np.sum(np.power(I_sq, ell) * a2 * 
                                      np.exp(-I_sq * np.pi**2 * t))

    # Norm of a function, should never be negative
    if f <= 0:
        return -1
    for s in reversed(range(2, ell)):
        # This could also be formulated using the double factorial n!!,
        # but this is faster so and requires an import less
        
        # Step one: estimate t_s from |f^(s+1)|^2
        odd_numbers_prod = np.product(np.arange(1, 2 * s + 1, 2, 
                                                dtype=np.float))
        K0 = odd_numbers_prod / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = np.power((2 * const * K0 / (N * f)), 
                        (2. / (3. + 2. * s)))
        
        # Step two: estimate |f^s| from t_s
        f = 2 * np.pi**(2 * s) * np.sum(np.power(I_sq, s) * a2 * 
                                        np.exp(-I_sq * np.pi**2 * time))
        
    # This is the minimizer of the AMISE
    t_opt = np.power(2 * N * np.sqrt(np.pi) * f, -2. / 5)
    
    # Return the difference between the original t and the optimal value
    return t - t_opt


def _root(function, N, args):
    """
    Root finding algorithm. Based on MATLAB implementation by Botev et al.
    
    >>> # From the matlab code
    >>> ints = np.arange(1, 51)
    >>> ans = _root(_fixed_point, N=50, args=(50, ints, ints))
    >>> assert np.allclose(ans, 5.203713947289470e-05)
    """
    # From the implementation by Botev, the original paper author
    # Rule of thumb of obtaining a feasible solution
    N = max(min(1050, N), 50)
    tol = 10e-12 + 0.01 * (N - 50) / 1000
    # While a solution is not found, increase the tolerance and try again
    found = 0
    while found == 0:
        try:
            # Other viable solvers include: [brentq, brenth, ridder, bisect]
            x, res = brentq(function, 0, tol, args=args, 
                            full_output=True, disp=False)
            found = 1 if res.converged else 0
        except ValueError:
            x = 0
            tol *= 2.
            found = 0
        if x <= 0:
            found = 0

        # If the tolerance grows too large, minimize the function
        if tol >= 1:
            raise ValueError('Root finding did not converge. Need more data.')
            
    if not x > 0:
        raise ValueError('Root finding failed to find positive solution.')
    return x


def improved_sheather_jones(data):
    """
    The Improved Sheater Jones (ISJ) algorithm from the paper by Botev et al.
    This algorithm computes the optimal bandwidth for a gaussian kernel,
    and works very well for bimodal data (unlike other rules). The
    disadvantage of this algorithm is longer computation time, and the fact
    that this implementation does not always converge if very few data
    points are supplied.

    Understanding this algorithm is difficult, see:
    https://books.google.no/books?id=Trj9HQ7G8TUC&pg=PA328&lpg=PA328&dq=
    sheather+jones+why+use+dct&source=bl&ots=1ETdKd_6EF&sig=jZk4R515GB1xsn-
    VZVnjr-JfjSI&hl=en&sa=X&ved=2ahUKEwi1_czNncTcAhVGhqYKHaPiBtcQ6AEwA3oEC
    AcQAQ#v=onepage&q=sheather%20jones%20why%20use%20dct&f=false
    """
    obs, dims = data.shape
    assert dims == 1
    
    n = 2**10
    # Setting `percentile` higher decreases the chance of overflow
    xmesh = autogrid(data, boundary_abs=6, num_points=n, boundary_rel=0.5)
    data = data.ravel()
    xmesh = xmesh.ravel()
    
    # Create an equidistant grid
    R = np.max(data) - np.min(data)
    # dx = R / (n - 1)
    data = data.ravel()
    N = len(np.unique(data))

    # Use linear binning to bin the data on an equidistant grid, this is a
    # prerequisite for using the FFT (evenly spaced samples)
    initial_data = linbin_numpy(data, xmesh)
    assert np.allclose(initial_data.sum(), 1)
    
    # Compute the type 2 Discrete Cosine Transform (DCT) of the data
    a = fftpack.dct(initial_data)
    
    # Compute the bandwidth
    I_sq = np.power(np.arange(1, n, dtype=np.float), 2)
    a2 = a[1:]**2 / 4

    # Solve for the optimal (in the AMISE sense) t
    t_star = _root(_fixed_point, N, args=(N, I_sq, a2))

    # The remainder of the algorithm computes the actual density
    # estimate, but this function is only used to compute the 
    # bandwidth, since the bandwidth may be used for other kernels
    # apart from the Gaussian kernel

    # Smooth the initial data using the computed optimal t  
    # Multiplication in frequency domain is convolution   
    # integers = np.arange(n, dtype=np.float)
    # a_t = a * np.exp(-integers**2 * np.pi ** 2 * t_star / 2)

    # Diving by 2 done because of the implementation of fftpack.idct
    # density = fftpack.idct(a_t) / (2 * R)
    
    # Due to overflow, some values might be smaller than zero, correct it
    # density[density < 0] = 0.
    bandwidth = np.sqrt(t_star) * R
    return bandwidth


def scotts_rule(data):
    """
    Scotts rule.
    
    Scott (1992, page 152)
    Scott, D.W. (1992) Multivariate Density Estimation. Theory, Practice and 
    Visualization. New York: Wiley. 
    
    Examples
    --------
    >>> data = np.arange(9).reshape(-1, 1)
    >>> scotts_rule(data)
    1.76474568962182
    """
    if not len(data.shape) == 2:
        raise ValueError('Data must be of shape (obs, dims).')
    
    obs, dims = data.shape
    assert dims == 1
    sigma = np.std(data, ddof=1)
    # scipy.norm.ppf(.75) - scipy.norm.ppf(.25) -> 1.3489795003921634
    IQR = ((np.percentile(data, q=75) - np.percentile(data, q=25)) / 
           1.3489795003921634)

    sigma = min(sigma, IQR)
    return sigma * np.power(obs, -1. / (dims + 4))


def silvermans_rule(data):
    """
    Returns optimal smoothing (standard deviation) if the data is close to 
    normal.
    
    TODO: Extend to multidimensional:
        https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.
        stats.gaussian_kde.html#r216
        
    Examples
    --------
    >>> data = np.arange(9).reshape(-1, 1)
    >>> silvermans_rule(data)
    1.8692607078355594
    """
    if not len(data.shape) == 2:
        raise ValueError('Data must be of shape (obs, dims).')
    obs, dims = data.shape
    assert dims == 1

    if obs == 1:
        return 1
    if obs < 1:
        raise ValueError('Data must be of length > 0.')
        
    sigma = np.std(data, ddof=1)
    # scipy.norm.ppf(.75) - scipy.norm.ppf(.25) -> 1.3489795003921634
    IQR = ((np.percentile(data, q=75) - np.percentile(data, q=25)) / 
           1.3489795003921634)

    sigma = min(sigma, IQR)
    return sigma * (obs * 3 / 4.) ** (-1 / 5)

    
_bw_methods = {'silverman': silvermans_rule,
               'scott': scotts_rule,
               'ISJ': improved_sheather_jones}
