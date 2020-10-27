#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for kernel functions. A kernel function is a radial basis function which
is everywhere non-negative and whose integral evalutes to unity. Every kernel
function takes an `x` of shape (obs, dims) and returns a y of shape (obs, 1).
"""

import numpy as np
import collections.abc
import numbers
import functools
from scipy.special import gamma, factorial2
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
    r"""
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
        raise ValueError("n must be odd or even.")


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
    The p-norm of an array of shape (obs, dims)

    Examples
    --------
    >>> x = np.arange(9).reshape((3, 3))
    >>> p = 2
    >>> np.allclose(p_norm(x, p), euclidean_norm(x))
    True
    """
    if np.isinf(p):
        return infinity_norm(x)
    elif p == 2:
        return euclidean_norm(x)
    elif p == 1:
        return taxicab_norm(x)
    return np.power(np.power(np.abs(x), p).sum(axis=1), 1 / p)


def euclidean_norm(x):
    """
    The 2 norm of an array of shape (obs, dims)
    """
    return np.sqrt((x * x).sum(axis=1))


def euclidean_norm_sq(x):
    """
    The squared 2 norm of an array of shape (obs, dims)
    """
    return (x * x).sum(axis=1)


def infinity_norm(x):
    """
    The infinity norm of an array of shape (obs, dims)
    """
    return np.abs(x).max(axis=1)


def taxicab_norm(x):
    """
    The taxicab norm of an array of shape (obs, dims)
    """
    return np.abs(x).sum(axis=1)


def volume_unit_ball(d, p=2):
    """
    Volume of d-dimensional unit ball under the p-norm. When p=1 this is called
    a cross-polytype, when p=2 it's called a hypersphere, and when p=infty it's
    called a hypercube.

    Notes
    -----
    See the following paper for a very general result related to this:

    - Wang, Xianfu. “Volumes of Generalized Unit Balls.”
      Mathematics Magazine 78, no. 5 (2005): 390–95.
      https://doi.org/10.2307/30044198.
    """
    return 2.0 ** d * gamma(1 + 1 / p) ** d / gamma(1 + d / p)


def epanechnikov(x, dims=1):
    normalization = 2 / (dims + 2)
    dist_sq = x ** 2
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = (1 - dist_sq)[mask] / normalization
    return out


def gaussian(x, dims=1):
    normalization = dims * gauss_integral(dims - 1)
    dist_sq = x ** 2
    return np.exp(-dist_sq / 2) / normalization


def box(x, dims=1):
    normalization = 1
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = 1 / normalization
    return out


def exponential(x, dims=1):
    normalization = gamma(dims) * dims
    return np.exp(-x) / normalization


def tri(x, dims=1):
    normalization = 1 / (dims + 1)
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = np.maximum(0, 1 - x)[mask] / normalization
    return out


def biweight(x, dims=1):
    normalization = 8 / ((dims + 2) * (dims + 4))
    dist_sq = x ** 2
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = np.maximum(0, (1 - dist_sq) ** 2)[mask] / normalization
    return out


def triweight(x, dims=1):
    normalization = 48 / ((dims + 2) * (dims + 4) * (dims + 6))
    dist_sq = x ** 2
    out = np.zeros_like(dist_sq)
    mask = dist_sq < 1
    out[mask] = np.maximum(0, (1 - dist_sq) ** 3)[mask] / normalization
    return out


def tricube(x, dims=1):
    normalization = 162 / ((dims + 3) * (dims + 6) * (dims + 9))
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = np.maximum(0, (1 - x ** 3) ** 3)[mask] / normalization
    return out


def cosine(x, dims=1):
    Is, Ic = trig_integral(dims - 1)
    normalization = Ic
    out = np.zeros_like(x)
    mask = x < 1
    out[mask] = np.cos((np.pi * x) / 2)[mask] / (normalization * dims)
    return out


def logistic(x, dims=1):
    return 1 / (2 + 2 * np.cosh(x))


def sigmoid(x, dims=1):
    return 1 / (np.pi * np.cosh(x))


class Kernel(collections.abc.Callable):
    def __init__(self, function, var=1, support=3):
        """
        Initialize a new kernel function.

        function: callable, numpy.arr -> numpy.arr, should integrate to 1
        expected_value : peak, typically 0
        support: support of the function.

        Example
        -------
        >>> from scipy.special import gamma
        >>> # Normalized function of x
        >>> def exp(x, dims=1):
        ...     normalization = gamma(dims) * dims
        ...     return np.exp(-x) / normalization
        >>> kernel = Kernel(exp, var=4, support=np.inf)
        >>> # The function is scaled so that the standard deviation (bw) = 1
        >>> kernel(0, bw=1, norm=2)[0] > kernel(1, bw=1, norm=2)[0]
        True
        >>> np.allclose(kernel(np.array([0, 1, 2])), kernel([0, 1, 2]))
        True
        >>> np.allclose(kernel(0), kernel([0]))
        True
        >>> np.allclose(kernel(0), kernel.evaluate(0))
        True
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
        Return the support for practical purposes. Used to find a support value
        for computations for kernel functions without finite (bounded) support.
        """
        # If the kernel has finite support, return the support accounting for
        # the bw
        if self.finite_support:
            return self.support * bw

        # If the function does not have finite support, find a practical value
        else:

            def f(x):
                return self.evaluate(x, bw=bw) - atol

            try:
                xtol = 1e-3
                ans = brentq(f, a=0, b=8 * bw, xtol=xtol, full_output=False)
                return ans + xtol
            except ValueError:
                msg = (
                    "Unable to solve for support numerically. Use a "
                    + "kernel with finite support or scale data to smaller bw."
                )
                raise ValueError(msg)

    def evaluate(self, x, bw=1, norm=2):
        """
        Evaluate the kernel.

        Parameters
        ----------
        x : array-like
            Should have shape (obs, dims).
        bw : array-like
            Must have shape (obs, ), or float.
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

        # Set the volume function
        volume_func = functools.partial(volume_unit_ball, p=norm)

        if dims > 1:
            distances = p_norm(x, norm).ravel()
        else:
            distances = np.abs(x).ravel()

        return self.function(distances / real_bw, dims) / ((real_bw ** dims) * volume_func(dims))

    __call__ = evaluate


gaussian = Kernel(gaussian, var=1, support=np.inf)
exp = Kernel(exponential, var=2, support=np.inf)
box = Kernel(box, var=1 / 3, support=1)
tri = Kernel(tri, var=1 / 6, support=1)
epa = Kernel(epanechnikov, var=1 / 5, support=1)
biweight = Kernel(biweight, var=1 / 7, support=1)
triweight = Kernel(triweight, var=1 / 9, support=1)
tricube = Kernel(tricube, var=35 / 243, support=1)
cosine = Kernel(cosine, var=(1 - (8 / np.pi ** 2)), support=1)
logistic = Kernel(logistic, var=(np.pi ** 2 / 3), support=np.inf)
sigmoid = Kernel(sigmoid, var=(np.pi ** 2 / 4), support=np.inf)

_kernel_functions = {
    "gaussian": gaussian,
    "exponential": exp,
    "box": box,
    "tri": tri,
    "epa": epa,
    "biweight": biweight,
    "triweight": triweight,
    "tricube": tricube,
    "cosine": cosine,
    # 'logistic': logistic,
    # 'sigmoid': sigmoid
}


if __name__ == "__main__" and False:
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[".", "--doctest-modules", "-v"])
