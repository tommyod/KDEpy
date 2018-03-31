#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests.
"""
import numpy as np
from scipy.integrate import quad
from KDEpy import KDE
import pytest


class TestKernelFunctions():
    
    @pytest.mark.parametrize("fname, function", 
                             list(KDE._available_kernels.items()))
    def test_integral_unity(self, fname, function):
        """
        Verify that all available kernel functions have an integral evaluating
        to unity. This is a requirement of the kernel functions.
        """

        if function.finite_support:
            # When bw=1, the function is scaled by the standard deviation
            # so that std(f) = 1. Divide by standard deviation to get
            # the integration limits.
            a, b = tuple(s / np.sqrt(function.var) for s in function.support)
        else:
            a, b = -30, 30
        integral, abserr = quad(function, a=a, b=b)
        assert np.isclose(integral, 1)
            
    @pytest.mark.parametrize("fname, function", 
                             list(KDE._available_kernels.items()))
    def test_monotonic_decreasing(self, fname, function):
        """
        Verify that all available kernel functions decrease away from their 
        mode.
        """

        if function.finite_support:
            a, b = tuple(s / np.sqrt(function.var) for s in function.support)
            x = np.linspace(a, b)
        else:
            x = np.linspace(-50, 50)
        y = function(x)
        diffs_left = np.diff(y[x <= 0])
        diffs_right = np.diff(y[x >= 0])
        assert np.all(diffs_right <= 0)
        assert np.all(diffs_left >= 0)
                     
    @pytest.mark.parametrize("fname, function", 
                             list(KDE._available_kernels.items()))
    def test_non_negative(self, fname, function):
        """
        Verify that all available kernel functions are non-negative.
        """

        if function.finite_support:
            a, b = tuple(s / np.sqrt(function.var) for s in function.support)
            x = np.linspace(a, b)
        else:
            x = np.linspace(-20, 20)
        y = function(x)
        assert np.all(y >= 0)
    
 
if __name__ == "__main__":
    import pytest
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])