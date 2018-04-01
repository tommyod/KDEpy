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
            a, b = function.support
        else:
            a, b = -5 * function.var, 5 * function.var
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
            x = np.linspace(*function.support)
        else:
            x = np.linspace(-5 * function.var, 5 * function.var)
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
            x = np.linspace(*function.support)
        else:
            x = np.linspace(-5 * function.var, 5 * function.var)
        y = function(x)
        assert np.all(y >= 0)
    
 
if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])