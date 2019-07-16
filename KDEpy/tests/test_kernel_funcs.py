#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the kernel functions K. Every kernel function is a radial basis function,
i.e. it's a composition of a norm and a function defined on positive reals.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy
from scipy.integrate import quad
from KDEpy.BaseKDE import BaseKDE
from KDEpy.kernel_funcs import gauss_integral, trig_integral
import pytest
import itertools


class TestKernelHelperFunctions:
    @pytest.mark.parametrize(
        "dim, expected",
        [
            (0, 1.25331),
            (1, 1),
            (2, 1.25331),
            (3, 2),
            (4, 3.75994),
            (5, 8),
            (6, 18.7997),
        ],
    )
    def test_gauss_integral(self, dim, expected):
        """
        Test that the results of the gauss integral are equal to those obtained
        using WolframAlpha for small test cases.
        """
        assert np.allclose(gauss_integral(dim), expected, rtol=10e-5)

    @pytest.mark.parametrize(
        "dim, expected", [(2, (0.29454, 0.12060)), (3, (0.23032, 0.074080))]
    )
    def test_trig_integral(self, dim, expected):
        """
        Test that the results of the trig integral are equal to those obtained
        using WolframAlpha for small test cases.
        """
        assert np.allclose(trig_integral(dim), expected, rtol=10e-5)


class TestKernelFunctions:
    @pytest.mark.parametrize(
        "fname, function", list(BaseKDE._available_kernels.items())
    )
    def test_integral_unity(self, fname, function):
        """
        Verify that all available kernel functions have an integral evaluating
        to unity in 1D. This is a basic requirement of the kernel functions.
        """

        if function.finite_support:
            a, b = -function.support, function.support
        else:
            a, b = -5 * function.var, 5 * function.var
        integral, abserr = quad(function, a=a, b=b)
        assert np.isclose(integral, 1)

    @pytest.mark.parametrize(
        "p, kernel_name",
        itertools.product([0.5, 1, 1.5, 2, 13, np.inf], ["triweight", "gaussian"]),
    )
    def test_integral_unity_2D_many_p_norms(self, p, kernel_name):
        """
        Verify that the triweight (random choice) has area 1 under
        several p-norms when p >= 1 in 2D. This gives confidence that the
        normalization under any p-norm is correct.
        """

        function = BaseKDE._available_kernels[kernel_name]
        a, b = -function.support, function.support

        # Perform integration 2D
        def int2D(x1, x2):
            return function([[x1, x2]], norm=p)

        ans, err = scipy.integrate.nquad(
            int2D, [[a, b], [a, b]], opts={"epsabs": 10e-2, "epsrel": 10e-2}
        )

        assert np.allclose(ans, 1, rtol=10e-4, atol=10e-4)

    @pytest.mark.skip(reason="Slow test.")
    @pytest.mark.parametrize("p", [0.5, 1, 1.5, 2, 10, 50, np.inf])
    def test_integral_unity_3D_many_p_norms(self, p):
        """
        Verify that the triweight (random choice) has area 1 under
        several p-norms when p >= 1 in 3D. This gives confidence that the
        normalization under any p-norm is correct.
        """
        function = BaseKDE._available_kernels["triweight"]
        a, b = -function.support, function.support

        # Perform integration 2D
        def int2D(x1, x2, x3):
            return function([[x1, x2, x3]], norm=p)

        ans, err = scipy.integrate.nquad(
            int2D, [[a, b], [a, b], [a, b]], opts={"epsabs": 10e-1, "epsrel": 10e-1}
        )

        assert np.allclose(ans, 1, rtol=10e-2, atol=10e-2)

    @pytest.mark.parametrize(
        "fname, function, p",
        [
            (a[0], a[1], b)
            for (a, b) in list(
                itertools.product(
                    BaseKDE._available_kernels.items(), [0.5, 1, 2, 5.5, np.inf]
                )
            )
        ],
    )
    def test_integral_unity_2D_p_norm(self, fname, function, p):
        """
        Verify that all available kernel functions have an integral evaluating
        to unity in 2D using various p-norms.
        """
        if fname in ("logistic", "sigmoid", "box"):
            return

        if function.finite_support:
            a, b = -function.support, function.support
        else:
            a, b = -4, 4

        # Perform integration 2D
        def int2D(x1, x2):
            return function([[x1, x2]], norm=p)

        ans, err = scipy.integrate.nquad(
            int2D, [[a, b], [a, b]], opts={"epsabs": 10e-1, "epsrel": 10e-1}
        )

        assert np.allclose(ans, 1, rtol=10e-3, atol=10e-3)

    @pytest.mark.skip(reason="Slow test.")
    @pytest.mark.parametrize(
        "fname, function, p",
        [
            (a[0], a[1], b)
            for (a, b) in list(
                itertools.product(
                    BaseKDE._available_kernels.items(), [1, 2, 5.5, np.inf]
                )
            )
        ],
    )
    def test_integral_unity_3D_p_norm(self, fname, function, p):
        """
        Verify that all available kernel functions have an integral evaluating
        to unity in 3D using various p-norms.
        """
        if fname in ("logistic", "sigmoid", "box"):
            return

        if function.finite_support:
            a, b = -function.support, function.support
        else:
            a, b = -4, 4

        # Perform integration 2D
        def int2D(x1, x2, x3):
            return function([[x1, x2, x3]], norm=p)

        ans, err = scipy.integrate.nquad(
            int2D, [[a, b], [a, b], [a, b]], opts={"epsabs": 10e-1, "epsrel": 10e-1}
        )

        assert np.allclose(ans, 1, rtol=10e-2, atol=10e-2)

    @pytest.mark.parametrize(
        "fname, function", list(BaseKDE._available_kernels.items())
    )
    def test_monotonic_decreasing(self, fname, function):
        """
        Verify that all available kernel functions decrease away from 0.
        """

        if function.finite_support:
            x = np.linspace(-function.support, function.support)
        else:
            x = np.linspace(-5 * function.var, 5 * function.var)
        y = function(x)
        diffs_left = np.diff(y[x <= 0])
        diffs_right = np.diff(y[x >= 0])
        assert np.all(diffs_right <= 0)
        assert np.all(diffs_left >= 0)

    @pytest.mark.parametrize(
        "fname, function", list(BaseKDE._available_kernels.items())
    )
    def test_non_negative(self, fname, function):
        """
        Verify that all available kernel functions are non-negative.
        """

        if function.finite_support:
            x = np.linspace(-function.support, function.support)
        else:
            x = np.linspace(-5 * function.var, 5 * function.var)
        y = function(x)
        assert np.all(y >= 0)


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "--durations=15"])
