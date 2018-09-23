#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the TreeKDE.
"""
import numpy as np
from KDEpy.TreeKDE import TreeKDE
import itertools
import pytest


args = list(itertools.product([[-1, 0, 1, 10], [1, 2, 3, 4], [1, 1, 1, 2]],
                              [1, 2, 3]))


@pytest.mark.parametrize("data, split_index", args)
def test_additivity(data, split_index):
    """
    Test the additive propery of the KDE.
    """
    x = np.linspace(-10, 10)

    # Fit to add data
    y = TreeKDE('epa').fit(data).evaluate(x)

    # Fit to splits, and compensate for smaller data using weights
    weight_1 = split_index / len(data)
    y_1 = TreeKDE('epa').fit(data[:split_index]).evaluate(x) * weight_1

    weight_2 = (len(data) - split_index) / len(data)
    y_2 = TreeKDE('epa').fit(data[split_index:]).evaluate(x) * weight_2

    # Additive property of the functions
    assert np.allclose(y, y_1 + y_2)


@pytest.mark.parametrize("data, split_index", args)
def test_additivity_with_weights(data, split_index):
    """
    Test the additive propery of the KDE.
    """

    x = np.linspace(-10, 15)
    weights = np.arange(len(data)) + 1
    weights = weights / np.sum(weights)

    # Fit to add data
    y = TreeKDE('epa').fit(data, weights).evaluate(x)

    # Split up the data and the weights
    data = list(data)
    weights = list(weights)
    data_first_split = data[:split_index]
    data_second_split = data[split_index:]
    weights_first_split = weights[:split_index]
    weights_second_split = weights[split_index:]

    # Fit to splits, and compensate for smaller data using weights
    y_1 = (TreeKDE('epa').fit(data_first_split, weights_first_split)
           .evaluate(x) * sum(weights_first_split))

    y_2 = (TreeKDE('epa').fit(data_second_split, weights_second_split)
           .evaluate(x) * sum(weights_second_split))

    # Additive property of the functions
    assert np.allclose(y, y_1 + y_2)


@pytest.mark.parametrize("kernel, bw, n, expected_result",
                         [('box', 0.1, 5, np.array([2.101278e-19,
                                                    3.469447e-18,
                                                    1.924501e+00,
                                                    0.000000e+00,
                                                    9.622504e-01])),
                          ('box', 0.2, 5, np.array([3.854941e-18,
                                                    2.929755e-17,
                                                    9.622504e-01,
                                                    0.000000e+00,
                                                    4.811252e-01])),
                          ('box', 0.6, 3, np.array([0.1603751,
                                                    0.4811252,
                                                    0.4811252])),
                          ('tri', 0.6, 3, np.array([0.1298519,
                                                    0.5098009,
                                                    0.3865535])),
                          ('epa', 0.1, 6, np.array([0.000000e+00,
                                                    7.285839e-17,
                                                    2.251871e-01,
                                                    1.119926e+00,
                                                    0.000000e+00,
                                                    1.118034e+00])),
                          ('biweight', 2, 5, np.array([0.1524078,
                                                       0.1655184,
                                                       0.1729870,
                                                       0.1743973,
                                                       0.1696706]))])
def test_against_R_density(kernel, bw, n, expected_result):
    """
    Test against the following function call in R:

        d <- density(c(0, 0.1, 1), kernel="{kernel}", bw={bw},
        n={n}, from=-1, to=1);
        d$y
    """
    data = np.array([0, 0.1, 1])
    x = np.linspace(-1, 1, num=n)
    y = TreeKDE(kernel, bw=bw).fit(data).evaluate(x)
    assert np.allclose(y, expected_result, atol=10**(-2.7))


@pytest.mark.parametrize("bw, n, expected_result",
                         [(1, 3, np.array([0.17127129,
                                           0.34595518,
                                           0.30233275])),
                          (0.1, 5, np.array([2.56493684e-22,
                                             4.97598466e-06,
                                             2.13637668e+00,
                                             4.56012216e-04,
                                             1.32980760e+00])),
                          (0.01, 3, np.array([0.,
                                              13.29807601,
                                              13.29807601]))])
def test_against_scipy_density(bw, n, expected_result):
    """
    Test against the following function call in SciPy:

        data = np.array([0, 0.1, 1])
        x = np.linspace(-1, 1, {n})
        bw = {bw}/np.asarray(data).std(ddof=1)
        density_estimate = gaussian_kde(dataset = data, bw_method = bw)
        y = density_estimate.evaluate(x)

    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    """
    data = np.array([0, 0.1, 1])
    x = np.linspace(-1, 1, num=n)
    y = TreeKDE(kernel='gaussian', bw=bw).fit(data).evaluate(x)
    error = np.mean((y - expected_result)**2)
    assert error < 1e-10


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=['.', '--doctest-modules', '-v'])
