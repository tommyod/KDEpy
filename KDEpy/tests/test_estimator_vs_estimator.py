#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the implemented estimators against each other on simple data sets.
"""
import numpy as np
from KDEpy.NaiveKDE import NaiveKDE
from KDEpy.TreeKDE import TreeKDE
from KDEpy.FFTKDE import FFTKDE
import itertools
import pytest

N = 2**5

estimators = [NaiveKDE, TreeKDE, FFTKDE]
estimators_2 = list(itertools.combinations(estimators, 2))
kernels = list(NaiveKDE._available_kernels.keys())


@pytest.mark.parametrize("est1, est2", estimators_2)
def test_vs_simple(est1, est2):
    """
    Test that mean error is low on default parameters.
    """

    np.random.seed(12)
    data = np.random.randn(N)
    x1, y1 = est1().fit(data)()
    x1, y2 = est2().fit(data)()
    assert np.sqrt(np.mean((y1 - y2) ** 2)) < 0.0001


@pytest.mark.parametrize("est1, est2", estimators_2)
def test_vs_simple_weighted(est1, est2):
    """
    Test that mean error is low on default parameters with weighted data.
    """

    np.random.seed(12)
    data = np.random.randn(N) * 10
    weights = np.random.randn(N) ** 2 + 1
    x1, y1 = est1().fit(data, weights)()
    x1, y2 = est2().fit(data, weights)()
    assert np.sqrt(np.mean((y1 - y2) ** 2)) < 0.0001


@pytest.mark.parametrize("estimators, kernel, bw", list(itertools.product(estimators_2, kernels, [0.1, 5])))
def test_vs_simple_weighted_kernels(estimators, kernel, bw):
    """
    Test every kernel function over every implementation.
    """
    est1, est2 = estimators

    np.random.seed(13)
    data = np.random.randn(N) * 10
    weights = np.random.randn(N) ** 2 + 1
    x1, y1 = est1(kernel, bw=bw).fit(data, weights)()
    x1, y2 = est2(kernel, bw=bw).fit(data, weights)()
    assert np.sqrt(np.mean((y1 - y2) ** 2)) < 0.01
    # TODO: Examine why error increases when bw -> 0


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    # pytest.main(args=['.', '--doctest-modules', '-v'])

    est1, est2 = NaiveKDE, TreeKDE

    np.random.seed(13)
    data = np.random.randn(2**8) * 10
    weights = np.random.randn(2**8) ** 2 + 1
    x1, y1 = est1(bw=100).fit(data, weights)()
    x1, y2 = est2(bw=100).fit(data, weights)()
    import matplotlib.pyplot as plt

    plt.plot(x1, y1 - y2)
