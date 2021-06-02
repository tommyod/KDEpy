#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the bandwidth selection.
"""

import pytest
import numpy as np

from KDEpy.bw_selection import (
    _bw_methods,
    improved_sheather_jones,
    k_nearest_neighbors,
    cross_val
)
from KDEpy.TreeKDE import TreeKDE


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return np.random.randn(100, 1)


@pytest.mark.parametrize("method", _bw_methods.values())
def test_equal_weights_dont_changed_bw(data, method):
    weights = np.ones_like(data).squeeze() * 2
    bw_no_weights = method(data, weights=None)
    bw_weighted = method(data, weights=weights)
    np.testing.assert_almost_equal(bw_no_weights, bw_weighted)


def test_isj_bw_weights_single_zero_weighted_point(data):
    data_with_outlier = np.concatenate((data.copy(), np.array([[1000]])))
    weights = np.ones_like(data_with_outlier).squeeze()
    weights[-1] = 0

    np.testing.assert_array_almost_equal(
        improved_sheather_jones(data),
        improved_sheather_jones(data_with_outlier, weights),
    )


# multiple runs to allow a good spread of catching errors
@pytest.mark.parametrize("execution_number", range(5))
def test_isj_bw_weights_same_as_resampling(data, execution_number):
    sample_weights = np.random.randint(low=1, high=100, size=len(data))
    data_resampled = np.repeat(data, repeats=sample_weights).reshape((-1, 1))
    np.testing.assert_array_almost_equal(
        improved_sheather_jones(data_resampled),
        improved_sheather_jones(data, sample_weights),
    )


@pytest.mark.parametrize("dims", [1,2,3])
def test_knn_with_2_points(dims):
    data = np.zeros((2,dims))
    data[0,0] = 0
    data[1,0] = 1
    # Distances must be [1,1]
    np.testing.assert_array_almost_equal(
        k_nearest_neighbors(data, k=1),
        np.array([1, 1])
    )


@pytest.mark.parametrize("dims", [1,2,3])
def test_cv_with_2_points(dims):
    data = np.zeros((2,dims))
    data[0,0] = 0
    data[1,0] = 1
    # The optimal bw can be found analytically by solving:
    # d log(kernel(1,bw)) / d bw = 0
    # For kernel="gaussian" and norm=2.0 it gives:
    bw_optimal = 1 / np.sqrt(dims)
    grid = np.logspace(-0.01, 0.01, 5)  # Make grid of factors around 1
    np.allclose(
        cross_val(TreeKDE(), data, seed=bw_optimal, grid=grid),
        bw_optimal
    )


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "--durations=15"])
