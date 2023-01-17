#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the bandwidth selection.
"""

import numpy as np
import pytest

from KDEpy.bw_selection import _bw_methods, improved_sheather_jones


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


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "--durations=15"])
