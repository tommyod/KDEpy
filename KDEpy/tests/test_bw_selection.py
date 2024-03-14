#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the bandwidth selection.
"""

import numpy as np
import pytest

from KDEpy.bw_selection import _bw_methods, improved_sheather_jones, silvermans_rule


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return np.random.randn(100, 1)

@pytest.fixture(scope="module")
def multidim_data() -> np.ndarray:
    return np.random.randn(100, 2)


@pytest.mark.parametrize("method", _bw_methods.values())
def test_equal_weights_dont_changed_bw(data, method):
    weights = np.ones_like(data).squeeze() * 2
    bw_no_weights = method(data, weights=None)
    bw_weighted = method(data, weights=weights)
    np.testing.assert_almost_equal(bw_no_weights, bw_weighted)


def test_multidim_silvermans_rule_weights_dont_changed_bw(multidim_data):
    weights = np.ones_like(multidim_data).squeeze() * 2
    bw_no_weights = silvermans_rule(multidim_data, weights=None)
    bw_weighted = silvermans_rule(multidim_data, weights=weights)
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


def test_onedim_silvermans_rule_shape(data):
    sr_res = silvermans_rule(data)
    # dims is a float
    assert isinstance(sr_res, float)


def test_multidim_silvermans_rule_shape(multidim_data):
    sr_res = silvermans_rule(multidim_data)
    # dims shape is 2
    dim = sr_res.shape[0]
    assert dim == 2


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "--durations=15"])
