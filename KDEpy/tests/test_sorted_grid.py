#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for verifying that grids obeys the sorting properties required for linear binning.
"""
import numpy as np
import pytest

from KDEpy.binning import grid_is_sorted  # Imported from .pyx to binning.py, then here


class TestGridSorted:
    def test_regression_issue_grids(self):
        """
        This test is based on Issue 15, raised by @blasern.

        https://github.com/tommyod/KDEpy/issues/15
        """

        # The original example by @blasern. Should NOT pass the verification.
        grid_size = 20
        min_X = np.array([-2.6, -4.0])
        max_X = np.array([3.2, 7.7])
        grid_margins = tuple(np.linspace(mn, mx, grid_size) for mn, mx in zip(min_X, max_X))
        grid = np.stack(np.meshgrid(*grid_margins), -1).reshape(-1, len(grid_margins))
        assert not grid_is_sorted(grid)

        # More minimal example, should also fail.
        grid_x = np.linspace(-2, 2, 2**5)
        grid_y = np.linspace(-2, 2, 2**4)
        grid = np.stack(np.meshgrid(grid_x, grid_y), -1).reshape(-1, 2)
        assert not grid_is_sorted(grid)

        # Changing the above slightly, should also fail.
        grid = np.stack(np.meshgrid(grid_y, grid_x), -1).reshape(-1, 2)
        assert not grid_is_sorted(grid)

        # Swapping the indices should work
        grid = np.stack(np.meshgrid(grid_y, grid_x), -1).reshape(-1, 2)
        grid[:, [0, 1]] = grid[:, [1, 0]]  # Swap indices.
        assert grid_is_sorted(grid)

    def test_regression_issue_code(self):
        """
        This test is based on Issue 15, raised by @blasern. Tests the full code.

        https://github.com/tommyod/KDEpy/issues/15
        """
        # imports
        import numpy as np

        import KDEpy

        # Create bimodal 2D data
        data = np.vstack((np.random.randn(2**8, 2), np.random.randn(2**8, 2) + (0, 5)))

        # Create 2D grid
        grid_size = 20
        min_X = np.min(data, axis=0) - 0.1
        max_X = np.max(data, axis=0) + 0.1
        grid_margins = tuple(np.linspace(mn, mx, grid_size) for mn, mx in zip(min_X, max_X))
        grid = np.stack(np.meshgrid(*grid_margins), -1).reshape(-1, len(grid_margins))

        # density estimates
        with pytest.raises(ValueError):
            KDEpy.FFTKDE(bw=0.2).fit(data).evaluate(grid)

    def test_on_good_grids(self):
        """
        Test on grids that are good.
        """

        grid = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]], dtype=float)
        assert grid_is_sorted(grid)

        grid = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
            dtype=float,
        )
        assert grid_is_sorted(grid)

        grid = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
        assert grid_is_sorted(grid)

        grid = np.array(
            [
                [-4.1, -4.1, -4.1],
                [-4.1, -4.1, 4.1],
                [-4.1, 0.0, -4.1],
                [-4.1, 0.0, 4.1],
                [-4.1, 4.1, -4.1],
                [-4.1, 4.1, 4.1],
                [-1.4, -4.1, -4.1],
                [-1.4, -4.1, 4.1],
                [-1.4, 0.0, -4.1],
                [-1.4, 0.0, 4.1],
                [-1.4, 4.1, -4.1],
                [-1.4, 4.1, 4.1],
                [1.4, -4.1, -4.1],
                [1.4, -4.1, 4.1],
                [1.4, 0.0, -4.1],
                [1.4, 0.0, 4.1],
                [1.4, 4.1, -4.1],
                [1.4, 4.1, 4.1],
                [4.1, -4.1, -4.1],
                [4.1, -4.1, 4.1],
                [4.1, 0.0, -4.1],
                [4.1, 0.0, 4.1],
                [4.1, 4.1, -4.1],
                [4.1, 4.1, 4.1],
            ],
            dtype=float,
        )
        assert grid_is_sorted(grid)

        grid = np.array(
            [
                [-4.1, -4.1, -4.1],
                [-4.1, -4.1, -2.0],
                [-4.1, -4.1, 0.0],
                [-4.1, -4.1, 2.0],
                [-4.1, -4.1, 4.1],
                [-4.1, -1.4, -4.1],
                [-4.1, -1.4, -2.0],
                [-4.1, -1.4, 0.0],
                [-4.1, -1.4, 2.0],
                [-4.1, -1.4, 4.1],
                [-4.1, 1.4, -4.1],
                [-4.1, 1.4, -2.0],
                [-4.1, 1.4, 0.0],
                [-4.1, 1.4, 2.0],
                [-4.1, 1.4, 4.1],
                [-4.1, 4.1, -4.1],
                [-4.1, 4.1, -2.0],
                [-4.1, 4.1, 0.0],
                [-4.1, 4.1, 2.0],
                [-4.1, 4.1, 4.1],
                [4.1, -4.1, -4.1],
                [4.1, -4.1, -2.0],
                [4.1, -4.1, 0.0],
                [4.1, -4.1, 2.0],
                [4.1, -4.1, 4.1],
                [4.1, -1.4, -4.1],
                [4.1, -1.4, -2.0],
                [4.1, -1.4, 0.0],
                [4.1, -1.4, 2.0],
                [4.1, -1.4, 4.1],
                [4.1, 1.4, -4.1],
                [4.1, 1.4, -2.0],
                [4.1, 1.4, 0.0],
                [4.1, 1.4, 2.0],
                [4.1, 1.4, 4.1],
                [4.1, 4.1, -4.1],
                [4.1, 4.1, -2.0],
                [4.1, 4.1, 0.0],
                [4.1, 4.1, 2.0],
                [4.1, 4.1, 4.1],
            ],
            dtype=float,
        )
        assert grid_is_sorted(grid)

        grid = np.array(
            [
                [-4.1, -4.1, -4.1, -4.1],
                [-4.1, -4.1, -4.1, 0.0],
                [-4.1, -4.1, -4.1, 4.1],
                [-4.1, -4.1, 0.0, -4.1],
                [-4.1, -4.1, 0.0, 0.0],
                [-4.1, -4.1, 0.0, 4.1],
                [-4.1, -4.1, 4.1, -4.1],
                [-4.1, -4.1, 4.1, 0.0],
                [-4.1, -4.1, 4.1, 4.1],
                [-4.1, 0.0, -4.1, -4.1],
                [-4.1, 0.0, -4.1, 0.0],
                [-4.1, 0.0, -4.1, 4.1],
                [-4.1, 0.0, 0.0, -4.1],
                [-4.1, 0.0, 0.0, 0.0],
                [-4.1, 0.0, 0.0, 4.1],
                [-4.1, 0.0, 4.1, -4.1],
                [-4.1, 0.0, 4.1, 0.0],
                [-4.1, 0.0, 4.1, 4.1],
                [-4.1, 4.1, -4.1, -4.1],
                [-4.1, 4.1, -4.1, 0.0],
                [-4.1, 4.1, -4.1, 4.1],
                [-4.1, 4.1, 0.0, -4.1],
                [-4.1, 4.1, 0.0, 0.0],
                [-4.1, 4.1, 0.0, 4.1],
                [-4.1, 4.1, 4.1, -4.1],
                [-4.1, 4.1, 4.1, 0.0],
                [-4.1, 4.1, 4.1, 4.1],
                [0.0, -4.1, -4.1, -4.1],
                [0.0, -4.1, -4.1, 0.0],
                [0.0, -4.1, -4.1, 4.1],
                [0.0, -4.1, 0.0, -4.1],
                [0.0, -4.1, 0.0, 0.0],
                [0.0, -4.1, 0.0, 4.1],
                [0.0, -4.1, 4.1, -4.1],
                [0.0, -4.1, 4.1, 0.0],
                [0.0, -4.1, 4.1, 4.1],
                [0.0, 0.0, -4.1, -4.1],
                [0.0, 0.0, -4.1, 0.0],
                [0.0, 0.0, -4.1, 4.1],
                [0.0, 0.0, 0.0, -4.1],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.1],
                [0.0, 0.0, 4.1, -4.1],
                [0.0, 0.0, 4.1, 0.0],
                [0.0, 0.0, 4.1, 4.1],
                [0.0, 4.1, -4.1, -4.1],
                [0.0, 4.1, -4.1, 0.0],
                [0.0, 4.1, -4.1, 4.1],
                [0.0, 4.1, 0.0, -4.1],
                [0.0, 4.1, 0.0, 0.0],
                [0.0, 4.1, 0.0, 4.1],
                [0.0, 4.1, 4.1, -4.1],
                [0.0, 4.1, 4.1, 0.0],
                [0.0, 4.1, 4.1, 4.1],
                [4.1, -4.1, -4.1, -4.1],
                [4.1, -4.1, -4.1, 0.0],
                [4.1, -4.1, -4.1, 4.1],
                [4.1, -4.1, 0.0, -4.1],
                [4.1, -4.1, 0.0, 0.0],
                [4.1, -4.1, 0.0, 4.1],
                [4.1, -4.1, 4.1, -4.1],
                [4.1, -4.1, 4.1, 0.0],
                [4.1, -4.1, 4.1, 4.1],
                [4.1, 0.0, -4.1, -4.1],
                [4.1, 0.0, -4.1, 0.0],
                [4.1, 0.0, -4.1, 4.1],
                [4.1, 0.0, 0.0, -4.1],
                [4.1, 0.0, 0.0, 0.0],
                [4.1, 0.0, 0.0, 4.1],
                [4.1, 0.0, 4.1, -4.1],
                [4.1, 0.0, 4.1, 0.0],
                [4.1, 0.0, 4.1, 4.1],
                [4.1, 4.1, -4.1, -4.1],
                [4.1, 4.1, -4.1, 0.0],
                [4.1, 4.1, -4.1, 4.1],
                [4.1, 4.1, 0.0, -4.1],
                [4.1, 4.1, 0.0, 0.0],
                [4.1, 4.1, 0.0, 4.1],
                [4.1, 4.1, 4.1, -4.1],
                [4.1, 4.1, 4.1, 0.0],
                [4.1, 4.1, 4.1, 4.1],
            ],
            dtype=float,
        )

        assert grid_is_sorted(grid)

        grid = np.array(
            [
                [-4.1, -4.1, -4.1, -4.1],
                [-4.1, -4.1, -4.1, 0.0],
                [-4.1, -4.1, -4.1, 4.1],
                [-4.1, -4.1, 0.0, -4.1],
                [-4.1, -4.1, 0.0, 4.1],
                [-4.1, -4.1, 4.1, -4.1],
                [-4.1, 0.0, -4.1, 0.0],
                [-4.1, 0.0, -4.1, 4.1],
                [-4.1, 0.0, 0.0, -4.1],
                [-4.1, 0.0, 0.0, 0.0],
                [-4.1, 4.1, -4.1, 0.0],
                [-4.1, 4.1, -4.1, 4.1],
                [-4.1, 4.1, 0.0, -4.1],
                [-4.1, 4.1, 0.0, 0.0],
                [-4.1, 4.1, 4.1, 4.1],
                [0.0, -4.1, -4.1, -4.1],
                [0.0, -4.1, -4.1, 0.0],
                [0.0, -4.1, -4.1, 4.1],
                [0.0, -4.1, 0.0, -4.1],
                [0.0, -4.1, 0.0, 0.0],
                [0.0, -4.1, 0.0, 4.1],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.1],
                [0.0, 0.0, 4.1, -4.1],
                [0.0, 0.0, 4.1, 0.0],
                [0.0, 0.0, 4.1, 4.1],
                [0.0, 4.1, -4.1, -4.1],
                [0.0, 4.1, 4.1, -4.1],
                [0.0, 4.1, 4.1, 0.0],
                [4.1, -4.1, -4.1, 4.1],
                [4.1, -4.1, 0.0, -4.1],
                [4.1, -4.1, 0.0, 0.0],
                [4.1, -4.1, 4.1, 4.1],
                [4.1, 0.0, 0.0, -4.1],
                [4.1, 0.0, 0.0, 0.0],
                [4.1, 0.0, 0.0, 4.1],
                [4.1, 0.0, 4.1, -4.1],
                [4.1, 4.1, -4.1, 4.1],
                [4.1, 4.1, 0.0, -4.1],
                [4.1, 4.1, 4.1, -4.1],
                [4.1, 4.1, 4.1, 4.1],
            ],
            dtype=float,
        )

        assert grid_is_sorted(grid)

    def test_on_bad_grids(self):
        """
        Test on grids that are good.
        """
        grid = np.array([[0], [0], [2], [1], [1], [1], [2], [2], [2]], dtype=float)
        assert not grid_is_sorted(grid)

        grid = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [1, 1], [2, 0], [2, 1], [2, 2]],
            dtype=float,
        )
        assert not grid_is_sorted(grid)

        grid = np.array([[1, 1], [3, 3], [2, 2]], dtype=float)
        assert not grid_is_sorted(grid)

        grid = np.array(
            [
                [-4.1, -4.1, -4.1],
                [-4.1, -4.1, 4.1],
                [-4.1, 0.0, -4.1],
                [-4.1, 0.0, 4.1],
                [-4.1, 4.1, -4.1],
                [-4.1, 4.1, 4.1],
                [-1.4, -4.1, -4.1],
                [-1.4, 0.0, -4.1],
                [-1.4, -4.1, 4.1],
                [-1.4, 0.0, 4.1],
                [-1.4, 4.1, -4.1],
                [-1.4, 4.1, 4.1],
                [1.4, -4.1, -4.1],
                [1.4, -4.1, 4.1],
                [1.4, 0.0, -4.1],
                [1.4, 0.0, 4.1],
                [1.4, 4.1, -4.1],
                [1.4, 4.1, 4.1],
                [4.1, -4.1, -4.1],
                [4.1, -4.1, 4.1],
                [4.1, 0.0, -4.1],
                [4.1, 0.0, 4.1],
                [4.1, 4.1, -4.1],
                [4.1, 4.1, 4.1],
            ],
            dtype=float,
        )
        assert not grid_is_sorted(grid)


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[".", "--doctest-modules", "-v", "--durations=15"])
