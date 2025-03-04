import pytest
import numpy as np
from itertools import product

from KDEpy.mirroring import mirror_data

boundaries_list = [((1, 3), (1, 3)), ((1, 3), (None, 3)), ((1, 3), None), ((1.5, 2.7), None)]


@pytest.mark.parametrize("boundaries", boundaries_list)
def test_mirror_axis(boundaries):
    """
    Test the mirror_data function with different boundary conditions.

    Parameters
    ----------
    boundaries : tuple of tuples
        Each tuple contains the lower and upper boundaries for each dimension.
        Use None for no boundary in that dimension.

    Notes
    -----
    - First test: 2D data, 2 boundaries each
    - Second test: 2D data, 2 and 1 boundaries
    - Third test: 2D data, 2 and None boundaries
    - Fourth test: 2D data, 2 and None boundaries, reflection on closest data point

    Expected Results
    ----------------
    - For boundaries ((1, 3), (1, 3)):
        [[1, 1], [1, 2], [1, 3],
        [2, 1], [2, 2], [2, 3],
        [3, 1], [3, 2], [3, 3]]
    - For boundaries ((1, 3), (None, 3)):
        [[1, -1], [1, 0], [1, 1], [1, 2], [1, 3],
        [2, -1], [2, 0], [2, 1], [2, 2], [2, 3],
        [3, -1], [3, 0], [3, 1], [3, 2], [3, 3]]
    - For boundaries ((1, 3), None):
        [[1, -1], [1, 0], [1, 1], [1, 2], [1, 3],
        [2, -1], [2, 0], [2, 1], [2, 2], [2, 3],
        [3, -1], [3, 0], [3, 1], [3, 2], [3, 3]]
    - For boundaries ((1.5, 2.7), None):
        [[2, -1], [2, 0], [2, 1], [2, 2], [2, 3]]
    """

    boundaries = tuple(tuple(b) if isinstance(b, list) else b for b in boundaries)
    data = np.array(list(product(range(-1, 4), repeat=2)))
    expected_results = {
        ((1, 3), (1, 3)): np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]),
        ((1, 3), (None, 3)): np.array(
            [
                [1, -1],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, -1],
                [2, 0],
                [2, 1],
                [2, 2],
                [2, 3],
                [3, -1],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
            ]
        ),
        ((1, 3), None): np.array(
            [
                [1, -1],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, -1],
                [2, 0],
                [2, 1],
                [2, 2],
                [2, 3],
                [3, -1],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
            ]
        ),
        ((1.5, 2.7), None): np.array([[2, -1], [2, 0], [2, 1], [2, 2], [2, 3]]),
    }

    mirrored_data, _ = mirror_data(data, boundaries)
    print(mirrored_data, expected_results[boundaries])
    assert np.array_equal(mirrored_data, expected_results[boundaries])


boundaries_list = [((1, 3), (1, 3)), ((1, 3), (None, 3)), ((1, 3), None), ((1.5, 2.7), None)]
pdf_values = [
    None,
    np.array(
        [
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.085,
            0.085,
            0.085,
            0.01,
            0.01,
            0.085,
            0.16,
            0.085,
            0.01,
            0.01,
            0.085,
            0.085,
            0.085,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ]
    ),
]


@pytest.mark.parametrize("boundaries, pdf_values", product(boundaries_list, pdf_values))
def test_mirror_pdf(boundaries, pdf_values):
    """
    Testing the PDF result from mirror_data function with different boundary conditions.
    Both over a uniform distribution (None) and a given PDF of a piramid (bivariate triangular distribution).
    Parameters
    ----------
    boundaries : tuple of tuples
        Each tuple contains the lower and upper boundaries for each dimension.
        Use None for no boundary in that dimension.
    pdf_values : list of lists, optional (default=None)
        PDF values for each data point. If None, assume uniform distribution over the multidimensional grid provided.

    Notes
    -----
    8 tests in total, 2 for each boundary condition, 4 for uniform distribution and 4 for the pyramid PDF.

    - First test: 2D data, 2 boundaries each
    - Second test: 2D data, 2 and 1 boundaries
    - Third test: 2D data, 2 and None boundaries
    - Fourth test: 2D data, 2 and None boundaries, reflection on closest data point
    """

    boundaries = tuple(tuple(b) if isinstance(b, list) else b for b in boundaries)
    data = np.array(list(product(range(-1, 4), repeat=2)))
    pyramid_pdf = np.array(
        [
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.085,
            0.085,
            0.085,
            0.01,
            0.01,
            0.085,
            0.16,
            0.085,
            0.01,
            0.01,
            0.085,
            0.085,
            0.085,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ]
    )

    if pdf_values is None:
        expected_results_pdf = {
            ((1, 3), (1, 3)): np.array([0.0625, 0.0625, 0.125, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.25]),
            ((1, 3), (None, 3)): np.array(
                [
                    0.04166667,
                    0.04166667,
                    0.04166667,
                    0.04166667,
                    0.08333333,
                    0.04166667,
                    0.04166667,
                    0.04166667,
                    0.04166667,
                    0.08333333,
                    0.08333333,
                    0.08333333,
                    0.08333333,
                    0.08333333,
                    0.16666667,
                ]
            ),
            ((1, 3), None): np.array(
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]
            ),
            ((1.5, 2.7), None): np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        }
    elif np.array_equal(pdf_values, pyramid_pdf):
        expected_results_pdf = {
            ((1, 3), (1, 3)): np.array(
                [
                    0.29906542,
                    0.1588785,
                    0.03738318,
                    0.1588785,
                    0.1588785,
                    0.03738318,
                    0.03738318,
                    0.03738318,
                    0.07476636,
                ]
            ),
            ((1, 3), (None, 3)): np.array(
                [
                    0.0130719,
                    0.11111111,
                    0.20915033,
                    0.11111111,
                    0.02614379,
                    0.0130719,
                    0.11111111,
                    0.11111111,
                    0.11111111,
                    0.02614379,
                    0.02614379,
                    0.02614379,
                    0.02614379,
                    0.02614379,
                    0.05228758,
                ]
            ),
            ((1, 3), None): np.array(
                [
                    0.0137931,
                    0.11724138,
                    0.22068966,
                    0.11724138,
                    0.0137931,
                    0.0137931,
                    0.11724138,
                    0.11724138,
                    0.11724138,
                    0.0137931,
                    0.02758621,
                    0.02758621,
                    0.02758621,
                    0.02758621,
                    0.02758621,
                ]
            ),
            ((1.5, 2.7), None): np.array([0.03636364, 0.30909091, 0.30909091, 0.30909091, 0.03636364]),
        }

    # Testing only the pdf now, as the mirrored data is already tested
    mirror, pdf = mirror_data(data, boundaries, pdf_values)
    print(data, boundaries, pdf_values, mirror, pdf)
    print(expected_results_pdf[boundaries])
    assert np.allclose(pdf, expected_results_pdf[boundaries], atol=1e-7)


if __name__ == "__main__":
    # --durations=10  <- May be used to show potentially slow tests
    # pytest.main(args=[".", "--doctest-modules", "-v", "--durations=15"])
    pytest.main(args=[".", "-k", "test_mirroring"])
