import pytest 
import numpy as np
from itertools import product

from KDEpy.mirroring import mirror_data

boundaries_list = [
    ((1, 3), (1, 3)),
    ((1, 3), (None, 3)),
    ((1, 3), None),
    ((1.5, 2.7), None)
]
@pytest.mark.parametrize("boundaries", boundaries_list)
def test_mirror_data(boundaries):
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
        [[1, -1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
        [2, -1], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
        [3, -1], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]]
    - For boundaries ((1.5, 2.7), None):
        [[2, -1], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]]
    """
    
    boundaries = tuple(tuple(b) if isinstance(b, list) else b for b in boundaries)
    data = np.array(list(product(range(-1, 5), repeat=2)))
    expected_results = {
    ((1, 3), (1, 3)): np.array([
        [1, 1], [1, 2], [1, 3],
        [2, 1], [2, 2], [2, 3],
        [3, 1], [3, 2], [3, 3]
    ]),
    ((1, 3), (None, 3)): np.array([
        [1, -1], [1, 0], [1, 1], [1, 2], [1, 3],
        [2, -1], [2, 0], [2, 1], [2, 2], [2, 3],
        [3, -1], [3, 0], [3, 1], [3, 2], [3, 3]
    ]),
    ((1, 3), None): np.array([
        [1, -1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
        [2, -1], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
        [3, -1], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]
    ]),
    ((1.5, 2.7), None): np.array([
        [2, -1], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]
    ]), 
    }

    mirrored_data, _ = mirror_data(data, boundaries)
    assert (mirrored_data == expected_results[boundaries]).all()

pytest.main(["-k", "test_mirror_data"])
