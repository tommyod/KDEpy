import numpy as np


def mirror_data(data, boundaries, pdf_values=None, decimals=10):
    """
    Mirrors the data based on the provided boundaries and adds up the values onto the unmirrored parts.
    Sets everything below or above the boundaries to 0 in the values.

    Reflection across the closest outer data point (pivot). Most of the time this value will be dropped
    after the reflection. Using this value as a pivot is the only way of keeping grid structure when using
    native KDEpy functions. By the nature of the folding and copying procedure, if there is any data exactly
    in the boundary it will be duplicated.

    Datapoints in 'data' should be unique determined, if there are repeated points, they will be aggregated,
    and the PDF values will be summed.

    Parameters
    ----------
    data : np.ndarray
        The input data array (N x D). N data points in D dimensions.
    boundaries : list of tuples
        A list of tuples specifying the lower and upper boundaries for each dimension.
        Use None for no boundary in that dimension.
        Valid examples:
            boundaries = [(1, 10), (1, 10)]  # Both dimensions are bounded between 1 and 10
            boundaries = [(1, 10), (None, 10)]  # Bounds (1, 10),  (-inf, 10)
            boundaries = [(1, 10), None]     # Bounds (1, 10), (-inf, inf)
    pdf_values : np.ndarray, optional
        Array (N,) of the evaluated PDF at each gridpoint. If not provided, assumes uniform distribution.
    decimals : int, optional
        Number of decimal places to round to for grouping. Default is 10.

    Returns
    -------
    np.ndarray
        The mirrored data array (N x D).
    np.ndarray
        The rescaled PDF values.

    Example
    -------
    >>> from KDEpy.mirroring import mirror_data
    >>> import numpy as np
    >>> data = np.arange(8).reshape(-1, 1)
    >>> boundaries = [(1, 7)]
    >>> mirror_data(data, boundaries)
    (array([[1],
           [2],
           [3],
           [4],
           [5],
           [6],
           [7]]), array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2]))
    >>> from KDEpy import FFTKDE
    >>> np.random.seed(42)
    >>> data = np.random.rand(1000, 2) * 10
    >>> grid_points = 100
    >>> boundaries = [[1, 10], (1, 10)] # 2-dimensional dataframe. Accepts tuple or list
    >>> X, Z = FFTKDE(bw=1).fit(data)((grid_points, grid_points))
    >>> X_mirrored, Z_mirrored = mirror_data(X, boundaries, Z)
    """

    def sum_together_np(mirrored_data, updated_values, decimals=10):
        """
        Group by unique rows in data and sum the corresponding values.

        Parameters
        ----------
        mirrored_data : np.ndarray
            The input data array (N x D).
        updated_values : np.ndarray
            The values array (N,).
        decimals : int
            Number of decimal places to round to for grouping.

        Returns
        -------
        np.ndarray
            The grouped data array (M x D).
        np.ndarray
            The summed values array (M,).
        """
        # Round the data to the specified number of decimals.
        # Important! Otherwise, due to numerical precision, the numbers won't match
        rounded_data = np.round(mirrored_data, decimals=decimals)

        unique_data, indices = np.unique(
            rounded_data, axis=0, return_inverse=True
        )  # Find unique rows and their indices
        summed_values = np.zeros(unique_data.shape[0])  # Preallocate the summed values array
        np.add.at(summed_values, indices, updated_values)  # Use np.add.at to sum the values for each unique row

        return unique_data, summed_values

    if pdf_values is None:
        pdf_values = (
            np.ones(data.shape[0]) / data.shape[0]
        )  # If no PDF values are provided, assume uniform distribution.

    # Check if the amount of rows of pdf_values matches the ones on data.
    assert pdf_values.shape[0] == data.shape[0], "PDF values must match the data size."

    # Check if the data and boundaries have the same dimensions
    assert data.shape[1] == len(boundaries), "Data dimensions must match the boundaries."

    mirrored_data = data.copy()
    updated_values = pdf_values.copy()

    # Each boundary is 'folding' the data; creating a copy, adding the PDF values,
    # and then deleting the values which are beyond the boundary.
    for dim, boundary in enumerate(boundaries):
        if boundary is not None:
            lower, upper = boundary
            if lower is not None:
                try:
                    closest_lower = np.max(
                        mirrored_data[mirrored_data[:, dim] <= lower, dim]
                    )  # Mirror using the closest inner data point as pivot
                    lower_mirror = 2 * closest_lower - mirrored_data[:, dim]
                    mirrored_points = np.column_stack(
                        [lower_mirror if i == dim else mirrored_data[:, i] for i in range(mirrored_data.shape[1])]
                    )
                    mirrored_data = np.vstack([mirrored_data, mirrored_points])
                    updated_values = np.concatenate([updated_values, updated_values])
                    mirrored_data, updated_values = sum_together_np(mirrored_data, updated_values)
                except ValueError:
                    print("No points below the lower boundary.")
                    pass  # If there are no points below the lower boundary, they can't be mirrored.
                    # Equivalent as mirroring -inf

                # Deleting the points that are out of the boundaries.
                mask = mirrored_data[:, dim] >= lower
                mirrored_data = mirrored_data[mask]
                updated_values = updated_values[mask]

            if upper is not None:
                try:
                    closest_upper = np.min(
                        mirrored_data[mirrored_data[:, dim] >= upper, dim]
                    )  # Mirror using the closest data point as pivot
                    upper_mirror = 2 * closest_upper - mirrored_data[:, dim]
                    mirrored_points = np.column_stack(
                        [upper_mirror if i == dim else mirrored_data[:, i] for i in range(mirrored_data.shape[1])]
                    )
                    mirrored_data = np.vstack([mirrored_data, mirrored_points])
                    updated_values = np.concatenate([updated_values, updated_values])
                    mirrored_data, updated_values = sum_together_np(mirrored_data, updated_values)
                except ValueError:
                    print("No points above the upper boundary.")
                    pass  # If there are no points above the upper boundary, they can't be mirrored.
                    # Equivalent as mirroring inf

                # Deleting the points that are out of the boundaries.
                mask = mirrored_data[:, dim] <= upper
                mirrored_data = mirrored_data[mask]
                updated_values = updated_values[mask]

    # Have to do a second pass. After the mirroring, drop values out of boundaries.
    for dim, boundary in enumerate(boundaries):
        if boundary is not None:
            lower, upper = boundary
            if lower is not None:
                mask = mirrored_data[:, dim] >= lower
                mirrored_data = mirrored_data[mask]
                updated_values = updated_values[mask]
            if upper is not None:
                mask = mirrored_data[:, dim] <= upper
                mirrored_data = mirrored_data[mask]
                updated_values = updated_values[mask]

    # Adjusting pdf to make the sum = 1
    updated_values = updated_values / updated_values.sum()

    return mirrored_data, updated_values



if __name__ == "__main__":
    import doctest
    doctest.testmod()