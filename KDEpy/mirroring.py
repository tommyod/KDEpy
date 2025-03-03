import numpy as np


def mirror_data(data, boundaries, pdf_values=None, decimals=10):
    """
    Mirrors the data based on the provided boundaries and adds up the values onto the unmirrored parts.
    Sets everything below or above the boundaries to 0 in the values.

    Parameters
    ----------
    data : np.ndarray
        The input data array (N x D).
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

    assert pdf_values.shape[0] == data.shape[0], "PDF values must match the data size."

    mirrored_data = data.copy()
    updated_values = pdf_values.copy()

    # Each boundary is 'folding' the data, and creating a copy, and adding the PDF values.
    for dim, boundary in enumerate(boundaries):
        if boundary is not None:
            lower, upper = boundary
            if lower is not None:
                try:
                    closest_lower = np.max(
                        data[data[:, dim] <= lower, dim]
                    )  # Mirror using the closest data point as pivot
                    lower_mirror = 2 * closest_lower - data[:, dim]
                    mirrored_points = np.column_stack(
                        [lower_mirror if i == dim else data[:, i] for i in range(data.shape[1])]
                    )
                    mirrored_data = np.vstack([mirrored_data, mirrored_points])
                    updated_values = np.concatenate([updated_values, pdf_values])

                    mirrored_data, updated_values = sum_together_np(mirrored_data, updated_values)
                except Exception as e:
                    print(f"Failed to mirror lower boundary: {e}")
                    pass
            if upper is not None:
                try:
                    closest_upper = np.min(
                        data[data[:, dim] >= upper, dim]
                    )  # Mirror using the closest data point as pivot
                    upper_mirror = 2 * closest_upper - data[:, dim]
                    mirrored_points = np.column_stack(
                        [upper_mirror if i == dim else data[:, i] for i in range(data.shape[1])]
                    )
                    mirrored_data = np.vstack([mirrored_data, mirrored_points])
                    updated_values = np.concatenate([updated_values, pdf_values])

                    mirrored_data, updated_values = sum_together_np(mirrored_data, updated_values)
                except Exception as e:
                    print(f"Failed to mirror upper boundary: {e}")
                    pass

    # Have to do a second pass. After the mirroring, drop values out of boundaries.
    for dim, boundary in enumerate(boundaries):
        if boundary is not None:
            lower, upper = boundary
            if lower is not None:
                try:
                    mask = mirrored_data[:, dim] >= lower
                    mirrored_data = mirrored_data[mask]
                    updated_values = updated_values[mask]
                except Exception as e:
                    print(f"Failed to drop lower boundary: {e}")
                    pass
            if upper is not None:
                try:
                    mask = mirrored_data[:, dim] <= upper
                    mirrored_data = mirrored_data[mask]
                    updated_values = updated_values[mask]
                except Exception as e:
                    print(f"Failed to drop upper boundary: {e}")
                    pass

    # Adjusting pdf to make the sum = 1
    updated_values = updated_values / updated_values.sum()

    return mirrored_data, updated_values
