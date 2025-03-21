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
    >>> # FIRST EXAMPLE (mirroring synthetic data):
    >>> from KDEpy.mirroring import mirror_data
    >>> from KDEpy.bw_selection import silvermans_rule
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> boundaries = [(0, 1), (0, 1)] # Both dimensions are bounded between 0 and 1
    >>> data = np.random.rand(15,2) # Generate random uniform data
    >>> kernel_std = [silvermans_rule(data[:, column].reshape(-1, 1)) for column in range(data.shape[1])]
    >>> synthetic_data_size = 1000
    >>> resampled_data = data[np.random.choice(data.shape[0], size=synthetic_data_size, replace=True)]
    >>> resampled_data = resampled_data + np.random.randn(synthetic_data_size, data.shape[1]) * kernel_std
    >>> mirrored_data, _ = mirror_data(resampled_data, boundaries, decimals=16)
    >>> # Mirrored the data with same boundaries as original data:
    >>> # SECOND EXAMPLE (mirroring KDE evaluation):
    >>> from KDEpy import FFTKDE
    >>> np.random.seed(42)
    >>> data = np.random.rand(15, 2)
    >>> grid_points = 100
    >>> boundaries = [[0, 1], (0, 1)] # 2-dimensional dataframe. Accepts tuple or list
    >>> X, Z = FFTKDE(bw=1).fit(data)((grid_points, grid_points)) # Fit KDE to data
    >>> X_mirrored, Z_mirrored = mirror_data(X, boundaries, Z) # Mirror KDE evaluation
    """

    def check_grid_and_width(data):
        """
        Check if the sequence of points in a multidimensional grid are living in a grid (equidistant)
        and find the width of the grid.

        Parameters
        ----------
        data : np.ndarray
            The input data array (N x D). N data points in D dimensions.

        Returns
        -------
        bool
            True if the points are living in a grid (equidistant) for each dimension, False otherwise.
        list
            A list of grid widths for each dimension if the points are equidistant, None otherwise.
        """
        # Initialize list to store grid widths for each dimension
        grid_widths = []

        # Iterate over each dimension (column)
        for dim in range(data.shape[1]):
            # Get unique values of the current column
            unique_values = np.unique(data[:, dim])

            # Calculate the differences between consecutive unique values
            differences = np.diff(unique_values)

            # Check if all differences are equal (equidistant)
            is_grid = np.all(np.isclose(differences, np.mean(differences), rtol=0.000001))

            # If the points are equidistant, store the width of the grid for the current dimension
            if is_grid:
                grid_widths.append(differences[0])  # Store the width of the grid for the current dimension
            else:
                return False, None

        return True, grid_widths

    def sum_together_np(mirrored_data, updated_values, decimals):
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

    # Check if dimension is 1 and reshape if necessary
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Check if the amount of rows of pdf_values matches the ones on data.
    assert pdf_values.shape[0] == data.shape[0], "PDF values must match the data size."

    # Check if the data and boundaries have the same dimensions
    assert data.shape[1] == len(boundaries), "Data dimensions must match the boundaries."

    # Check if the data is living in a grid (equidistant) for each dimension
    is_grid, grid_widths = check_grid_and_width(data)

    mirrored_data = data.copy()
    updated_values = pdf_values.copy()

    # Each boundary is 'folding' the data; creating a copy, adding the PDF values,
    # and then deleting the values which are beyond the boundary.
    for dim, boundary in enumerate(boundaries):
        if boundary is not None:
            lower, upper = boundary
            if lower is not None:
                try:
                    if is_grid:
                        closest_lower = np.max(
                            mirrored_data[mirrored_data[:, dim] <= lower, dim]
                        )  # Mirror using the closest outer data point as pivot
                    else:
                        closest_lower = lower
                    lower_mirror = 2 * closest_lower - mirrored_data[:, dim]
                    mirrored_points = np.column_stack(
                        [lower_mirror if i == dim else mirrored_data[:, i] for i in range(mirrored_data.shape[1])]
                    )
                    mirrored_data = np.vstack([mirrored_data, mirrored_points])
                    updated_values = np.concatenate([updated_values, updated_values])
                    mirrored_data, updated_values = sum_together_np(mirrored_data, updated_values, decimals)
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
                    if is_grid:
                        closest_upper = np.min(
                            mirrored_data[mirrored_data[:, dim] >= upper, dim]
                        )  # Mirror using the closest outer data point as pivot
                    else:
                        closest_upper = upper
                    upper_mirror = 2 * closest_upper - mirrored_data[:, dim]
                    mirrored_points = np.column_stack(
                        [upper_mirror if i == dim else mirrored_data[:, i] for i in range(mirrored_data.shape[1])]
                    )
                    mirrored_data = np.vstack([mirrored_data, mirrored_points])
                    updated_values = np.concatenate([updated_values, updated_values])
                    mirrored_data, updated_values = sum_together_np(mirrored_data, updated_values, decimals)
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
    # Adjusting pdf according to bin width if the data is in a grid
    if is_grid:
        updated_values = updated_values / np.prod(grid_widths)

    return mirrored_data, updated_values


if __name__ == "__main__":
    import subprocess
    import sys
    import os

    def run_command(command):
        result = subprocess.run(command, capture_output=True, text=True)
        print(f"{command} Output:")
        print(result.stdout)
        print(result.stderr)

    # Get the path to the virtual environment's Python executable
    venv_python = (
        os.path.join(sys.prefix, "bin", "python")
        if os.name != "nt"
        else os.path.join(sys.prefix, "Scripts", "python.exe")
    )

    # Run black check
    run_command([venv_python, "-m", "black", "KDEpy", "-l", "120"])

    # Run flake8 check
    run_command(
        [
            venv_python,
            "-m",
            "flake8",
            "--show-source",
            "--ignore=F811,W293,W391,W292,W291,W504,W503,E231",
            "--max-line-length=120",
            "--exclude=*examples.py,testing.py,*kde.py",
            "KDEpy",
        ]
    )

    # Run pytest
    run_command([venv_python, "-m", "pytest", "KDEpy", "--doctest-modules", "--capture=sys"])
