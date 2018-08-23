
Examples
=======

In 2D
-----

You are most likely here because you wonder what kernels are available.
Every available kernel is shown in the figure below.
Kernels with bounded support are annotated with **B**.

.. plot::
   :include-source:

    from KDEpy import FFTKDE

    # Create 2D data of shape (obs, dims)
    data = np.concatenate((np.random.randn(16).reshape(-1, 1),
                           np.random.randn(16).reshape(-1, 1)), axis=1)


    grid_points = 2**5  # Grid points in each dimension
    N = 16  # Number of contours

    for plt_num, norm in enumerate([1, 2, np.inf], 1):

        ax = fig.add_subplot(1, 3, plt_num)
        ax.set_title(f'Norm $p={norm}$')

        # Compute the kernel density estimate
        kde = FFTKDE(kernel='gaussian', norm=norm)
        grid, points = kde.fit(data).evaluate(grid_points)

        # The grid is of shape (obs, dims), points are of shape (obs, 1)
        x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        z = points.reshape(grid_points, grid_points).T

        # Plot the kernel density estimate
        ax.contour(x, y, z, N, linewidths=0.8, colors='k')
        ax.contourf(x, y, z, N, cmap="RdBu_r")
        ax.plot(data[:, 0], data[:, 1], 'ok', ms=3)

    plt.tight_layout()
