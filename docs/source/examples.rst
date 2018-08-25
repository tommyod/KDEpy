Examples
=======

Minimal working example with options
------------------------------------

.. plot::
   :include-source:

    from KDEpy import FFTKDE
    data = np.random.randn(2**6)

    # Notice how bw (variance), kernel, weights and grid points are set
    x, y = FFTKDE(bw=1, kernel='gaussian').fit(data, weights=None).evaluate(2**8)

    plt.plot(x, y); plt.tight_layout()

Three kernels in 1D
-------------------

.. plot::
   :include-source:

    from KDEpy import FFTKDE
    data = np.exp(np.random.randn(2**6))  # Lognormal data

    for plt_num, kernel in enumerate(['box', 'biweight', 'gaussian'], 1):

        ax = fig.add_subplot(1, 3, plt_num)
        ax.set_title(f'Kernel: "{kernel}"')

        x, y = FFTKDE(kernel=kernel, bw='silverman').fit(data).evaluate()
        ax.plot(x, y)

    fig.tight_layout()



Weighted data
-------------

.. plot::
   :include-source:

    from KDEpy import FFTKDE
    data = np.random.randn(2**6)  # Normal distribution
    weights = data**2  # Large weights away from zero

    x, y = FFTKDE(bw='silverman').fit(data, weights).evaluate()
    plt.plot(x, y, label=r'Weighted away from zero $w_i = x_i^2$')

    y2 = FFTKDE(bw='silverman').fit(data).evaluate(x)
    plt.plot(x, y2, label=r'Unweighted $w_i = 1/N$')

    plt.title('Weighted and unweighted KDE')
    plt.tight_layout(); plt.legend(loc='best');



The effect of norms in 2D
-------------------------

Below a non-smooth kernel is chosen to reveal the effect of the choice of norm more clearly.

.. plot::
   :include-source:

    from KDEpy import FFTKDE

    # Create 2D data of shape (obs, dims)
    data = np.random.randn(2**4, 2)

    grid_points = 2**7  # Grid points in each dimension
    N = 16  # Number of contours

    for plt_num, norm in enumerate([1, 2, np.inf], 1):

        ax = fig.add_subplot(1, 3, plt_num)
        ax.set_title(f'Norm $p={norm}$')

        # Compute the kernel density estimate
        kde = FFTKDE(kernel='box', norm=norm)
        grid, points = kde.fit(data).evaluate(grid_points)

        # The grid is of shape (obs, dims), points are of shape (obs, 1)
        x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        z = points.reshape(grid_points, grid_points).T

        # Plot the kernel density estimate
        ax.contour(x, y, z, N, linewidths=0.8, colors='k')
        ax.contourf(x, y, z, N, cmap="RdBu_r")
        ax.plot(data[:, 0], data[:, 1], 'ok', ms=3)

    plt.tight_layout()
