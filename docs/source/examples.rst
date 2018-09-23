Examples
=======

Minimal working example with options
------------------------------------

This *minimal working example* shows how to compute a KDE in one line of code.

.. plot::
   :include-source:

    from KDEpy import FFTKDE
    data = np.random.randn(2**6)

    # Notice how bw (variance), kernel, weights and grid points are set
    x, y = FFTKDE(bw=1, kernel='gaussian').fit(data, weights=None).evaluate(2**8)

    plt.plot(x, y); plt.tight_layout()

Three kernels in 1D
-------------------

This example shows the effect of different kernel functions :math:`K`.

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

A *weight* :math:`w_i` may be associated with every data point :math:`x_i`.

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


Multimodal distributions
------------------------

The *Improved Sheather Jones* (ISJ) algorithm for automatic bandwidth selection is implemented in KDEpy.
It does not assume normality, and is **robust to multimodal distributions**.
The disadvantage is that it requires more data to make accurate assessments, and that the running time is slower.

.. plot::
   :include-source:

    from scipy import stats
    from KDEpy import FFTKDE

    # Create a bimodal distribution from two Gaussians and draw data
    dist1 = stats.norm(loc=0, scale=1)
    dist2 = stats.norm(loc=20, scale=1)
    data = np.hstack([dist1.rvs(10**3), dist2.rvs(10**3)])

    # Plot the true distribution and KDE using Silverman's Rule
    x, y = FFTKDE(bw='silverman').fit(data)()
    plt.plot(x, (dist1.pdf(x) + dist2.pdf(x)) / 2, label='True distribution')
    plt.plot(x, y, label="FFTKDE with Silverman's rule")

    # KDE using ISJ - robust to multimodality, but needs more data
    y = FFTKDE(bw='ISJ').fit(data)(x)
    plt.plot(x, y, label="FFTKDE with Improved Sheather Jones (ISJ)")

    plt.title('Silverman vs. Improved Sheather Jones')
    plt.tight_layout(); plt.legend(loc='best');


Boundary correction using mirroring
-----------------------------------

If the domain is bounded (e.g. :math:`\mathbb{R}_+`) and you expect observations to fall near the boundary, a KDE might put density outside of the domain.
Mirroring the data about the boundary is an elementary way to reduce this unfortunate effect.
If :math:`\hat{g}(x)` is the original KDE, then :math:`\hat{g}_*(x)=\hat{g}(x-2a)` is the KDE obtained when mirroring the data about :math:`x=a`.
Note that at the boundary :math:`a`, the derivative of the final estimate :math:`\hat{f}(x)` is zero, since

.. math::

   \hat{f}'(a) = \hat{g}'(x) + \hat{g}_*'(x)  \bigr |_a  = \hat{g}'(x) - \hat{g}'(2a - x)  \bigr |_a = \hat{g}'(a) - \hat{g}'(a) = 0,

where the change of sign is due to the chain rule of calculus.
The reduction of boundary bias and the fact that the derivative is zero is demonstrated graphically in the example below.

.. plot::
   :include-source:

    from scipy import stats
    from KDEpy import FFTKDE

    # Beta distribution, where x=1 is a hard lower limit
    dist = stats.beta(a=1.05, b=3, loc=1, scale=10)

    data = dist.rvs(10**2)
    kde = FFTKDE(bw='silverman', kernel='triweight')
    x, y = kde.fit(data)(2**10)  # Two-step proceudure to get bw
    plt.plot(x, dist.pdf(x), label='True distribution')
    plt.plot(x, y, label='FFTKDE')
    plt.scatter(data, np.zeros_like(data), marker='|')

    # Mirror the data about the domain boundary
    low_bound = 1
    data = np.concatenate((data, 2 * low_bound - data))

    # Compute KDE using the bandwidth found, and twice as many grid points
    x, y = FFTKDE(bw=kde.bw, kernel='triweight').fit(data)(2**11)
    y[x<=low_bound] = 0  # Set the KDE to zero outside of the domain
    y = y * 2  # Double the y-values to get integral of ~1

    plt.plot(x, y, label='Mirrored FFTKDE')
    plt.title('Mirroring data to help overcome boundary bias')
    plt.tight_layout(); plt.legend();


Estimating density on the circle
--------------------------------

If the data is bounded on a circle and the domain is known, the data can be *repeated* instead of *reflected*.
The result of this is shown graphically below.
The derivative of :math:`\hat{f}(x)` at the lower and upper boundary will have the same value.

.. plot::
   :include-source:

    from scipy import stats
    from KDEpy import FFTKDE

    # The Von Mises distribution - normal distribution on a circle
    dist = stats.vonmises(kappa=0.5)
    data = dist.rvs(10**2)

    # Plot the normal KDE and the true density
    kde = FFTKDE(bw='silverman', kernel='triweight')
    x, y = kde.fit(data).evaluate()
    plt.plot(x, dist.pdf(x), label='True distribution')
    plt.plot(x, y, label='FFTKDE')
    plt.xlim([np.min(x), np.max(x)])

    # Repeat the data and fit a KDE to adjust for boundary effects
    a, b = (-np.pi, np.pi)
    data = np.concatenate((data - (b - a), data, data + (b - a)))
    x, y = FFTKDE(bw=kde.bw, kernel='biweight').fit(data).evaluate()
    y = y * 3  # Multiply by three since we tripled data observations

    plt.plot(x, y, label='Repeated FFTKDE')
    plt.plot([a, a], list(plt.ylim()), '--k', label='Domain lower bound')
    plt.plot([b, b], list(plt.ylim()), '--k', label='Domain upper bound')
    plt.tight_layout(); plt.legend();



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

.. comment:
  Kernel regression via KDE
  -------------------------

  Here's how a weighted KDE can be used for 1D kernel regression.
  Beware of boundary effects--the estimate will fall to zero.


  .. plot::
     :include-source:

      from scipy.integrate import trapz
      from KDEpy import FFTKDE

      N = 2**6
      # Sampe the function on equidistant points
      x = np.linspace(0, 25, num=N)
      y = np.sin(x/3) + np.random.randn(N) / 6

      # Compute the area (integral), used to normalize later on
      area = trapz(y, x)

      plt.scatter(x, y, marker='x', alpha=0.5, label='Noisy samples')
      plt.plot(x, np.sin(x/3), label='True function')

      # Weight data by y-values, normalize using the area
      x, y = FFTKDE(bw=0.5).fit(x, weights=y).evaluate()
      plt.plot(x, y * area, label='Kernel regression')

      plt.title('Kernel regression via KDE')
      plt.tight_layout(); plt.legend(loc='best');
