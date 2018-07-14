
Kernels
=======

Basic properties
----------------

We will impose some requirements on the kernel functions :math:`K(x)`.

* Normalization: :math:`\int K(x) \, dx = 1`.
* Unit variance: :math:`\operatorname{Var}[K(x)] = 1` when the bandwidth :math:`h` is 1.
* Symmetry: :math:`K(-x) = K(x)` for every :math:`x`.

Furthermore, a kernel may have have bounded support or not.
A kernel with compact (or bounded) support is defined on a domain such as :math:`[-1, 1]`,
while a non-compact kernel is defined on :math:`[-\infty, \infty]`.

Below we plot the *Guassian kernel* and the *Epanechnikov kernel*.

* The Gaussian kernel is non-compact.
* The Epanechnikov is compact.

Both kernels have unit variance.

.. plot::
   :include-source:

   from KDEpy import *

   plt.figure(figsize=(7, 3))
   x, y1 = NaiveKDE(kernel='gaussian', bw=1).fit([0]).evaluate()
   y2 = NaiveKDE(kernel='epa', bw=1).fit([0]).evaluate(x)
   plt.plot(x, y1, label='Gaussian kernel')
   plt.plot(x, y2, label='Epanechnikov kernel')
   plt.grid(True, ls='--', zorder=-15); plt.legend(); plt.show()


Higher dimensional kernels
--------------------------

The one-dimensional example is deceptively simple, since every :math:`p`-norm is the same.
In general, the :math:`p`-norm is a measure of distance, defined by

.. math::

   \left\| x \right\| _p = \bigg( \sum_{i=1}^d \left| x_i \right| ^p \bigg) ^{1/p}.

Consider :math:`p=2` and :math:`p=\infty`, corresponding to the familiar Euclidean norm
:math:`\left\| x \right\| _2 = \sqrt{x_1^2 + x_2^2 + \dots + x_d^2}` and the maximum norm
:math:`\left\| x \right\| _\infty = \max_{i} \left| x_i \right|`.

.. plot::
   :include-source:

   from KDEpy.BaseKDE import BaseKDE
   from mpl_toolkits.mplot3d import Axes3D
   import numpy as np

   kernel = BaseKDE._available_kernels['tri']

   n = 64
   p = np.linspace(-3, 3, num=n)
   obs_x_dims = np.array(np.meshgrid(p, p)).T.reshape(-1, 2)

   fig = plt.figure(figsize=(7, 3))
   ax = fig.add_subplot(1, 2, 1, projection='3d')
   z = kernel(obs_x_dims, norm=np.inf).reshape((n, n))
   surf = ax.plot_surface(*np.meshgrid(p, p), z)
   ax.set_title('Using the $\max$-norm')

   ax = fig.add_subplot(1, 2, 2, projection='3d')
   z = kernel(obs_x_dims, norm=2).reshape((n, n))
   surf = ax.plot_surface(*np.meshgrid(p, p), z)
   ax.set_title('Using the $2$-norm')

Euclidean normalization
~~~~~~~~~~~~~~~~~~~~~~~

We would like to normalize the kernel functions in higher dimensions for the
most common norms. To accomplish this, we start with the equation for the volume
of a :math:`d`-dimensional hypersphere. The equation is

.. math::

   V_d = \int_0^1 S_{d-1} r^{d-1} \, dr,

where :math:`r` is a positive radius and :math:`S_{d-1}` is a constant which
depends on the dimension. Integrating, we find that :math:`d \cdot V_d = S_{d-1}`,
and in the case of a unit hypersphere

.. math::

   V_d = \frac{\pi^{d/2}}{\Gamma\left ( \frac{d}{2} + 1 \right )}.

This allows us to compute normalization coefficients. For instance, the
linear kernel is normalized by

.. math::

   \int_0^1 \left ( 1 - r \right )S_{d-1} r^{d-1} \, dr = V_d - \frac{S_{d-1}}{d+1} = V_d - \frac{d \cdot V_d}{d+1} = V_d \left ( \frac{1}{d+1} \right )

and the biweight kernel is similarily normalized by

.. math::

   \int_0^1 \left ( 1 - r^2 \right )^2 S_{d-1} r^{d-1} \, dr = V_d \left ( 1 - \frac{2n}{n+2} + \frac{n}{n+4} \right ) = V_d \left ( \frac{8}{(n+2)(n+4)} \right )


Max-norm normalization
~~~~~~~~~~~~~~~~~~~~~~

In this case, :math:`V_d = 2^d` and :math:`S_{d-1} = d \cdot 2^d = d \cdot V_d`.
The constants are similar, except that :math:`V_d` has a simpler form than in
the Euclidean case.






.. currentmodule:: KDEpy.kde

:class:`KDE` will take in its ``fit`` method arrays X, y
and will store the coefficients :math:`w` of the linear model in its
``coef_`` member.

Histograms
----------

By centering histograms bins, the data controls the estimate.

.. plot::
   :include-source:

   from KDEpy import *
   from scipy import stats

   # Generate a distribution and some data
   dist = stats.norm(loc=0, scale=1)
   data = dist.rvs(32)

   # Compute kernel density estimate on a grid
   plt.figure(figsize=(7, 3))

   x, y = NaiveKDE(kernel='box', bw='silverman').fit(data).evaluate()
   plt.plot(x, y, label='KDE estimate')
   plt.scatter(data, np.zeros_like(data), marker='x', label='Data', color='k')
   plt.plot(x, dist.pdf(x), ls='--', label='True distribution')
   plt.grid(True, ls='--', zorder=-15); plt.legend(); plt.show()


Fixed width kernels
-------------------

.. plot::
   :include-source:

   from KDEpy import *
   from scipy import stats

   # Generate a distribution and some data
   dist = stats.norm(loc=0, scale=1)
   data = dist.rvs(16)

   plt.figure(figsize=(7, 3))

   # Kernel density estimate with too small bandwidth
   x, y = NaiveKDE(bw=0.1).fit(data).evaluate()
   plt.plot(x, y, label='KDE estimate')

   # Kernel density estimate with too large bandwidth
   x, y = NaiveKDE(bw=2).fit(data).evaluate()
   plt.plot(x, y, label='KDE estimate')

   plt.plot(x, dist.pdf(x), ls='--', label='True distribution')
   plt.scatter(data, np.zeros_like(data), marker='x', label='Data', color='k')
   plt.grid(True, ls='--', zorder=-15); plt.legend(); plt.show()


Variable width kernels
----------------------

.. plot::
   :include-source:

   from KDEpy import *
   from scipy import stats

   # Generate a distribution and some data
   dist = stats.lognorm(s=1)
   data = dist.rvs(160)

   plt.figure(figsize=(7, 3))

   # Kernel density estimate with too small bandwidth
   x, y = NaiveKDE(bw=data).fit(data).evaluate()
   plt.plot(x, y, label='KDE estimate')

   plt.plot(x, dist.pdf(x), ls='--', label='True distribution')
   plt.scatter(data, np.zeros_like(data), marker='x', label='Data', color='k')
   plt.grid(True, ls='--', zorder=-15); plt.legend(); plt.show()


.. topic:: References

    * "Notes on Regularized Least Squares", Rifkin & Lippert (`technical report
      <http://cbcl.mit.edu/projects/cbcl/publications/ps/MIT-CSAIL-TR-2007-025.pdf>`_,
      `course slides
      <http://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf>`_).
