
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



sdf


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
