
What is kernel density estimation?
==================================

Testing
-------


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
