Introduction
============

.. currentmodule:: KDEpy.kde

- what is it
- why use it
- histograms
- centered histograms
- the KDE
- choices
- computation and implementations

A Kernel Density Estimator (KDE) may be thought of as an extension to the familiar histogram.
The purpose of the KDE is to estimate an unknown probability density function given points drawn from it.
A natural first thought is to use a histogram -- it's well known, simple to understand and works reasonably well.


The histogram
-------------

To see how the histogram performs, we'll generate some data from a normal distribution and plot it alongside the histogram.
As seen below, the histogram does a fairly poor job.
The location of the bins and the number of bins both seem arbitrary, and the estimated distribution is discontinuous.

.. plot::
  :include-source:

  from scipy import stats

  # Generate a distribution and some data
  dist = stats.norm(loc=0, scale=1)
  data = dist.rvs(2**4)

  plt.hist(data, bins='auto', density=True, edgecolor='k', zorder=10)
  plt.scatter(data, np.zeros_like(data), marker='x', color='red', zorder=15)
  x = np.linspace(-3, 3)
  plt.plot(x, dist.pdf(x), ls='--', color='red', zorder=20)
  plt.grid(True, ls='--', zorder=-15);

Centering the histogram
-----------------------

In an effort to reduce the arbitrary placement of the histogram bins, we center a "box" on each data point and sum those boxes to obtain a distribution.
This is a kernel density estimate, with a `rectangular function <https://en.wikipedia.org/wiki/Rectangular_function>`_ as the kernel.
The method is more data driven and less arbitrary than the histogram, but two problems still remain -- the result is discontinuous, and the choice of the *bandwidth* (the width of the rectangular kernel function) must be chosen carefully.

.. plot::
  :include-source:

  from scipy import stats
  from KDEpy import TreeKDE

  dist = stats.norm(loc=0, scale=1)
  data = dist.rvs(2**4)

  x, y = TreeKDE(kernel='box', bw=1).fit(data).evaluate()
  plt.plot(x, y, zorder=10)
  plt.scatter(data, np.zeros_like(data), marker='x', color='red', zorder=15)
  plt.plot(x, dist.pdf(x), ls='--', color='red', zorder=20)
  plt.grid(True, ls='--', zorder=-15);


Choosing a smooth kernel
------------------------

To alleviate the problem of discontinuity, we substitute the rectangular function used above for a `gaussian function <https://en.wikipedia.org/wiki/Gaussian_function>`_.
The gaussian is smooth, and so the result of our estimate will also be smooth.

.. plot::
  :include-source:

  from scipy import stats
  from KDEpy import TreeKDE

  dist = stats.norm(loc=0, scale=1)
  data = dist.rvs(2**4)

  x, y = TreeKDE(kernel='gaussian', bw=1).fit(data).evaluate()
  plt.plot(x, y, zorder=10)
  plt.scatter(data, np.zeros_like(data), marker='x', color='red', zorder=15)
  plt.plot(x, dist.pdf(x), ls='--', color='red', zorder=20)
  plt.grid(True, ls='--', zorder=-15);


Selecting a good bandwidth
--------------------------

.. plot::
  :include-source:

  from scipy import stats
  from KDEpy import TreeKDE

  dist = stats.norm(loc=0, scale=1)
  data = dist.rvs(2**4)

  x, y = TreeKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()
  plt.plot(x, y, zorder=10)
  plt.scatter(data, np.zeros_like(data), marker='x', color='red', zorder=15)
  plt.plot(x, dist.pdf(x), ls='--', color='red', zorder=20)
  plt.grid(True, ls='--', zorder=-15);



Methods of computation
----------------------

.. plot::
  :include-source:

  from scipy import stats
  from KDEpy import TreeKDE, FFTKDE

  dist = stats.norm(loc=0, scale=1)
  data = dist.rvs(2**4)

  x, y = TreeKDE(kernel='gaussian', bw='silverman').fit(data)()
  plt.plot(x, y, zorder=10, lw=5, label='TreeKDE')

  y = FFTKDE(kernel='gaussian', bw='silverman').fit(data).evaluate(x)
  plt.plot(x, y, zorder=10, lw=2, label='FFTKDE')

  plt.scatter(data, np.zeros_like(data), marker='x', color='red', zorder=15)
  plt.plot(x, dist.pdf(x), ls='--', color='red', zorder=20)
  plt.grid(True, ls='--', zorder=-15); plt.legend();


Extensions to the problem
-------------------------

sdf
