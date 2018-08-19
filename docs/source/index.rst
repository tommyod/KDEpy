.. KDEpy documentation master file, created by
   sphinx-quickstart on Sat Mar 31 09:23:26 2018.

KDEpy
=====

.. currentmodule:: KDEpy

This Python package implements various Kernel Density Esimators (KDE).
The long-term goal is to support state-of-the-art KDE algorithms, and eventually have the most complete implementation in the scientific Python universe.
As of now, three algorithms are implemented through the same API: :class:`~KDEpy.NaiveKDE.NaiveKDE`, :class:`~KDEpy.TreeKDE.TreeKDE` and :class:`~KDEpy.FFTKDE.FFTKDE`.

.. image:: _static/img/example3.png
   :target: #

The code generating the above graph is found in `KDEpy/examples.py <https://github.com/tommyod/KDEpy/blob/master/KDEpy/examples.py>`_.

Installation
------------

KDEpy is available through `PyPI <https://pypi.org/project/KDEpy/>`_, and may be installed using ``pip``: ::

   $ pip install KDEpy


Example
-----------------------

Here's an example showing the usage of :class:`~KDEpy.FFTKDE.FFTKDE`.
Notice how the *kernel* and *bandwidth* are set, and how *weights* may be passed.
The other classes share this common API, and are used the same way.

.. plot::
   :include-source:

   from KDEpy import FFTKDE
   from scipy.stats import norm

   # Generate a distribution and 2**6 data points
   dist = norm(loc=0, scale=1)
   data = dist.rvs(2**6)

   # Compute kernel density estimate on a grid using Silverman's rule for bw
   x, y1 = FFTKDE(bw='silverman').fit(data)()

   # Compute a weighted estimate on the same grid, using verbose API
   estimator = FFTKDE(kernel='biweight', bw=0.8)
   y2 = estimator.fit(data, weights=np.arange(len(data)) + 1).evaluate(x)

   plt.plot(x, y1, label='KDE estimate with defaults')
   plt.plot(x, y2, label='KDE estimate with verbose API')
   plt.plot(x, dist.pdf(x), label='True distribution')
   plt.grid(True, ls='--', zorder=-15); plt.legend()


Navigation
----------

.. toctree::
   :maxdepth: 1

   index.rst
   introduction.ipynb
   bandwidth.rst
   examples.rst
   literature.rst
   API.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
