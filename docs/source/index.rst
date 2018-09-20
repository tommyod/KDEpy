KDEpy
=====

.. currentmodule:: KDEpy

This Python 3.5+ package implements various Kernel Density Estimators (KDE).
Three algorithms are implemented through the same API: :class:`~KDEpy.NaiveKDE.NaiveKDE`, :class:`~KDEpy.TreeKDE.TreeKDE` and :class:`~KDEpy.FFTKDE.FFTKDE`.
The :class:`~KDEpy.FFTKDE.FFTKDE` outperforms other popular implementations, see the `comparison page <comparison.rst>`_.

.. note:: KDEpy is relatively stable, and the plan is to finish active development by the end of 2018.
   If you have feedback, please report an `Issue <https://github.com/tommyod/KDEpy/issues>`_ on GitHub.
   Contributions to code and documentation is welcome too.


.. image:: _static/img/showcase.png
   :target: #

The code generating the above graph is found in `KDEpy/examples.py <https://github.com/tommyod/KDEpy/blob/master/KDEpy/examples.py>`_.

Installation
------------

KDEpy is available through `PyPI <https://pypi.org/project/KDEpy/>`_, and may be installed using ``pip``: ::

   $ pip install KDEpy


Example
-----------------------

Here's an example showing the usage of :class:`~KDEpy.FFTKDE.FFTKDE`, the fastest algorithm implemented.
Notice how the *kernel* and *bandwidth* are set, and how the *weights* argument is used.
The other classes share this common API of instantiating, fitting and finally evaluating.

.. plot::
   :include-source:

   from KDEpy import FFTKDE
   from scipy.stats import norm

   # Generate a distribution and draw 2**6 data points
   dist = norm(loc=0, scale=1)
   data = dist.rvs(2**6)

   # Compute kernel density estimate on a grid using Silverman's rule for bw
   x, y1 = FFTKDE().fit(data)(2**10)

   # Compute a weighted estimate on the same grid, using verbose API
   weights = np.arange(len(data)) + 1
   estimator = FFTKDE(kernel='biweight', bw='silverman')
   y2 = estimator.fit(data, weights=weights).evaluate(x)

   plt.plot(x, y1, label='KDE estimate with defaults')
   plt.plot(x, y2, label='KDE estimate with verbose API')
   plt.plot(x, dist.pdf(x), label='True distribution')
   plt.grid(True, ls='--', zorder=-15); plt.legend()


Table of contents
-----------------

.. toctree::
   :maxdepth: 2

   introduction.ipynb
   comparison.rst
   bandwidth.rst
   kernels.rst
   examples.rst
   literature.rst
   API.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
