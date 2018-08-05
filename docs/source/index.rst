.. KDEpy documentation master file, created by
   sphinx-quickstart on Sat Mar 31 09:23:26 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

KDEpy
=====

KDEpy is a Python library which implements kernel density estimators.

Highlights
----------

KDEpy is a Python library for kernel density estimation.

Currently, the following features are implemented:

* **Many kernels**: There are 10 kernel functions implemented.
* **Data weights**: Individual data points might be weighed.
* **Variable bandwidth**: The kernel bandwidth :math:`h` may be adjusted.



Minimal working example
-----------------------

Here's a minimal working example::

    >>> from KDEpy import NaiveKDE
    >>> import numpy as np
    >>> from scipy.stats import norm
    >>> data = norm(loc=0, scale=1).rvs(100) # Generate 100 points
    >>> x, y = NaiveKDE(kernel='gaussian', bw=0.5).fit(data).evaluate()

.. image:: _static/img/minimal_working_example.png
   :width: 400 px
   :target: #

It's really that simple.



.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from KDEpy import NaiveKDE
   from scipy.stats import norm

   # Generate a distribution and some data
   np.random.seed(42)
   dist = norm(loc=0, scale=1)
   data = dist.rvs(100) # Generate 100 points

   # Compute kernel density estimate on a grid
   x, y = NaiveKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()

   plt.figure(figsize=(7, 3))
   plt.plot(x, y, label='KDE estimate')
   plt.plot(x, dist.pdf(x), label='True distribution')
   plt.grid(True, ls='--', zorder=-15); plt.legend(); plt.show()


Navigation
----------

.. toctree::
   :maxdepth: 1

   index.rst
   intro_kde.rst
   bandwidth.rst
   kernels.rst
   examples.rst
   notebook.ipynb
   literature.rst
   API.rst


Contribute
----------

You are very welcome to contribute.
To do so, please go to GitHub.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
