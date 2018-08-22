Comparison
==========

In this section we compare s

.. note::

   Times will vary depending on computer.

Features and timings
--------------------

+----------------------------------+--------------+----------+-------------+--------------+
| Feature / Library                | scipy        | sklearn  | statsmodels | KDEpy.FFTKDE |
+==================================+==============+==========+=============+==============+
| Number of kernels                | 1 (gauss)    | 6        | 7 (6 slow)  |      9       |
+----------------------------------+--------------+----------+-------------+--------------+
| Number of dimensions             | Any          | Any      |  Any        |     Any      |
+----------------------------------+--------------+----------+-------------+--------------+
| Weighted data points             | No           | No       |     No      |     Yes      |
+----------------------------------+--------------+----------+-------------+--------------+
| Automatic bandwidth              | Normal rule  | ...      |             |              |
+----------------------------------+--------------+----------+-------------+--------------+
| Time  :math:`N=10^6`             | 42 s         | 22 s     |   0.27 s    |    0.01 s    |
+----------------------------------+--------------+----------+-------------+--------------+
| Time  :math:`N=10^2 \times 10^2` | 0.5 s        | 1.6 s    |   1.3 s     |    0.005 s   |
+----------------------------------+--------------+----------+-------------+--------------+

The times for the 1D :math:`N = 10^6` data points were computed taking the median of 5 runs.
The kernel was Gaussian and the number of grid points were :math:`n=2^{10}`.


Speed in 1D
-----------

We run the algorithms 20 times on random data, and compare the medians of the running times.
The plot below compares the speed of the implementations with a **Gaussian kernel**.
The 1D statsmodels implementation uses a similar algorithm when the kernel is Gaussian, and the performance is somewhat comparable.

.. image:: _static/img/profiling_1D_gauss.png
   :scale: 100 %
   :align: center

Switching to the **Epanechnikov kernel** (scipy falls back to Gaussian, since it only implements this kernel) the picture is very different.
Statsmodels now uses a naive computation, it's speed is no longer comparable.

.. image:: _static/img/profiling_1D_epa.png
   :scale: 100 %
   :align: center


Speed in 2D
-----------

Here are the result in 2D.

.. image:: _static/img/profiling_2D_gauss.png
   :scale: 100 %
   :align: center
