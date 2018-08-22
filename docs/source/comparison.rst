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



The choice of bandwidth is more important than the choice of kernel function.
Consider a kernel density estimator based on :math:`N` points, weighting the
data points :math:`X_i` with weights :math:`w_i`.

.. math::

   \widehat{f}(x) = h^{-D} \sum_{i=1}^{N} w_i K \left( \frac{\left \| x - X_i \right \|_p}{h} \right)

We extend by diving by a scaling factor :math:`h` called the *bandwidth*.
When diving with :math:`h`, every dimension :math:`D` is stretched, so we must
re-scale with :math:`h^D` so that the integral of :math:`\widehat{f}(x)`
evaluates to unity.

.. math::

   \widehat{f}(x) = h^{-D} \sum_{i=1}^{N} w_i K \left( \frac{\left \| x - X_i \right \|_p}{h} \right)

We will now briefly explain two ways to find a good :math:`h`.

Normal reference rules
----------------------

If the data is unimodal and close to normal, *silverman's rule of thumb* or
*scott's rule of thumb* may be used. They are computationally very fast.

An example using ``bw='silverman'`` is shown in the code snippet below.

.. plot::
   :include-source:

   from KDEpy import FFTKDE
   from scipy.stats import norm

   # Generate a distribution and some data
   dist = norm(loc=0, scale=1)
   data = dist.rvs(2**8) # Generate 2**8 points

   # Compute density estimates using 'silverman' and 'scott'
   x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()
   plt.plot(x, y, label='KDE estimate /w silverman')

   plt.plot(x, dist.pdf(x), label='True distribution')
   plt.grid(True, ls='--', zorder=-15); plt.legend();

The Improved Sheather-Jones algorithm
-------------------------------------

The *Improved Sheather-Jones algorithm* (ISJ) is a *plug-in selector*.
The mean integrated square error (MISE) is given by

.. math::

  \operatorname{MISE} (h) = \mathbb{E}\int \left ( \hat{f}(x;h) - f(x) \right )^2 \, dx.

The ISJ algorithm attempts to find :math:`h` to minimize the asymptotic mean
integrated square error (AMISE), which depends on the unknown quantity :math:`\left \| f''(x) \right \|^2`.
Using a recursive formula, this is accomplished by computing the following
sequence of estimates.

.. math::

   \dots \to _*\hat{t}_{\ell +1} \to \left \| f^{(\ell + 1)} \right \|^2 \to
   _*\hat{t}_{\ell } \to \left \| f^{(\ell )} \right \|^2 \to \dots

The details are found in [1]

.. topic:: References

    *  "Kernel density estimation via diffusion", Botev, Z.I.; Grotowski, J.F.; Kroese, D.P. (2010).
    `arXiv <https://arxiv.org/abs/1011.2602>`_.

    `Link text <https://domain.invalid/>`_




asdfasdfasdfsdfsd
-----------------

You are most likely here because you wonder what kernels are available.
Every available kernel is shown in the figure below.
Kernels with bounded support are annotated with **B**.

.. plot::
   :include-source:

   from KDEpy import *


   for name, func in NaiveKDE._available_kernels.items():
      x, y = NaiveKDE(kernel=name).fit([0]).evaluate()
      plt.plot(x, y, label=name + (' (B)' if func.finite_support else ''))

   plt.grid(True, ls='--', zorder=-15); plt.legend();

Basic properties
----------------

Let us discuss the basic properties of kernel functions.
We will impose the following requirements on the kernel functions :math:`K(x)`.

* Normalization: :math:`\int K(x) \, dx = 1`.
* Unit variance: :math:`\operatorname{Var}[K(x)] = 1` when the bandwidth :math:`h` is 1.
* Symmetry: :math:`K(-x) = K(x)` for every :math:`x`.

Furthermore, a kernel may have have bounded support or not.
A kernel with bounded (or compact) support is defined on a domain such as :math:`[-1, 1]`,
while a non-bounded kernel is defined on :math:`[-\infty, \infty]`.

Below we plot the *Guassian kernel* and the *Epanechnikov kernel*.

* The Gaussian kernel is not bounded.
* The Epanechnikov is bounded.

The reason why kernels are normalized to unit variance is so bounded and non-bounded
kernel functions are more easy compared.

.. plot::
   :include-source:

   from KDEpy import *

   x, y1 = NaiveKDE(kernel='gaussian', bw=1).fit([0]).evaluate()
   y2 = NaiveKDE(kernel='epa', bw=1).fit([0]).evaluate(x)
   plt.plot(x, y1, label='Gaussian kernel')
   plt.plot(x, y2, label='Epanechnikov kernel')
   plt.grid(True, ls='--', zorder=-15); plt.legend();


Higher dimensional kernels
--------------------------

The one-dimensional example is deceptively simple, since in one dimension every
:math:`p`-norm is equivalent. In higher dimensions, this is not true.
The general :math:`p`-norm is a measure of distance in :math:`\mathbb{R}^d`,
defined by

.. math::

   \left\| x \right\| _p := \bigg( \sum_{i=1} \left| x_i \right| ^p \bigg) ^{1/p}.

The three most common :math:`p`-norms are

* The Manhattan norm :math:`\left\| x \right\| _1 = \sum_{i} \left| x_i \right|`
* The Euclidean norm :math:`\left\| x \right\| _2 = \sqrt{x_1^2 + x_2^2 + \dots + x_d^2}`
* The max-norm :math:`\left\| x \right\| _\infty = \max_{i} \left| x_i \right|`

In higher dimensions, a norm must be chosen in addition to a kernel.
Let :math:`r := \left\| x \right\| _p` be a general radius, then a kernel
function is a function such that :math:`\partial_r K(r) < 0`.
Normalization is still necessary, but symmetry is guaranteed since
:math:`\left\| -x \right\| _p = \left\| x \right\| _p`.
The figure below shows the effect of choosing different norms with the same kernel.


.. plot::
   :include-source:

   from KDEpy.BaseKDE import BaseKDE
   from mpl_toolkits.mplot3d import Axes3D

   kernel = BaseKDE._available_kernels['tri']

   n = 64
   p = np.linspace(-3, 3, num=n)
   obs_x_dims = np.array(np.meshgrid(p, p)).T.reshape(-1, 2)

   ax = fig.add_subplot(1, 2, 1, projection='3d')
   z = kernel(obs_x_dims, norm=np.inf).reshape((n, n))
   surf = ax.plot_surface(*np.meshgrid(p, p), z)
   ax.set_title('Using the $\max$-norm')

   ax = fig.add_subplot(1, 2, 2, projection='3d')
   z = kernel(obs_x_dims, norm=2).reshape((n, n))
   surf = ax.plot_surface(*np.meshgrid(p, p), z)
   ax.set_title('Using the $2$-norm')


Kernel normalization
~~~~~~~~~~~~~~~~~~~~

Kernels are normalized by the software when :math:`p \in \{1, 2, \infty \}`.
For other choices of :math:`p`, the kernels are not normalized.
To explain how a high-dimensional kernel is normalized, we first examine
volumes in high dimension. The :math:`d`-dimensional volume :math:`V_d(r)` is
proportional to :math:`r^d`, where :math:`r` is the distance from the origin
in a norm. We will now examine the unit :math:`d`-dimensional
volume :math:`V_d := V_d(1)`.

In general, we integrate over the :math:`V_{d-1}(r)` to obtain :math:`V_{d}` using

.. math::

   V_d = \int_0^1 V_{d-1}(r) \, dr.

Since :math:`V_{d-1}(r) \propto r^{d-1}`, we write it as :math:`V_{d-1}(r) = K(d-1) r^{d-1}`,
where :math:`K(d-1)` is a constant. Pulling this out of the integral, we are left with

.. math::

   V_d = K(d-1) \int_0^1 r^{d-1} \, dr.

Furthermore, since :math:`V_{d-1}(1) = K(d-1)`, we see that :math:`K(d-1) = V_{d-1}`.
In summary, if we know the unit volume is given by

.. math::

   V_d = V_{d-1} \int_0^1 r^{d-1} \, dr.

Integrating this relationship gives :math:`V_{d-1} = V_{d} \cdot d`.
The following table shows :math:`V_d` for arbitrary dimensions :math:`d` for common norms.

.. table:: High dimensional volumes
   :widths: auto

   ==============  ==============  ================================================================
   :math:`p`       Name            Unit volume :math:`V_d`
   ==============  ==============  ================================================================
   :math:`1`       Cross-polytope  :math:`\frac{2^d}{d!}`
   :math:`2`       Hypersphere     :math:`\frac{\pi^{d/2}}{\Gamma\left ( \frac{d}{2} + 1 \right )}`
   :math:`\infty`  Hypercube       :math:`2^d`
   ==============  ==============  ================================================================



Example - Euclidean normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We would like to normalize the kernel functions in higher dimensions for the
most common norms. To accomplish this, we start with the equation for the volume
of a :math:`d`-dimensional hypersphere. The equation is

.. math::

   V_d = V_{d-1} \int_0^1 r^{d-1} \, dr = V_{d} \cdot d \int_0^1 r^{d-1} \, dr.

The integral of the kernel over the :math:`d`-dimensional space is then given by

.. math::

   V_{d} \cdot d \int_0^1 K(r) \, r^{d-1} \, dr.

Which we can compute. For instance, the linear kernel :math:`K(r) = (1-r)` is
normalized by

.. math::

   V_{d} \cdot d \int_0^1 \left ( 1 - r \right ) r^{d-1} \, dr = V_{d} \cdot d \left ( \frac{1}{d} - \frac{1}{d+1} \right )= V_d \left ( \frac{1}{d+1} \right )

The biweight kernel :math:`K(r) = \left ( 1 - r^2 \right )^2` is similarly normalized by

.. math::

   V_{d} \cdot d \int_0^1 \left ( 1 - r^2 \right )^2 r^{d-1} \, dr = V_d \left ( 1 - \frac{2d}{d+2} + \frac{d}{d+4} \right ) = V_d \left ( \frac{8}{(d+2)(d+4)} \right ).



.. plot::
    :include-source:

    from KDEpy.BaseKDE import BaseKDE
    from mpl_toolkits.mplot3d import Axes3D

    n = 64
    p = np.linspace(-3, 3, num=n)
    obs_x_dims = np.array(np.meshgrid(p, p)).T.reshape(-1, 2)

    # fig = plt.figure() is already set, adjust the size
    fig.set_figwidth(7); fig.set_figheight(5);

    selected_kernels = ['box', 'tri', 'exponential', 'gaussian']
    for i, kernel_name in enumerate(selected_kernels, 1):

      kernel = BaseKDE._available_kernels[kernel_name]
      ax = fig.add_subplot(2, 2, i, projection='3d')
      z = kernel(obs_x_dims, norm=2).reshape((n, n))
      surf = ax.plot_surface(*np.meshgrid(p, p), z)
      ax.set_title(f"'{kernel_name}', $2$-norm")







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

   x, y = NaiveKDE(kernel='box', bw='silverman').fit(data).evaluate()
   plt.plot(x, y, label='KDE estimate')
   plt.scatter(data, np.zeros_like(data), marker='x', label='Data', color='k')
   plt.plot(x, dist.pdf(x), ls='--', label='True distribution')
   plt.grid(True, ls='--', zorder=-15); plt.legend();


Fixed width kernels
-------------------

.. plot::
   :include-source:

   from KDEpy import *
   from scipy import stats

   # Generate a distribution and some data
   dist = stats.norm(loc=0, scale=1)
   data = dist.rvs(16)

   # Kernel density estimate with too small bandwidth
   x, y = NaiveKDE(bw=0.1).fit(data).evaluate()
   plt.plot(x, y, label='KDE estimate')

   # Kernel density estimate with too large bandwidth
   x, y = NaiveKDE(bw=2).fit(data).evaluate()
   plt.plot(x, y, label='KDE estimate')

   plt.plot(x, dist.pdf(x), ls='--', label='True distribution')
   plt.scatter(data, np.zeros_like(data), marker='x', label='Data', color='k')
   plt.grid(True, ls='--', zorder=-15); plt.legend();


Variable width kernels
----------------------

.. plot::
   :include-source:

   from KDEpy import *
   from scipy import stats

   # Generate a distribution and some data
   dist = stats.lognorm(s=1)
   data = dist.rvs(160)

   # Kernel density estimate with too small bandwidth
   x, y = NaiveKDE(bw=data).fit(data).evaluate()
   plt.plot(x, y, label='KDE estimate')

   plt.plot(x, dist.pdf(x), ls='--', label='True distribution')
   plt.scatter(data, np.zeros_like(data), marker='x', label='Data', color='k')
   plt.grid(True, ls='--', zorder=-15); plt.legend();


.. topic:: References

    * "Notes on Regularized Least Squares", Rifkin & Lippert (`technical report
      <http://cbcl.mit.edu/projects/cbcl/publications/ps/MIT-CSAIL-TR-2007-025.pdf>`_,
      `course slides
      <http://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf>`_).
