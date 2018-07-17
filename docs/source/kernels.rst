
Kernels
=======

Available kernels
-----------------

You are most likely here because you wonder what kernels are available.
Every available kernel is shown in the figure below.
Kernels with bounded support are annotated with **B**.

.. plot::
   :include-source:

   from KDEpy import *

   plt.figure(figsize=(7, 3))

   for name, func in NaiveKDE._available_kernels.items():
      x, y = NaiveKDE(kernel=name).fit([0]).evaluate()
      plt.plot(x, y, label=name + (' (B)' if func.finite_support else ''))

   plt.grid(True, ls='--', zorder=-15); plt.legend(); plt.show()

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

   plt.figure(figsize=(7, 3))
   x, y1 = NaiveKDE(kernel='gaussian', bw=1).fit([0]).evaluate()
   y2 = NaiveKDE(kernel='epa', bw=1).fit([0]).evaluate(x)
   plt.plot(x, y1, label='Gaussian kernel')
   plt.plot(x, y2, label='Epanechnikov kernel')
   plt.grid(True, ls='--', zorder=-15); plt.legend(); plt.show()


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

   fig = plt.figure(figsize=(7, 3))
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

    fig = plt.figure(figsize=(7, 5))

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
