Kernels
=======

Available kernels
-----------------

You are likely reading this because you wonder which kernels are available.
Every available kernel is shown in the figure below.
Kernels with finite support are annotated with **F**.
A listing of every available option is found in ``_available_kernels.items()``.

.. plot::
   :include-source:

   from KDEpy import NaiveKDE

   for name, func in NaiveKDE._available_kernels.items():
      x, y = NaiveKDE(kernel=name).fit([0]).evaluate()
      plt.plot(x, y, label=name + (' (F)' if func.finite_support else ''))

   plt.grid(True, ls='--', zorder=-15); plt.legend();

Kernel properties
-----------------

The kernels implemented in KDEpy obey some properties.

* Normalization: :math:`\int K(x) \, dx = 1`.
* Unit variance: :math:`\operatorname{Var}[K(x)] = 1` when the bandwidth :math:`h` is 1.
* Symmetry: :math:`K(-x) = K(x)` for every :math:`x`.

Kernels are radial basis functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Symmetry follows from a more general property, namely the fact that every kernel implemented is a radial basis function.
A *radial basis function* is a function which evaluates to the same value whenever the distance from the origin is the same.
In other words, it is the composition of a norm :math:`\left\| \cdot \right\| _p: \mathbb{R}^d \to \mathbb{R}_+` and a function :math:`\kappa: \mathbb{R}_+ \to \mathbb{R}_+`.

.. math::

   K(x) = \kappa \left( \left\| x \right\| _p \right)

.. note::

   If you have high dimensional data with vastly different scales, consider standardizing the data before feeding it to a KDE.

Kernels may have finite support, or not
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A given kernel may or may not have have finite support.
A kernel with finite (or compact) support is defined on a domain such as :math:`[-1, 1]`,
while a kernel without finite support is defined on :math:`[-\infty, \infty]`.

Below we plot the *Guassian kernel* and the *Epanechnikov kernel*.

* The Gaussian kernel is not bounded.
* The Epanechnikov is bounded.

The reason why kernels are normalized to unit variance is so that bounded and non-bounded
kernel functions are more easily compared.

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
Let :math:`r := \left\| x \right\| _p` be a measure of distance (:math:`r` stands for radius here).
Normalization is necessary, but symmetry is guaranteed since
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
^^^^^^^^^^^^^^^^^^^^

Kernels in any dimension are normalized so that the integral is unity for any :math:`p`.
To explain how a high-dimensional kernel is normalized, we first examine
high dimensional volumes.

Let :math:`r := \left\| x \right\| _p` be the distance from the origin, as measured by some :math:`p`-norm.
The :math:`d`-dimensional volume :math:`V_d(r)` is proportional to :math:`r^d`.
We will now examine the unit :math:`d`-dimensional volume :math:`V_d := V_d(1)`.

We integrate over :math:`V_{d-1}(r)` to obtain :math:`V_{d}` using

.. math::

   V_d = \int_0^1 V_{d-1}(r) \, dr.

Since :math:`V_{d-1}(r) \propto r^{d-1}`, we write it as :math:`V_{d-1}(r) = K_{d-1} r^{d-1}`,
where :math:`K_{d-1}` is a constant. Pulling this out of the integral, we are left with

.. math::

   V_d = K_{d-1} \int_0^1 r^{d-1} \, dr = K_{d-1} / d = V_{d-1} / d,

where the last equality follows from :math:`V_{d-1}(1) = K_{d-1} (1)^{d-1}`.

What is the volume of a unit ball :math:`V_d` in the :math:`p` norm in :math:`d` dimensions?
Fortunately an analytical expression exists, it's given by

.. math::

   2^d \frac{\Gamma \left( 1 + \frac{1}{p} \right)^d}{\Gamma \left(1 + \frac{d}{p} \right)}.

For more information about this, see for instance the paper by Wang in :ref:`literature`.
The equation above reduces to more well-known cases when :math:`p` takes common values, as shown in the table below.

.. table:: High dimensional volumes
   :widths: auto

   ==============  ==============  ================================================================
   :math:`p`       Name            Unit volume :math:`V_d`
   ==============  ==============  ================================================================
   :math:`1`       Cross-polytope  :math:`\frac{2^d}{d!}`
   :math:`2`       Hypersphere     :math:`\frac{\pi^{d/2}}{\Gamma\left ( \frac{d}{2} + 1 \right )}`
   :math:`\infty`  Hypercube       :math:`2^d`
   ==============  ==============  ================================================================



Example - normalization
^^^^^^^^^^^^^^^^^^^^^^^

We would like to normalize the kernel functions in higher dimensions any norm.
To accomplish this, we start with the equation for the volume of a :math:`d`-dimensional volume.
The equation is

.. math::

   V_d = V_{d-1} \int_0^1 r^{d-1} \, dr = V_{d} \cdot d \int_0^1 r^{d-1} \, dr.

The integral of the kernel :math:`\kappa: \mathbb{R}_+ \to \mathbb{R}_+` over the :math:`d`-dimensional space is then given by

.. math::

   V_{d} \cdot d \int_0^1 \kappa(r) \, r^{d-1} \, dr,

which we can compute.
For instance, the *linear kernel* :math:`\kappa(r) = (1-r)` is
normalized by

.. math::

   V_{d} \cdot d \int_0^1 \left ( 1 - r \right ) r^{d-1} \, dr = V_{d} \cdot d \left ( \frac{1}{d} - \frac{1}{d+1} \right )= V_d \left ( \frac{1}{d+1} \right ).

The *biweight kernel* :math:`\kappa(r) = \left ( 1 - r^2 \right )^2` is similarly normalized by

.. math::

   V_{d} \cdot d \int_0^1 \left ( 1 - r^2 \right )^2 r^{d-1} \, dr = V_d \left ( 1 - \frac{2d}{d+2} + \frac{d}{d+4} \right ) = V_d \left ( \frac{8}{(d+2)(d+4)} \right ).


Some 2D kernels
^^^^^^^^^^^^^^^

Let's see what the kernels look like in 2D when :math:`p=2`.

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
