.. _literature:

Literature review
=================

Books
-----

- Silverman, B. W. "*Density Estimation for Statistics and Data Analysis.*" Chapman and Hall, 1986.

  - A general introduction to the subject.
    Hints to the use of k-d trees for computational efficiency in section 5.2.3, and adaptive kernel density estimators in section 5.3.
    Does not discuss practical algorithmic considerations.

- Wand, M. P., and M. C. Jones. "*Kernel Smoothing.*" Chapman and Hall, 1995.

  - Not as approachable as the book by Silverman.
    Appendix D contains computational hints, where linear binning and computation by means of convolution is explained.

- Kroese, Dirk P., Thomas Taimre, and Zdravko I. Botev.
  "*Handbook of Monte Carlo Methods.*" 1 edition. Hoboken, N.J: Wiley, 2011.

  - Some of the ideas used in the paper by Botev et al. are explained in detail here.
    In particular, see chapter 8.5 on Kernel Density Estimation.

Papers
------

..
  - Friedman, Jerome H., Jon Louis Bentley, and Raphael Ari Finkel.
  "*An Algorithm for Finding Best Matches in Logarithmic Expected Time.*"
  ACM Trans. Math. Softw. 3, no. 3 (September 1977): 209–226.
  https://doi.org/10.1145/355744.355745.
  - An early paper explaining a k-d tree.

- Fan, Jianqing, and James S. Marron.
  "*Fast Implementations of Nonparametric Curve Estimators.*"
  Journal of Computational and Graphical Statistics 3, no. 1 (March 1, 1994).
  https://doi.org/10.1080/10618600.1994.10474629.

  - Explains how linear binning may be computed in :math:`\mathcal{O}(N)` time using the quotient and remainder of division.
    This idea is extended to a fast :math:`d`-dimensional :math:`\mathcal{O}(N2^d)` algorithm in KDEpy.

- Maneewongvatana, Songrit, and David M. Mount.
  "*It’s Okay to Be Skinny, If Your Friends Are Fat.*"
  In Center for Geometric Computing 4th Annual Workshop on Computational
  Geometry, 2:1–8, 1999. https://www.cs.umd.edu/~mount/Papers/cgc99-smpack.pdf

  - Explains the sliding-midpoint rule for k-d trees.
    The ``scipy`` implementation uses this rule.

- Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
  "*Kernel density estimation via diffusion.*"
  Annals of Statistics, Volume 38, Number 5, pages 2916-2957. 2010.
  https://arxiv.org/pdf/1011.2602.pdf

  - Introduces the improved Sheather-Jones (ISJ) algorithm for bandwidth selection, which does not assume normality (unlike the Silverman rule of thumb).

- Wang, Xianfu.
  "*Volumes of Generalized Unit Balls.*"
  Mathematics Magazine 78, no. 5 (2005): 390–95.
  https://doi.org/10.2307/30044198.

  - Gives a formula for unit balls in any dimension and any :math:`p`-norm.
    This result is used to normalize kernels in any dimension and norm in KDEpy.

Other resources
---------------

- Jake VanderPlas. "*Kernel Density Estimation in Python.*". 2013.
  https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

- ISJ implementation by Daniel B. Smith, PhD, found at https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py.
