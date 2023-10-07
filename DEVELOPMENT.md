## Deployment

The KDEpy project uses GitHub Actions to build wheels for Linux, Mac and Windows.
Wheels are built using the [cibuildwheel](https://github.com/joerick/cibuildwheel) Python package.
After developing, the following will push a tagged commit -- from there CI will build wheels to distribute to PyPI automatically.

```
$ <run tests and linting>
$ git commit -m "Release of v0.X.Y"
$ git tag v0.X.Y
$ git push origin V0.X.Y
```

## Milestones

The list below roughly shows what needs to be done.

- [X] univariate BaseKDE (todo: check if more common code can be moved)
- [X] univariate NaiveKDE
- [X] univariate TreeKDE
- [X] univariate FFTKDE (implement linbin even faster in cython)
- [ ] univariate DiffusionKDE
- [X] Refactor kernel funcs - add solver for effective bandwidth

- [X] Implement univariate, fixed bandwidth KDEs naively
- [X] Implement **weighted**, fixed bandwidth, univariate KDEs naively
- [X] Implement variable bandwidth KDEs naively
- [X] Implement TreeKDE, test against other implementations
- [X] Implement Scott and Silverman rules for bandwidth selection
- [X] Make sure that speed and functionally matches `statsmodels`, `scikit-learn` and `scipy`
- [X] Implement methods taking care of boundaries
- [X] Make sure TreeKDE works without finite support too

## Misc

### General guidelines for this project

I hope to follow these guidelines for this project:
- Import as few external dependencies as possible, ideally only NumPy.
- Use test driven development, have tests and docs for every method.
- Cite literature and implement recent methods.
- Unless it's a bottleneck computation, readability trumps speed.
- Employ object orientation, but resist the temptation to implement
  many methods - stick to the basics.
- Follow PEP8

---------------


# Theory

# Existing implementations

## Implementations in Python

### Implementations in conda packages

- `sklearn/neighbors/kde.py`
- `scipy/stats/kde.py`
- `statsmodels/nonparametric/*`
- `seaborn/distributions.py`

### Other Python implementations

- https://github.com/cooperlab/AdaptiveKDE
- https://github.com/tillahoffmann/asymmetric_kde
- http://pythonhosted.org/PyQt-Fit/KDE_tut.html
- https://github.com/Daniel-B-Smith/KDE-for-SciPy

## Implementations in other languages

- [MATLAB: adaptive kernel density estimation in one-dimension](https://se.mathworks.com/matlabcentral/fileexchange/58309-adaptive-kernel-density-estimation-in-one-dimension?s_tid=gn_loc_drop)
- [MATLAB: Kernel Density Estimator for High Dimensions](http://se.mathworks.com/matlabcentral/fileexchange/58312-kernel-density-estimator-for-high-dimensions)


# References

## Books

- Silverman, B. W. Density Estimation for Statistics and Data Analysis. Boca Raton: Chapman and Hall, 1986. -- Page 99 for reference to kd-tree
- Wand, M. P., and M. C. Jones. Kernel Smoothing. London ; New York: Chapman and Hall/CRC, 1995. -- Page 182 for computation using linbin and fft

## Wikipedia and other articles

- [Wiki - Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation)
- [Wiki - Variable kernel density estimation](https://en.wikipedia.org/wiki/Variable_kernel_density_estimation)
- [Wiki - Kernel (statistics)](https://en.wikipedia.org/wiki/Kernel_(statistics))
- [Histograms and kernel density estimation KDE 2](https://mglerner.github.io/posts/histograms-and-kernel-density-estimation-kde-2.html?p=28)
- [Jakevdp - Kernel Density Estimation in Python](https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/)

## Papers

- [arXiv - Efficient statistical classification of satellite
measurements](https://arxiv.org/pdf/1202.2194.pdf)
- [arXiv - UNIFIED TREATMENT OF THE ASYMPTOTICS OF ASYMMETRIC KERNEL DENSITY ESTIMATORS](https://arxiv.org/pdf/1512.03188.pdf)
- [arXiv - A Review of Kernel Density Estimation with Applications to Econometrics](https://arxiv.org/pdf/1212.2812.pdf)
- [A Reliable Data-Based Bandwidth Selection Method for Kernel Density Estimation](https://www.researchgate.net/profile/Simon_Sheather/publication/224817413_A_Reliable_Data-Based_Bandwidth_Selection_Method_for_Kernel_Density_Estimation/links/0046352bc8b276ba1c000000/A-Reliable-Data-Based-Bandwidth-Selection-Method-for-Kernel-Density-Estimation.pdf)
- [KERNEL DENSITY ESTIMATION VIA DIFFUSION](https://projecteuclid.org/download/pdfview_1/euclid.aos/1281964340)
- [Variable Kernel Density Estimation](https://projecteuclid.org/download/pdf_1/euclid.aos/1176348768)
- [ Bayesian Approach to Bandwidth Selection for Multivariate Kernel Density Estimation](https://robjhyndman.com/papers/mcmckernel.pdf)
- [BOOTSTRAP BANDWIDTH SELECTION IN KERNEL DENSITY ESTIMATION](http://www.ism.ac.jp/editsec/aism/pdf/056_1_0019.pdf)
- [Kernel Estimator and Bandwidth Selection for Density and its Derivatives](https://cran.r-project.org/web/packages/kedd/vignettes/kedd.pdf)

## Misc

- [Variable Kernel Density Estimation - 20 slides](https://pdfs.semanticscholar.org/96c6/d421342631e3005cc85a330fedc729c8298b.pdf)
- [Lecture Notes on Nonparametrics - 25 pages](https://pdfs.semanticscholar.org/2c36/60a1844f55935f798b10a48197a665d1a825.pdf)
- [APPLIED SMOOTHING TECHNIQUES - Part 1: Kernel Density Estimation - 20 pages](http://staff.ustc.edu.cn/~zwp/teach/Math-Stat/kernel.pdf)
- [Kernel density estimation - 26 slides](http://research.cs.tamu.edu/prism/lectures/pr/pr_l7.pdf)
- [Density Estimation - 32 pages](http://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf)

# Computation

- *An Algorithm for Finding Best Matches in Logarithmic Expected Time*, Friedman et al, DOI 10.1145/355744.355745

https://www.ics.uci.edu/~ihler/code/kde.html

http://www-stat.wharton.upenn.edu/~lzhao/papers/MyPublication/Fast_jcgs.2010.pdf


https://indico.cern.ch/event/397113/contributions/1837849/attachments/1213965/1771772/main.pdf

http://www.cs.ubc.ca/~nando/papers/empirical.pdf

http://iopscience.iop.org/article/10.1088/1742-6596/762/1/012042/pdf

https://arxiv.org/pdf/1206.5278.pdf


https://www.researchgate.net/publication/228773329_Insights_on_fast_kernel_density_estimation_algorithms
