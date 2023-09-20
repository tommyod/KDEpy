#!/usr/bin/env python3

"""KDEpy installer script using setuptools.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import numpy as np
from Cython.Distutils import build_ext
from setuptools import Extension, setup

setup(
    packages=["KDEpy"],
    cmdclass={"build_ext": build_ext},
    include_dirs=[np.get_include()],
    ext_modules=[Extension("KDEpy._cutils", ["KDEpy/cutils_ext/cutils.pyx"])],
)
