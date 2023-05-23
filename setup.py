#!/usr/bin/env python3

"""KDEpy installer script using setuptools.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import os

import numpy as np
from Cython.Distutils import build_ext
from setuptools import Extension, setup

HERE = os.path.abspath(os.path.dirname(__file__))


def read(fname):
    return open(os.path.join(HERE, fname)).read()


setup(
    packages=["KDEpy"],
    cmdclass={"build_ext": build_ext},
    include_dirs=[np.get_include()],
    ext_modules=[Extension("cutils", ["KDEpy/cutils.pyx"])],
)
