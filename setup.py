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
    # name="KDEpy",
    # version=VERSION,
    description="Kernel Density Estimation in Python.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/tommyod/KDEpy",
    author="tommyod",
    author_email="tommy.odland@gmail.com",
    license="new BSD",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=["KDEpy"],
    install_requires=["numpy>=1.14.2", "scipy>=1.0.1", "matplotlib>=2.2.0"],
    cmdclass={"build_ext": build_ext},
    include_dirs=[np.get_include()],
    ext_modules=[Extension("cutils", ["KDEpy/cutils.pyx"])],
)
