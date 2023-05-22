#!/usr/bin/env python3

"""KDEpy installer script using setuptools.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import Extension, setup
import os
import re


try:
    from Cython.Distutils import build_ext
    import numpy as np
except ImportError:
    can_build_ext = False
else:
    can_build_ext = True

HERE = os.path.abspath(os.path.dirname(__file__))


def read(fname):
    return open(os.path.join(HERE, fname)).read()


cmdclass = {}
ext_modules = []
include_dirs = []

if can_build_ext:
    cmdclass["build_ext"] = build_ext
    ext_modules.append(Extension("cutils", ["KDEpy/cutils.pyx"]))
    include_dirs.append(np.get_include())
else:
    # Build extension with previously Cython generated source.
    ext_modules.append(Extension("cutils", ["KDEpy/cutils.c"]))


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
    cmdclass=cmdclass,
    include_dirs=include_dirs,
    ext_modules=ext_modules,
)
