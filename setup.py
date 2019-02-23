#!/usr/bin/env python3

"""KDEpy installer script using setuptools.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import Extension, setup

try:
    from Cython.Distutils import build_ext
    import numpy as np
except ImportError:
    can_build_ext = False
else:
    can_build_ext = True


# If this is incremented, also increment in __init__.py
VERSION = "0.6.10"

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
    name="KDEpy",
    version=VERSION,
    description="Kernel Density Estimation in Python.",
    long_description="Kernel Density Estimation in Python.",
    url="https://github.com/tommyod/KDEpy",
    author="tommyod",
    author_email="tod001@uib.no",
    license="GNU GPLv3",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["KDEpy"],
    install_requires=["numpy>=1.14.2", "scipy>=1.0.1", "matplotlib>=2.2.0"],
    cmdclass=cmdclass,
    include_dirs=include_dirs,
    ext_modules=ext_modules,
)
