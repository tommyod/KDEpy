#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
import numpy as np
# To use a consistent encoding
from codecs import open
from os import path
# from KDEpy import __version__
VERSION = '0.5.6' # __version__


def read(fname):
    return open(path.join('.', fname)).read()


try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [Extension("cutils", [path.join('KDEpy', 'cutils.pyx')]),]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [Extension("cutils", [path.join('KDEpy', 'cutils.c')]),]


setup(
    name='KDEpy',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=VERSION,

    description='Kernel Density Estimation in Python.',
    long_description='Kernel Density Estimation in Python.',
    # read('README.rst'),
    # The project's main homepage.
    url='https://github.com/tommyod/KDEpy',

    # Author details
    author='tommyod',
    author_email='tod001@uib.no',
    # Choose your license
    license='GNU GPLv3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        # 'Intended Audience :: End Users/Desktop',
        # 'Intended Audience :: Healthcare Industry',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages('.', exclude=['contrib', 'docs', 'tests']),
    package_dir={'KDEpy': 'KDEpy'},

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
    'numpy>=1.14.2',
    'scipy>=1.0.1',
    'setuptools>=39.2.0'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # include_package_data=True,
    package_data={
        '': ['templates/*', '*.tex', '*.html'],
    },

    # For cython, see: http://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
    # ext_modules = cythonize(path.join(".", "KDEpy", "cutils.pyx")),
    cmdclass=cmdclass,
    include_dirs=[np.get_include()],
    ext_modules=ext_modules,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #    'console_scripts': [
    #         'sample=sample:main',
    #    ],
    # }
)
