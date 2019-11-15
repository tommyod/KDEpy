#!/bin/sh
$PIP install cibuildwheel;
$PYTHON -m cibuildwheel --output-dir wheelhouse;
$PIP install twine;
$PYTHON -m twine upload dist/* -u tommyod -p $TWINE_PASSWORD --skip-existing;
