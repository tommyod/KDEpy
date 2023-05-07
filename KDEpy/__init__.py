#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.metadata


from KDEpy.FFTKDE import FFTKDE
from KDEpy.NaiveKDE import NaiveKDE
from KDEpy.TreeKDE import TreeKDE


__version__ = importlib.metadata.version("KDEpy")
__author__ = "tommyod"

__all__ = ["TreeKDE", "NaiveKDE", "FFTKDE"]
