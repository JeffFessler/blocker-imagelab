""" Conveniene Module for quick command-line experiments.
Please use explicit import list in any permanent code.

This is very bad practice. To use:
>>> from imagelab.lab import *
"""

from warnings import warn as _warn

_import_list = """
import numpy as np
import matplotlib.pyplot as plt
import imagelab as il
from imagelab import noise
from imagelab.plot import show_im, play_vid
"""

_warn("Importing:" + _import_list)
exec(_import_list)
