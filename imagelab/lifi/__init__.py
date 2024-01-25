"""imagelab/lifi/__init__.py

This package contains code specific to working with
light field data.

With a few, mostly developmental, exceptions, lightfield data
is assumed to take the matrix style row-major form of
lf[v, u, y, x]
where u is the horizontal view index and x is the horizontal image index (v,y similarly).
Note this is functional form (lf[x,y,u,v]) transposed, and so switching between the two
can be easily done without changing the memory layout
(the functional form then being column-major).

this package uses gantry-style axis directions:
X increases to the right
Y increases down
U increases as the perspective shifts right
V increases as the perspective shifts down
"""
from . import io, sim, system_model

# flake8: noqa
from ..utils import in_notebook as _in_notebook

# import numpy as np

### Reshaping


def draw_subaperture_matrix(lf):
    v, u, y, x = 0, 1, 2, 3
    Nv, Nu, Ny, Nx = lf.shape
    return lf.transpose([v, y, u, x]).reshape(Nv * Ny, Nu * Nx)


def draw_lenselet_image(lf):
    v, u, y, x = 0, 1, 2, 3
    Nv, Nu, Ny, Nx = lf.shape
    return lf.transpose([y, v, x, u]).reshape(Nv * Ny, Nu * Nx)


def focal_plane_image(lf):
    return lf.sum(axis=(0, 1))


def corner_views(lf):
    return lf[[[0, 0], [-1, -1]], [[0, -1], [0, -1]]]


def extreme_views(lf):
    return lf[[0, -1], [0, -1]]


def mid_views(lf):
    nV, nU = lf.shape[:2]
    cV, cU = nV // 2, nU // 2
    return lf[
        [[0, 0, 0], [cV, cV, cV], [-1, -1, -1]], [[0, cU, -1], [0, cU, -1], [0, cU, -1]]
    ]


if _in_notebook():
    # convenience access in notebooks
    from .plot import *
