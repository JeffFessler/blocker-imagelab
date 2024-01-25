"""imagelab/tv.py

Edge-preserving regularizers and finite-difference
operators

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""

import numpy as np
from scipy.sparse import diags

from .linop import AbstractLinOp, Matrix, VStackLinOp
from .mat import circshift


class Cdiff1_circshift(AbstractLinOp):
    """1D Periodic Finite-Difference Operator
    implemented using circular shifts
    based on Jeff Fessler's ir toolbox"""

    def __init__(self, displacement=0, in_shape=None):
        super(Cdiff1_circshift, self).__init__(in_shape=in_shape, out_shape=in_shape)
        if isinstance(displacement, int):
            axis = displacement
            displacement = [0] * (axis + 1)
            displacement[axis] = 1
        self.displacement = displacement
        self.coef = -1

    def forward_project(self, x):
        return x + self.coef * circshift(x, self.displacement)

    def back_project(self, y):
        return y + self.coef * circshift(y, [-d for d in self.displacement])

    def abs(self):
        absC = Cdiff1_circshift(self.displacement, self.in_shape)
        absC.coef = np.abs(self.coef)
        return absC

    def opnorm(self):
        return 4


class Cdiff1_npdiff(AbstractLinOp):
    """1D Finite-Difference Operator
    implemented using np.diff"""

    def __init__(self, axis=0, in_shape=None, **kwargs):
        out_shape = None
        if in_shape is not None:
            out_shape = [dim for dim in in_shape]
            out_shape[axis] -= 1
        super(Cdiff1_npdiff, self).__init__(
            in_shape=in_shape, out_shape=out_shape, **kwargs
        )
        # if isinstance(displacement, int):
        #     axis = displacement
        #     displacement = [0]*(axis+1)
        #     displacement[axis] = 1
        self.axis = axis
        self.coef = -1

    def forward_project(self, x):
        # slc = [slice(None)]*len(x.shape)
        # slc[self.axis] = -1
        return self.backend.diff(x, axis=self.axis)  # , append=x[slc])

    def back_project(self, y):
        slc = [slice(None)] * len(y.shape)
        slc.insert(self.axis + 1, np.newaxis)  # since we lose an axis
        slc[self.axis] = 0
        y0 = -y[tuple(slc)]
        slc[self.axis] = -1
        yend = y[tuple(slc)]
        ymid = -self.backend.diff(y, axis=self.axis)
        return self.backend.concatenate([y0, ymid, yend], axis=self.axis)

    def abs(self):
        # Can't implement here, so use indexing!
        absC = Cdiff1_indexing(self.axis, self.in_shape)
        absC.coef = self.backend.abs(self.coef)
        return absC

    def opnorm(self):
        return 4


class Cdiff1_indexing(AbstractLinOp):
    """1D Finite-Difference Operator
    implemented using array indexing"""

    def __init__(self, axis=0, in_shape=None):
        out_shape = None
        if in_shape is not None:
            out_shape = [dim for dim in in_shape]
            out_shape[axis] -= 1
        super(Cdiff1_indexing, self).__init__(in_shape=in_shape, out_shape=out_shape)

        self.axis = axis
        self.coef = -1

    def forward_project(self, x):
        slc = [slice(None)] * len(x.shape)
        slc[self.axis] = slice(1, None)
        x0 = x[tuple(slc)]
        slc[self.axis] = slice(0, -1)
        x1 = x[tuple(slc)]
        return x0 + self.coef * x1

    def back_project(self, y):
        slc = [slice(None)] * len(y.shape)
        slc.insert(self.axis + 1, np.newaxis)  # since we lose an axis
        slc[self.axis] = 0
        y0 = self.coef * y[tuple(slc)]
        slc[self.axis] = -1
        yend = y[tuple(slc)]
        ymid = self.coef * self.forward_project(y)
        return np.concatenate([y0, ymid, yend], axis=self.axis)

    def abs(self):
        absC = Cdiff1_indexing(self.axis, self.in_shape)
        absC.coef = np.abs(self.coef)
        return absC

    def opnorm(self):
        return 4


def Cdiff1_spdiag(axis=0, in_shape=None):
    out_shape = [dim for dim in in_shape]
    out_shape[axis] -= 1
    N = np.prod(in_shape)
    M = np.prod(out_shape)
    mtx = diags([-np.ones(M), np.ones(M)], [0, 1], shape=(M, N))
    return Matrix(mtx)


def Cdiffn(ndim, in_shape=None):
    if isinstance(ndim, int):
        ndim = range(ndim)
    return VStackLinOp([Cdiff1_npdiff(n, in_shape=in_shape) for n in ndim])
