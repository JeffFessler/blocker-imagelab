"""imagelab/sparsity.py
A collection of convenience function for doing
sparsity regularization and analysis.

Example:
>>> import sparsity
>>> W = sparsity.dct2dmtx(8)
>>> plot_dictionary(W.T)

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""

from collections.abc import Iterable as _Iterable
from functools import reduce as _reduce
import itertools as _it

import numpy as np
from numpy.linalg import norm
import pywt

from .linop import AbstractLinOp
from .utils import export


@export
class OWTLinOp(AbstractLinOp):
    """The 2D Orthoganl Wavelet Transform as a LinOp"""

    def __init__(self, in_shape, level=3, wtype="haar"):
        L = 2 ** level
        self.padding = [L - ((dim - 1) % L + 1) for dim in in_shape]
        out_shape = [dim + pad for dim, pad in zip(in_shape, self.padding)]
        super(OWTLinOp, self).__init__(in_shape=in_shape, out_shape=out_shape)
        self.level = level
        self.type = wtype
        self.n_coeffs = 3 * level + 1
        self._init_coeff_map()
        self.clip_shape = tuple(slice(shp) for shp in in_shape)

    def _init_coeff_map(self):
        box = self.out_shape
        coeff_map = np.zeros(box)
        coeff_shapes = []
        for scale in reversed(range(3, self.n_coeffs, 3)):
            xdim = box[1]
            ydim = box[0]
            box = ((ydim) // 2, (xdim) // 2)
            coeff_shapes.extend([(ydim - box[0], xdim - box[1])] * 3)
            coeff_map[box[0] : ydim, box[1] : xdim] = scale
            coeff_map[box[0] : ydim, : box[1]] = scale - 1
            coeff_map[: box[0], box[1] : xdim] = scale - 2
        self._coeff_shapes = [coeff_shapes[-1]] + list(reversed(coeff_shapes))
        assert (
            len(self._coeff_shapes) == self.n_coeffs
        ), f"{self.n_coeffs}, {self._coeff_shapes}"
        self.coeff_map = coeff_map

    def pad_input(self, x):  # what is the best boundary condition? Does it matter?
        # or reflect?
        return np.pad(x, [(0, pad) for pad in self.padding], mode="edge")

    def forward_project(self, x):
        x = self.pad_input(x)
        coeffs = pywt.wavedec2(x, wavelet=self.type, level=self.level)
        y = np.zeros(self.coeff_map.shape)
        y[self.coeff_map == 0] = coeffs[0].reshape(-1)
        for ii, cc in enumerate(_it.chain.from_iterable(coeffs[1:]), 1):
            y[self.coeff_map == ii] = cc.reshape(-1)
        return y

    def back_project(self, y):
        coeffs = [y[self.coeff_map == 0].reshape(self._coeff_shapes[0])]
        for ii in range(1, self.n_coeffs - 2, 3):
            H = y[self.coeff_map == ii].reshape(self._coeff_shapes[ii])
            V = y[self.coeff_map == ii + 1].reshape(self._coeff_shapes[ii + 1])
            D = y[self.coeff_map == ii + 2].reshape(self._coeff_shapes[ii + 2])
            coeffs.append((H, V, D))
        return pywt.waverec2(coeffs, wavelet=self.type)[self.clip_shape]

    def _get_pad_amt(self, wtype):  # TODO: make larger wavelets work
        if wtype == "haar":
            return 1
        elif "db" in wtype:
            return int(wtype.strip("db"))
        else:
            ValueError("Unknown wavelet type wtype={}".format(wtype))


def _isiterable(obj):
    if isinstance(obj, _Iterable):
        return True
    elif hasattr(obj, "__getitem__") and hasattr(obj, "__len__"):
        return True
    else:
        return False


def _accept_tuple_or_args(args):
    if len(args) == 1:
        if _isiterable(args[0]):
            return args[0]
        else:
            return (args,)
    else:
        return args


@export
def dctmtx(K, N=None):
    """Creates an KxN matrix that performs the
    DCT(-II) transform on a N-point vector

    [wiki](https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II)
    """
    if N is None:
        N = K
    n = np.arange(0, N) + 0.5
    k = np.arange(0, K)
    dctm = np.cos(np.pi * np.outer(k, n) / K)
    # orthogonalize
    if N > 1:
        dctm[1:, :] -= dctm[1:, :].mean(axis=1, keepdims=True)
    dctm /= norm(dctm, axis=1, keepdims=True)
    return dctm


@export
def oc_nd_dctmtx(K_shape, N_shape):
    """Overcomplete ND DCT Matrix"""
    return _reduce(np.kron, (dctmtx(K, N) for K, N in zip(K_shape, N_shape)))


@export
def nd_dctmtx(*args):
    """Creates a square Nd - DCT Matrix"""
    args = _accept_tuple_or_args(args)
    return oc_nd_dctmtx(args, args)
