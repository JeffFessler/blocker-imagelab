"""imagelab/color/__init__.py

Basic colorimetry functions. This is not meant to
compete with more complete color science packages
such as `colour` but provides basics for moving
about and visualizing different color spaces.

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""

import numpy as np

# from scipy.signal import convolve
from scipy.ndimage import convolve

from . import cie  # noqa: F401
from .. import linop


def random_mask(shape, cdensity="RGBW", seed=None):
    rng = np.random.default_rng(seed)
    digit_mask = np.mod(rng.permutation(np.prod(shape)), len(cdensity)).reshape(shape)
    pattern = np.array([ch for ch in cdensity.upper()])
    mask = pattern[digit_mask]
    w_mask = mask == "W"
    return [w_mask + c_mask for c_mask in [mask == "R", mask == "G", mask == "B"]]


def bayer_mask(shape, pattern="RGGB"):
    pattern = np.array([ch for ch in pattern.upper()]).reshape((2, 2))
    Y, X = np.mgrid[: shape[0], : shape[1]]

    mask = pattern[np.mod(Y, 2), np.mod(X, 2)]
    return [mask == "R", mask == "G", mask == "B"]


def bilinear_bayer_demosaic(raw, pattern="RGGB"):
    r_mask, g_mask, b_mask = bayer_mask(raw.shape[-2:], pattern)
    if "cupy" in type(raw).__module__:
        import cupy

        xp = cupy
        import cupyx.scipy.ndimage

        conv = cupyx.scipy.ndimage.convolve
        r_mask, g_mask, b_mask = xp.array(r_mask), xp.array(g_mask), xp.array(b_mask)
    else:
        xp = np
        conv = convolve

    r_ker = (
        xp.array(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ],
            dtype=raw.dtype,
        )
        / 4
    )

    g_ker = (
        xp.array(
            [
                [0, 1, 0],
                [1, 4, 1],
                [0, 1, 0],
            ],
            dtype=raw.dtype,
        )
        / 4
    )

    b_ker = r_ker

    newaxes = tuple([None] * (raw.ndim - r_ker.ndim))
    r_img = conv(raw * r_mask, r_ker[newaxes], mode="constant")
    g_img = conv(raw * g_mask, g_ker[newaxes], mode="constant")
    b_img = conv(raw * b_mask, b_ker[newaxes], mode="constant")

    return xp.array([r_img, g_img, b_img])


def std_to_linear(img, gamma=2.2):
    return img ** (gamma)


def linear_to_std(img, gamma=2.2):
    return img ** (1 / gamma)


def sRGB_to_lRGB(img):
    if np.issubdtype(img.dtype, np.integer):
        raise ValueError("This function assumes float valued images")
    img = np.clip(img, a_min=0, a_max=1)
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def lRGB_to_sRGB(img):
    # this seems to work for cupy as well
    img = np.clip(img, a_min=0, a_max=1)
    return np.where(
        img <= 0.04045 / 12.92, img * 12.92, (img ** (1 / 2.4)) * 1.055 - 0.055
    )


class BayerCFA(linop.AbstractLinOp):
    """Bayer Color Filter Array
    Input should be at least 3 dimensions,
    and the last two dimensions that are
    not the color dimension should be
    the sensor dimensions.
    """

    def __init__(self, in_shape, color_dim=-1):
        assert in_shape[color_dim] == 3, "color_dim is not length 3"
        out_shape = list(in_shape)
        del out_shape[color_dim]
        super(BayerCFA, self).__init__(in_shape=in_shape, out_shape=out_shape)
        self.cfa_mask = self.backend.array(bayer_mask(out_shape[-2:]))
        self.color_dim = color_dim

    def forward_project(self, x):
        imgiter = np.moveaxis(x, self.color_dim, 0)
        y = self.backend.zeros(self.out_shape)
        for clr, msk in zip(imgiter, self.cfa_mask):
            y += clr * msk
        return y

    def back_project_old(self, y):
        x = self.backend.zeros(self.in_shape)
        imgiter = self.backend.moveaxis(x, self.color_dim, 0)  # view into x
        for clr, mask in zip(imgiter, self.cfa_mask):
            print("clr", type(clr), clr.shape)
            print("y", type(y), y.shape)
            print("mask", type(mask), mask.shape)
            clr[..., mask] = y[..., mask]
        return x

    def back_project(self, y):
        color_mask = self.backend.moveaxis(self.cfa_mask, 0, self.color_dim)
        return color_mask * self.backend.expand_dims(y, self.color_dim)

    def det(self):
        return 0

    def pinv(self):
        return self.T

    def opnorm(self):
        return 1

    def abs(self):
        return self

    def guess_inv(self, y):
        z = bilinear_bayer_demosaic(y)
        z = np.moveaxis(z, 0, self.color_dim)
        return z
