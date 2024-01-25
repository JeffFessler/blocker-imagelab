import itertools

import numpy as np
import scipy

from . import pix


def downsample(x, scale, *, axis=None, reduce="mean", color="auto"):
    if scale == 1:
        return x
    if axis is None:
        axis = list(range(len(x.shape)))  # all axes
        if (color is True) or (color == "auto" and pix.is_color(x)):
            axis = axis[:-1]  # is this too complicated of a default?

    if not hasattr(scale, "__iter__"):
        scale = [scale] * len(axis)
    elif len(scale) < len(axis):
        # only apply to last axes
        axis = axis[-len(scale) :]
    elif len(scale) > len(axis):
        raise ValueError("More scales given than axes")

    if reduce in ["mean", "avg", "average"]:
        pool = np.mean
    elif reduce in ["max"]:
        pool = np.max
    elif reduce in ["min"]:
        pool = np.min
    elif reduce in ["sum"]:
        pool = np.sum
    elif reduce in ["med", "median"]:
        pool = np.median
    elif callable(reduce):
        pool = reduce
    else:
        raise ValueError(f"Unrecognized Reduction method, reduce={reduce}")

    for sc, ax in zip(scale, axis):
        if sc % 1 != 0:
            raise ValueError("Only integer scales supported")
        sc = int(sc)
        if sc == 1:
            continue
        dim = x.shape[ax]
        m1 = sc * np.floor(dim / sc)
        if m1 < dim:  # i.e. m1 != dim
            # we need to crop to a multiple of scale
            x = ccrop1(x, m1, ax)

        assert m1 == x.shape[ax]
        new_shape = list(x.shape[:ax]) + [int(m1 // sc), sc] + list(x.shape[ax + 1 :])
        x = pool(x.reshape(new_shape), axis=ax + 1)
    return x


@np.vectorize
def jinc(r):
    """The Jinc function.

    Parameters
    ----------
        r: radial distance.

    Return
    ------
        the value of j1(x)/x for x != 0, 0.5 at 0.

    """
    r = np.fabs(r)  # only necessary for first part
    if r <= 1e-8:  # numerical precision issues
        return 0.5
    else:
        return scipy.special.j1(r) / r


def ccrop1(x, clen, axis):
    slc = [slice(None)] * len(x.shape)
    dim = x.shape[axis]
    clip = int(dim - clen)
    if clip == 0:
        pass
    else:
        slc[axis] = slice(clip // 2, -clip // 2)
        x = x[tuple(slc)]
    assert clen == x.shape[axis], f"{clen} != {x.shape[axis]}"
    return x


def ccrop(x, shape):
    for axis, clen in enumerate(shape):
        x = ccrop1(x, clen, axis)
    return x


def c_support_trim(filt, axes=None, step=5, tol=1e-8):
    """Centered Support trim
    Removes zeros from both sides of each axis in `axes
    so long as the sum of the removed margins is less than
    `tol*filt.max()`. Useful for making psfs with a small support
    use less memory.
    """
    tol *= filt.max()
    if axes is None:
        axes = [ii for ii in range(filt.ndim)]
    elif isinstance(axes, int):
        axes = [axes]
    for axis in axes:
        stp = step
        slc = [slice(None) for shp in filt.shape]
        while True:
            slc[axis] = slice(0, stp)
            margin = filt[tuple(slc)].sum()
            slc[axis] = slice(-stp, None)
            margin += filt[tuple(slc)].sum()

            # print(margin, stp)
            if margin > tol:
                if stp == 1:
                    break
                else:
                    stp = stp // 2
                    while filt.shape[axis] < 2 * stp:
                        stp = stp // 2
                    if stp < 1:
                        break
            elif filt.shape[axis] > 2 * stp:
                slc[axis] = slice(stp, -stp)
                filt = filt[tuple(slc)]
            else:
                if stp == 1:
                    break
                else:
                    stp = stp // 2
    return filt.copy()  # so old can be gc


def create_mask(percent, shape, seed=None):
    rng = np.random.default_rng(seed)
    return (rng.permutation(np.prod(shape)) < percent * np.prod(shape)).reshape(shape)


def cubic_inpaint2D(y, mask):
    if y.shape == mask.shape:
        y = y[mask]
    points = np.mgrid[tuple(slice(None, shp) for shp in mask.shape)]
    points = tuple(p.reshape(-1) for p in points)
    x0 = scipy.interpolate.griddata(
        mask.nonzero(), y, points, method="cubic", fill_value=np.NaN
    ).reshape(*mask.shape)
    z = np.isfinite(x0)
    if not z.all():
        x0 = scipy.interpolate.griddata(
            z.nonzero(), x0[z], points, method="nearest"
        ).reshape(*mask.shape)
    return x0


def cubic_inpaint(y, mask):
    out_shape = y.shape
    if mask.shape != y.shape and mask.shape == y.shape[-2:]:
        raise ValueError("Mask shape must match image")
    y = y.reshape((-1, *y.shape[-2:]))
    mask = mask.reshape((-1, *y.shape[-2:]))
    x = np.empty_like(y)
    for ii, jj in itertools.zip_longest(
        range(y.shape[0]), range(mask.shape[0]), fillvalue=0
    ):
        x[ii] = cubic_inpaint2D(y[ii], mask[jj])
    return x.reshape(out_shape)
