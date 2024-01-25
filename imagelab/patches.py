from numbers import Number

import numpy as np

from . import utils


class PatchConfig(object):
    """docstring for PatchConfig"""

    def __init__(
        self,
        arr_shape,
        patch_shape,
        stride=1,
        pad=None,
        extent="same",
        no_pad_axes=(),
        backend=np,
    ):
        super(PatchConfig, self).__init__()

        arr_ndim = len(arr_shape)
        pat_ndim = len(patch_shape)

        self.arr_shape = arr_shape
        if isinstance(stride, Number):
            stride = [stride] * pat_ndim
        self.stride = [1] * (arr_ndim - len(stride)) + list(stride)
        self.patch_shape = [1] * (arr_ndim - pat_ndim) + list(patch_shape)
        self.pad = (
            pad if extent.lower() != "valid" else None
        )  # None, circ, mirror, constant
        self.pad = {"zero": "constant", "circ": "wrap"}.get(self.pad, self.pad)
        self.extent = (
            extent.lower() if self.pad is not None else "valid"
        )  # full, same, valid
        self.no_pad_axes = no_pad_axes
        self.backend = backend

        # initialize padding
        if self.extent == "same":
            self.padding = [
                (p := ((blk - 1) // 2), (blk - 1) - p) for blk in self.patch_shape
            ]
            for axis in self.no_pad_axes:
                self.padding[axis] = (0, 0)
        elif self.extent == "full":
            self.padding = [(p := (blk - 1), p) for blk in self.patch_shape]
            for axis in self.no_pad_axes:
                self.padding[axis] = (0, 0)
        elif self.extent == "valid":
            self.padding = [(0, 0)] * arr_ndim
            self.no_pad_axes = list(range(arr_ndim))
        else:
            raise ValueError(f"Unknown 'extent' parameter: {self.extent}")

        self.padded_shape = [
            left + size + right
            for size, (left, right) in zip(self.arr_shape, self.padding)
        ]

        patch_idx_shape = [
            m - p + 1 for m, p in zip(self.padded_shape, self.patch_shape)
        ]
        self.strd_patch_idx_shape = [
            int(np.ceil(shp / strd)) for shp, strd in zip(patch_idx_shape, self.stride)
        ]
        self.num_patches = np.prod(self.strd_patch_idx_shape)

        # self._padded_buffer = None

        assert arr_ndim == len(self.patch_shape)
        assert arr_ndim == len(self.stride)
        assert arr_ndim == len(self.padding)
        assert arr_ndim == len(self.padded_shape)
        assert arr_ndim == len(self.strd_patch_idx_shape)

    # def padded_buffer(self, dtype):
    #     if self._padded_buffer is None or self._padded_buffer.dtype != dtype:
    #         self._padded_buffer = self.backend.empty(self.padded_shape, dtype)
    #     return self._padded_buffer

    def as_patches(self, arr, mode=0):
        """Image to patch windows

        Extracts overlapping patches with given stride and size
        from the image.

        Parameters
        ----------
        arr: nD array to extract patches from
        patch_shape: tuple of patch dimensions
        stride: shift of subsequent extracted patches. Default stride=1
        pad: defines the end conditions. Default pad='constant' which is
            zeros outside of the given image area.
        no_pad_axes: If pad is not None, exclude these axes from padding

        Returns
        -------
        patches: 2D array of patches [prod(patch_shape) x ~arr.size/prod(strides)]
        """
        if self.pad is not None:
            arr = self.backend.pad(
                arr, self.padding, mode=self.pad  # , out=self.padded_buffer(arr.dtype)
            )
        # Parameters
        if mode == 0:
            shp = self.strd_patch_idx_shape + self.patch_shape
            strd = [s * t for s, t in zip(self.stride, arr.strides)] + list(arr.strides)
        elif mode == 1:
            shp = self.patch_shape + self.strd_patch_idx_shape
            strd = list(arr.strides) + [s * t for s, t in zip(self.stride, arr.strides)]
        else:
            raise ValueError(f"Unknown 'mode' parameter {mode}")

        out_view = self.backend.lib.stride_tricks.as_strided(
            arr, shape=shp, strides=strd
        )
        return out_view

    def as_patch_mtx(self, arr):
        return self.as_patches(arr, mode=1).reshape(
            np.prod(self.patch_shape), self.num_patches
        )

    def random_patches(self, arr, J, seed=None):
        rng = np.random.default_rng(seed)
        out_view = self.as_patches(arr, mode=0)
        choices = np.unravel_index(
            rng.choice(
                self.num_patches,
                size=min(J, self.num_patches),
                replace=False,
            ),
            self.strd_patch_idx_shape,
        )
        return out_view[choices]

    def accumulate_patches(self, patch, T=None, out=None):
        """Patch columns to image

        Places patches obtained via im2col back where they came from
        in an image, adding pixels that are represented multiple
        times in the patch matrix.

        Parameters
        ----------
        patches: 2D array of patches [prod(patch_shape) x prod(shape)]
        shape: shape of the 2D array to place patches in
        patch_shape: tuple of patch dimensions
        stride: shift of subsequent extracted patches. Default stride=1
        pad: defines the end conditions. Default pad='constant' which is
            zeros outside of the given image area.
        no_pad_axes: parameter passed to im2col needed for reconstruction
        out: if provided an array to place output in.

        Returns
        -------
        out: array of collected pixels
        """
        if self.pad not in ["constant", "wrap", None]:
            # need to add more
            raise NotImplementedError(
                "Only pad in 'constant', None or 'wrap' are currently supported"
            )

        if out is None:
            out = self.backend.zeros(self.arr_shape, dtype=patch.dtype)
        else:
            out.fill(0)

        if T is None:
            imgs = patch.reshape((*self.patch_shape, *self.strd_patch_idx_shape))
        else:
            T = T.reshape((*self.patch_shape, -1))

        axes = list(range(len(self.arr_shape)))
        slc = [slice(None, None, strd) for strd in self.stride]

        # the idea here is that each row of a patch column matrix is a shifted version of
        #   the original image. So we reshape each image and de-shift it.
        for indxs in np.ndindex(*self.patch_shape):
            # ^^ equivalent to len(self.patch_shape) nested for loops ^^
            shifts = [(ii - (n - 1) // 2) for ii, n in zip(indxs, self.patch_shape)]
            for axis in self.no_pad_axes:
                slc[axis] = slice(
                    indxs[axis],
                    indxs[axis] + self.arr_shape[axis] - (self.patch_shape[axis] - 1),
                    self.stride[axis],
                )
                shifts[axis] = 0

            if T is None:
                img = imgs[indxs]
            else:
                img = (T[indxs] @ patch).reshape(self.strd_patch_idx_shape)

            if self.pad is not None:
                out[tuple(slc)] += self.backend.roll(
                    img, shift=shifts, axis=axes
                )  # undesirable copy...
            else:
                out[tuple(slc)] += img  # we could forego the if, but why waste a copy?
        return out

    def weighting(self, squeeze=True):
        """im2col Weights

        Weight matrix representing how many times each pixel in an image
        is represented in a patch matrix obtained via im2col.
        """
        patches = self.as_patch_mtx(self.backend.ones(self.arr_shape))
        weights = self.accumulate_patches(patches)
        if squeeze:
            weights = minimize_dims(weights)
        return weights


def minimize_dims(arr):
    was_reduced = False
    for ii in range(arr.ndim):
        z = arr.take([0], axis=ii)
        if (z == arr).all():
            arr = z
            was_reduced = True
    if was_reduced:
        arr = arr.copy()  # release reference
    return arr


def random_patches(
    arrs, J, patch_shape, stride=1, seed=None, pad=None, no_pad_axes=(), backend=np
):
    rng = np.random.default_rng(seed)
    pcs = [
        PatchConfig(
            arr_shape=arr.shape,
            patch_shape=patch_shape,
            stride=stride,
            pad=pad,
            no_pad_axes=no_pad_axes,
            backend=backend,
        )
        for arr in arrs
    ]

    # compute how many patches to collect from each array
    num_patches = np.array([pc.num_patches for pc in pcs])
    J = np.sum(num_patches) if J is None else min(J, np.sum(num_patches))
    draws = rng.multivariate_hypergeometric(num_patches, J)

    # draw patches from each array
    dtype = arrs[0].dtype
    patches = backend.empty((J, *patch_shape), dtype=dtype)
    idx = 0
    for ii in range(len(pcs)):
        end = idx + draws[ii]
        patches[idx:end] = pcs[ii].random_patches(arrs[ii], draws[ii], seed=rng)
        idx = end
    return patches


def random_patch_mtx(
    arrs, J, patch_shape, stride=1, seed=None, pad=None, no_pad_axes=(), backend=np
):
    return (
        random_patches(arrs, J, patch_shape, stride, seed, pad, no_pad_axes, backend)
        .reshape(-1, np.prod(patch_shape))
        .T
    )


# edited from https://stackoverflow.com/a/30110497/5026175
def im2win(A, patch_shape, stride=1, pad="constant", no_pad_axes=()):
    """Image to patch windows

    Extracts overlapping patches with given stride and size
    from the image.

    Parameters
    ----------
    A: nD array to extract patches from
    patch_shape: tuple of patch dimensions
    stride: shift of subsequent extracted patches. Default stride=1
    pad: defines the end conditions. Default pad='constant' which is
        zeros outside of the given image area.
    no_pad_axes: If pad is not None, exclude these axes from padding

    Returns
    -------
    patches: 2D array of patches [prod(patch_shape) x ~A.size/prod(strides)]
    """
    backend = utils.get_backend(A)  # numpy or cupy
    return PatchConfig(
        patch_shape=patch_shape,
        arr_shape=A.shape,
        stride=stride,
        pad=pad,
        no_pad_axes=no_pad_axes,
        backend=backend,
    ).as_patches(A, mode=1)


def im2col(A, patch_shape, stride=1, pad="constant", no_pad_axes=()):
    """Image to patch columns

    Extracts overlapping patches with given stride and size
    from the image and vectorizes them.

    Parameters
    ----------
    A: nD array to extract patches from
    patch_shape: tuple of patch dimensions
    stride: shift of subsequent extracted patches. Default stride=1
    pad: defines the end conditions. Default pad='constant' which is
        zeros outside of the given image area.
    no_pad_axes: If pad is not None, exclude these axes from padding

    Returns
    -------
    patches: 2D array of patches [prod(patch_shape) x ~A.size/prod(strides)]
    """
    return im2win(A, patch_shape, stride, pad, no_pad_axes).reshape(
        np.prod(patch_shape), -1
    )


def col2im(
    patch, shape, patch_shape, stride=1, pad="constant", no_pad_axes=(), out=None
):
    """Patch columns to image

    Places patches obtained via im2col back where they came from
    in an image, adding pixels that are represented multiple
    times in the patch matrix.

    Parameters
    ----------
    patches: 2D array of patches [prod(patch_shape) x prod(shape)]
    shape: shape of the 2D array to place patches in
    patch_shape: tuple of patch dimensions
    stride: shift of subsequent extracted patches. Default stride=1
    pad: defines the end conditions. Default pad='constant' which is
        zeros outside of the given image area.
    no_pad_axes: parameter passed to im2col needed for reconstruction
    out: if provided an array to place output in.

    Returns
    -------
    out: array of collected pixels
    """
    backend = utils.get_backend(patch)  # numpy or cupy
    return PatchConfig(
        patch_shape=patch_shape,
        arr_shape=shape,
        stride=stride,
        pad=pad,
        no_pad_axes=no_pad_axes,
        backend=backend,
    ).accumulate_patches(patch, out=out)


def tcol2im(
    W, patch, shape, patch_shape, stride=1, pad="constant", no_pad_axes=(), out=None
):
    """Transform Patch columns to image

    Like col2im(W@patch) but uses memory more efficiently by accumulating
    the result of each row of W multplied by patch into the output.

    Parameters
    ----------
    W: transform matrix (assumed to be square?)
    patches: 2D array of patches [prod(patch_shape) x prod(shape)]
    shape: shape of the 2D array to place patches in
    patch_shape: tuple of patch dimensions
    stride: shift of subsequent extracted patches. Default stride=1
    pad: defines the end conditions. Default pad='constant' which is
        zeros outside of the given image area.
    no_pad_axes: parameter passed to im2col needed for reconstruction
    out: if provided an array to place output in.

    Returns
    -------
    out: array of collected pixels
    """
    backend = utils.get_backend(patch)  # numpy or cupy
    return PatchConfig(
        patch_shape=patch_shape,
        arr_shape=shape,
        stride=stride,
        pad=pad,
        no_pad_axes=no_pad_axes,
        backend=backend,
    ).accumulate_patches(patch, T=W, out=out)


def im2col_weights(
    shape, patch_shape, stride=1, pad="constant", no_pad_axes=(), backend=np
):
    """im2col Weights

    Weight matrix representing how many times each pixel in an image
    is represented in a patch matrix obtained via im2col.
    """
    return PatchConfig(
        patch_shape=patch_shape,
        arr_shape=shape,
        stride=stride,
        pad=pad,
        no_pad_axes=no_pad_axes,
        backend=backend,
    ).weighting(squeeze=False)
