""" Unitary Transform Learning
see docstring for `utl.utl` for more information
"""
from numbers import Number

import numpy as np

from .. import sparsity, utils
from ..prox import get_prox_op_by_name as _get_prox_op_by_name
from imagelab import fft

# Guides to Variable Names
#   T is a Transform
#   x is an image/signal, not its patches
#   xd is a denoised version of x
#   X_patches are patches extracted from x
#   gamma is the threshold of the proximal operator
#   niter is number of iterations
#   ninner is number of inner iterations
#   patch_shape refers to a window extracted
#       from x for processing, in a filtering
#       interpretation, this is the filter kernel shape
#   stride is the space corresponding pixels in
#       neighboring patches


def utl_fit(T, x, gamma, prox="l0", **patch_args):
    """computes ||TX - Z||_F^2 + gamma^2 ||Z||_0 """
    # this could be optimized for less memory usage
    # by special casing the prox ops and computing
    # the residual directly with the sparsity
    prox_op = _get_prox_op_by_name(prox, inplace=False)
    X_patch = utils.im2col(x, **patch_args)
    B = T @ X_patch
    Z = prox_op(B, gamma)
    diff = np.sum(np.abs(B - Z) ** 2)
    return diff + (gamma ** 2) * (np.count_nonzero(Z))


def blind_utl(
    x0,
    data_update=lambda xd, wt: xd / wt,
    gamma=None,
    T0=None,
    patch_config=None,
    niter=120,
    ninner=1,
    prox="l0",
    callback=lambda X, itr: None,
):
    """Blind Unitary Transform Learning [RB15]_, [RB12]_, [WRB15]_

    Algorithm is globally convergent in cost function value from
    any starting point

    .. math::
        x = argmin_X f(x) + R(x)

        where R(x) = min_{T,Z} sum_j ||TPjx-Z||_2^2 + gam^2*||Z||_0

                     subject to T'T = I

    where Pj extracts the 'jth' patch from the image x.

    The unitary constraint allows for efficient closed form updates
    of both T and Z.

    Parameters
    ----------
    x0:
        initial iterate
    data_update(xd,wt):
        function to apply data update with denoised image and weight.
                xd = sum_j Pj'T'Z    wt = diag(sum_j Pj'Pj)
                For example if f(X) is lamb*||A(x) - Y||_2^2, then
                data_update(xd,wt) = inv(lamb*A'A + wt)@(lamb*A'(Y) + xd)
                which can be computed efficiently when A'A is diagonal
                such as inpainting, denoising and single-coil MRI.
                Default is for f(X) = 0, i.e. inv(diag(wt))@xd.
    gamma:
        scalar or vector of length niter of thresholds for each iteration.
            Default is a geometric progression from 0.1 to 0.05.
            Consider roughly 1/sigma_noise (ref?).
    patch_shape:
        shape of patches to extract and filter
    niter:
        number of iterations to run. Default = 120
    ninner:
        number of transform update/sparse code updates to run
        before data update. Default = 1
    stride:
        stride between subsequent patches. Default = 1
    pad:
        padding to apply before extracting patches, equivalently convolution
        end conditions. Default = 'wrap'
    np_pad_axes:
        axes to not pad when extracting patches. Default = ()
    prox:
        which proximal operator to use. 'l1' for soft-thresholding,
        'l0' (default) for hard-thresholding.
    callback(x, itr):
        user provided function that is called with initial iterate and
        at the end of every iteration with image iterate and iteration number.

    Example
    -------
    >>> import utl
    >>> X_true  = imread(...)
    >>> X_true = X_true/X_true.max()
    >>> X0  = X_true + 0.1*np.random.randn(*X_true.shape)
    >>> data_update = lambda X, wt: (X + 1e-6*X0)/(wt + 1e-6) # denoising
    >>> X_hat = utl.blind_utl(X0, data_update)

    Notes
    -----
    This algorithm is patented by University of Illinois'
        Transform Learning group.See United States Patent 9734601,
        "Highly accelerated imaging and image reconstruction using
        adaptive sparsifying transforms"

    References
    ----------
    .. [RB15] Ravishankar S, Bresler Y
            "Efficient Blind Compressed Sensing Using Sparsifying Transforms
            with Convergence Guarantees and Application to MRI"
            SIAM J Imaging Science, arXiv:1501.02923
    .. [RB12] Ravishankar S, Bresler Y
            "Closed-form solutions within sparsifying transform learning"
            ICASSP 2013, 5378-5382
    .. [WRB15] Wen B, Ravishankar S, Bresler Y
            "Video Denoising by Online 3D sparsifying Transform Learning"
            ICIP 2015, doi:10.1109/ICIP.2015.7350771

    """
    backend = utils.get_backend(x0)  # numpy or cupy
    svd = utils.get_backend_linalg(x0).svd
    gamma = _init_gamma(niter, gamma)

    prox_op = _get_prox_op_by_name(prox)

    wt = patch_config.weighting().astype(x0.dtype)

    x = x0
    X_patch = patch_config.as_patch_mtx(x)  # redundant with first iter...
    T0 = (
        backend.array(sparsity.nd_dctmtx(*patch_config.patch_shape)).astype(x.dtype)
        if T0 is None
        else T0
    )
    T = T0
    B = T @ X_patch
    Z = prox_op(B, gamma[0], backend)

    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):

        X_patch = patch_config.as_patch_mtx(x)

        for _inner in range(ninner):
            # Transform Update
            U, _, Vh = svd(X_patch @ Z.conj().T, full_matrices=True)
            # T = Vh.conj().T@U.conj().T
            # T = np.dot(U, Vh, out=T).conj().T
            T = (U @ Vh).conj().T
            # T = np.matmul(U,Vh, out=T).conj().T

            # Sparse Code Update
            B = T @ X_patch
            Z = prox_op(B, gamma[itr - 1], backend)
            # Z = B*(np.abs(B) >= gamma[itr-1])

        xd = patch_config.accumulate_patches(T=T.conj().T, patch=Z, out=x)

        x = data_update(xd, wt)

        if callback(x, itr):
            break
    T0[:] = T.real if backend.isrealobj(T0) else T
    return x


def _init_gamma(niter, gamma=None, warmup=-30, factor=16):
    if isinstance(gamma, dict):
        warmup = gamma.get("warmup", warmup)
        factor = gamma.get("factor", factor)
        gamma = gamma.get("gamma", None)
    if warmup < 0:
        warmup = warmup + niter
    if gamma is None:
        gamma = 0.05
    if isinstance(gamma, Number):
        gamma = gamma * np.ones(niter)
        if warmup > 0:
            gamma[:warmup] = np.geomspace(
                start=gamma[-1] * factor, stop=gamma[-1], num=warmup
            )
    return gamma


def get_mri_data_update(Y, mask, lamb, **kwargs):
    """Assumes every pixel is in equal number of patches
    i.e. Pj'Pj is a scaled identity
    """
    inpainting_dup = get_inpainting_data_update(Y, mask, lamb)

    device = "gpu" if "cupy" in type(mask).__module__ else "cpu"
    fft2, ifft2 = fft.get_fft2_pair(device=device, **kwargs)

    def data_update(xd, wt):
        Yd = fft2(xd)
        Yi = inpainting_dup(Yd, wt)
        return ifft2(Yi)

    return data_update


def get_inpainting_data_update(Y, mask, lamb=1e8):
    lamb_mask2 = lamb * mask ** 2
    lamb_Y = lamb * mask * Y

    def data_update(xd, wt):
        return (xd + lamb_Y) / (wt + lamb_mask2)

    return data_update


def get_denoising_data_update(Y, lamb=1e-4):
    return get_inpainting_data_update(Y, 1, lamb)


def get_generic_data_update(Y, A, lamb, niter=5):
    import imagelab as il
    import imagelab.linop
    import imagelab.optim

    B = [A, il.linop.I]
    y = Y.reshape(-1)
    # backend = utils.get_backend(y)

    def data_update(xd, wt):
        # xbar = xd.reshape(-1)
        # D = wt.reshape(-1)
        grad = [
            lambda x: lamb * (x - y),
            lambda x: (wt * x.reshape(xd.shape) - xd).reshape(-1),
            # lambda x: D * x - xbar,
        ]
        Lgf = [lamb, np.max(wt)]
        x0 = xd / wt
        x0[np.isnan(x0)] = 0
        return il.optim.ncg_inv(
            B,
            grad,
            Lgf,
            x0=x0,
            niter=niter,
            ninner=1,  # b/c quadratic!
        )

    return data_update


def utlmri(Y, mask, lamb, *args, **kwargs):  # Not Tested!
    """Unitary Transform Learning for Single-Coil Undersampled MRI

    This is a convenience function that sets up the data_update for
    the utl function.

    Parameters
    ----------
    Y:
        The zero filled k-space data with the origin at the center
    mask:
        the k-space sampling mask
    lamb:
        data-fidelity hyperparameter ≥ 0
            large-values correspond to greater confidence in data (less noise)
            so if data is (assumed to be) noiseless, use ~ 1e8
    *args:
        args for `utl`
    **kwargs:
        keyword args for `utl`
    """
    X0 = np.fft.ifft2(np.fft.ifftshift(Y))  # A'Y
    return blind_utl(X0, get_mri_data_update(Y, mask, lamb), *args, **kwargs)


# non-blind utl


def learn_transform(
    X,
    gamma=None,
    T0=None,
    patch_shape=(8, 8),
    niter=20,
    stride=1,
    pad="wrap",
    no_pad_axes=(),
    prox="l0",
    callback=lambda T, itr, cost: None,
):
    """Non-Blind Unitary Transform Learning [RB15v]_, [RB12v]_, [WRB15v]_

    Algorithm is globally convergent in cost function value from any
    starting point

    .. math::
        T = argmin_{T,Z} sum_t sum_j ||TPjXt-Zt||_2^2 + gam^2*||Zt||_0

                     subject to T'T = I

    where Pj extracts the 'jth' patch from the image 't-th' image Xt.

    The unitary constraint allows for efficient closed form updates
    of both T and Z.

    Parameters
    ----------
    X:
        clean training data, list or array of training arrays
    gamma:
        scalar threshold
            if gamma is too large, Z will be zero and
                the T update will return an identity matrix
            if gamma is too small, the Z update will be a no-op (for l0)
                and T will stay close to T0.
    patch_shape:
        shape of patches to extract and filter
    niter:
        number of iterations to run. Default = 20
    stride:
        stride between subsequent patches. Default = 1
    pad:
        padding to apply before extracting patches, equivalently convolution
        end conditions. Default = 'wrap'
    np_pad_axes:
        axes to not pad when extracting patches. Default = ()
    prox:
        which proximal operator to use. 'l1' for soft-thresholding,
        'l0' (default) for hard-thresholding.
    callback(T, itr, cost):
        user provided function that is called with initial iterate and
        at the end of every iteration with transform iterate, iteration
        number, a function which will compute the cost.

    Notes
    -----
    This algorithm is patented by University of Illinois'
        Transform Learning group.See United States Patent 9734601,
        "Highly accelerated imaging and image reconstruction using
        adaptive sparsifying transforms"

    References
    ----------
    ..  [RB15v] Ravishankar S, Bresler Y
            "Efficient Blind Compressed Sensing Using Sparsifying Transforms
            with Convergence Guarantees and Application to MRI"
            SIAM J Imaging Science, arXiv:1501.02923
    ..  [RB12v] Ravishankar S, Bresler Y
            "Closed-form solutions within sparsifying transform learning"
            ICASSP 2013, 5378-5382
    ..  [WRB15v] Wen B, Ravishankar S, Bresler Y
            "Video Denoising by Online 3D sparsifying Transform Learning"
            ICIP 2015, doi:10.1109/ICIP.2015.7350771

    """
    backend = utils.get_backend(X[0])  # numpy or cupy
    svd = utils.get_backend_linalg(X[0]).svd
    prox_op = _get_prox_op_by_name(prox)
    gamma = _init_gamma(niter, gamma)

    # redundant with first iter...
    X_patch = [
        utils.im2col(Xt, patch_shape, stride=stride, pad=pad, no_pad_axes=no_pad_axes)
        for Xt in X
    ]
    T = (
        backend.array(sparsity.nd_dctmtx(*patch_shape)).astype(X[0].dtype)
        if T0 is None
        else T0
    )
    B = [T @ Xt_patch for Xt_patch in X_patch]
    Z = [prox_op(Bt, gamma[0], backend) for Bt in B]

    def cost(T):
        return backend.sum(
            sum(
                [
                    backend.abs(T @ Xt_patch - Zt) ** 2
                    for Xt_patch, Zt in zip(X_patch, Z)
                ]
            )
            + (gamma[-1] ** 2) * backend.sum([backend.count_nonzero(Zt) for Zt in Z])
        )

    if callback(T, 0, cost=cost):
        return T
    for itr in range(1, niter + 1):

        # Transform Update
        A = sum([Xt_patch @ Zt.conj().T for Xt_patch, Zt in zip(X_patch, Z)])
        U, _, Vh = svd(A, full_matrices=True)
        T = (U @ Vh).conj().T

        # Sparse Code Update
        B = [T @ Xt_patch for Xt_patch in X_patch]
        Z = [prox_op(Bt, gamma[itr - 1], backend) for Bt in B]

        if callback(T, itr, cost=cost):
            break
    return T


# this isn't UTL specific (just TL), so it may move?
def apply_transform_reg(
    T,
    x0,
    data_update=lambda xd, wt: xd / wt,
    gamma=None,
    patch_config=None,
    niter=120,
    prox="l0",
    callback=lambda X, itr: None,
):
    """ see utl.learn_transform """
    backend = utils.get_backend(x0)  # numpy or cupy

    gamma = _init_gamma(niter, gamma)

    prox_op = _get_prox_op_by_name(prox)

    wt = patch_config.weighting().astype(x0.dtype)

    x = x0
    X_patch = patch_config.as_patch_mtx(x)  # redundant with first iter...
    B = T @ X_patch
    Z = prox_op(B, gamma[0], backend)

    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):

        X_patch = patch_config.as_patch_mtx(x)

        # Sparse Code Update
        B = T @ X_patch
        Z = prox_op(B, gamma[itr - 1], backend)

        xd = patch_config.accumulate_patches(T=T.conj().T, patch=Z, out=x)
        x = data_update(xd, wt)

        if callback(x, itr):
            break
    return x
