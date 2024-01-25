import numpy as np

from ..prox import get_prox_op_by_name as _get_prox_op_by_name
from .utl import _init_gamma
from imagelab import patches, sparsity, utils


def _init_dictionary(D0, patch_shape, K_shape=None):
    if D0 is not None:
        D0 /= (abs(D0) ** 2).sum(axis=0, keepdims=True)
        return D0
    K_shape = (
        K_shape
        if K_shape is not None
        else [np.ceil(np.sqrt(2) * s) for s in patch_shape]
    )
    return sparsity.oc_nd_dctmtx(K_shape, patch_shape).T


def blind_dl(
    x0,
    gamma,
    data_update=lambda x, wt: x / wt,
    D0=None,
    Z0=None,
    patch_config=None,
    K_shape=None,
    L=None,
    niter=120,
    prox="l0",
    callback=lambda D, itr, cost: False,
):
    """Sum of Outer Products Dictionary Learning [RNJ16]_, [RNJ17]_

    Minimizes the Blind Dictionary Learning Problem::
        x = argmin_x min_D min_Z lamb norm(Ax-y) + norm(X-DZ)_F^2 + gamma^2 nnz(Z)

    via a block coordinate minimization scheme on columns of D and rows of Z

    Parameters
    ----------
    x0:
        initial image
    gamma:
        scalar threshold
    D0:
        initial dictionary. Default overcomplete iDCT
    Z0:
        initial sparse codes. Default zeros
    patch_shape:
        shape of patches to extract and filter
    K_shape:
        shape of sparse patches, prod(K_shape) is number of atoms
    niter:
        number of iterations to run. Default = 20
    stride:
        stride between subsequent patches. Default = 1
    pad:
        padding to apply before extracting patches, equivalently convolution end conditions. Default = 'wrap'
    np_pad_axes:
        axes to not pad when extracting patches. Default = ()
    prox:
        which proximal operator to use. 'l1' for soft-thresholding, 'l0' (default) for hard-thresholding.
    callback(D,itr, cost):
        user provided function that is called with inital iterate and at the end
            of every iteration with dictionary iterate, iteration number, a function which will compute the cost.

    References
    ----------
    .. [RNJ16] S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler,
            “Sum of Outer Products Dictionary Learning for Inverse Problems,”
            IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2016, pp. 1142-1146.
    .. [RNJ17] S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler,
            “Efficient Sum of Outer Products Dictionary Learning (SOUP-DIL)
            and Its Application to Inverse Problems,”
            IEEE Transactions on Computational Imaging, vol. 3, no. 4, pp. 694-709, Dec. 2017.

    """
    backend = utils.get_backend(x0)  # numpy or cupy
    linalg = utils.get_backend_linalg(x0)

    prox_op = _get_prox_op_by_name(prox)
    gamma = _init_gamma(niter, gamma)

    wt = patch_config.weighting().astype(x0.dtype)
    # initialize X
    x = x0
    X_patch = patch_config.as_patch_mtx(x0)
    # initialize D
    D = backend.array(_init_dictionary(D0, patch_config.patch_shape, K_shape)).astype(
        x0.dtype
    )
    n, K = D.shape
    L = X_patch.shape[1]
    Z = Z0 if Z0 is not None else backend.zeros((K, L))
    d0 = backend.zeros_like(D[:, 0])
    d0[0] = 1

    if callback(x0, 0):
        return x0
    for itr in range(1, niter + 1):
        X_patch = patch_config.as_patch_mtx(x)
        for kk in range(K):
            zk_old = Z[kk, :]  # <- can optimize to pull only non-zero values
            dk_old = D[:, kk]

            # sparse coding
            # note: zk_new = dk_old'R, i.e. this eqn depends on updating z before d
            # so that dk_old'dk_old = 1 can be used   # dk'dk*zk (=1*zk)
            zk_new = dk_old.conj() @ X_patch + zk_old - (dk_old.conj() @ D) @ Z
            zk_new = prox_op(zk_new, gamma[itr - 1], backend)

            # dictionary update
            if not zk_new.any():
                dk_new = d0  # or set to a random unit vector?
            else:
                dk_new = (  # = Rzk_new'
                    (X_patch @ (zk_new.conj()))
                    - D @ (Z @ (zk_new.conj()))
                    + dk_old * (zk_old @ (zk_new.conj()))
                )  # we add back dk_old zk_old because that is what is in DZ still
                dk_new /= linalg.norm(dk_new)  # update kth column of D

            Z[kk, :] = zk_new
            D[:, kk] = dk_new

        xd = patch_config.accumulate_patches(T=D, patch=Z, out=x)

        x = data_update(xd, wt)

        if callback(x, itr):
            break
    return x


def learn_dictionary(
    x0,
    gamma,
    D0=None,
    Z0=None,
    patch_shape=(8, 8),
    K_shape=None,
    L=None,
    niter=20,
    stride=1,
    pad=None,
    no_pad_axes=(),
    seed=None,
    prox="l0",
    callback=lambda D, itr, cost: False,
):
    """One-stage Dictionary Learning (OS-DL) [SBJ14]_.

    Further explored and redubbed as
    Sum of Outer Products Dictionary Learning (SOUP)
    in [RNJ16]_, [RNJ17]_

    Minimizes the Dictionary Learning Problem::
        D = argmin_D min_Z norm(X-DZ)_F^2 + gamma^2 nnz(Z)

    via a block coordinate minimization scheme on columns of D and rows of Z

    Parameters
    ----------
    x0:
        clean training data, list or array of training arrays
    gamma:
        scalar threshold
    D0:
        initial dictionary. Default overcomplete iDCT
    Z0:
        initial sparse codes. Default zeros
    patch_shape:
        shape of patches to extract and filter
    K_shape:
        shape of sparse patches, prod(K_shape) is number of atoms
    L:
        number of patches to learn from
    niter:
        number of iterations to run. Default = 20
    stride:
        stride between subsequent patches. Default = 1
    pad:
        padding to apply before extracting patches, equivalently convolution end conditions. Default = 'wrap'
    np_pad_axes:
        axes to not pad when extracting patches. Default = ()
    prox:
        which proximal operator to use. 'l1' for soft-thresholding, 'l0' (default) for hard-thresholding.
    callback(D,itr, cost):
        user provided function that is called with inital iterate and at the end
            of every iteration with dictionary iterate, iteration number, a function which will compute the cost.

    References
    ----------
    .. [SBJ14] M. Sadeghi, M. Babie-Zadeh, C. Jutten
            "Learning Overcomplete Dictionaries Based on Atom-by-Atom Updating,"
            IEEE Trans. on Singal Processing, vol. 62, no. 4, page 883-891, Feb. 2014
    .. [RNJ16] S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler,
            “Sum of Outer Products Dictionary Learning for Inverse Problems,”
            IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2016, pp. 1142-1146.
    .. [RNJ17] S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler,
            “Efficient Sum of Outer Products Dictionary Learning (SOUP-DIL)
            and Its Application to Inverse Problems,”
            IEEE Transactions on Computational Imaging, vol. 3, no. 4, pp. 694-709, Dec. 2017.

    """
    backend = utils.get_backend(x0)  # numpy or cupy
    linalg = utils.get_backend_linalg(x0)

    prox_op = _get_prox_op_by_name(prox)
    # initialize X
    X_patch = patches.random_patch_mtx(
        x0,
        L,
        patch_shape,
        stride=stride,
        pad=pad,
        no_pad_axes=no_pad_axes,
        backend=backend,
        seed=seed,
    )
    # initialize D
    D = backend.array(_init_dictionary(D0, patch_shape, K_shape)).astype(x0[0].dtype)
    n, K = D.shape
    L = X_patch.shape[1]
    Z = Z0 if Z0 is not None else backend.zeros((K, L))
    d0 = backend.zeros_like(D[:, 0])
    d0[0] = 1

    def cost(D):
        return float(
            linalg.norm(X_patch - D @ Z, "fro") ** 2 + gamma ** 2 * backend.sum(Z != 0)
        )

    if callback(D, 0, cost=cost):
        return D
    for itr in range(1, niter + 1):
        for kk in range(K):
            zk_old = Z[kk, :]  # <- can optimize to pull only non-zero values
            dk_old = D[:, kk]

            # sparse coding
            # note: zk_new = dk_old'R, i.e. this eqn depends on updating z before d
            # so that dk_old'dk_old = 1 can be used   # dk'dk*zk (=1*zk)
            zk_new = dk_old.conj() @ X_patch + zk_old - (dk_old.conj() @ D) @ Z
            zk_new = prox_op(zk_new, gamma, backend)

            # dictionary update
            if not zk_new.any():
                dk_new = d0  # or set to a random unit vector?
            else:
                dk_new = (  # = Rzk_new'
                    (X_patch @ (zk_new.conj()))
                    - D @ (Z @ (zk_new.conj()))
                    + dk_old * (zk_old @ (zk_new.conj()))
                )  # we add back dk_old zk_old because that is what is in DZ still
                dk_new /= linalg.norm(dk_new)  # update kth column of D

            Z[kk, :] = zk_new
            D[:, kk] = dk_new
        if callback(D, itr, cost=cost):
            break
    return D


# This should be equivalent, but slower
# (but easier to follow from derivations)
def learn_dictionary_simple(
    x0,
    gamma,
    D0=None,
    Z0=None,
    patch_shape=(8, 8),
    K_shape=None,
    L=None,
    niter=20,
    stride=1,
    pad=None,
    no_pad_axes=(),
    seed=None,
    prox="l0",
    callback=lambda D, itr, cost: False,
):
    backend = utils.get_backend(x0)  # numpy or cupy
    linalg = utils.get_backend_linalg(x0)

    prox_op = _get_prox_op_by_name(prox)
    X_patch = patches.random_patch_mtx(
        x0,
        L,
        patch_shape,
        stride=stride,
        pad=pad,
        no_pad_axes=no_pad_axes,
        backend=backend,
        seed=seed,
    )
    D = backend.array(_init_dictionary(D0, patch_shape, K_shape)).astype(x0[0].dtype)
    n, K = D.shape
    L = X_patch.shape[1]
    Z = Z0 if Z0 is not None else backend.zeros((K, L))
    d0 = backend.zeros_like(D[:, 0])
    d0[0] = 1

    R = X_patch - D @ Z

    def cost(D):
        return float(linalg.norm(R, "fro") ** 2 + gamma ** 2 * backend.sum(Z != 0))

    if callback(D, 0, cost=cost):
        return D
    for itr in range(1, niter + 1):
        for kk in range(K):
            zk = Z[kk, :]  # <- can optimize to pull only non-zero values
            dk = D[:, kk]

            R += backend.outer(dk, zk)

            # sparse coding
            zk = prox_op(dk.conj() @ R, gamma, backend)

            # dictionary update
            if not zk.any():
                dk = d0  # or set to a random unit vector?
            else:
                dk = R @ zk.conj()
                dk /= linalg.norm(dk)  # update kth column of D

            Z[kk, :] = zk
            D[:, kk] = dk
            R -= backend.outer(dk, zk)
        if callback(D, itr, cost=cost):
            break
    return D


def sparse_code(
    D, X_patch, gamma, Z0=None, prox="l0", niter=1, callback=lambda D, itr, cost: False
):
    """ D must have unit norm columns!"""
    backend = utils.get_backend(X_patch)  # numpy or cupy
    linalg = utils.get_backend_linalg(X_patch)
    prox_op = _get_prox_op_by_name(prox)
    if not np.allclose(
        D, D / linalg.norm(D, axis=0, keepdims=True)
    ):  # ensure normalization
        raise ValueError("Dictionary does not have unit norm columns")
    n, K = D.shape
    L = X_patch.shape[1]
    Z = Z0 if Z0 is not None else backend.zeros((K, L))

    def cost(Z):
        return (
            linalg.norm(X_patch - D @ Z, "fro") ** 2 + gamma ** 2 * backend.sum(Z != 0)
        ).get()

    if callback(Z, 0, cost=cost):
        return Z
    for itr in range(1, niter + 1):
        for kk in range(K):
            zk = Z[kk, :]
            dk = D[:, kk]
            zk = dk.conj() @ X_patch - (dk.conj() @ D) @ Z + zk  # dk'dk*zk (=1*zk)
            zk = prox_op(zk, gamma, backend)
            Z[kk, :] = zk
        if callback(Z, itr, cost=cost):
            break
    return Z


def denoise(
    D,
    y,
    gamma,
    Z0=None,
    prox="l0",
    niter=10,
    patch_config=None,
    callback=lambda D, itr, cost: False,
):
    Y_patch = patch_config.as_patch_mtx(y)
    wts = patch_config.weighting().astype(y.dtype)
    Z = sparse_code(D, Y_patch, gamma, Z0, prox, niter, callback)
    xd = patch_config.accumulate_patches(T=D, patch=Z)
    return xd / wts
