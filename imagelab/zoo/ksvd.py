import numpy as np

from .soupdil import _init_dictionary
from .soupdil import sparse_code as soup_sc
from imagelab import patches, utils


def _sparse_code(D, X_patch, Z0=None):
    return soup_sc(
        D,
        X_patch,
        gamma=0.1,
        Z0=Z0,
        prox="l0",
        niter=3,
    )


def learn_dictionary(
    x0,
    gamma,
    D0=None,
    Z0=None,
    patch_shape=(8, 8),
    K_shape=(21, 21),
    L=None,
    niter=20,
    stride=1,
    pad=None,
    no_pad_axes=(),
    seed=None,
    callback=lambda D, itr, cost: False,
):
    """K-SVD [AEB06]

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
    callback(D,itr, cost):
        user provided function that is called with inital iterate and at the end
            of every iteration with dictionary iterate, iteration number, a function which will compute the cost.

    References
    ----------
    .. [AEB06] M. Aharon, M. Elad and A. Bruckstein,
            “K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation,”
            IEEE Transactions on Signal Processing, vol. 54, no. 11, pp. 4311-4322, Nov. 2006.
            DOI: 10.1109/TSP.2006.881199

    """
    backend = utils.get_backend(x0)  # numpy or cupy
    linalg = utils.get_backend_linalg(x0)

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

    R = X_patch

    def cost(D):
        return float(linalg.norm(R, "fro") ** 2 + gamma ** 2 * backend.sum(Z != 0))

    if callback(D, 0, cost=cost):
        return D
    for itr in range(1, niter + 1):
        Z = _sparse_code(D, X_patch, Z0=Z)
        R = X_patch - D @ Z
        for kk in range(K):
            zk = Z[kk, :]  # <- can optimize to pull only non-zero values
            dk = D[:, kk]

            R += backend.outer(dk, zk)

            # dictionary update
            if not zk.any():
                dk = d0  # or set to a random unit vector?
            else:
                # rest = R[:, zk != 0]
                # print(rest.shape, rest.sum())
                omega = zk != 0
                try:
                    # this is slow, it computes more than just the first left/right
                    # singular vector and value that we end up not using
                    U, s, Vh = linalg.svd(R[:, omega], full_matrices=False)
                except Exception as e:
                    if "CUSOLVER" not in type(e).__name__:
                        raise e  # avoid importing cupy to catch specific exception
                    # Cupy fails when R is too large :(
                    U, s, Vh = [
                        backend.array(A)
                        for A in np.linalg.svd(R[:, omega].get(), full_matrices=False)
                    ]
                dk = U[:, 0]
                zk[omega] = s[0] * Vh[0, :]
                # ^depends on l0 norm not changing value since support doesn't change

            Z[kk, :] = zk
            D[:, kk] = dk
            R -= backend.outer(dk, zk)
        if callback(D, itr, cost=cost):
            break
    return D


def approx_ksvd(
    x0,
    gamma,
    D0=None,
    Z0=None,
    patch_shape=(8, 8),
    K_shape=None,
    L=None,
    niter=20,
    ninner1=1,
    ninner2=1,
    stride=1,
    pad=None,
    no_pad_axes=(),
    seed=None,
    callback=lambda D, itr, cost: False,
):
    """Approximate K-SVD [RZE08]

    see also PAU-DL [SBJ14] for variations on inner loops

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
    ninner1:
        number of alternations of all k before performing sparse coding
        see PAU-DL [SBJ14] on why increasing this is better than `ninner2`.
        Default = 1
    ninner2:
        number of alternations on dk zk before moving onto next k. Default = 1
    stride:
        stride between subsequent patches. Default = 1
    pad:
        padding to apply before extracting patches, equivalently convolution end conditions. Default = 'wrap'
    np_pad_axes:
        axes to not pad when extracting patches. Default = ()
    callback(D,itr, cost):
        user provided function that is called with inital iterate and at the end
            of every iteration with dictionary iterate, iteration number, a function which will compute the cost.

    References
    ----------
    .. [RZE08] R. Rubinstein, M. Zibulevsky and M. Elad,
            “Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit,”
            Technical Report, 2008
    .. [SBJ14] M. Sadeghi, M. Babie-Zadeh, C. Jutten
            "Learning Overcomplete Dictionaries Based on Atom-by-Atom Updating,"
            IEEE Trans. on Singal Processing, vol. 62, no. 4, page 883-891, Feb. 2014

    """
    backend = utils.get_backend(x0)  # numpy or cupy
    linalg = utils.get_backend_linalg(x0)

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

    def cost(D):
        return float(
            linalg.norm(X_patch - D @ Z, "fro") ** 2 + gamma ** 2 * backend.sum(Z != 0)
        )

    if callback(D, 0, cost=cost):
        return D
    for itr in range(1, niter + 1):
        Z = _sparse_code(D, X_patch, Z0=Z)
        for _ii in range(ninner1):
            for kk in range(K):
                zk = Z[kk, :]
                D[:, kk] = 0  # this removes the kth outer product

                # dictionary update
                if not zk.any():
                    dk = d0  # or set to a random unit vector?
                else:
                    omega = zk != 0
                    for _jj in range(ninner2):
                        # should approximate svd as `ninner2` increases
                        dk = (X_patch[:, omega] @ (zk[omega].conj())) - D @ (
                            Z[:, omega] @ (zk[omega].conj())
                        )
                        dk /= linalg.norm(dk)
                        zk[omega] = (
                            dk.conj() @ X_patch[:, omega]
                            - (dk.conj() @ D) @ Z[:, omega]
                        )

                Z[kk, :] = zk
                D[:, kk] = dk
        if callback(D, itr, cost=cost):
            break
    return D
