""" Based on an implementation in Raj Rao Nadakuditi's EECS 505 class
at the University of Michigan.
"""
import numpy as np
from scipy.linalg import svd


def optshrink(Y, r):
    """
    Perform rank-r denoising of data matrix Y using OptShrink [N14]_

    Parameters
    ----------
    Y : 2D array where Y = X + noise, and goal is to estimate X
    r : estimated rank of X

    Returns
    -------
    Xh : rank-r estimate of X using OptShrink weights for SVD components

    Notes
    -----
    This version works even if one of the dimensions of Y is large,
    as long as the other is sufficiently small. (We need to find
    a truncated SVD of Y).

    References
    ---------_
    ..  [N14] Nadakuditi, R. R.
            "OptShrink: An Algorithm for Improved Low-Rank Signal Matrix
            Denoising by Optimal, Data-Driven Singular Value Shrinkage"
            IEEE Trans. on Information Theory, Vol 60, 5, May 2014
            http://doi.org/10.1109/TIT.2014.2311661

    """
    if len(Y.shape) != 2:
        raise ValueError(
            f"Input Y must have only 2 dimensions, but the provided array has {len(Y.shape)}"
        )

    U, s, Vh = svd(Y, full_matrices=False)

    r = min([r, *Y.shape])

    sn = s[r + 1 :]
    m = max(Y.shape) - r  # what if r = Y.shape[0]?
    n = min(Y.shape) - r
    sm = np.zeros(sn.shape[0] + m - n, dtype=sn.dtype)
    sm[: -(m - n)] = sn

    w = np.zeros(r)
    for k in range(r):  # get rid of this loop with broadcasting?
        z = s[k]

        inv_n = 1 / (z ** 2 - sn ** 2)
        inv_m = 1 / (z ** 2 - sm ** 2)

        D1 = np.sum(z * inv_n) / n
        D2 = np.sum(z * inv_m) / m

        D = D1 * D2  # eq (16a)

        # derivative of D transform
        D1_der = np.sum(-2 * z ** 2 * inv_n ** 2 + inv_n) / n
        D2_der = np.sum(-2 * z ** 2 * inv_m ** 2 + inv_m) / m

        D_der = D1 * D2_der + D2 * D1_der  # eq (16b) in paper
        w[k] = -2 * D / D_der

    Xh = (U[:, :r] * w) @ Vh[:r, :]

    return Xh
