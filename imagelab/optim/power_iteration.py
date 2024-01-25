from typing import Callable

import numpy as np
from scipy.linalg import norm

from .. import linop
from ..utils import export


@export
@linop.utils.ensure_vec_out("A")
def power_iteration(
    A, niters: int = 500, seed=None, callback: Callable = lambda b, u, itr: False
):
    """Computes the dominant eigenvector and eigenvalue of a square,
    diagonizable matrix A.

    Parameters
    ----------
    A: usually a matrix/2D array, but anything that supports
        __matmul__ and .shape will work
    niters: number of iterations to run (default: 500)
    callback: a function taking two arguments bk, uk where
        bk is the estimate of the leading eigenvector
        at iteration k, uk the leading eigenvalue.
        Default callback(bk,uk) -> False

    Returns
    -------
    b: the leading eigenvector, or principal component
    u: the leading eigenvalue, or spectral radius

    Note
    ----
    If the two leading eigenvalues are equal, the leading
    eigen vector returned will be a random vector in the
    span of the two leading eigenvectors

    There is a non-zero probability that the original
    estimate will be orthogonal to the leading eigenvector
    causing the algorithm to return the 2nd leading eigenpair

    """
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.default_rng(seed).random(A.shape[1])
    u = 1

    if callback(b_k, u, 0):
        return b_k, u
    for itr in range(niters):
        # calculate the matrix-by-vector product Ab
        b_k1 = A @ b_k
        # Rayleigh quotient
        u = b_k1 @ b_k / (b_k @ b_k)
        # re normalize the vector
        b_k = b_k1 / norm(b_k1)
        if callback(b_k, u, itr):
            break

    return b_k, u
