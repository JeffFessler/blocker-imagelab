"""imagelab/optim/newton.py

2nd Order and quasi-newton optimization methods

"""
from collections import deque
from typing import Callable, List

import numpy as np
from scipy.linalg import norm

from .. import linop
from ..linop import LinearOperator as _LinearOperator  # typing
from ..utils import export
from .grad import gd

_Identity = linop.NullIdentity()
_Identity.vec_out = True


@export
@linop.utils.ensure_vec_out("H0")
def lbfgs(
    grd: Callable,
    Lg: float,
    x0,
    *,
    niter: int = 100,
    ninner: int = 10,
    H0: _LinearOperator = _Identity,
    m: int = 5,
    callback: Callable = lambda x, itr: False
):
    """Minimizes a smooth cost function using the limited-memory
    Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm [N80]_.

    x = x - inv(H)@grad(x)

    where H is the approximate Hessian given recursively by
    H_k+1 = (I-ro_k*s_k y_k')H_k(I - ro_k*y_k s_k') + ro_k*s_k s_k'
    Note that the product inv(H)@grad(x) is computed efficiently
    without forming H

    Parameters
    ----------
    grad:
        function that computes gradient of cost function
    x0:
        initial guess

    Other Parameters
    ----------------
    niter:
        number of iterations (default 100)
    m:
        number of iterates to store to approximate Hessian
    callback:
        user-defined function to be evaluated with two arguments (x,itr)
           it is evaluated at (x0,0) and then after each iteration

    Returns
    -------
    final iterate

    References
    ----------
    ..  [N80] Nocedal, J,
         "Updating Quasi-Newton Matrices With Limited Storage",
            Mathematics of Computation, Vol 35, #151, July 1980, 773-782
    """

    memory = deque([], m)
    x = x0.reshape(-1)
    g = 0
    gamma = 1
    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        g_old = g
        g = grd(x)

        # Compute BFGS update d=inv(H)@g
        d = -g  # descent direction
        # compute the right product
        alfs = []
        for ro, s, y in reversed(memory):
            alf = ro * s @ d
            d -= alf * y
            alfs.append(alf)
        # compute center product
        d = H0 @ d
        d *= gamma  # scale by gamma
        # compute left product
        for alf, (ro, s, y) in zip(reversed(alfs), memory):
            beta = ro * y @ d
            d += (alf - beta) * s
        # End BFGS update
        # d = -d

        # Line search
        # derivative of h(a) = cost(x + a * dir)
        def dh(alf):
            return d.conj().T @ grd(x + alf * d)

        Ldh = norm(d) ** 2 * Lg  # Lipschitz constant for dh
        alf = gd(dh, Ldh, 0, niter=ninner)

        # Update params
        y = g - g_old
        s = alf * d
        ro = 1 / (y @ s)
        gamma = (s @ y) / (y @ y)

        x += s

        memory.append((ro, s, y))

        if callback(x, itr):
            break
    return x.reshape(x0.shape)


@export
@linop.utils.ensure_vec_out("B", "H0")
def lbfgs_inv(
    B: List[_LinearOperator],
    gf: List[Callable],
    Lgf: List[float],
    x0,
    *,
    niter: int = 100,
    ninner: int = 10,
    H0: _LinearOperator = _Identity,
    m: int = 5,
    callback: Callable = lambda x, itr: False
):
    """
    Minimizes a smooth cost function using the limited-memory
    Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm

    Parameters
    ----------
    B:
        array of J blocks B_1,...,B_J
    gf:
        array of J functions for computing gradients of f_1,...,f_J
    Lgf:
        array of J Lipschitz constants for those gradients
    x0:
        initial guess

    Other Parameters
    ----------------
    niter:
        number of outer PSD iterations (default 100)
    ninner:
        number of inner iterations of GD for line search (default 10)
    H0:
        initial estimate of hessian (default _Identity)
    m:
        number of iterations to remember for hessian approximation
    callback:
        user-defined function to be evaluated with two arguments (x,iter).
              It is evaluated at (x0,0) and then after each iteration

    Returns
    -------
    final iterate:
    """

    J = len(B)
    memory = deque([], m)
    g = 0
    gamma = 1
    x = x0.reshape(-1)
    Bx = [B[j] @ x for j in range(J)]

    def grad(Bx):
        return sum([B[j].conj().T @ gf[j](Bx[j]) for j in range(J)])

    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        g_old = g
        g = grad(Bx)

        # Compute BFGS update d=inv(H)@g
        d = -g  # search direction
        # compute the right product
        alfs = []
        for ro, s, y in reversed(memory):
            alf = ro * s @ d
            d -= alf * y
            alfs.append(alf)
        # compute center product
        d = H0 @ d
        d *= gamma  # scale by gamma
        # compute left product
        for alf, (ro, s, y) in zip(reversed(alfs), memory):
            beta = ro * y @ d
            d += (alf - beta) * s
        # End BFGS update
        # d = -d # search direction

        Bd = [B[j] @ d for j in range(J)]

        # derivative of h(a) = cost(x + a * dir)
        def dh(alf):
            return np.sum(
                [Bd[j].conj().T @ gf[j](Bx[j] + alf * Bd[j]) for j in range(J)]
            )

        # Lipschitz constant for dh
        Ldh = np.sum([Lgf[j] * norm(Bd[j]) ** 2 for j in range(J)])
        alf = gd(dh, Ldh, 0, niter=ninner)  # GD-based line search

        y = g - g_old
        s = alf * d
        ro = 1 / (y @ s)
        gamma = (s @ y) / (y @ y)
        memory.append((ro, s, y))

        x += s
        for j in range(J):
            Bx[j] += alf * Bd[j]
        if callback(x, itr):
            break

    return x.reshape(x0.shape)


@export
def newton(
    grad, ihess, x0, *, niter: int = 100, callback: Callable = lambda x, itr: False
):
    """
    Performs Newton's method to minimize a cost function

    Parameters
    ----------
    grad:
        function that computes gradient g(x) of cost function
    ihess:
        function that computes the inverse hessian of the cost function
            at the point x, ihess(x) = inv(H(x))
    x0:
        initial guess

    Other Parameters
    ----------------
    niter:
        number of iterations (default 100)
    callback:
        user-defined function to be evaluated with two arguments (x,iter)
          it is evaluated at (x0,0) and then after each iteration

    Returns
    -------
    final iterate:
    """
    x = x0.reshape(-1)
    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        x -= ihess(x) @ grad(x)  # ! Inplace update !
        if callback(x, itr):
            break
    return x.reshape(x0.shape)
