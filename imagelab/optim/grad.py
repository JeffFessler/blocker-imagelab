"""General-First Order optimization methods for smooth cost function
minimization.

The functions provided in the module mostly accept either
the gradient of your cost function or a list of gradients
and a list of LinOps/matrices to composite into those
gradients.

Many of these are inspired by homework problems in
Jeff Fessler's EECS 598 course on Optimization
Methods for Signal Processing and Machine Learning at UMich.
"""
from typing import Callable, List

import numpy as np
from scipy.linalg import norm

from .. import linop
from ..linop import LinearOperator as _LinearOperator
from ..utils import export, get_backend

_Identity = linop.NullIdentity()
_Identity.vec_out = True


@export
@linop.utils.ensure_vec_out("P")
def gd(
    g: Callable,
    L: float,
    x0,
    *,
    P: _LinearOperator = _Identity,
    niter: int = 100,
    callback: Callable = lambda x, itr: False
):
    """Performs gradient descent to "solve" a minimization problem
    having a L-Lipschitz smooth gradient

    Parameters
    ----------
    g:
        function that computes gradient g(x) of cost function
    L:
        Lipschitz constant of cost function gradient
    x0:
        initial guess

    Other Parameters
    ----------------
    niter:
        number of iterations (default 100)
    callback:
        user-defined function to be evaluated with two arguments (x,itr)
            it is evaluated at (x0,0) and then after each iteration

    Returns
    -------
    x:
        final iterate
    """
    # Gradient descent
    x = x0.reshape(-1) if hasattr(x0, "shape") else x0
    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        x -= (P @ g(x)) / L  # ! Inplace update !
        if callback(x, itr):
            break
    return x.reshape(x0.shape) if hasattr(x0, "shape") else x


@export
@linop.utils.ensure_vec_out("P")
def psd(
    g: Callable,
    Lg: float,
    x0,
    *,
    niter: int = 100,
    ninner: int = 10,
    P: _LinearOperator = _Identity,
    callback: Callable = lambda x, itr: False
):
    """Performs preconditioned steepest descent (PSD)
    to minimize a convex cost function having a Lg-Lipschitz smooth gradient.

    Parameters
    ----------
    g:
        function that computes gradient g(x) of a convex cost function
    Lg:
        Lipschitz constant of cost function gradient
    x0:
        initial guess

    Other Parameters
    ----------------
    niter:
        number of outer PSD iterations (default 100)
    ninner:
        number of inner iterations of GD for line search (default 10)
    P:
        preconditioner (default I)
    callback:
        user-defined function to be evaluated with two arguments (x,itr).
                It is evaluated at (x0,0) and then after each iteration.

    Returns
    -------
    x:
        final iterate
    """

    x = x0.reshape(-1)
    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        dirn = -(P @ g(x))  # search direction
        # derivative of h(a) = cost(x + a * dir)
        def dh(alf):
            return dirn.conj().T @ g(x + alf * dirn)

        Ldh = norm(dirn) ** 2 * Lg  # Lipschitz constant for dh
        alf = gd(dh, Ldh, 0, niter=ninner) if Ldh != 0 else 0
        x += alf * dirn
        if callback(x, itr):
            break

    return x.reshape(x0.shape)


@export
@linop.utils.ensure_vec_out("B", "P")
def psd_inv(
    B: List[_LinearOperator],
    gf: List[Callable],
    Lgf: List[float],
    x0,
    *,
    niter: int = 100,
    ninner: int = 10,
    P: _LinearOperator = _Identity,
    callback: Callable = lambda x, itr: False
):
    """Performs preconditioned steepest descent to "solve" a
    general "inverse problem" cost function sum_{j=1}^J f_j(B_j x)
    where each f_j has a Lgf_j-Lipschitz smooth gradient.

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
    P:
        preconditioner (default I)
    callback:
        user-defined function to be evaluated with two arguments (x,itr).
                It is evaluated at (x0,0) and then after each iteration

    Returns
    -------
    x:
        final iterate
    """
    backend = get_backend(x0)
    J = len(B)

    x = x0.reshape(-1)
    Bx = [B[j] @ x for j in range(J)]

    def grad(Bx):
        return sum([B[j].conj().T @ gf[j](Bx[j]) for j in range(J)])

    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        dirn = -(P @ grad(Bx))  # search direction
        Bd = [B[j] @ dirn for j in range(J)]

        # derivative of h(a) = cost(x + a * dir)
        def dh(alf):
            return np.sum(
                [Bd[j].conj().T @ gf[j](Bx[j] + alf * Bd[j]) for j in range(J)]
            )

        # Lipschitz constant for dh
        Ldh = backend.sum(
            backend.array([Lgf[j] * (Bd[j].conj() @ Bd[j]) for j in range(J)])
        )
        alf = gd(dh, Ldh, 0, niter=ninner)  # GD-based line search

        x += alf * dirn
        for j in range(J):
            Bx[j] += alf * Bd[j]
        if callback(x, itr):
            break

    return x.reshape(x0.shape)


@export
@linop.utils.ensure_vec_out("P")
def ncg(
    g: Callable,
    Lg: float,
    x0,
    *,
    niter: int = 100,
    ninner: int = 10,
    P: _LinearOperator = _Identity,
    betahow: str = "dai_yuan",
    callback: Callable = lambda x, itr: False
):
    """Performs nonlinear (preconditioned) conjugate gradient (PCG)
    to minimize a convex cost function having a Lg-Lipschitz smooth gradient.

    Parameters
    ----------
    g:
        function that computes gradient g(x) of a convex cost function
    Lg:
        Lipschitz constant of cost function gradient
    x0:
        initial guess

    Other Parameters
    ----------------
    niter:
        number of outer PCG iterations (default 100)
    ninner:
        number of inner iterations of GD for line search (default 10)
    P:
        preconditioner (default I)
    betahow:
        "beta" method for the search direction (default 'dai_yuan')
    callback:
        User-defined function to be evaluated with two arguments (x,iter).
              It is evaluated at (x0,0) and then after each iteration.

    Returns
    -------
    x:
        final iterate
    """

    x = x0.reshape(-1)
    grad_old = None  # to silence the linter

    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        grad_new = g(x)  # gradient
        pgrad_new = P @ grad_new
        neg_pgrad = -(pgrad_new)  # negative preconditioned gradient
        if itr == 1:
            dirn = neg_pgrad
        else:
            if betahow == "dai_yuan":
                betaval = (
                    grad_new.conj().T
                    @ (pgrad_new)
                    / ((grad_new - grad_old).conj().T @ dirn)
                )
            elif betahow == "fletcher_reeves":
                betaval = (
                    grad_new.conj().T
                    @ (pgrad_new)
                    / (grad_old.conj().T @ (P @ grad_old))
                )
            elif betahow == "polak_ribiere":
                betaval = (
                    (grad_new - grad_old).conj().T
                    @ pgrad_new
                    / (grad_old.conj().T @ (P @ grad_old))
                )
            elif betahow == "hesteness_stiefel":
                betaval = (
                    (grad_new - grad_old).conj().T
                    @ pgrad_new
                    / ((grad_new - grad_old).conj().T @ dirn)
                )
            elif betahow == "hager_zhang":
                gdiff = grad_new - grad_old
                zeta = 2 * gdiff.conj().T @ (P @ gdiff) / gdiff.conj().T @ dirn
                betaval = (
                    (P @ gdiff - zeta * dirn).conj().T
                    @ (grad_new)
                    / gdiff.conj().T
                    @ dirn
                )
            else:
                raise ValueError("unknown beta choice")
            dirn = neg_pgrad + betaval * dirn  # search direction
        grad_old = grad_new

        # derivative of h(a) = cost(x + a * dirn)
        def dh(alf):
            return dirn.conj().T @ g(x + alf * dirn)

        Ldh = norm(dirn) ** 2 * Lg  # Lipschitz constant for dh
        alf = gd(dh, Ldh, 0, niter=ninner)
        x += alf * dirn  # ! Inplace update !
        if callback(x, itr):
            break

    return x.reshape(x0.shape)


@export
@linop.utils.ensure_vec_out("B", "P")
def ncg_inv(
    B: List[_LinearOperator],
    gf: List[Callable],
    Lgf: List[float],
    x0,
    *,
    niter: int = 50,
    ninner: int = 10,
    P: _LinearOperator = _Identity,
    betahow: str = "dai_yuan",
    callback: Callable = lambda x, itr: False
):
    """Nonlinear preconditioned conjugate gradient algorithm
    to minimize a general "inverse problem" cost function
        sum_{j=1}^J f_j(B_j x)
    where each f_j has a Lgf_j-Lipschitz smooth gradient.

    Parameters
    ----------
    B:
        array of J blocks B_1,...,B_J
    gf:
        array of J functions for computing gradients of f_1,...,f_J
    Lgf:
        array of J Lipschitz constants for those gradients
    x0:
        initial guess (Warning: This is changed in place!)

    Other Parameters
    ----------------
    niter:
        number of outer PSD iterations (default 100)
    ninner:
        number of inner iterations of GD for line search (default 10)
    P:
        preconditioner (default I)
    betahow:
        "beta" method for the search direction (default 'dai_yuan')
    callback:
        User-defined function to be evaluated with two arguments (x,iter).
              It is evaluated at (x0,0) and then after each iteration.

    Returns
    -------
    x:
        final iterate
    """
    backend = get_backend(x0)
    J = len(B)
    grad_old = None  # to silence the linter

    x = x0.reshape(-1)
    # dirn = [] # dirn is the conjugate direction
    # grad_old = []
    # grad_new = []

    Bx = [B[j] @ x for j in range(J)]

    def grad(Bx):
        return sum([B[j].conj().T @ gf[j](Bx[j]) for j in range(J)])

    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        grad_new = grad(Bx)  # gradient
        pgrad_new = P @ grad_new
        neg_pgrad = -(pgrad_new)
        if itr == 1:
            dirn = neg_pgrad
        else:
            if betahow == "dai_yuan":
                betaval = (
                    grad_new.conj().T
                    @ (pgrad_new)
                    / ((grad_new - grad_old).conj().T @ dirn)
                )
            elif betahow == "fletcher_reeves":
                betaval = (
                    grad_new.conj().T
                    @ (pgrad_new)
                    / (grad_old.conj().T @ (P @ grad_old))
                )
            elif betahow == "polak_ribiere":
                betaval = (
                    (grad_new - grad_old).conj().T
                    @ pgrad_new
                    / (grad_old.conj().T @ (P @ grad_old))
                )
            elif betahow == "hesteness_stiefel":
                betaval = (
                    (grad_new - grad_old).conj().T
                    @ pgrad_new
                    / ((grad_new - grad_old).conj().T @ dirn)
                )
            elif betahow == "hager_zhang":
                gdiff = grad_new - grad_old
                zeta = 2 * gdiff.conj().T @ (P @ gdiff) / gdiff.conj().T @ dirn
                betaval = (
                    (P @ gdiff - zeta * dirn).conj().T
                    @ (grad_new)
                    / gdiff.conj().T
                    @ dirn
                )
            else:
                raise ValueError("unknown beta choice")
            if not np.isfinite(betaval):
                break
            dirn = neg_pgrad + betaval * dirn  # search direction
        grad_old = grad_new

        Bd = [B[j] @ dirn for j in range(J)]

        # derivative of h(a) = cost(x + a * dirn)
        def dh(alf):
            return backend.sum(
                backend.array(
                    [Bd[j].conj().T @ gf[j](Bx[j] + alf * Bd[j]) for j in range(J)]
                )
            )

        # Lipschitz constant for dh
        Ldh = backend.sum(
            backend.array([Lgf[j] * (Bd[j].conj() @ Bd[j]) for j in range(J)])
        )
        alf = gd(dh, Ldh, 0, niter=ninner)  # GD-based line search

        x += alf * dirn  # ! Inplace update !
        for j in range(J):
            Bx[j] += alf * Bd[j]
        if callback(x, itr):
            break

    return x.reshape(x0.shape)


@export
@linop.utils.ensure_vec_out("B", "P")
def ncg_inv_mm(
    B: List[_LinearOperator],
    gf: List[Callable],
    curvf: List[Callable],
    x0,
    *,
    niter: int = 50,
    ninner: int = 5,
    P: _LinearOperator = _Identity,
    betahow: str = "dai_yuan",
    callback: Callable = lambda x, itr: False
):
    r"""Nonlinear preconditioned conjugate gradient algorithm
    to minimize a general "inverse problem" cost function
        sum_{j=1}^J f_j(B_j x)
    where each f_j(v) has a quadratic majorizer of the form
    q_j(v;u) = f_j(u) + \nabla f_j(u) (v - u) + 1/2 \|v - \u|^2_C
    where C is diagonal matrix of curvatures, with MM line search.

    Parameters
    ----------
    B:
        array of J blocks B_1,...,B_J
    gf:
        array of J functions for computing gradients of f_1,...,f_J
    curvf:
        array of J things, each of which is either a Lipschitz constant
            for the gradient of f_j, or a function z -> curv(z) that returns a
            vector of curvature values for each element of z
    x0:
        initial guess (Warning: This is changed in place!)

    Other Parameters
    ----------------
    niter:
        number of outer iterations; default 50
    ninner:
        number of inner iterations of MM line search; default 5
    P:
        preconditioner (default I)
    betahow:
        "beta" method for the search direction (default 'dai_yuan')
    callback:
        user-defined function to be evaluated with two arguments (x,itr).
                It is evaluated at (x0,0) and then after each iteration.

    Returns
    -------
    x:
        final iterate
    """
    backend = get_backend(x0)
    grad_old = None  # to silence the linter

    J = len(B)

    curvf = [
        curvfi
        if callable(curvfi)
        else lambda z, curvfi=curvfi: curvfi * backend.ones_like(z)
        for curvfi in curvf
    ]
    x = x0.reshape(-1)
    # dirn = []
    # grad_old = []
    # grad_new = []

    Bx = [B[j] @ x for j in range(J)]

    def grad(Bx):
        return sum([B[j].conj().T @ gf[j](Bx[j]) for j in range(J)])

    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        grad_new = grad(Bx)  # gradient
        pgrad_new = P @ grad_new
        neg_pgrad = -(pgrad_new)
        if itr == 1:
            dirn = neg_pgrad
        else:
            if betahow == "dai_yuan":
                # this is the only one I'm sure about Preconditioned
                betaval = (
                    grad_new.conj().T
                    @ (pgrad_new)
                    / ((grad_new - grad_old).conj().T @ dirn)
                )
            elif betahow == "fletcher_reeves":
                betaval = (
                    grad_new.conj().T
                    @ (pgrad_new)
                    / (grad_old.conj().T @ (P @ grad_old))
                )
            elif betahow == "polak_ribiere":
                betaval = (
                    pgrad_new.conj().T
                    @ (pgrad_new - P @ grad_old)
                    / (grad_old.conj().T @ (P @ grad_old))
                )
            elif betahow == "hesteness_stiefel":
                betaval = (
                    pgrad_new.conj().T
                    @ (pgrad_new - grad_old)
                    / ((grad_new - grad_old).conj().T @ dirn)
                )
            else:
                raise ValueError("unknown beta choice")
            dirn = neg_pgrad + betaval * dirn  # search direction
        grad_old = grad_new

        Bd = [B[j] @ dirn for j in range(J)]

        # derivative of h(a) = cost(x + a * dirn)
        def dh(alf):
            return backend.sum(
                backend.array(
                    [Bd[j].conj().T @ gf[j](Bx[j] + alf * Bd[j]) for j in range(J)]
                )
            )

        def curvh(alf):
            return backend.sum(
                backend.array(
                    [
                        backend.abs(Bd[j]) ** 2 @ curvf[j](Bx[j] + alf * Bd[j])
                        for j in range(J)
                    ]
                )
            )

        alf = 0
        for _inner in range(ninner):
            alf -= dh(alf) / curvh(alf)

        x += alf * dirn  # ! Inplace update !
        for j in range(J):
            Bx[j] += alf * Bd[j]
        if callback(x, itr):
            break

    return x.reshape(x0.shape)


@export
@linop.utils.ensure_vec_out("B")
def ogm_inv(
    B: List[_LinearOperator],
    gf: List[Callable],
    Lgf: List[float],
    x0,
    *,
    niter: int = 50,
    ninner: int = 10,
    callback: Callable = lambda x, itr: False
):
    """OGM with line search [DT18]_
    to minimize a general "inverse problem" cost function
        sum_{j=1}^J f_j(B_j x)
    where each f_j has a Lgf_j-Lipschitz smooth gradient.
    Uses 1D GD for the line search.

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
        number of outer iterations (default 50)
    ninner:
        number of inner iterations of GD for line search (default 10)
    callback:
        User-defined function to be evaluated with two arguments (x,itr).
               It is evaluated at (x0,0) and then after each iteration.

    Returns
    -------
    x:
        final iterate

    References
    ----------
    .. [DT18] Drori Y, Taylor A
                "Efficient First-order Methods for Convex Minimization: a
                Constructive Approach", arxiv 1803.05676, (v2, Feb 2019)
    """

    J = len(B)

    x = x0.reshape(-1)
    # dirn = []
    # grad_old = []
    # grad_new = []
    grad_sum = np.zeros(x0.shape)
    ti = 1
    thetai = 1

    B0 = [B[j] @ x0 for j in range(J)]
    Bx = [B0[j].copy() for j in range(J)]
    By = [B0[j].copy() for j in range(J)]

    def grad(Bx):
        return sum([B[j].conj().T @ gf[j](Bx[j]) for j in range(J)])

    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        grad_new = grad(Bx)  # gradient of x_{iter-1}
        grad_sum += ti * grad_new  # sum_{j=0}^{iter-1} t_j * gradient_j

        thetai = (1 + np.sqrt(8 * ti ** 2 + 1)) / 2  # theta_{i+1}
        ti = (1 + np.sqrt(4 * ti ** 2 + 1)) / 2  # t_{i+1}

        # use theta_i factor for last iteration
        tt = ti if (itr < niter) else thetai
        yi = (1 - 1 / tt) * x + (1 / tt) * x0

        for j in range(J):  # update Bj * yi
            By[j] = (1 - 1 / tt) * Bx[j] + (1 / tt) * B0[j]

        dirn = -(1 - 1 / tt) * grad_new - (2 / tt) * grad_sum  # -d_i

        # line search of h(a) = cost(yi + a * dirn)
        Bd = [B[j] @ dirn for j in range(J)]

        # derivative of h(a) = cost(x + a * dirn)
        def dh(alf):
            return np.sum(
                [Bd[j].conj().T @ gf[j](By[j] + alf * Bd[j]) for j in range(J)]
            )

        # Lipschitz constant for dh
        Ldh = np.sum([Lgf[j] * norm(Bd[j]) ** 2 for j in range(J)])
        alf = gd(dh, Ldh, 0, niter=ninner)  # GD-based line search

        x = yi + alf * dirn

        if itr < niter:
            for j in range(J):  # update Bj * x
                Bx[j] = By[j] + alf * Bd[j]

        #   for j=1:J # recursive update Bj * yi ???
        #       By[j] = (1 - 1/ti) * (By[j] + alf * Bd[j]) + (1/ti) * B0[j]
        #   end

        if callback(x, itr):
            break

    return x.reshape(x0.shape)
