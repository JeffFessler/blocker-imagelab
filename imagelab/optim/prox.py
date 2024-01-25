"""imagelab/optim/prox.py
Methods for minimizing composite cost functions that
are the the sum of a convex smooth term and a convex
prox-friendly term
    argmin_x  f(x)  +   g(x)
            [_____]   [_____]
            smooth     prox-friendly

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""
from typing import Callable, Optional

import numpy as np
from scipy.linalg import norm

from ..utils import export, get_backend


def _gr_restart(Fgrad, ydiff, cutoff):
    return -Fgrad @ ydiff <= cutoff * norm(Fgrad) * norm(ydiff)


@export  # not OISTA (oista has no c3 or zeta)
def pogm(
    f_grad: Callable,
    g_prox: Callable,
    x0,
    Lf: float,
    *,
    restart: Optional["str"] = None,
    cost: Callable = lambda x: None,
    gr_cutoff: float = 0,
    niter: int = 100,
    bsig: float = 1,
    f_mu: float = 0,
    callback: Callable = lambda x, itr, cost, is_restart: False
):
    """Proximal Optimized Gradient Method [THG15] (with Restart [KF18])
    tldr; fista, but faster!

    Minimizes the composite cost function
        argmin_x F(x) := f(x)  +   g(x)
                        [____]   [_____]
                        smooth  prox-friendly
    with an optimal worst-case bound for such cost functions
        F(x) - F(x*) < Lf|| x0 - x* ||_2^2 / niter^2
    i.e. a convergence rate of O(1/k^2)

    Parameters
    ----------
    f_grad(x): function to compute the gradient of the smooth term f(x)
        of the composite cost function at each iterate
    g_prox(z,c): the proximal operator of scaled prox-friendly term c*g(x)
        i.e. g_prox(z,c) = argmin_x 1/2||x-z||_2^2 + c*g(x)
    x0: the initial starting point
    Lf: the Lipshitz constant of the smooth term f(x), i.e. satisfying
        ||f_grad(x) - f_grad(y)|| ≤ Lf || x - y ||
        Note if f is twice differentiable, this is a bound on the second
        derivative

    Returns
    -------
    x: the final iterate of algorithm (of the seconday sequence)

    Other Parameters
    ----------------
    restart: method of restart to use for resetting momentum terms
            None (default) - do not perform restart
            'fr' - Function restart, restart when cost(x) - cost(x_old) > 0
            'gr' - Gradient restart, restart when
                    <-grad, y-y_old> ≤ cutoff*||grad||*||y-y_old||
    cost: function to compute the cost if restart=='fr'
    gr_cutoff: if restart=='gr', then restart when gradient change > angle
                where cos(angle) = gr_cutoff. Default 0
    niter: Number of iterations to run. Default 100
    bsig: "bar sigma" from [KF18]. If restart != None,
            gradient decay between (0,1]. Default = 1
    f_mu: Strong convexity parameter of f(x)
    callback(x, itr, cst, is_restart): User provided function that is given
        - x: the current iterate of the secondary sequence
        - itr: the current iteration number in [0,niter]
        - cst: the evaluated cost function if restart='fr', otherwise None
        - is_restart: flag if current iteration restarted

    References
    ----------
    .. [THG15] A.B. Taylor, J.M. Hendrickx, F. Glineur,
        "Exact worst-case performance of first-order algorithms
        for composite convex optimization," Arxiv:1512.07516, 2015,
        SIAM J. Opt. 2017
        [http://doi.org/10.1137/16m108104x]
    .. [KF18] D. Kim, J.A. Fessler,
        "Adaptive restart of the optimized gradient method
        for convex optimization," 2018
        Arxiv:1703.04641, [http://doi.org/10.1007/s10957-018-1287-4]
    """
    #  it doesn't make sense to: (raise ValueError?
    #                             no, its nice to turn restart on and off)
    #   provide restart, cost, bsig, gr_cutoff if f_mu is given
    #   provide a gr_cutoff that is not in [-1,1]
    #   provide gr_cutoff if restart != 'gr'
    #   provide cost if restart != 'fr'
    #   provide bsig if not restart
    backend = get_backend(x0)
    x = x0.reshape(-1)
    w = x
    y = x
    z = x  # inital value mult by zero
    sig = 1
    tt = 1
    zeta = 1  # inital value mult by zero
    q = f_mu / Lf  # like 1/condition(A)
    beta2 = (2 + q - np.sqrt(q ** 2 + 8 * q)) ** 2 / 4.0 / (1 - q)
    gamma2 = (2 + q - np.sqrt(q ** 2 + 8 * q)) / 2
    if restart == "fr":
        cost_k = cost(x)
    else:
        cost_k = None
    if restart:
        G = backend.zeros(x.shape)
    if callback(x, 0, cost=cost_k, is_restart=False):
        return x0
    for itr in range(1, niter + 1):
        # save last iter
        w_old = w
        x_old = x  # only for restart part
        tt_old = tt
        cost_old = cost_k
        is_restart = False

        # compute constants for this iter
        tt = 0.5 * (1 + np.sqrt((4 if itr != niter else 8) * tt_old ** 2 + 1))
        beta = (tt_old - 1) / tt if not f_mu else beta2  # "nesterov momentum"
        gamma = sig * tt_old / tt if not f_mu else gamma2  # "ogm momentum"
        c3 = beta / (Lf * zeta)  # note "zeta_old". a new kind of "pogm momentum"
        zeta = (2 * tt_old + tt - 1) / (Lf * tt)  # 1/L (1+beta+gamma) ?

        # perform update
        grad = f_grad(x)
        w = x - grad / Lf  # primary sequence
        z = w + beta * (w - w_old) + gamma * (w - x) + c3 * (z - x)
        x = g_prox(z, zeta, backend)  # secondary sequence

        # check for restart (all not necessary for worst case bounds)
        if restart:  # (but tends to help in practice)
            G_old = G  # G is grad_f + subgrad_g ~~
            y_old = y
            G = grad - (x - z) / zeta
            y = x_old - G / Lf
            cost_k = cost(x)
            if (
                restart == "gr"
                and _gr_restart(G, y - y_old, gr_cutoff)
                or restart == "fr"
                and cost_k is not None
                and cost_k > cost_old
            ):
                tt = 1
                sig = 1
                is_restart = True
                # we don't need to reset zeta since (tt_old - 1) = 0 = c3
            if G @ G_old < 0:  # does this need to be higher precision?
                sig = bsig * sig

        # pass user info
        if callback(x, itr, cost=cost_k, is_restart=is_restart):
            break

    return x.reshape(x0.shape)


@export  # aka FPGM
def fista(
    f_grad: Callable,
    g_prox: Callable,
    x0,
    Lf: float,
    *,
    restart: Optional["str"] = None,
    cost: Callable = lambda x: None,
    gr_cutoff: float = 0,
    niter: int = 100,
    f_mu: float = 0,
    callback: Callable = lambda x, itr, cost, is_restart: False
):
    """FISTA aka FPGM
    Fast Iterative Shrinkage/Thresholding Algorithm [BT09](with Restart [OC15])
    NOTE: This algorithm is slower than POGM in both worst-case bounds and
        empirical real world problems, consider using `pogm` instead of `fista`

    Minimizes the composite cost function
        argmin_x F(x) := f(x)  +   g(x)
                        [____]   [_____]
                        smooth  prox-friendly
    with an optimal worst-case bound for such cost functions
        F(x) - F(x*) < 2Lf|| x0 - x* ||_2^2 / niter^2
    i.e. a convergence rate of O(1/k^2)

    Parameters
    ----------
    f_grad(x): function to compute the gradient of the smooth term f(x)
        of the composite cost function at each iterate
    g_prox(z,c): the proximal operator of scaled prox-friendly term c*g(x)
        i.e. g_prox(z,c) = argmin_x 1/2||x-z||_2^2 + c*g(x)
    x0: the initial starting point
    Lf: the Lipshitz constant of the smooth term f(x), i.e. satisfying
        ||f_grad(x) - f_grad(y)|| ≤ Lf || x - y ||
        Note if f is twice differentiable, this is a bound on the second
        derivative

    Returns
    -------
    x: the final iterate of algorithm (of the primary sequence)

    Other Parameters
    ----------------
    restart: method of restart to use for resetting momentum terms
            None (default) - do not perform restart
            'fr' - Function restart, restart when cost(x) - cost(x_old) > 0
            'gr' - Gradient restart, restart when
                    <-grad, y-y_old> < cutoff*||grad||*||y-y_old||
    cost: function to compute the cost if restart=='fr'
    gr_cutoff: if restart=='gr', then restart when gradient change > angle
                where cos(angle) = gr_cutoff. Default 0
    niter: Number of iterations to run. Default 100
    f_mu: Strong convexity parameter of f(x)
    callback(x, itr, cst, is_restart): User provided function that is given
        - x: the current iterate of the secondary sequence
        - itr: the current iteration number in [0,niter]
        - cst: the evaluated cost function if restart='fr', otherwise None
        - is_restart: flag if current iteration restarted

    References
    ----------
    .. [BT09] A. Beck, M. Teboulle:
        "A fast iterative shrinkage-thresholding algorithm
        for linear inverse problems,"
        SIAM J. Imaging Sci., 2009.
    .. [OC15] B. O'Donoghue, E. Candès:
        "Adaptive Restart for Accelerated Gradient Schemes"
        Foundations of Computational Mathematics,
        Vol. 15, 3, June 2015, 715-732

    """
    # kwargs.pop("bsig", None)  # for pogm compatibility

    # initialize variables
    y = x0.reshape(-1)
    x = y
    tt = 1
    q = f_mu / Lf  # like 1/condition(A)
    beta2 = (2 + q - np.sqrt(q ** 2 + 8 * q)) ** 2 / 4.0 / (1 - q)
    if restart == "fr":
        cost_k = cost(x)
    else:
        cost_k = None
    if callback(x, 0, cost=cost_k, is_restart=False):
        return x0
    for itr in range(1, niter + 1):
        # store last iter
        x_old = x
        y_old = y  # for restart
        tt_old = tt
        is_restart = False
        cost_old = cost_k

        # compute params for iter
        tt = (1 + np.sqrt(1 + 4 * tt ** 2)) / 2
        beta = (tt_old - 1) / tt if not f_mu else beta2

        # perform update
        grad = f_grad(y)
        x = g_prox(y - grad / Lf, 1 / Lf)
        y = x + beta * (x - x_old)

        # check for restart
        if restart:
            # G is grad_f + subgrad_g ~~
            G = grad - Lf * (x - y_old)
            cost_k = cost(x)
            if (
                restart == "gr"
                and _gr_restart(G, x - x_old, gr_cutoff)
                or restart == "fr"
                and cost_k is not None
                and cost_k > cost_old
            ):
                tt = 1
                is_restart = True

        # pass user info
        if callback(x, itr, cost=cost_k, is_restart=is_restart):
            break
    return x.reshape(x0.shape)


@export  # aka PGM
def ista(
    f_grad: Callable,
    g_prox: Callable,
    x0,
    Lf: float,
    *,
    niter: int = 100,
    f_mu: float = 0,
    callback: Callable = lambda x, itr: False
):
    """ ISTA """
    x = x0.reshape(-1)
    alpha = (2 if f_mu else 1) / (Lf + f_mu)  # strong convexity case
    if callback(x, 0):
        return x0
    for itr in range(1, niter + 1):
        x = g_prox(x - alpha * f_grad(x), alpha)
        if callback(x, itr):
            break
    return x.reshape(x0.shape)


@export
def mfista(
    f_grad: Callable,
    g_prox: Callable,
    x0,
    Lf: float,
    *,
    cost: Callable = lambda x: 0,
    niter: int = 100,
    callback: Callable = lambda x, itr, cost: False
):
    """MFISTA
    Monotone FISTA
    """

    # initialize variables
    y = x0.reshape(-1)
    x = y
    tt = 1
    fx = cost(x)
    if callback(x, 0, cost=fx):
        return x0
    for itr in range(1, niter + 1):
        # store last iter
        x_old = x
        tt_old = tt

        # compute params for iter
        tt = (1 + np.sqrt(1 + 4 * tt ** 2)) / 2
        beta = (tt_old - 1) / tt
        gamma = tt_old / tt

        # perform update
        z = g_prox(y - f_grad(y) / Lf, 1 / Lf)
        fz = cost(z)
        x, fx = (z, fz) if fz <= fx else (x_old, fx)
        y = x + beta * (x - x_old) + gamma * (z - x)

        # pass user info
        if callback(x, itr, cost=fx):
            break
    return x.reshape(x0.shape)


@export
def mfistava(
    f_grad: Callable,
    g_prox: Callable,
    x0,
    Lf: float,
    f_cost: Callable,
    mu=1.5,
    *,
    cost: Callable = lambda x: 0,
    niter: int = 100,
    callback: Callable = lambda x, itr, cst: False
):
    """MFISTA-VA
    Monotone FISTA with Variable Acceleration [ZHRH19]

    This algorithm is not optimized for runtime,
    it needs to be written as a _inv type to save
    the number of A@x.

    Reference
    ---------
    .. [ZHRH19] M.V.W. Zibetti, E.S. Helou, R.R. Regatte, G.T. Herman
        "Monotone FISTA With Variable Acceleration for Compressed
        Sensing Magnetic Resonance Imaging"
        IEEE Trans. on Comp. Imaging, Vol 5, #1, March 2019
    """

    # initialize variables
    y = x0.reshape(-1)
    x = y
    tt = 1
    fx = cost(x)
    if callback(x, 0, cost=fx):
        return x0
    for itr in range(1, niter + 1):
        # store last iter
        x_old = x
        tt_old = tt

        # compute params for iter
        tt = (1 + np.sqrt(1 + 4 * tt ** 2)) / 2
        beta = (tt_old - 1) / tt
        gamma = tt_old / tt

        # perform update (oh boy...)
        grad = f_grad(y)
        z = g_prox(y - grad / Lf, 1 / Lf)
        fz = cost(z)
        # propose a better point
        xbar = (1 - mu) * x_old + mu * z
        fxbar = cost(xbar)
        # ensure monotonicity
        x, fx = (z, fz) if fz <= fx else (x_old, fx)
        x, fx = (xbar, fxbar) if fxbar < fx else (x, fx)
        # compute majorizer gap
        ksi = f_cost(y) - f_cost(z) + grad @ (z - y) + Lf / 2 * norm(z - y) ** 2
        delta = cost(z) - cost(x)
        eta = 1 + 2 * (ksi + delta) / (Lf * norm(z - y) ** 2)
        # take modified momentum step
        y = x + beta * (x - x_old) + gamma * (z - x) + gamma * (eta - 1) * (z - y)

        # pass user info
        if callback(x, itr, cost=fx):
            break
    return x.reshape(x0.shape)
