"""imagelab/optim/prox.py
Methods for minimizing composite cost functions via
the dual problem.
    argmin_x,y  f(x)  +   g(y) s.t. x=y
               [_____]   [_____]
               smooth     prox-friendly

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""
from typing import Callable

import numpy as np

from .. import linop
from ..utils import export


# @export
def admm(f_grad, g_prox, x0, Lf, mu):  # a prox method?
    """Alternating Direction Method of Multipliers"""
    raise NotImplementedError


# @export
def lalm(f_grad, g_prox, x0, Lf, mu):
    """Linearized Augmented Lagrangian Method
    aka Augmented ADMM, "Majorized ADMM", split inexact Uzawa method
    """
    raise NotImplementedError


@export
@linop.utils.ensure_vec_out("A", "T")
def pdhg(
    A,
    y,
    T,
    f_prox: Callable,
    g_prox: Callable,
    x0,
    alpha: float = 1,
    beta: float = 1,
    *,
    Maj=1,
    niter: int = 100,
    callback: Callable = lambda x, itr: False
):
    """Primal-Dual Hybrid Gradient (with Majorization [RKW18])
    aka first-order primal-dual, aka forward-backward primal-dual

    Minimizes a cost function of the form::
            f(Ax-y) + g(Tx)
    where f(z) and g(w) have efficient prox operators and are both
    convex. Note this function could be written as h(Bx-b), g is
    for convenience.

    Cost converges with a worst-case rate O{1/sqrt(k)}

    A common example would be the TV problem::
            1/2 ||Ax-y||_2^2 + c ||Dx||_1

    Parameters
    ----------
    A:      Matrix operator in compostion of function f
    y:      bias or offset in composition of function f
    T:      Matrix operator in compostion of function g
    f_prox: Proximal operator of the function f
    g_prox: Proximal operator of the function g
    x0:     initial starting point, should be a best estimate
    alpha:  step-size parameter?
    beta:   step-size parameter?

    Other Parmeters
    ---------------
    Maj:    A Majorizer such that M ≥ alpha*(A'A + (beta/alpha)**2 T'T)
    niter:  number of iterations to run
    callback(x, itr): User provided function that is given
        - x: the current iterate
        - itr: the current iteration number in [0,niter]

    Returns
    -------
    x: the final iterate

    References
    ----------
    ..[RKW18] E. K. Ryu, S. Ko, J. Won
            "Splitting with Near-Circulant Linear Systems: Applications
            to Total Variation CT and PET", Arxiv:1810.13100, 2018

    """

    def f_conj_prox(y, c):
        return y - c * f_prox(y / c, 1 / c)

    def g_conj_prox(y, c):
        return y - c * g_prox(y / c, 1 / c)

    x = x0
    u = f_conj_prox(-alpha * y)  # np.zeros(A.shape[0])
    v = np.zeros(T.shape[0])
    if callback(x0, 0):
        return x0
    for itr in range(1, niter + 1):
        x_old = x
        x = x_old - (A.H @ u + (beta / alpha) * T.H @ v) / Maj
        u = f_conj_prox(u + alpha * (A @ (2 * x - x_old) - y), alpha)
        v = g_conj_prox(v + beta * (T @ (2 * x - x_old)), beta / alpha)

        if callback(x, itr):
            break
    return x
