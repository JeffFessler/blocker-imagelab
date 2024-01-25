import numpy as np
import pytest

# from imagelab.linop import I
from scipy.linalg import lstsq, norm

from imagelab.optim.grad import gd, ncg, ncg_inv, ncg_inv_mm, ogm_inv, psd, psd_inv
from imagelab.optim.newton import lbfgs, lbfgs_inv  # also smooth


@pytest.fixture(scope="module")  # to compute xh once
def ridge_regression_problem():
    M = 400
    N = 100
    np.random.seed(345)
    A = np.random.randn(M, N)
    y = np.random.randn(M)
    reg = 5
    xh = lstsq(A.T @ A + reg * np.eye(N), A.T @ y)[0]
    return A, y, reg, xh


@pytest.mark.parametrize(
    "opt_alg,niter",
    [
        (gd, 200),
        (psd, 100),
        (ncg, 30),
        (lbfgs, 100),
    ],
)
def test_ridge_regression_convergence(ridge_regression_problem, opt_alg, niter):
    A, y, reg, xh = ridge_regression_problem

    def grad(x):
        return A.T @ (A @ x - y) + reg * x

    L = norm(A, 2) ** 2 + reg  # impractical for large problems
    ii = []

    def func(x, itr):
        return ii.append(itr)

    xk = opt_alg(grad, L, np.zeros_like(xh), niter=niter, callback=func)
    assert np.allclose(xh, xk)
    assert (np.r_[0 : niter + 1] == np.array(ii)).all()


@pytest.mark.parametrize(
    "opt_alg_inv,niter",
    [(psd_inv, 100), (ncg_inv, 30), (ncg_inv_mm, 50), (ogm_inv, 200), (lbfgs_inv, 100)],
)
def test_ridge_regression_composite_convergence(
    ridge_regression_problem, opt_alg_inv, niter
):
    A, y, reg, xh = ridge_regression_problem
    B = [A, np.eye(xh.shape[0])]
    gf = [lambda u: u - y, lambda v: reg * v]  # of functions gradients
    Lgf = [1, reg]
    ii = []

    def func(x, itr):
        return ii.append(itr)

    xk = opt_alg_inv(B, gf, Lgf, np.zeros_like(xh), niter=niter, callback=func)
    assert np.allclose(xh, xk)
    assert (np.r_[0 : niter + 1] == np.array(ii)).all()
