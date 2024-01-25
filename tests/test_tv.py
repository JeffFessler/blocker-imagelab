import numpy as np
import pytest

import imagelab.linop
from imagelab.tv import (
    Cdiff1_circshift,
    Cdiff1_indexing,
    Cdiff1_npdiff,
    Cdiff1_spdiag,
    Cdiffn,
)


@pytest.mark.parametrize(
    "linOp", [Cdiff1_spdiag, Cdiff1_npdiff, Cdiff1_circshift, Cdiff1_indexing]
)
def test_forward_adjoint_via_full_matrix(linOp):
    C = linOp(in_shape=(10,))
    c_full = C.full_matrix()
    c_full_t = C.T.full_matrix()
    assert np.allclose(c_full, c_full_t.T)

    Ca = C.abs()
    c_full_a = Ca.full_matrix()
    c_full_t_a = Ca.T.full_matrix()
    assert np.allclose(c_full_a, c_full_t_a.T)
    assert np.allclose(c_full_a, np.abs(c_full))


@pytest.mark.parametrize(
    "linOp", [Cdiff1_spdiag, Cdiff1_npdiff, Cdiff1_circshift, Cdiff1_indexing]
)
def test_forward_adjoint_via_inner_products(linOp):
    C = linOp(0, in_shape=(50, 10))
    assert imagelab.linop.test_forward_adjoint_consistency(C)
    C = linOp(1, in_shape=(50, 10))
    assert imagelab.linop.test_forward_adjoint_consistency(C)


# The Tests below check for correct implementation


@pytest.mark.parametrize(
    "linOp",
    [
        pytest.param(
            Cdiff1_spdiag, marks=pytest.mark.xfail(reason="Doesn't support 2D inputs")
        ),
        Cdiff1_npdiff,
        pytest.param(
            Cdiff1_circshift, marks=pytest.mark.xfail(reason="periodic end conditions")
        ),
        Cdiff1_indexing,
    ],
)
def test_diff_on_gradient_img(linOp):
    img = np.outer(np.ones(10), np.r_[0:15])
    C = linOp(0, in_shape=img.shape)
    res = C @ img
    assert np.allclose(res, np.zeros_like(res))

    C = linOp(1, in_shape=img.shape)
    res = C @ img
    assert np.allclose(res, np.ones_like(res))


@pytest.mark.parametrize(
    "linOp",
    [
        Cdiff1_spdiag,
        Cdiff1_npdiff,
        pytest.param(
            Cdiff1_circshift, marks=pytest.mark.xfail(reason="periodic end conditions")
        ),
        Cdiff1_indexing,
    ],
)
def test_diff_on_gradient_line(linOp):
    line = np.r_[0:15]
    C = linOp(0, in_shape=line.shape)
    res = C @ line
    assert np.allclose(res, np.ones_like(res))


@pytest.mark.parametrize(
    "linOp", [Cdiff1_spdiag, Cdiff1_npdiff, Cdiff1_circshift, Cdiff1_indexing]
)
def test_diff_on_constant_line(linOp):
    line = np.ones(10)
    C = linOp(0, in_shape=line.shape)
    res = C @ line
    assert np.allclose(res, np.zeros_like(res))


# A simpler test that doesn't care if we are using periodic end conditions
@pytest.mark.parametrize(
    "linOp", [Cdiff1_spdiag, Cdiff1_npdiff, Cdiff1_circshift, Cdiff1_indexing]
)
def test_diff_on_pyramid(linOp):
    line = np.concatenate([np.r_[0:11], np.r_[9:-1:-1]])
    C = linOp(0, in_shape=line.shape)
    res = C @ line
    assert np.sum(res) == 0
    assert res[3] == 1
    assert res[-3] == -1


def test_diff2_on_gradient_img():
    N = 10
    M = 15
    img = np.outer(np.ones(N), np.r_[0:M])
    C = Cdiffn([0, 1], in_shape=img.shape)
    res = C @ img
    split = np.prod(C.op_list[0].out_shape)
    assert np.allclose(res[:split], 0)
    assert np.allclose(res[split:], 1)
