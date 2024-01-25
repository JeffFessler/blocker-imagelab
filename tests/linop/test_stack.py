import numpy as np

from .test_linop import _test_LinOp_against_matrix
from imagelab import linop
from imagelab.linop.stack import DStackLinOp, VStackLinOp


def test_vstack_against_matrix():
    np.random.seed(435)
    A = np.random.randn(9, 4) + 1j * np.random.randn(9, 4)
    A_top = A[:4]
    A_bot = A[4:]
    lin_op = VStackLinOp([linop.Matrix(A_top.copy()), linop.Matrix(A_bot.copy())])
    _test_LinOp_against_matrix(A, lin_op)
    _test_LinOp_against_matrix(A.conj().T, lin_op.H)


def test_vstack_against_identity():
    np.random.seed(848)
    x = np.random.randn(9, 4, 5) + 1j * np.random.randn(9, 4, 5)
    linOp = VStackLinOp([linop.Identity() for _ in range(3)])
    y = linOp @ x
    assert y.size == 3 * x.size
    assert len(y.shape) == 1

    xbp = linOp.H @ y
    assert x.size == xbp.size
    assert (xbp == 3 * x.reshape(-1)).all()


def test_dstack_against_identity():
    np.random.seed(848)
    x = np.random.randn(9, 4, 5) + 1j * np.random.randn(9, 4, 5)
    linOp = DStackLinOp([linop.Identity() for _ in range(3)])
    y = linOp @ x
    assert y.size == 3 * x.size
    assert len(y.shape) == 4
    assert y.shape[0] == 3

    xbp = linOp.H @ y
    assert x.shape == xbp.shape
    assert (xbp == 3 * x).all()


def test_dstack_vstack_equality():
    np.random.seed(43422)
    dstack = DStackLinOp([linop.Identity()])
    vstack = VStackLinOp([linop.Identity()])
    for _ii in range(3):
        A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        dstack = DStackLinOp([dstack, linop.Matrix(A)])
        vstack = VStackLinOp([vstack, linop.Matrix(A)])
    x = np.random.randn(4) + 1j * np.random.randn(4)
    yd = dstack @ x
    yv = vstack @ x
    assert (yd == yv.reshape(yd.shape)).all()
    xd = dstack.H @ yd
    xv = vstack.H @ yv
    assert (xd == xv).all()

    assert dstack.shape == vstack.shape
