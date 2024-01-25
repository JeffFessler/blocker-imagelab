import numpy as np
from scipy.linalg import det, inv, norm, pinv

import imagelab.linop
from imagelab.linop import Diagonal, Identity, Matrix, Scalar


def _test_LinOp_against_matrix(A, lin_op):
    if lin_op.in_shape is None:
        lin_op.in_shape = (A.shape[1],)
        lin_op.out_shape = (A.shape[0],)

    x = np.random.randn(A.shape[1]) + 1j * np.random.randn(A.shape[1])
    y = np.random.randn(A.shape[0]) + 1j * np.random.randn(A.shape[0])
    x0 = x.copy()
    y0 = y.copy()

    assert np.allclose(A @ x, lin_op @ x)  # AbstractLinOp __matmul__
    assert np.all(x == x0)  # make sure we didn't change the input

    assert np.allclose(x @ (A.conj().T), x @ lin_op.H)  # AdjointView __rmatmul__
    assert np.all(x == x0)  # make sure we didn't change the input

    assert np.allclose(A.conj().T @ y, lin_op.H @ y)  # Adjoint_View __matmul__
    assert np.all(y == y0)  # make sure we didn't change the input

    assert np.allclose(y @ A, y @ lin_op)  # AbstractLinOp __rmatmul__
    assert np.all(y == y0)  # make sure we didn't change the input

    assert imagelab.linop.test_forward_adjoint_consistency(lin_op)

    for ii in range(A.shape[0]):
        for jj in range(A.shape[1]):
            assert np.allclose(A[ii, jj], lin_op[ii, jj]), "ii = {}, jj = {}".format(
                ii, jj
            )

    assert np.allclose(A, lin_op.full_matrix())


def test_MatrixLinOp_against_Matrix_complex():
    np.random.seed(333)
    A = np.random.randn(9, 4) + 1j * np.random.randn(9, 4)
    lin_op = Matrix(A.copy())

    # Rectangular Matrices
    _test_LinOp_against_matrix(A, lin_op)
    _test_LinOp_against_matrix(A.conj(), lin_op.conj())
    _test_LinOp_against_matrix(A.conj().T, lin_op.H)
    _test_LinOp_against_matrix(np.abs(A), lin_op.abs())
    _test_LinOp_against_matrix(pinv(A), lin_op.pinv())
    assert np.isclose(norm(A, 2), lin_op.opnorm())

    B = np.random.randn(6, 6) + 1j * np.random.randn(6, 6)
    lin_op2 = Matrix(B.copy())

    _test_LinOp_against_matrix(B, lin_op2)
    _test_LinOp_against_matrix(B.conj().T, lin_op2.H)
    _test_LinOp_against_matrix(np.abs(B), lin_op2.abs())
    _test_LinOp_against_matrix(inv(B), lin_op2 ** -1)

    assert np.isclose(det(B), lin_op2.det())
    assert np.isclose(np.trace(B), lin_op2.trace())
    assert np.isclose(norm(B, 2), lin_op2.opnorm())


def test_CompositeLinOp_against_Matrix_complex():
    np.random.seed(435)
    A = np.random.randn(9, 4) + 1j * np.random.randn(9, 4)
    lin_op = Matrix(A.copy())
    _test_LinOp_against_matrix(A @ A.conj().T, lin_op @ lin_op.H)  # Testing composite
    _test_LinOp_against_matrix(A.conj().T @ A, lin_op.H @ lin_op)  # Testing composite

    I1 = Identity()
    I2 = Identity()
    comp = I1 @ lin_op @ I2.T @ I2 @ lin_op.H
    assert len(comp.op_list) == 2


def test_ScalarLinOp_against_Matrix_complex():
    np.random.seed(988)
    sclr = np.random.randn() + 1j * np.random.randn()
    A = np.diag(sclr * np.ones(7))
    lin_op = Scalar(sclr, in_shape=7, out_shape=7)
    _test_LinOp_against_matrix(A, lin_op)
    _test_LinOp_against_matrix(A.conj().T, lin_op.H)
    _test_LinOp_against_matrix(np.abs(A), lin_op.abs())
    _test_LinOp_against_matrix(inv(A), lin_op ** -1)


def test_DiagonalLinOp_against_Matrix_complex():
    np.random.seed(586)
    diag_array = np.random.randn(9) + 1j * np.random.randn(9)
    A = np.diag(diag_array)
    lin_op = Diagonal(diag_array)
    _test_LinOp_against_matrix(A, lin_op)
    _test_LinOp_against_matrix(A.conj().T, lin_op.H)
    _test_LinOp_against_matrix(np.abs(A), lin_op.abs())
    _test_LinOp_against_matrix(inv(A), lin_op ** -1)

    assert np.isclose(det(A), lin_op.det())
    assert np.isclose(np.trace(A), lin_op.trace())
    assert np.isclose(norm(A, 2), lin_op.opnorm())


def test_IdentityLinOp_against_Matrix():
    Idty = Identity(in_shape=10, out_shape=10)
    eye = np.eye(10)
    _test_LinOp_against_matrix(eye, Idty)
    _test_LinOp_against_matrix(eye.conj().T, Idty.H)
    _test_LinOp_against_matrix(np.abs(eye), Idty.abs())
    _test_LinOp_against_matrix(inv(eye), Idty ** -1)

    assert np.isclose(det(eye), Idty.det())
    assert np.isclose(np.trace(eye), Idty.trace())
    assert np.isclose(norm(eye, 2), Idty.opnorm())


def test_set_shape():
    np.random.seed(875)
    A = np.random.randn(9, 4) + 1j * np.random.randn(9, 4)
    lin_op = Matrix(A.copy())

    assert lin_op.shape[0] == lin_op.H.shape[1]
    assert lin_op.shape[1] == lin_op.H.shape[0]
    assert A.shape[0] == lin_op.shape[0]
    assert A.shape[1] == lin_op.shape[1]
    lin_op.H.in_shape = (lin_op.H.in_shape[0] + 1,)
    assert lin_op.shape[0] == lin_op.H.shape[1]
    assert lin_op.shape[1] == lin_op.H.shape[0]
    assert A.shape[0] + 1 == lin_op.shape[0]
    assert A.shape[1] == lin_op.shape[1]
