import numpy as np
import pytest
import scipy.fft

from imagelab.sparsity import (  # dct2dmtx,; oc_dctmtx,; overcomplete_dct2dmtx,
    dctmtx,
    nd_dctmtx,
    oc_nd_dctmtx,
)


@pytest.mark.parametrize(
    "size",
    [
        4,
        8,
        13,
        21,
    ],
)
def test_unitary_dct(size):
    W = dctmtx(size)

    assert W.shape == (size, size)

    assert np.allclose(W.T @ W, np.eye(size))
    assert np.allclose(W @ W.T, np.eye(size))


@pytest.mark.parametrize(
    "size",
    [
        4,
        8,
        13,
        21,
    ],
)
def test_dct_values(size):
    assert np.allclose(dctmtx(size), scipy.fft.dct(np.eye(size), axis=0, norm="ortho"))


def test_oc_nd_dct_singleton_dims():
    K_shape = (1, 5, 4, 1, 5)
    N_shape = (5, 3, 1, 6, 1)
    mtx = oc_nd_dctmtx(K_shape, N_shape)
    assert mtx.shape == (np.prod(K_shape), np.prod(N_shape))
    assert np.allclose(mtx.T, mtx.T / (abs(mtx.T) ** 2).sum(axis=0, keepdims=True))


@pytest.mark.parametrize(
    "N,M",
    [
        (4, 7),
        (8, 8),
        (13, 13),
        (23, 21),
    ],
)
def test_unitary_2d_dct(N, M):
    size = N * M
    W = nd_dctmtx(N, M)

    assert W.shape == (size, size)

    assert np.allclose(W.T @ W, np.eye(size))
    assert np.allclose(W @ W.T, np.eye(size))


@pytest.mark.parametrize(
    "K, N",
    [
        (4, 7),
        (8, 8),
        (13, 13),
        (23, 21),
    ],
)
def test_overcomplete_dct(K, N):
    W = dctmtx(K, N)

    assert W.shape == (K, N)
