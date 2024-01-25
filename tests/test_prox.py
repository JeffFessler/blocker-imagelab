import numpy as np

from imagelab import prox


def test_hard_thresholding_inplace_real():
    gamma = 0.5  # low precision required for exact representation
    # we test real separetly since it branches in inplace function
    B = np.random.rand(50, 61)
    # make sure we are consistent at boundaries
    B[34, 34] = gamma
    B[24, 14] = -gamma
    B[0, 0] = gamma / 2

    Z1 = prox.hard_thresholding(B, 0.5)
    Z2 = prox.hard_thresholding_inplace(B.copy(), 0.5)

    assert Z1[0, 0] == 0
    # assert np.allclose(Z1, Z2)
    assert (Z1 == Z2).all()


def test_hard_thresholding_inplace_complex():
    gamma = 0.5  # low precision required for exact representation
    # make sure we handle complex numbers
    B = np.random.rand(50, 61) + 1j * np.random.rand(50, 61)
    # make sure we are consistent at boundaries
    B[34, 34] = gamma
    B[24, 14] = 1j * gamma
    B[44, 50] = np.sqrt(gamma) + 1j * np.sqrt(gamma)
    B[45, 50] = np.sqrt(gamma) - 1j * np.sqrt(gamma)
    # trying here to catch both sides of float precision rounding
    B[15, 15] = -np.sqrt(gamma) - 1j * np.sqrt(gamma)

    B[0, 0] = gamma / 2

    Z1 = prox.hard_thresholding(B, 0.5)
    Z2 = prox.hard_thresholding_inplace(B.copy(), 0.5)

    assert Z1[0, 0] == 0
    # assert np.allclose(Z1, Z2)
    assert (Z1 == Z2).all()


def test_soft_thresholding_inplace_complex():
    gamma = 0.5  # low precision required for exact representation
    # make sure we handle complex numbers
    B = np.random.rand(50, 61) + 1j * np.random.rand(50, 61)
    # make sure we are consistent at boundaries
    B[34, 34] = gamma
    B[24, 14] = 1j * gamma
    B[44, 50] = np.sqrt(gamma) + 1j * np.sqrt(gamma)
    B[45, 50] = np.sqrt(gamma) - 1j * np.sqrt(gamma)
    # trying here to catch both sides of float precision rounding
    B[15, 15] = -np.sqrt(gamma) - 1j * np.sqrt(gamma)

    B[0, 0] = gamma / 2

    Z1 = prox.soft_thresholding(B, 0.5)
    Z2 = prox.soft_thresholding_inplace(B.copy(), 0.5)

    assert Z1[0, 0] == 0
    # assert np.allclose(Z1, Z2)
    assert (Z1 == Z2).all()
