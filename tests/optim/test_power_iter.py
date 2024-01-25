import numpy as np
from scipy.linalg import svd

from imagelab.optim.power_iteration import power_iteration


def test_power_iteration():
    A = np.random.randn(8, 5)
    A = A.T @ A
    U, s, Vh = svd(A)
    b_true = U[:, 0]
    s_true = s[0]
    b_hat, s_hat = power_iteration(A)

    assert np.isclose(s_true, s_hat)
    assert np.allclose(b_true, b_hat)
