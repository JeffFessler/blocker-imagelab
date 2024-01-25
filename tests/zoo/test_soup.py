import numpy as np
import pytest

from imagelab.patches import PatchConfig
from imagelab.zoo import soupdil as soup

# from imagelab.sparsity import nd_dctmtx


@pytest.mark.parametrize(
    "patch_shape,prox,pad",
    [
        ((4, 7), "l0", "wrap"),
        ((6, 6), "l0", None),
        ((5, 5), "l1", "wrap"),
        ((7, 5), "l0", "wrap"),
        ((1, 7), "l1", "constant"),
    ],
)
def test_blind_soup_runs(patch_shape, prox, pad):  # will it even run?
    iters = []

    def func(x, itr):
        return iters.append(itr)

    np.random.seed(643)
    Xtest_true = np.random.randn(150, 223).astype(np.float32)
    Xtest0 = Xtest_true + 0.1 * np.random.randn(*Xtest_true.shape).astype(np.float32)

    def data_update(X, weight):
        return (X + 1e-6 * Xtest0) / (weight + 1e-6)  # denoising

    Xhat = soup.blind_dl(
        Xtest0,
        gamma=None,
        data_update=data_update,
        patch_config=PatchConfig(
            arr_shape=Xtest0.shape, patch_shape=patch_shape, pad=pad, stride=1
        ),
        niter=5,
        prox=prox,
        callback=func,
    )
    assert len(iters) == 6
    assert (np.array(iters) == np.r_[0:6]).all()
    assert Xhat.shape == Xtest0.shape
    assert Xhat.dtype == Xtest_true.dtype
