import numpy as np
import pytest

from imagelab import fft

try:
    import cupy as cp

    cuda_en = True
except ImportError:
    cuda_en = False


@pytest.mark.parametrize("norm", ["ortho", "forward", "backward"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(not cuda_en, reason="Cupy could not be imported"),
        ),
    ],
)
@pytest.mark.parametrize("shift_x", [True, False])
@pytest.mark.parametrize("shift_f", [True, False])
def test_fft2_pairs(norm, device, shift_x, shift_f):
    fft2, ifft2 = fft.get_fft2_pair(
        norm=norm, device=device, shift_x=shift_x, shift_f=shift_f
    )
    if device == "cpu":
        f = np.random.randn(333, 512)
    else:
        f = cp.random.randn(333, 512)
    fhat = ifft2(fft2(f))
    assert np.allclose(f, fhat)
