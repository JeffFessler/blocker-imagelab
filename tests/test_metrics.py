import numpy as np
import numpy.linalg
import pytest

from imagelab.metrics import (
    MAE,
    MSE,
    NRMSE,
    PSNR,
    RMSE,
    SNR,
    SSIM,
    MaxErr,
    mae,
    maxerr,
    mse,
    nrmse,
    psnr,
    rmse,
    snr,
    ssim,
)


def test_snr_psnr_diff():
    np.random.seed(252)
    x = np.random.rand(200, 300)
    xtrue = np.random.rand(200, 300)
    snr_x = snr(x, xtrue)
    psnr_x = psnr(x, xtrue)

    expected_diff = 10 * np.log10((np.max(xtrue) ** 2) / ((np.abs(xtrue) ** 2).mean()))
    actual_diff = psnr_x - snr_x

    assert np.allclose(actual_diff, expected_diff)


def test_nrmse():
    np.random.seed(563)
    x = np.random.rand(200, 300)
    xtrue = np.random.rand(200, 300)

    nrmse_x = nrmse(x, xtrue)

    rmse_x = rmse(x, xtrue)
    rms_xtrue = rmse(xtrue, np.zeros_like(xtrue))

    assert np.allclose(nrmse_x, rmse_x / rms_xtrue)


def test_psnr():
    np.random.seed(678)
    x = np.random.rand(200, 300)
    xtrue = np.random.rand(200, 300)
    psnr_x = psnr(x, xtrue)

    other_psnr = 20 * np.log10(np.max(xtrue) / rmse(x, xtrue))

    assert np.allclose(psnr_x, other_psnr)


def test_nrmse_mse_psnr_conversion():
    np.random.seed(678)
    x = np.random.rand(200, 300)
    xtrue = np.random.rand(200, 300)
    psnr_x = psnr(x, xtrue)
    mse_x = mse(x, xtrue)
    nrmse_x = nrmse(x, xtrue)
    norm_diff = nrmse_x * np.linalg.norm(xtrue.reshape(-1))
    other_mse = (norm_diff ** 2) / xtrue.size
    assert np.allclose(mse_x, other_mse)

    peak = np.max(xtrue) ** 2
    other_psnr = 10 * np.log10(peak / other_mse)
    assert np.allclose(psnr_x, other_psnr)


@pytest.mark.parametrize(
    "func,Metric",
    [
        (mse, MSE),
        (rmse, RMSE),
        (nrmse, NRMSE),
        (mae, MAE),
        (maxerr, MaxErr),
        (ssim, SSIM),
        (snr, SNR),
        (psnr, PSNR),
    ],
)
def test_metric_equivalence(func, Metric):
    np.random.seed(218)
    x = np.random.rand(200, 300)
    xtrue = np.random.rand(200, 300)
    res1 = func(x, xtrue)

    metric = Metric(xtrue)
    res2 = float(metric(x))
    assert np.allclose(res1, res2)
