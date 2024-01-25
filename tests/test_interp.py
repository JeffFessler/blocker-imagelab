import numpy as np
import pytest

from imagelab.interp import downsample


@pytest.mark.parametrize(
    "reduce", ["mean", "median", "max", "min"]
)  # these arguments all return the same for kron ones data
def test_downsample_on_kron(reduce):
    truth = np.r_[0:4].reshape(2, 2)
    # downsample by 2, data multiple of 2
    x = np.kron(truth, np.ones((2, 2)))
    y = downsample(x, 2, reduce=reduce)
    assert (truth == y).all()

    y = downsample(x, (2, 2), reduce=reduce)
    assert (truth == y).all()

    # downsample by 3, data multiple of 3
    x = np.kron(truth, np.ones((3, 3)))
    y = downsample(x, 3, reduce=reduce)
    assert (truth == y).all()

    y = downsample(x, (3, 3), reduce=reduce)
    assert (truth == y).all()

    # downsample by (3,2), data multiple of 3,2
    x = np.kron(truth, np.ones((3, 2)))
    y = downsample(x, (3, 2), reduce=reduce)
    assert (truth == y).all()

    x = np.kron(truth, np.ones((6, 3, 2)))[np.newaxis, np.newaxis]
    y = downsample(x, (6, 3, 2), reduce=reduce)[0, 0]
    assert (truth == y).all()

    x = np.kron(truth, np.ones((7, 3, 3)))
    y = downsample(x, 3, reduce=reduce)
    assert truth.shape == y.shape[1:]
    assert y.shape[0] == 2
    assert (truth == y[0]).all()


def test_downsample_mirt():
    x = [6, 5, 2]
    x = np.transpose(2 * np.r_[1 : np.prod(x) + 1].reshape(2, 5, 6), (2, 1, 0))
    y = downsample(x, 1)
    assert (y == x).all()
    y = downsample(x, 2)
    assert y[0, 0, 0] == 39
    assert (y == np.array([[39, 63], [43, 67], [47, 71]])[:, :, np.newaxis]).all()

    x = np.r_[1 : 24 + 1].reshape(6, 4).T
    y = downsample(x, 2)
    assert (y == np.array([[3.5, 11.5, 19.5], [5.5, 13.5, 21.5]])).all()
