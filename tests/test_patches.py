import numpy as np
import pytest

from imagelab import patches


def assert_common_base(a, b):
    assert a.__array_interface__["data"][0] == b.__array_interface__["data"][0]


@pytest.mark.parametrize(
    "pad,stride",
    [
        ("wrap", 1),
        ("constant", 1),
        (None, 1),
        (None, 2),
        pytest.param(
            "wrap",
            2,
            marks=pytest.mark.xfail(
                reason="im2col_weights doesn't support stride != 1"
            ),
        ),
        pytest.param(
            "constant",
            3,
            marks=pytest.mark.xfail(
                reason="im2col_weights doesn't support stride != 1"
            ),
        ),
        pytest.param(
            "edge",
            1,
            marks=pytest.mark.xfail(reason="col2im doesn't support pad=='edge'"),
        ),
    ],
)
def test_im2col_col2im_weights_recovery(pad, stride):
    # an integration test
    patch_shape = (4, 8)
    np.random.seed(4848)
    img = np.random.rand(256, 500) + 1j * np.random.rand(256, 500)

    img_patch = patches.im2col(img, patch_shape, stride=stride, pad=pad)

    img_sum = patches.col2im(img_patch, img.shape, patch_shape, stride=stride, pad=pad)

    img_recon = img_sum / patches.im2col_weights(
        img.shape, patch_shape, stride=stride, pad=pad
    )

    # Note floating point division cause some error in recon,
    # thus we check for approximate equality
    assert np.allclose(img, img_recon)


@pytest.mark.parametrize(
    "pad,stride",
    [
        ("wrap", 1),
        ("constant", 1),
    ],
)
def test_im2col_col2im_weights_recovery_color(pad, stride):
    # an integration test
    patch_shape = (3, 4, 3)
    np.random.seed(4832)
    img = np.random.rand(128, 150, 3) + 1j * np.random.rand(128, 150, 3)

    img_patch = patches.im2col(
        img, patch_shape, stride=stride, pad=pad, no_pad_axes=(-1,)
    )

    img_sum = patches.col2im(
        img_patch, img.shape, patch_shape, stride=stride, pad=pad, no_pad_axes=(-1,)
    )

    img_recon = img_sum / patches.im2col_weights(
        img.shape, patch_shape, stride=stride, pad=pad, no_pad_axes=(-1,)
    )

    # Note floating point division cause some error in recon,
    # thus we check for approximate equality
    assert np.allclose(img, img_recon)


@pytest.mark.parametrize(
    "patch_shape,pad,stride",
    [
        ((3, 3), "wrap", 1),
        ((3, 4, 3, 3), "constant", 1),
        ((3, 1, 3, 1), "wrap", 1),
        ((1, 2, 1, 3), "constant", 1),
        pytest.param(
            (3, 3),
            "wrap",
            (3, 3),
            marks=pytest.mark.xfail(reason="col2im doesn't support stride != 1"),
        ),
        pytest.param(
            (3, 4, 3, 3),
            "constant",
            (3, 4, 3, 3),
            marks=pytest.mark.xfail(reason="col2im doesn't support stride != 1"),
        ),
        pytest.param(
            (3, 1, 3, 1),
            "wrap",
            (3, 1, 3, 1),
            marks=pytest.mark.xfail(reason="col2im doesn't support stride != 1"),
        ),
        pytest.param(
            (1, 2, 1, 3),
            "constant",
            (1, 2, 1, 3),
            marks=pytest.mark.xfail(reason="col2im doesn't support stride != 1"),
        ),
    ],
)
def test_im2col_col2im_weights_recovery_lf(patch_shape, pad, stride):
    # an integration test
    np.random.seed(9876)
    shape = (3, 4, 50, 51)
    img = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    img_patch = patches.im2col(
        img, patch_shape, stride=stride, pad=pad, no_pad_axes=(0, 1)
    )

    img_sum = patches.col2im(
        img_patch, img.shape, patch_shape, stride=stride, pad=pad, no_pad_axes=(0, 1)
    )

    img_recon = img_sum / patches.im2col_weights(
        img.shape, patch_shape, stride=stride, pad=pad, no_pad_axes=(0, 1)
    )

    # Note floating point division cause some error in recon,
    # thus we check for approximate equality
    assert np.allclose(img, img_recon)


def test_nonoverlapping():
    wts = patches.im2col_weights((6, 6), (3, 3), stride=(3, 3), pad=None)
    assert np.allclose(wts, 1)


@pytest.mark.parametrize(
    "pad",
    [
        "wrap",
        "constant",
    ],
)
def test_im2col(pad):
    delta = np.zeros((101, 101))
    delta[50, 50] = 1

    patch = patches.im2col(delta, (1, 5), stride=1, pad=pad)

    assert patch.shape[1] == 101 ** 2  # True for stride = 1, pad != None

    # The center patch should contain a delta in the center
    center_patch = patch.shape[1] // 2
    assert (patch[:, center_patch] == np.array([0, 0, 1, 0, 0])).all()

    # Should see an inverted identity in the patch matrix
    assert (patch[::-1, center_patch - 2 : center_patch + 3] == np.eye(5)).all()

    # All others should be zero
    assert (patch[:, : center_patch - 2] == 0).all()
    assert (patch[:, center_patch + 3 :] == 0).all()


@pytest.mark.parametrize(
    "patch_shape,pad,stride",
    [
        ((3, 3), "wrap", 1),
        ((3, 4, 3, 3), "constant", 1),
        ((1, 4, 3, 1), None, 1),
        ((3, 1, 3, 1), "wrap", 1),
        ((1, 2, 1, 3), "constant", 1),
    ],
)
def test_tcol2im_against_col2im(patch_shape, pad, stride):
    n = np.prod(patch_shape)
    W = np.random.rand(n, n)
    X = np.random.rand(5, 5, 20, 30)
    Xp = patches.im2col(X, patch_shape, stride=stride, pad=pad)

    B1 = patches.col2im(W @ Xp, X.shape, patch_shape, pad=pad, stride=stride)
    B2 = patches.tcol2im(W, Xp, X.shape, patch_shape, pad=pad, stride=stride)

    assert np.allclose(B1, B2)


def test_no_copy_if_no_pad():  # can't be done for im2col :<
    X = np.random.randn(100, 200)
    Xp = patches.im2win(X, (5, 7), stride=1, pad=None)
    assert_common_base(X, Xp)
    # assert_common_base(X, Xp.reshape(5, 7, -1))


def test_minimize_dims():
    a = np.outer(np.ones((3,)), [1, 2, 3])
    b = patches.minimize_dims(a)
    assert (a == b).all()
    assert a.ndim == b.ndim
    assert a.shape != b.shape
    assert b.shape == (1, 3)
    assert (b == [[1, 2, 3]]).all()

    a = a.T
    b = patches.minimize_dims(a)
    assert (a == b).all()
    assert a.ndim == b.ndim
    assert a.shape != b.shape
    assert b.shape == (3, 1)
    assert (b == [[1], [2], [3]]).all()
