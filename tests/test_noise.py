import numpy as np

from imagelab import noise

# need more tests that check autoscale works properly


def test_unique_seeds():
    eta = noise()
    S = set()
    for _ii in range(100_000):
        S.add(eta.seed)
        eta = eta()
    assert len(S) == 100_000


def test_constant_seeds():
    eta = noise()
    S = {abs(eta).seed}
    for _ii in range(50):
        S.add(eta.seed)
        eta = eta + 5
        S.add(eta.seed)
        eta = 3 * eta
    assert len(S) == 1


def test_multiply():
    # There is a slight difference in these tests
    # between the right and left, namely when the
    # dtype conversion happens, thus the allclose...
    zz = np.zeros(10, np.float32)
    eta = noise(10)
    # right multiply
    assert np.allclose(5.0 * (zz + eta), (zz + 5.0 * eta))
    # left multiply
    assert np.allclose((zz + eta) * 5.0, (zz + eta * 5.0))
    # self multiply
    assert np.allclose((zz + eta) ** 2, zz + eta * eta)
    eps = noise(5) + 5
    assert np.allclose((zz + eta) * eps, zz + eta * eps)


def test_add():
    zz = np.zeros(10)
    eta = noise(10)
    # right add
    assert (5 + (zz + eta) == (zz + (5 + eta))).all()
    # left add
    assert ((zz + eta) + 5 == (zz + (eta + 5))).all()
    # self add
    assert ((zz + eta) + eta == (zz + (eta + eta))).all()
    eps = noise(5) + 5
    assert ((zz + eta) + eps == (zz + (eta + eps))).all()


def test_subtract():
    zz = np.zeros(10)
    eta = noise(10)
    # right sub
    assert (5 - (zz + eta) == (zz + (5 - eta))).all()
    # left sub
    assert ((zz + eta) - 5 == (zz + (eta - 5))).all()
    # self sub
    assert (zz == zz + (eta - eta)).all()
    eps = noise(5) + 5
    assert ((zz + eta) - eps == (zz + (eta - eps))).all()


def test_neg():
    zz = np.zeros(10)
    eta = noise(10)
    assert ((zz - eta) == (zz + (-eta))).all()
    assert ((zz + -1 * eta) == (zz + (-eta))).all()


def test_pos():
    zz = np.zeros(10)
    eta = noise(10)
    assert ((zz + eta) == (zz + (+eta))).all()
    assert ((zz + 1 * eta) == (zz + (+eta))).all()


def test_abs():
    zz = np.zeros(10)
    eta = noise(10)
    assert (abs(zz + eta) == (zz + abs(eta))).all()


def test_pow():
    zz = np.zeros(10)
    eta = noise(10)
    assert ((zz + eta ** 2) == (zz + eta) ** 2).all()


def test_draw():
    eta = noise(10, mean=5)
    eps = ~eta
    assert eps.scale == eta.scale
    assert eps.mean == eta.mean
    assert eps.autoscale == eta.autoscale
    assert eps.seed != eta.seed


def test_complicated():
    zz = np.zeros(10)
    eta = noise(10)
    # while theoretically this could fail, it is improbable
    assert np.allclose(10 * abs(zz - eta + 100), (zz + 2 * abs(5 * (-eta + 100))))


def test_matmul():
    A = np.eye(15)[:15, :10]
    oo = np.ones(15)
    oo[10:15] = 0
    zz = np.zeros(10)
    eta = noise(10)
    assert (A @ eta == eta * oo).all()
    assert (eta @ A == zz + eta).all()


def test_string_scale():
    eta = noise("40db")
    assert eta.scale == 0.01
    assert eta.autoscale is True
    eps = noise("1%")
    assert eps.scale == 0.01
    assert eps.autoscale is True
    zz = 10 * np.random.randn(5) + 7
    eta.seed = eps.seed
    assert (eps + zz == zz + eta).all()
    zz = np.zeros(10)
    assert (zz == zz + eta).all()
