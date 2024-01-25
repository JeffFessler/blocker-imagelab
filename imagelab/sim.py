"""imagelab/sim.py

Code for creating simulation test data.
"""

import numpy as np


def continuous_disk(R=20, f1=0.5, f2=1 / 3):
    return lambda y, x: (
        1
        + (np.cos(f1 * np.pi * np.sqrt(x ** 2 + y ** 2)) > 0) * (x > -y)
        + (np.cos(f2 * np.pi * np.sqrt(x ** 2 + y ** 2)) > 0) * (x <= -y)
    ) * (x ** 2 + y ** 2 < R ** 2)


def continuous_target(R=20, f1=0.5):
    return lambda y, x: (1 + (np.cos(f1 * np.pi * np.sqrt(x ** 2 + y ** 2)) > 0)) * (
        x ** 2 + y ** 2 < R ** 2
    )


def continuous_ellipse(R=20, ecc=1):
    return lambda y, x: 1 * (x ** 2 + (ecc * y) ** 2 < R ** 2)


def continuous_rectangle(w=10, h=10):
    return lambda y, x: (np.abs(x) < w / 2) * (np.abs(y) < h / 2) * 1


def continuous_norm_target(R=20, f1=0.5, norm="inf"):
    if norm == "inf":
        return lambda y, x: (
            1 + (np.cos(f1 * np.pi * np.maximum(np.abs(x), np.abs(y))) > 0)
        ) * (np.maximum(np.abs(x), np.abs(y)) < np.abs(R))
    else:
        return lambda y, x: (
            1
            + (
                np.cos(
                    f1 * np.pi * (np.abs(x) ** norm + np.abs(y) ** norm) ** (1 / norm)
                )
                > 0
            )
        ) * (np.abs(x) ** norm + np.abs(y) ** norm < np.abs(R) ** norm)


def discrete_stair_case(N=100, step_width=25, step_height=1):
    """An example signal from Jeff Fessler's EECS 598 class at UMich
    Options
        N: length of signal
        step_width: width of each step in staircase
        step_height: height of each step in staircase
    Output
        x: 1D Numpy array
    """
    return step_height * (np.cumsum([np.mod(t + 1, step_width) == 0 for t in range(N)]))


def discrete_UM(enlarge=13):
    """An example image from Jeff Fessler's EECS 598 class at UMich
    Options
        enlarge: Increase image size by this factore so that it is
            enlarge*6 x enlarge*20
    Output
        x: 2D Numpy array
    """
    tmp = np.array(
        [
            np.zeros(20),
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            np.zeros(20),
        ]
    )
    x = np.kron(tmp, np.ones((enlarge, enlarge)))
    return x


def discrete_rect_face():
    """ An example image from Jeff Fessler's EECS 598 class at UMich"""
    M, N = 256, 192
    X = np.zeros((M, N))
    X[19:90, 29:50] = 1  # left eye
    X[99:110, 89:100] = 1  # nose
    X[19:90, 129:150] = 1  # right eye
    X[149:200, 19:170] = 1  # mouth
    X[159:161, 149:151] = 0  # spec in mouth
    return X


def rand_rect(shape, num=10, seed=None):
    """Random Sum of Rectangles

    Generates a sum of `num` rectangles
    of random width.

    Parameters
    ----------
    shape: shape of generated signal, if int, returns a 1D
            signal, if 2-tuple, returns an image
    num: number of rectangles to add together

    """
    rng = np.random.default_rng(seed)
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, tuple):
        if len(shape) == 1:
            M, N = shape[0], 1
        else:
            M, N = shape
    l, w = M // 5, N // 5
    img = np.zeros((M, N))
    for _ii in range(num):
        # tl - top left, br - bottom right
        tl_y = rng.integers(M - 2 * l) if M > 1 else 0
        tl_x = rng.integers(N - 2 * w) if N > 1 else 0
        br_y = rng.integers(tl_y + l, M) if M > 1 else 1
        br_x = rng.integers(tl_x + w, N) if N > 1 else 1
        img[tl_y:br_y, tl_x:br_x] += 1
    return img / img.max() if len(shape) > 1 else np.squeeze(img / img.max())


def batch_rand_rect(shape, num=10, seed=None):
    """Returns a generator that yields shape[0] examples
    one-by-one. If you would like all of the examples then
    just pass it to a list or array like so:

    >>> all_train = list(batch_rand_rect((500,32,32)))
    >>> all_train = np.array(list(batch_rand_rect((500,32,32))))
    or process this one by one:
    >>> for example in batch_rand_rect((500,32,32)):
    >>>    assert example.shape == (32,32)
    >>> img_gen = batch_rand_rect((500,32,32))
    >>> example1 = next(img_gen)
    >>> example2 = next(img_gen)

    This allows us to process a lot of examples without
    keeping them all in memory.
    """
    rng = np.random.default_rng(seed)
    batch_size = shape[0]
    shape = shape[1:]
    return (rand_rect(shape, num, rng) for _ in range(batch_size))


def rand_mod_pulse(shape, num=3, seed=None):
    """Sum of Random Modulated Pulses

    Generates a sum of `num` modulated gaussian pulses of
    random width and random frequency.

    Parameters
    ----------
    shape: shape of generated signal, if int, returns a 1D
            signal, if 2-tuple, returns an image
    num: number of pulses to add together

    """
    rng = np.random.default_rng(seed)
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, tuple):
        if len(shape) == 1:
            M, N = shape[0], 1
        else:
            M, N = shape
    X, Y = np.meshgrid(np.r_[0:N], np.r_[0:M])
    img = np.zeros((M, N))
    for _ii in range(num):
        rx = rng.integers(N - 1) + 1 if N > 1 else 1
        ry = rng.integers(M - 1) + 1 if M > 1 else 1
        cx = rng.integers(rx // 2, N - rx // 2)
        cy = rng.integers(ry // 2, M - ry // 2)
        phi = rng.random() * 2 * np.pi
        wx = rng.random() * N / 3
        wy = rng.random() * M / 3
        img += np.exp(-((X - cx) ** 2 / (15 * rx) + (Y - cy) ** 2 / (15 * ry))) * (
            1 + np.cos((wx * X / N + wy * Y / M) + phi)
        )
    return img / img.max() if len(shape) > 1 else np.squeeze(img / img.max())


def batch_rand_mod_pulse(shape, num=3, seed=None):
    """Returns a generator that yields shape[0] examples
    one-by-one. If you would like all of the examples then
    just pass it to a list or array like so:

    >>> all_train = list(batch_rand_mod_pulse((500,32,32)))
    >>> all_train = np.array(list(batch_rand_mod_pulse((500,32,32))))
    or process this one by one:
    >>> for example in batch_rand_mod_pulse((500,32,32)):
    >>>    assert example.shape == (32,32)
    >>> img_gen = batch_rand_mod_pulse((500,32,32))
    >>> example1 = next(img_gen)
    >>> example2 = next(img_gen)

    This allows us to process a lot of examples without
    keeping them all in memory.
    """
    rng = np.random.default_rng(seed)
    batch_size = shape[0]
    shape = shape[1:]
    return (rand_mod_pulse(shape, num, rng) for _ in range(batch_size))
