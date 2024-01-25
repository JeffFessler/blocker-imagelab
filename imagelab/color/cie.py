import os

import numpy as np
from scipy.io import loadmat

here = os.path.abspath(os.path.dirname(__file__))
spectral_locus = loadmat(here + "/locus.mat", variable_names=["locus"])["locus"]


def show_cie31():
    from matplotlib import pyplot as plt  # lazy import for now...

    fig, ax = plt.subplots()
    ax.plot(spectral_locus[:, 0], spectral_locus[:, 1])
    ax.plot(spectral_locus[::10, 0], spectral_locus[::10, 1], "o")

    m = np.linspace(10 ** 3 / 1667, 10 ** 3 / 25000, 100)
    T = 10 ** 3 / m
    px, py = temp_to_xy(T)
    ax.plot(px, py, "r")

    ax.set_aspect("equal")
    ax.set_xlim([0.0, 0.85])
    ax.set_ylim([0.0, 0.85])
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def temp_to_xy(T):
    """A cubic spline interpolation of the mired scale
    as shown on wikipedia.
    https://en.wikipedia.org/wiki/Planckian_locus
    """
    m = (10 ** 3) / T
    x = (
        -0.2661239 * (m ** 3) - 0.23435890 * (m ** 2) + 0.87769560 * m + 0.17991000
    ) * (T >= 1667) * (T < 4000) + (
        -3.0258469 * (m ** 3) + 2.10703790 * (m ** 2) + 0.22263470 * m + 0.24039000
    ) * (
        T >= 4000
    ) * (
        T < 25001
    )
    y = (
        (-1.1063814 * (x ** 3) - 1.34811020 * (x ** 2) + 2.18555832 * x - 0.20219683)
        * (T >= 1667)
        * (T < 2222)
        + (-0.9549476 * (x ** 3) - 1.37418593 * (x ** 2) + 2.09137015 * x - 0.16748867)
        * (T >= 2222)
        * (T < 4000)
        + (3.0817580 * (x ** 3) - 5.87338670 * (x ** 2) + 3.75112997 * x - 0.37001483)
        * (T >= 4000)
        * (T < 25001)
    )
    return x, y
