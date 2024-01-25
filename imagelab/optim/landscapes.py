from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import norm

__all__ = [
    "ArtificialLanscape",
    "quadratic",
    "uquadratic",
    "diamond",
    "rosenbrock",
    "rosenbrock2",
    "peaks",
    "quartic",
    "rastrigin",
    "ackley",
    "himmelblau",
]
# Goal is to implement a majority of
# https://en.wikipedia.org/wiki/Test_functions_for_optimization


class ArtificialLanscape(object):
    """docstring for AbstractLanscape"""

    def __init__(
        self,
        func,
        grad,
        x_star=None,
        xlim=(-5, 5),
        ylim=(-5, 5),
        L=None,
        x0=None,
        title=None,
        ccount=30,
    ):
        super(ArtificialLanscape, self).__init__()
        self.fig = None
        self.ax = None
        self.CS = None
        self.func = func
        self.grad = grad
        self.xlim = xlim
        self.ylim = ylim
        self.x_star = x_star
        self.x0 = x0
        self.L = L
        self.title = title
        self.ccount = ccount

    def init_contours(self, ax=None, figsize=(10, 6)):
        delta = 0.025
        x = np.arange(self.xlim[0], self.xlim[1] + delta, delta)
        y = np.arange(self.ylim[0], self.ylim[1] + delta, delta)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
        self.CS = self.ax.contour(X, Y, np.sign(Z) * np.sqrt(np.abs(Z)), self.ccount)
        self.CS.levels = [np.sign(lev) * (lev ** 2) for lev in self.CS.levels]
        # ax.clabel(CS, inline=1, fontsize=10)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("$x_1$")
        self.ax.set_ylabel("$x_2$")
        if self.title is not None:
            self.ax.set_title(self.title)

    def plot_iterates(self, opt_func, L=None, x0=None, **kwargs):
        if not self.ax:
            self.init_contours(
                ax=kwargs.pop("ax", None), figsize=kwargs.pop("figsize", (10, 6))
            )
        if x0 is None:
            x0 = self.x0
        if L is None:
            L = self.L

        itrs = []

        def func(x, itr):
            itrs.append(x.copy())
            return False

        opt_func(self.grad, L, x0.copy(), callback=func, **kwargs)
        itrs = np.array(itrs)
        x_itrs = itrs[:, 0]
        y_itrs = itrs[:, 1]
        self.ax.plot(x_itrs, y_itrs, "-o", label=opt_func.__name__)
        if self.x_star is not None:
            self.ax.plot(self.x_star[0], self.x_star[1], "r*")
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        return itrs

    __call__ = plot_iterates


####################


A = np.array([[1.0, 0.5], [0.0, 0.5]])
y0 = np.array([1.0, 1.0])
xstar = np.array([0, 2])
opnorm = norm(A, 2)


def func(x1, x2):
    return 0.5 * (
        (A[0, 0] * x1 + A[0, 1] * x2 - y0[0]) ** 2
        + (A[1, 1] * x2 + A[1, 0] * x1 - y0[1]) ** 2
    )


def grad(x):
    return A.T @ (A @ x - y0)


quadratic_scape = {
    "title": "Quadratic",
    "func": func,
    "grad": grad,
    "xlim": [-1.5, 1.5],
    "ylim": [-0.3, 2.3],
    "x0": np.array([0.0, 0.0]),
    "x_star": xstar,
    "L": opnorm ** 2,
}

quadratic = ArtificialLanscape(**quadratic_scape)

# ####################

# A = np.array([  [1.0, 1.0],
#                 [1.0, 1.0]])
# y0 = np.array([1,1])
# xstar = np.array([0.5,0.5])
# opnorm = norm(A,2)
# func = lambda x1, x2: 0.5*((A[0,0]*x1+A[0,1]*x2-y0[0])**2 \
#                       + (A[1,1]*x2+A[1,0]*x1-y0[1])**2)
# grad = lambda x: A.T@(A@x-y0)

# uquadratic_scape = {
#     'title': 'Underdetermined Quadratic',
#     'func': func,
#     'grad': grad,
#     'xlim': [-0.1, 1],
#     'ylim': [-0.1, 1],
#     'x0'  : np.array([0.0,0.0]),
#     'x_star': xstar,
#     'L': opnorm**2
# }

# uquadratic = ArtificialLanscape(**uquadratic_scape)

####################

uA = np.array([[0.5, 1.0], [0.5, 1.0]])
uy0 = np.array([1, 1])
xstar = None
opnorm = norm(uA, 2)


def func(x1, x2):
    return 0.5 * (
        (uA[0, 0] * x1 + uA[0, 1] * x2 - uy0[0]) ** 2
        + (uA[1, 1] * x2 + uA[1, 0] * x1 - uy0[1]) ** 2
    )


def grad(x):
    return uA.T @ (uA @ x - uy0)


uquadratic_scape = {
    "title": "Underdetermined Quadratic",
    "func": func,
    "grad": grad,
    "xlim": [-0.1, 2.1],
    "ylim": [-0.1, 1.1],
    "x0": np.array([0.0, 0.0]),
    "x_star": xstar,
    "L": opnorm ** 2,
}

uquadratic = ArtificialLanscape(**uquadratic_scape)

####################

pA = np.array([[-3.0, 3.0], [1.0, 1.0]])
xstar = np.array([0, 0])


def func(x1, x2):
    return (3 * np.abs(x2 - x1) + np.abs(x1 + x2)) ** 2  # squared for visual


def grad(x):
    return np.norm(pA @ x, 1)


# from https://en.wikipedia.org/wiki/Coordinate_descent 3/27/19
diamond_scape = {
    "title": "Diamond",
    "func": func,
    "grad": grad,
    "xlim": [-3, 3],
    "ylim": [-3, 3],
    "x0": np.array([-2, -2.5]),
    "x_star": xstar,
    "L": None,
    "ccount": 10,
}

diamond = ArtificialLanscape(**diamond_scape)

####################


def f_rosenbrock(x1, x2, a=1, b=100):
    return (a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2


def grad_rosenbrock(x1, x2=None, a=1, b=100):
    if x2 is None:
        x2 = x1[1]
        x1 = x1[0]
    return np.array([-2 * b * (x2 - x1 ** 2) * x1 - 2 * (1 - x1), b * (x2 - x1 ** 2)])


def hess_rosenbrock(x1, x2=None, a=1, b=100):
    if x2 is None:
        x2 = x1[1]
        x1 = x1[0]
    return np.array([[-b * (x2 - 3 * x1 ** 2) + 2, -b * x1], [-b * x1, b / 2]])


rosenbrock_scape = {
    "title": "Rosenbrock Function",
    "func": f_rosenbrock,
    "grad": grad_rosenbrock,
    "xlim": [-1.1, 1.1],
    "ylim": [-0.5, 1.1],
    "x0": np.array([-0.5, 0.9]),
    "x_star": np.array([1, 1]),
    "L": 446,
}

rosenbrock = ArtificialLanscape(**rosenbrock_scape)


def grad_ros2(x1, x2=None):
    if x2 is None:
        x2 = x1[1]
        x1 = x1[0]
    return np.array([1, 5]) * grad_rosenbrock(x1, 5 * x2, a=1, b=0.5)


rosenbrock_scape2 = {
    "title": "Rosenbrock Function",
    "func": lambda x1, x2: f_rosenbrock(x1, 5 * x2, a=1, b=0.5),
    "grad": grad_ros2,
    "xlim": [-2, 2],
    "ylim": [-0.5, 1.5],
    "x0": np.array([-1.5, 1.25]),
    "x_star": np.array([1.0, 1.0 / 5]),
    "ccount": 20,
}

rosenbrock2 = ArtificialLanscape(**rosenbrock_scape2)

#####################


def f_peaks(x1, x2):
    return (
        3 * (1 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1) ** 2)
        - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.exp(-(x1 ** 2) - x2 ** 2)
        - 1 / 3 * np.exp(-((x1 + 1) ** 2) - x2 ** 2)
    )


peaks_scape = {
    "title": "Peaks Function",
    "func": f_peaks,
    "grad": None,
    "xlim": [-3, 3],
    "ylim": [-3, 3],
    "x0": np.array([0.0, 0.0]),
    "x_star": None,
    "L": None,
}

peaks = ArtificialLanscape(**peaks_scape)

#####################


def convex_quartic_wikipedia(x1, x2):
    return (x2 + 1) ** 4 / 25 + (x1 - 1) ** 4 / 10 + x1 ** 2 + x2 ** 2


def grad_convex_quartic_wikipedia(x1, x2=None):
    if x2 is None:
        x2 = x1[1]
        x1 = x1[0]
    u = (2 / 5) * (x1 - 1) ** 3 + 2 * x1
    v = (4 / 25) * (x2 + 1) ** 3 + 2 * x2
    return np.array([u, v])


convex_quaritc_wikipedia_scape = {
    "title": "Convex Quartic Function",
    "func": convex_quartic_wikipedia,
    "grad": grad_convex_quartic_wikipedia,
    "xlim": [-2, 2],
    "ylim": [-2, 2],
    "x0": np.array([-1.7, -1.7]),
    "x_star": np.array([0.1311699, -0.06532408]),
}

quartic = ArtificialLanscape(**convex_quaritc_wikipedia_scape)

#####################


def f_rastrigin(x1, x2, A=10):
    return (
        2 * A
        + (x1 ** 2 - A * np.cos(2 * np.pi * x1))
        + (x2 ** 2 - A * np.cos(2 * np.pi * x2))
    )


rastrigin_scape = {
    "title": "Rastrigin Function",
    "func": f_rastrigin,
    "grad": None,
    "xlim": [-5.12, 5.12],
    "ylim": [-5.12, 5.12],
    "x0": np.array([-5.0, -5]),
    "x_star": np.array([0.0, 0.0]),
    "L": None,
    "ccount": 10,
}

rastrigin = ArtificialLanscape(**rastrigin_scape)

#####################


def f_ackley(x1, x2):
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
        + np.exp(1)
        + 20
    )


ackley_scape = {
    "title": "Ackley Function",
    "func": f_ackley,
    "grad": None,
    "xlim": [-5, 5],
    "ylim": [-5, 5],
    "x0": np.array([-5.0, -5]),
    "x_star": np.array([0.0, 0.0]),
    "L": None,
}

ackley = ArtificialLanscape(**ackley_scape)

#####################


def f_himmelblau(x1, x2):
    return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2


himmelblau_scape = {
    "title": "Himmelblau Function",
    "func": f_himmelblau,
    "grad": None,
    "xlim": [-5, 5],
    "ylim": [-5, 5],
    "x0": np.array([0.0, 0]),
    "x_star": np.array([3.0, 3.0]),
    "L": None,
}

himmelblau = ArtificialLanscape(**himmelblau_scape)

#####################
