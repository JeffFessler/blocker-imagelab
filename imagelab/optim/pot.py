from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import numpy as np

from ..prox import csign
from ..utils import export


@export
class AbstractPotentialFunction(AbstractBaseClass):
    """docstring for PotentialFunction"""

    def __init__(self, reg=1):
        super(AbstractPotentialFunction, self).__init__()
        self.reg = reg

    @abstractmethod
    def eval(self, x):
        raise NotImplementedError

    @abstractmethod
    def grad(self, x):
        raise NotImplementedError

    def curv(self, x):
        raise NotImplementedError

    def shrink(self, x):
        raise NotImplementedError

    @property
    def L(self):
        if hasattr(self, "_L"):
            return self._L
        else:
            return self.curv(0)

    def __call__(self, x):
        return self.eval(x)

    def der(self, x):
        return self.grad(x)


@export
class FairPotential(AbstractPotentialFunction):
    def __init__(self, reg=1, delta=1):
        super(FairPotential, self).__init__(reg)

        self.delta = delta
        self.delta2 = delta ** 2
        self._L = self.reg

    def eval(self, x):
        z = np.abs(x) / self.delta
        return self.reg * (self.delta2) * (z - np.log(1 + z))

    def grad(self, x):
        return self.reg * x / (1 + np.abs(x) / self.delta)

    def curv(self, x):
        return self.reg / (1 + np.abs(x) / self.delta)

    def shrink(self, y, beta=1):
        """
        Compute minimizer over x of 1/2 |y - x|^2 + reg fair(x,delta)
        where fair(x,delta) = delta^2 (|x/delta| - log(1 + |x/delta|))

        Input
        y     scalar, vector, or array of input values

        Output
        xh    solution to minimization problem for each element of y
              (same size as y)
        """
        b1 = beta * self.reg + 1
        return (
            csign(y)
            * (
                (np.abs(y) - self.delta * b1)
                + np.sqrt(
                    (self.delta * b1 - np.abs(y)) ^ 2 + 4 * self.delta * np.abs(y)
                )
            )
            / 2
        )

    def _repr_latex_(self):
        """ Display in notebooks as Latex """
        return (
            r"$\psi_{2}(\ x\ ; \beta={0},\ \delta={1} )".format(
                self.reg, self.delta, "{fair}"
            )
            + r" = \beta \delta^2 (\frac{|x|}{\delta} - \log(1 + \frac{|x|}{\delta}))$"
        )


@export
class HyperbolaPotential(AbstractPotentialFunction):
    def __init__(self, reg=1, delta=1):
        super(HyperbolaPotential, self).__init__(reg)

        self.delta = delta
        self.delta2 = delta ** 2
        self._L = self.reg

    def eval(self, x):
        z = np.abs(x) / self.delta
        return self.reg * (self.delta2 * (np.sqrt(1 + z ** 2) - 1))

    def grad(self, x):
        return self.reg * (x / np.sqrt(1 + np.abs(x / self.delta) ** 2))

    def curv(self, x):
        return self.reg * (1 / np.sqrt(1 + np.abs(x / self.delta) ** 2))

    def shrink(self, y, beta=1):
        """
        Compute minimizer over x of 1/2 |y - x|^2 + reg hyper(x,delta)
        where fair(x,delta) = delta^2 (|x/delta| - log(1 + |x/delta|))

        Input
        y     scalar, vector, or array of input values

        Output
        xh    solution to minimization problem for each element of y
              (same size as y)
        """
        raise NotImplementedError

    def _repr_latex_(self):
        """ Display in notebooks as Latex """
        return (
            r"$\psi_{2}(\ x \ ; \beta={0},\ \delta={1} )".format(
                self.reg, self.delta, "{hyper}"
            )
            + r" = \beta \delta^2 (\sqrt{1 + (\frac{|x|}{\delta})^2} - 1)$"
        )


@export
class HuberPotential(AbstractPotentialFunction):
    def __init__(self, reg=1, delta=1):
        super(HuberPotential, self).__init__(reg)

        self.delta = delta
        self.delta2 = delta ** 2
        self._L = self.reg

    def eval(self, x):
        h = np.abs(x) ** 2 / 2
        ii = np.abs(x) > self.delta
        h[ii] = self.delta * np.abs(x[ii]) - self.delta2 / 2
        return self.reg * h

    def grad(self, x):
        g = 1 * x
        ii = np.abs(x) > self.delta
        g[ii] = self.delta * csign(x[ii])
        return self.reg * g

    def curv(self, x):
        w = np.ones(x.shape)
        ii = np.abs(x) > self.delta
        w[ii] = self.delta / np.abs(x[ii])
        return self.reg * w

    def shrink(self, y, beta=1):
        """
        Compute minimizer over x of 1/2 |y - x|^2 + reg huber(x,delta)
        where huber(x,delta) =

        Input
        y     scalar, vector, or array of input values

        Output
        xh    solution to minimization problem for each element of y
              (same size as y)
        """
        out = y / (1 + beta)
        big = self.delta * (1 + beta) < np.abs(y)
        if np.numel(beta) > 1:
            beta = beta(big)
        y = y[big]
        out[big] = y * (1 - beta * self.delta / np.abs(y))
        return out

    def _repr_latex_(self):
        """ Display in notebooks as Latex """
        return (
            r"$\psi_{2}(\ x \ ; \beta={0},\ \delta={1} )".format(
                self.reg, self.delta, "{huber}"
            )
            + r" = \beta \frac12 x^2 or \beta\delta|x| - \frac{\delta^2}2$"
        )


@export
class QuadraticPotential(AbstractPotentialFunction):
    def __init__(self, reg=1, delta=None):
        super(QuadraticPotential, self).__init__(reg)
        self._L = self.reg

    def eval(self, x):
        return self.reg * np.abs(x) ** 2 / 2

    def grad(self, x):
        return self.reg * x

    def curv(self, x):
        return self.reg

    def shrink(self, y, beta=1):
        """
        Compute minimizer over x of 1/2 |y - x|^2 + reg L2(x)
        where L2(x) = sum(abs(x)**2)

        Input
        y     scalar, vector, or array of input values

        Output
        xh    solution to minimization problem for each element of y
              (same size as y)
        """
        out = y / (1 + beta)
        return out

    def _repr_latex_(self):
        """ Display in notebooks as Latex """
        return (
            r"$\psi_{1}(\ x \ ; \beta={0})".format(self.reg, "{L2}")
            + r" = \beta \frac12 x^2$"
        )


### Helper functions


def get_pot_func_by_name(str_name):
    if isinstance(str_name, AbstractPotentialFunction):
        return str_name
    str_name = str_name.lower()
    if str_name in ["fair"]:
        return FairPotential
    elif str_name in ["hyper", "hyperbola"]:
        return HyperbolaPotential
    elif str_name in ["huber"]:
        return HuberPotential
    elif str_name in ["l2", "parabola", "quadratic"]:
        return QuadraticPotential


def plot_pot(pot_func):
    pot = get_pot_func_by_name(pot_func)()

    from matplotlib import pyplot as plt

    x = np.r_[-2:2:0.01]
    plt.plot(x, pot(x), label="Potential")
    plt.plot(x, pot.grad(x), label="Derivative")
    plt.plot(x, pot.curv(x), label="Weighting")
    plt.legend()
    plt.title(type(pot).__name__)
    return pot


def contour_pot(pot_func):
    pot = get_pot_func_by_name(pot_func)()

    from matplotlib import pyplot as plt

    x = np.r_[-2:2:0.01]
    y = x
    XX, YY = np.meshgrid(x, y)
    ZZ = pot(XX) + pot(YY)
    plt.contour(XX, YY, ZZ)
    plt.title(type(pot).__name__)
    return pot
