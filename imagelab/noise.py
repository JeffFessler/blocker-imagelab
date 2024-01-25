"""imagelab/noise.py
Implements broadcasting noise that becomes the required shape when
operated on.

This is mostly a convenience class, so that instead of wrting
>> y = A@x
>> y = y + y.max()*sigma*np.random.rand(*y.shape)
I can instead write (using default sigma):
>> y = A@x + noise
or with another sigma
>> y = A@x + sigma*noise

Note how it even autoscales to y.max(). (well really y.max() - y.min()).
Calling noise will return noise without autoscaling
>> y = x + noise() # no scaling to range of x!

Also note that noise represents a realization, not a random number
generator (thought it does that as necessary under the hood).
Thus we have:
>>> eta = noise(5)
>>> eps = 1*eta
>>> zz = np.ones(3)
>>> (zz + eta) - eps == zz
True

Everytime you call noise() or any instantiantions (eta()), you
get a new object with its own seed. To draw a new realization
from the same distribution, use ~
>>> eta = noise(5)
>>> eps = ~eta
>>> zz = np.ones(3)
>>> (zz + eta) - eps != zz
True

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""
import numbers

import numpy as np

from . import linop

_A = 1664525
_C = 1013904223
_MOD = 2 ** 32


def _lcg(seed, mod=_MOD, a=_A, c=_C):
    # a Linear congruential generator to generate
    # pseudo-random but fixed seeds. While not
    # necessary, this alleviates any
    # worry that sequential seeds may give
    # correlated output.
    for _ in range(_MOD):
        seed = (a * seed + c) % mod
        yield seed
    raise RuntimeError("No more unique seeds")


_seedgen = _lcg(9898)  # first million are at least unique


def get_seed():
    return next(_seedgen)


class BroadcastingNoise(object):
    """a noise function that broadcast to the size"""

    __array_ufunc__ = None  # we need this to override rmul, radd

    def __init__(
        self,
        scale=1,
        mean=0,
        dist=np.random.standard_normal,
        seed=9898,
        autoscale=False,
    ):
        super(BroadcastingNoise, self).__init__()
        if isinstance(scale, str):
            if "db" in scale.lower():
                scale = 10 ** (-float(scale.lower().replace("db", "").strip()) / 20)
                autoscale = True
            elif "%" in scale:
                scale = float(scale.replace("%", "").strip()) / 100
                autoscale = True
            else:
                scale = float(scale)
        self.scale = scale
        self.mean = mean
        self.dist = dist
        self.seed = seed
        self.autoscale = autoscale

    def realize(self, other=None, shape=None, dtype=None):
        scale = self.scale
        if self.autoscale:
            scale *= other.max() - other.min()
        if shape is None:
            shape = other.shape
        if dtype is None and other is not None:
            dtype = other.dtype
        np.random.seed(self.seed)
        return (self.mean + scale * self.dist(shape)).astype(dtype)

    def __call__(self, *args, **kwargs):
        seed = kwargs.pop("seed", get_seed())
        return BroadcastingNoise(*args, seed=seed, **kwargs)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            new_scale = self.scale * other
            new_mean = self.mean * other
            return BroadcastingNoise(
                scale=new_scale,
                dist=self.dist,
                seed=self.seed,
                mean=new_mean,
                autoscale=self.autoscale,
            )
        elif isinstance(other, linop.Arraylike):
            return other * self.realize(other)
        elif isinstance(other, BroadcastingNoise):
            if self.autoscale != other.autoscale:
                raise NotImplementedError
            else:

                def new_dist(shp):
                    return self.realize(shape=shp) * other.realize(shape=shp)

                # the seed is meaningless now...
                return BroadcastingNoise(
                    scale=1,
                    dist=new_dist,
                    seed=self.seed,
                    mean=0,
                    autoscale=self.autoscale,
                )
        else:
            raise NotImplementedError

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            new_mean = self.mean + other
            return BroadcastingNoise(
                scale=self.scale,
                dist=self.dist,
                seed=self.seed,
                mean=new_mean,
                autoscale=self.autoscale,
            )
        elif isinstance(other, linop.Arraylike):
            return other + self.realize(other)
        elif isinstance(other, BroadcastingNoise):
            if self.autoscale != other.autoscale:
                raise NotImplementedError
            else:

                def new_dist(shp):
                    return self.realize(shape=shp) + other.realize(shape=shp)

                # the seed is meaningless now...
                return BroadcastingNoise(
                    scale=1,
                    dist=new_dist,
                    seed=self.seed,
                    mean=0,
                    autoscale=self.autoscale,
                )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __rsub__(self, other):
        if isinstance(other, numbers.Number):
            new_mean = other - self.mean
            return BroadcastingNoise(
                scale=-self.scale,
                dist=self.dist,
                seed=self.seed,
                mean=new_mean,
                autoscale=self.autoscale,
            )
        elif isinstance(other, linop.Arraylike):
            return other - self.realize(other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            new_mean = self.mean - other
            return BroadcastingNoise(
                scale=self.scale,
                dist=self.dist,
                seed=self.seed,
                mean=new_mean,
                autoscale=self.autoscale,
            )
        elif isinstance(other, linop.Arraylike):
            return self.realize(other) - other
        elif isinstance(other, BroadcastingNoise):
            if self.autoscale != other.autoscale:
                raise NotImplementedError
            else:

                def new_dist(shp):
                    return self.realize(shape=shp) - other.realize(shape=shp)

                # the seed is meaningless now...
                return BroadcastingNoise(
                    scale=1,
                    dist=new_dist,
                    seed=self.seed,
                    mean=0,
                    autoscale=self.autoscale,
                )
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        """Act as a random vector when matrix multiplied"""
        if isinstance(other, linop.Arraylike):
            return other @ self.realize(other, shape=other.shape[-1])
        elif isinstance(other, linop.AbstractLinOp):
            return other @ self.realize(np.array([1.0, 0]), shape=other.shape[-1])
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        """Act as a random vector when matrix multiplied"""
        if isinstance(other, linop.Arraylike):
            return self.realize(other, shape=other.shape[0]) @ other
        elif isinstance(other, linop.AbstractLinOp):
            return self.realize(np.array([1.0, 0.0]), shape=other.shape[0]) @ other
        else:
            raise NotImplementedError

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return 1 * self

    def __abs__(self):
        return BroadcastingNoise(
            scale=1,
            dist=lambda shp: abs(self.realize(shape=shp)),
            autoscale=self.autoscale,
            seed=self.seed,
            mean=0,
        )

    def __pow__(self, num):
        return BroadcastingNoise(
            scale=1,
            dist=lambda shp: self.realize(shape=shp) ** num,
            autoscale=self.autoscale,
            seed=self.seed,
            mean=0,
        )

    def __invert__(self):
        # We are overloading this symbol as draw
        # from same distribution ~, i.e. get a new seed
        return BroadcastingNoise(
            scale=self.scale, dist=self.dist, autoscale=self.autoscale, mean=self.mean
        )


noise = BroadcastingNoise(autoscale=True)
