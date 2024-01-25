from abc import ABC as AbstractBaseClass
from numbers import Number


class AbstractGeometry(AbstractBaseClass):
    """docstring for AbstractGeometry"""

    def __init__(self, keys, dXnX):
        super(AbstractGeometry, self).__init__()
        self._keys = keys
        self._dXnX = dXnX

    def keys(self):  # so we can use **
        return self._keys

    def __getitem__(self, key):
        if key in self.keys():  # so we can use **
            return getattr(self, key)
        if isinstance(key, tuple):
            return self.cti(*key)
        else:
            return self.cti(key)

    def oversample(self, factor):
        obj = dict(self)
        coords = self._dXnX
        if isinstance(factor, int):
            factor = [factor, factor]
        coords = reversed(coords)
        factor = reversed(factor)
        for s, (dx, nx) in zip(factor, coords):
            obj[nx] = obj[nx] * s
            obj[dx] = obj[dx] / s
        return type(self)(**obj)

    def subsample(self, factor):
        obj = dict(self)
        coords = self._dXnX
        if isinstance(factor, int):
            factor = [factor, factor]
        coords = reversed(coords)
        factor = reversed(factor)
        for s, (dx, nx) in zip(factor, coords):
            old = obj[nx]
            obj[nx] = obj[nx] // s
            obj[dx] = obj[dx] * (old / obj[nx])
        return type(self)(**obj)

    def _get_coord(self):
        return [(getattr(self, dX), getattr(self, nX)) for dX, nX in self._dXnX]

    def cti(self, *args):
        """ Continuous Coordinate to Index """
        dcnc = self._get_coord()
        try:
            ei = args.index(Ellipsis)
        except ValueError:
            ei = None
        res = []
        for arg in args[:ei]:
            res.append(self._cti(arg, *dcnc.pop(0)))
        bres = []
        if ei is not None:
            for arg in reversed(args[ei + 1 :]):
                bres.append(self._cti(arg, *dcnc.pop(-1)))
            bres.append(args[ei])
        return tuple(res + bres[::-1])

    def _cti(self, ci, di, ni):
        if isinstance(ci, Number):
            ci = round(ci / di)
            ctr = ni // 2
            return ctr + ci
        elif isinstance(ci, slice):
            start = self._cti(ci.start, di, ni)
            stop = self._cti(ci.stop, di, ni)
            step = self._cti(ci.step, di, ni)
            return slice(start, stop, step)
        else:
            return ci
