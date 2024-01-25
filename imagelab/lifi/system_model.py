"""imagelab/lifi/system_model.py
Contains models which generate data from light-field inputs

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""

import numpy as np
from scipy.signal import convolve

from imagelab import linop
from imagelab.geom import AbstractGeometry
from imagelab.interp import c_support_trim
from imagelab.utils import kdelta_like


class CameraGeometry(AbstractGeometry):
    """A convenience class for collecting camera parameters
    that are often passed to functions together.
    Supports keyword unpacking into functions like dictionaries."""

    def __init__(self, *, nX, dX, nU, dU, f, F, nY=None, dY=None, nV=None, dV=None):
        super(CameraGeometry, self).__init__(
            [
                "nX",
                "nY",
                "dX",
                "dY",
                "nU",
                "nV",
                "dU",
                "dV",
                "f",
                "F",
            ],  # for keys() and **
            [("dV", "nV"), ("dU", "nU"), ("dY", "nY"), ("dX", "nX")],
        )  # for cti and [...]
        self.nX = nX
        self.nY = nY if nY is not None else nX
        self.dX = dX
        self.dY = dY if dY is not None else dX
        self.nU = nU
        self.nV = nV if nV is not None else nU
        self.dU = dU
        self.dV = dV if dV is not None else dU
        self.f = f
        self.F = -1 / (1 / f - 1 / F) if F > 0 else F
        self.units = "mm"

    def stc(self, *points):
        return tuple(-1 / (1 / self.f - 1 / p) if p > 0 else p for p in points)

    def cts(self, *points):
        return tuple(1 / (1 / self.f + 1 / p) if p < 0 else p for p in points)

    def shifts(self, *depths):
        """Number of pixels an object at given depths moves
        for one unit in the aperture plane"""
        return [
            -(self.dU / self.dX) * (1 - d / self.f + d / -self.F) / (d / -self.F)
            for d in depths
        ]

    def depths(self, *shifts):
        return [
            -self.F / (-s * (self.dX / self.dU) - 1 + -self.F / self.f) for s in shifts
        ]

    @property
    def sensor(self):
        return (self.nY * self.dY, self.nX * self.dX)

    @property
    def aperture(self):
        return (self.nV * self.dV, self.nU * self.dU)

    def sensor_diag(self):
        a, b = self.sensor
        return np.sqrt(a ** 2 + b ** 2)

    def crop_factor(self):
        if self.units == "mm":
            return 43.3 / self.sensor_diag()

    def resolution(self):
        return self.nX * self.nY / 1e6  # MegaPixels

    @property
    def fnum(self):
        """For a circular aperture this is the
        ratio of the focal length to the aperture
        diameter. For square apertures, it is the
        diameter of the circle with equal area.
        .. [HS80] M. R. Hatch, D. E. Stoltzmann,
            "The f-stops here", Optical Spectra, 1980, 88-91
        """
        A = self.aperture[0] * self.aperture[1]
        return self.f / np.sqrt(4 * A / np.pi)

    def __repr__(self):
        u = f"\u200B\x1b[97m{self.units}\x1b[0m"
        return f"""
Camera Geometry:
    Image sensor:
        Ny, Nx = {self.nY:d}, {self.nX:d}
        Δy, Δx = {self.dY:7.5f}{u}, {self.dX:7.5f}{u}
        Wy, Wx = {self.sensor[0]:4.2f}{u}, {self.sensor[1]:4.2f}{u}
        res    = {self.resolution():4.2f}\u200B\x1b[97mMP\x1b[0m
        diag   = {self.sensor_diag():4.2f}{u}
        crop   = {self.crop_factor():4.2f}
        F      = {self.F:6.3f}{u}
    Aperture:
        Nv, Nu = {self.nV:d}, {self.nU:d}
        Δv, Δu = {self.dV:4.2f}{u}, {self.dU:4.2f}{u}
        Wv, Wu = {self.aperture[0]:4.2f}{u}, {self.aperture[1]:4.2f}{u}
        f      = {self.f:5.3f}{u}
        f/#    = f/{self.fnum:3.1f}
        Ef. FF = {self.crop_factor()*self.f:5.3f}{u}, f/{self.crop_factor()*self.fnum:3.1f}
        range  = {self.depths(-1)[0]:5.2f}{u}, {self.depths(1)[0]:7.2f}{u}

"""

    @classmethod
    def iphone6(cls, nU, F):
        """https://www.devicespecifications.com/en/model/5d342ce2"""
        nX, nY = 3264, 2448
        wX, wY = 4.8, 3.6
        f = 4.15
        fnum = 2.2
        wU = (f / fnum) * (np.sqrt(np.pi) / 2)
        return cls(nX=nX, nY=nY, dX=wX / nX, dY=wY / nY, nU=nU, dU=wU / nU, f=f, F=F)

    @classmethod
    def iphoneX(cls, nU, F):
        """https://www.devicespecifications.com/en/model/36ea45ae"""
        nX, nY = 4032, 3024
        diag = 7.06
        wY = diag / np.sqrt(((nX / nY) ** 2 + 1))
        wX = (nX / nY) * wY
        f = 3.99
        fnum = 1.8
        wU = (f / fnum) * (np.sqrt(np.pi) / 2)
        return cls(nX=nX, nY=nY, dX=wX / nX, dY=wY / nY, nU=nU, dU=wU / nU, f=f, F=F)

    @classmethod
    def nikon_d600(cls, nU, F, fnum=2, f=50):
        """https://photographylife.com/reviews/nikon-d600"""
        wX, wY = 35.9, 24
        nX, nY = 6016, 4016
        wU = (f / fnum) * (np.sqrt(np.pi) / 2)
        return cls(nX=nX, nY=nY, dX=wX / nX, dY=wY / nY, nU=nU, dU=wU / nU, f=f, F=F)


class RefocusTransform(linop.AbstractLinOp):
    """docstring for FocalStackCamera"""

    def __init__(self, nX, nY, dX, dY, nU, nV, dU, dV, focal_length, Fstart, Fend):
        super(RefocusTransform, self).__init__(
            in_shape=(nV, nU, nY, nX), out_shape=(nV, nU, nY, nX)
        )
        self.nX = nX
        self.nY = nY
        self.dX = dX
        self.dY = dY
        self.nU = nU
        self.nV = nV
        self.dU = dU
        self.dV = dV
        self.focal_length = focal_length
        self.Fstart = Fstart
        self.Fend = Fend
        self._g = self._generate_g()
        # self.shape = (nU*nV*nX*nY, nU*nV*nX*nY)

    def _generate_g(self):
        x = self.dX * np.r_[-((self.nX - 1) // 10) : (self.nX - 1) // 10 + 1]
        y = self.dY * np.r_[-((self.nY - 1) // 10) : (self.nY - 1) // 10 + 1]
        u = self.dU * np.r_[-((self.nU - 1) // 2) : (self.nU - 1) // 2 + 1]
        v = self.dV * np.r_[-((self.nV - 1) // 2) : (self.nV - 1) // 2 + 1]
        # print(x)
        # simulate ndgrid with meshgrid
        [X, Y, U, V] = np.meshgrid(x, y, u, v)
        X = np.transpose(X, [1, 0, 2, 3])
        Y = np.transpose(Y, [1, 0, 2, 3])
        U = np.transpose(U, [1, 0, 2, 3])
        V = np.transpose(V, [1, 0, 2, 3])

        D = self.Fstart

        z = 1 / (1 / self.focal_length - 1 / self.Fend)
        F = 1 / (1 / z + 1 / D)

        b = D / F - D / self.focal_length

        ab = np.abs(b)
        # print(f'Fstart={self.Fstart}, Fend={self.Fend}, D={D}, z={z}, F={F}')
        if b == 0:

            def f(x, u, dX):
                return (
                    (np.fmin(dX / 2, x + dX / 2) - np.fmax(-dX / 2, x - dX / 2))
                    * (np.fmin(dX / 2, x + dX / 2) > np.fmax(-dX / 2, x - dX / 2))
                    * u
                )

        else:

            def f(x, u, dX):
                return (
                    dX * (u - x / b) - ab / 2 * np.sign(u - x / b) * (u - x / b) ** 2
                ) * (np.abs(u - x / b) <= dX / ab) + (
                    np.sign(u - x / b) * dX ** 2 / 2 / ab
                ) * (
                    np.abs(u - x / b) > dX / ab
                )

        st_h = f(X, U + self.dU / 2, self.dX) - f(X, U - self.dU / 2, self.dX)
        st_v = f(Y, V + self.dV / 2, self.dY) - f(Y, V - self.dV / 2, self.dY)

        g = st_h * st_v

        g = np.ascontiguousarray(g.T[:, :, ::-1, ::-1])
        g /= np.sum(g)
        g *= self.nU * self.nV
        g.flags.writeable = False
        return g

    def forward_project(self, lf):

        relf = np.zeros((self.nV, self.nU, self.nY, self.nX))
        for iV in range(self.nV):
            for iU in range(self.nU):
                # zero-padding conv
                relf[iV, iU] = convolve(
                    lf[iV, iU], self._g[-1 - iV, -1 - iU], mode="same", method="auto"
                )

        return relf

    def back_project(self, relf):

        lf = np.zeros((self.nV, self.nU, self.nY, self.nX))

        for iV in range(self.nU):
            for iU in range(self.nV):
                lf[iV, iU] = convolve(
                    relf[iV, iU],
                    self._g[-1 - iV, -1 - iU, ::-1, ::-1],
                    mode="same",
                    method="auto",
                )

        return lf

    def abs(self):
        return self


class ApertureSum(linop.AbstractLinOp):
    """docstring for ApertureSum"""

    def __init__(self, nV, nU, nY=-1, nX=-1):
        if not (nY == -1 and nX == -1):
            super(ApertureSum, self).__init__(
                in_shape=(nV, nU, nY, nX), out_shape=(nV, nU, nY, nX)
            )
        else:
            super(ApertureSum, self).__init__()

        self.nU = nU
        self.nV = nV

    def forward_project(self, x):
        return np.sum(x, axis=(0, 1))

    def back_project(self, y):
        return np.repeat(np.repeat(y[None], self.nU, axis=0)[None], self.nV, axis=0)

    def abs(self):
        return self


def ApertureProject(*args, **kwargs):
    return ApertureSum(*args, **kwargs).T


class FocalStackCamera(linop.AbstractLinOp):
    """docstring for FocalStackCamera"""

    def __init__(self, nX, nY, dX, dY, nU, nV, dU, dV, f, z, F=None):
        self.stack_distances = np.array(
            [1 / (1 / f - 1 / zi) if zi > 0 else np.abs(zi) for zi in z]
        )
        self._z = z
        self.nF = len(self.stack_distances)
        self.F = (
            (1 / (1 / f - 1 / F) if F > 0 else np.abs(F))
            if F is not None
            else max(self.stack_distances)
        )
        super(FocalStackCamera, self).__init__(
            in_shape=(nV, nU, nY, nX), out_shape=(self.nF, nY, nX)
        )
        self.nX = nX
        self.nY = nY
        self.dX = dX
        self.dY = dY
        self.nU = nU
        self.nV = nV
        self.dU = dU
        self.dV = dV
        self.focal_length = f
        self._g = self._generate_g()

    def _generate_g(self):
        x = self.dX * np.r_[-((self.nX - 1) // 2) : (self.nX - 1) // 2 + 1]
        y = self.dY * np.r_[-((self.nY - 1) // 2) : (self.nY - 1) // 2 + 1]
        u = self.dU * np.r_[-((self.nU - 1) // 2) : (self.nU - 1) // 2 + 1]
        v = self.dV * np.r_[-((self.nV - 1) // 2) : (self.nV - 1) // 2 + 1]
        # print(x)
        # simulate ndgrid with meshgrid
        [X, Y, U, V] = np.meshgrid(x, y, u, v)
        X = np.transpose(X, [1, 0, 2, 3])
        Y = np.transpose(Y, [1, 0, 2, 3])
        U = np.transpose(U, [1, 0, 2, 3])
        V = np.transpose(V, [1, 0, 2, 3])

        D = self.F

        g = np.zeros((len(x), len(y), self.nU, self.nV, self.nF))

        z = 1 / (1 / self.focal_length - 1 / self.stack_distances)
        for iF in range(self.nF):
            F = 1 / (1 / z[iF] + 1 / D)

            b = D / F - D / self.focal_length

            ab = np.abs(b)

            if b == 0:
                f = (
                    lambda x, u, dX: (
                        np.fmin(dX / 2, x + dX / 2) - np.fmax(-dX / 2, x - dX / 2)
                    )
                    * (np.fmin(dX / 2, x + dX / 2) > np.fmax(-dX / 2, x - dX / 2))
                    * u
                )
            else:

                def f(x, u, dX):
                    return (
                        dX * (u - x / b)
                        - ab / 2 * np.sign(u - x / b) * (u - x / b) ** 2
                    ) * (np.abs(u - x / b) <= dX / ab) + (
                        np.sign(u - x / b) * dX ** 2 / 2 / ab
                    ) * (
                        np.abs(u - x / b) > dX / ab
                    )

            st_h = f(X, U + self.dU / 2, self.dX) - f(X, U - self.dU / 2, self.dX)
            st_v = f(Y, V + self.dV / 2, self.dY) - f(Y, V - self.dV / 2, self.dY)

            g[:, :, :, :, iF] = st_h * st_v
        g = np.ascontiguousarray(
            c_support_trim(g.T[:, :, :, ::-1, ::-1], axes=(-1, -2))
        )
        g.flags.writeable = False
        return g

    def forward_project(self, lf):

        focalStack = np.zeros((self.nF, self.nY, self.nX), dtype=lf.dtype)

        for iF in range(self.nF):
            for iV in range(self.nV):
                for iU in range(self.nU):
                    # zero-padding conv
                    focalStack[iF] += convolve(
                        lf[iV, iU],
                        self._g[iF, -1 - iV, -1 - iU],
                        mode="same",
                        method="auto",
                    )

        return focalStack

    def back_project(self, focalStack):

        lf = np.zeros((self.nV, self.nU, self.nY, self.nX), dtype=focalStack.dtype)

        for iU in range(self.nU):
            for iV in range(self.nV):
                img = np.zeros((self.nY, self.nX), dtype=focalStack.dtype)
                for iF in range(self.nF):
                    # zero-padding conv.
                    img += convolve(
                        focalStack[iF],
                        self._g[iF, -1 - iV, -1 - iU, ::-1, ::-1],
                        mode="same",
                        method="auto",
                    )
                lf[iV, iU] = img
        return lf

    def abs(self):
        return self

    def hessian(self):
        dlta = np.zeros(self.in_shape)
        dlta[self.nV // 2, self.nU // 2, self.nY // 2, self.nX // 2] = 1
        hg = self.H @ self @ dlta

        dlta2 = np.zeros(self.in_shape)
        dlta2[0, 0, self.nY // 2, self.nX // 2] = 1
        ht = self.H @ self @ dlta2

        psf = np.zeros((2 * ht.shape[0] - 1, 2 * ht.shape[1] - 1, *ht.shape[2:]))
        psf[self.nV - 1 :, self.nU - 1 :, :, :] = ht[:, :, :, :]
        psf[: self.nV, self.nU - 1 :, :, :] = ht[::-1, :, ::-1, :]
        psf[self.nV - 1 :, : self.nU, :, :] = ht[:, ::-1, :, ::-1]
        psf[: self.nV, : self.nU, :, :] = ht[::-1, ::-1, ::-1, ::-1]
        psf[
            self.nV - 1 - self.nV // 2 : self.nV + self.nV // 2,
            self.nU - 1 - self.nU // 2 : self.nU + self.nU // 2,
        ] = hg
        psf = c_support_trim(psf, axes=(-2, -1))

        def forw(x):
            return convolve(x, psf, mode="same", method="auto")

        return linop.create_custom_linop(
            forw, forw, in_shape=self.in_shape, out_shape=self.in_shape
        )

    def hessian_old(self):
        # g = self._g
        # hg = np.zeros(g.shape)
        # for iF in range(self.nF):
        #     for iV in range(self.nV):
        #         for iU in range(self.nU):
        #             hg[iF, iV,iU] = convolve(g[iF, -1-iV, -1-iU], g[iF, -1-iV, -1-iU, ::-1, ::-1], mode='same', method='auto')
        # hg = convolve(hg, np.ones((1,2*self.nU+1, 2*self.nV+1,1,1)), mode='same', method='auto').sum(axis=0)
        dlta = np.zeros(self.in_shape)
        dlta[self.nV // 2, self.nU // 2, self.nY // 2, self.nX // 2] = 1
        ht = c_support_trim(self.H @ self @ dlta, axes=(-1, -2))

        def forw(x):
            return convolve(x, ht, mode="same", method="auto")

        return linop.create_custom_linop(
            forw, forw, in_shape=self.in_shape, out_shape=self.in_shape
        )

    def hessian2(self):
        dlta = np.zeros(self.in_shape)
        dlta[self.nV // 2, self.nU // 2, self.nY // 2, self.nX // 2] = 1
        hg = c_support_trim(self.H @ self @ dlta, axes=(-1, -2))

        def forw(x):
            y = np.zeros(x.shape)
            for iV in range(self.nV):
                for iU in range(self.nU):
                    y[iV, iU] = convolve(
                        x[iV, iU], hg[iV, iU], mode="same", method="auto"
                    )
            z = np.sum(y, axis=(0, 1))
            return np.repeat(np.repeat(z[None], self.nU, axis=0)[None], self.nV, axis=0)

        return linop.create_custom_linop(
            forw, forw, in_shape=self.in_shape, out_shape=self.in_shape
        )

    def circ_precond(self, beta):
        psf = (self.H @ self @ kdelta_like(self.in_shape)).reshape(self.in_shape)
        OTF = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psf))).real
        OTF = np.fmax(OTF, 0)
        filt = OTF + beta
        filt = np.fft.ifftshift(filt)

        def forw(x):
            return np.fft.ifftn(np.fft.fftn(x) / filt).real

        return linop.create_custom_linop(
            forw, forw, in_shape=self.in_shape, out_shape=self.in_shape
        )

    def circ_precond_pad(self, beta):
        dlta = np.zeros(self.in_shape)
        dlta[self.nV // 2, self.nU // 2, self.nY // 2, self.nX // 2] = 1
        hg = self.H @ self @ dlta

        dlta2 = np.zeros(self.in_shape)
        dlta2[0, 0, self.nY // 2, self.nX // 2] = 1
        ht = self.H @ self @ dlta2

        psf = np.zeros((2 * ht.shape[0] - 1, 2 * ht.shape[1] - 1, *ht.shape[2:]))
        psf[self.nV - 1 :, self.nU - 1 :, :, :] = ht[:, :, :, :]
        psf[: self.nV, self.nU - 1 :, :, :] = ht[::-1, :, ::-1, :]
        psf[self.nV - 1 :, : self.nU, :, :] = ht[:, ::-1, :, ::-1]
        psf[: self.nV, : self.nU, :, :] = ht[::-1, ::-1, ::-1, ::-1]
        psf[
            self.nV - 1 - self.nV // 2 : self.nV + self.nV // 2,
            self.nU - 1 - self.nU // 2 : self.nU + self.nU // 2,
        ] = hg

        OTF = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psf))).real
        OTF = np.fmax(OTF, 0)
        filt = OTF + beta
        filt = np.fft.ifftshift(filt)
        pad = (p - x for p, x in zip(psf.shape, self.in_shape))

        def forw(x):
            return np.fft.ifftn(np.fft.fftn(np.pad(x, pad)) / filt).real

        return linop.create_custom_linop(
            forw, forw, in_shape=self.in_shape, out_shape=self.in_shape
        )


def shiftView(image, shift):
    """Shift images using linear interpolation

    Parameters
    ----------
    I: image with 2 (gray) or 3 (color) dimensions
    shift: 2-tuple with shift in y and x

    from shiftView.m
    """
    # Diagram
    # -------
    #         |___a1___               |
    #        s1d      :              s1u
    #  --[s0d-o-----------------------o--
    #    [    |       :               |
    #  a0[    |       :               |
    #    [    |       :               |
    #    [ . .|. . . .* shift         |
    #         |                       |
    #         |                       |
    #         |                       |
    #  ---s0u-o-----------------------o--
    #         |                       |
    if shift[0] == 0 and shift[1] == 0:
        return image
    if "cupy" in type(image).__module__:
        import cupy

        xp = cupy
    else:
        xp = np

    if image.ndim == 3:
        no_color = False
    else:
        image = image[:, :, None]
        no_color = True
    n, m, c = image.shape

    a0 = shift[0] - np.floor(shift[0])
    a1 = shift[1] - np.floor(shift[1])

    s0d = int(np.floor(shift[0]))
    s0u = s0d + 1
    s1d = int(np.floor(shift[1]))
    s1u = s1d + 1

    # zero padding by max shift
    ms0 = max(np.abs([s0u, s0d]))
    ms1 = max(np.abs([s1u, s1d]))

    pI = xp.zeros((n + ms0 * 2, m + ms1 * 2, c), dtype=image.dtype)
    pI[ms0:-ms0, ms1:-ms1, :] = image

    I1d2d = pI[ms0 + s0d : ms0 + s0d + n, ms1 + s1d : ms1 + s1d + m, :]
    I1u2d = pI[ms0 + s0u : ms0 + s0u + n, ms1 + s1d : ms1 + s1d + m, :]
    I1d2u = pI[ms0 + s0d : ms0 + s0d + n, ms1 + s1u : ms1 + s1u + m, :]
    I1u2u = pI[ms0 + s0u : ms0 + s0u + n, ms1 + s1u : ms1 + s1u + m, :]

    nI = (
        I1d2d * (1 - a0) * (1 - a1)
        + I1d2u * (1 - a0) * (a1)
        + I1u2d * (a0) * (1 - a1)
        + I1u2u * (a0) * (a1)
    )
    if no_color:
        nI = nI[..., 0]
    return nI


def refocus(s, lf):

    relf = np.zeros(lf.shape)
    nV, nU = lf.shape[0:2]
    for iV in range((-nV + 1) // 2, (nV + 1) // 2):
        for iU in range((-nU + 1) // 2, (nU + 1) // 2):
            cV = iV - (-nV + 1) // 2
            cU = iU - (-nU + 1) // 2
            relf[cV, cU] = shiftView(
                lf[cV, cU],
                [s * iV, s * iU],
            )

    return relf


class ShiftSumFocalStackCamera(linop.AbstractLinOp):
    """docstring for FocalStackCamera"""

    def __init__(self, nX, nY, dX, dY, nU, nV, dU, dV, f, z, F=None, nC=None):
        self.stack_distances = np.array(
            [1 / (1 / f - 1 / zi) if zi > 0 else np.abs(zi) for zi in z]
        )
        self.depths = np.array([zi if zi > 0 else 1 / (1 / f + 1 / zi) for zi in z])
        self._z = z
        self.nF = len(self.stack_distances)
        self.F = (
            (1 / (1 / f - 1 / F) if F > 0 else np.abs(F))
            if F is not None
            else max(self.stack_distances)
        )
        if nC is None:
            super(ShiftSumFocalStackCamera, self).__init__(
                in_shape=(nV, nU, nY, nX), out_shape=(self.nF, nY, nX)
            )
        else:
            super(ShiftSumFocalStackCamera, self).__init__(
                in_shape=(nV, nU, nY, nX, nC), out_shape=(self.nF, nY, nX, nC)
            )

        self.nX = nX
        self.nY = nY
        self.dX = dX
        self.dY = dY
        self.nU = nU
        self.nV = nV
        self.dU = dU
        self.dV = dV
        self.nC = nC
        self.focal_length = f
        self.sU = [
            -(self.dU / self.dX) * (1 - d / f + d / self.F) / (d / self.F)
            for d in self.depths
        ]
        self.sV = [
            -(self.dV / self.dY) * (1 - d / f + d / self.F) / (d / self.F)
            for d in self.depths
        ]

    def forward_project(self, lf):

        if self.nC is None:
            focalStack = self.backend.zeros((self.nF, self.nY, self.nX), dtype=lf.dtype)
        else:
            focalStack = self.backend.zeros(
                (self.nF, self.nY, self.nX, self.nC), dtype=lf.dtype
            )

        for iF in range(self.nF):
            if self.nC is None:
                img = self.backend.zeros((self.nY, self.nX), dtype=lf.dtype)
            else:
                img = self.backend.zeros((self.nY, self.nX, self.nC), dtype=lf.dtype)
            for iV in range((-self.nV + 1) // 2, (self.nV + 1) // 2):
                for iU in range((-self.nU + 1) // 2, (self.nU + 1) // 2):
                    # zero-padding conv
                    img += shiftView(
                        lf[iV - (-self.nV + 1) // 2, iU - (-self.nU + 1) // 2],
                        [self.sV[iF] * iV, self.sU[iF] * iU],
                    )

            focalStack[iF] = img
        return focalStack

    def back_project(self, focalStack):

        if self.nC is None:
            lf = self.backend.zeros(
                (self.nV, self.nU, self.nY, self.nX), dtype=focalStack.dtype
            )
        else:
            lf = self.backend.zeros(
                (self.nV, self.nU, self.nY, self.nX, self.nC), dtype=focalStack.dtype
            )

        for iV in range((-self.nV + 1) // 2, (self.nV + 1) // 2):
            for iU in range((-self.nU + 1) // 2, (self.nU + 1) // 2):
                if self.nC is None:
                    img = self.backend.zeros((self.nY, self.nX), dtype=focalStack.dtype)
                else:
                    img = self.backend.zeros(
                        (self.nY, self.nX, self.nC), dtype=focalStack.dtype
                    )
                for iF in range(self.nF):
                    # zero-padding conv.
                    img += shiftView(
                        focalStack[iF], [-self.sV[iF] * iV, -self.sU[iF] * iU]
                    )
                lf[iV - (-self.nV + 1) // 2, iU - (-self.nU + 1) // 2] = img
        return lf

    def abs(self):
        return self

    def guess_inv(self, y):
        x = self.back_project(y)
        return x / (self.nV * self.nU * self.nF)

    @classmethod
    def init(cls, in_shape, s):
        """ init function when I don't know physical geometry """
        try:
            nC = in_shape[4]
        except IndexError:
            nC = 1
        A = cls(
            nX=in_shape[3],
            nY=in_shape[2],
            nU=in_shape[1],
            nV=in_shape[0],
            dX=1,
            dY=1,
            dU=1,
            dV=1,
            nC=nC,
            f=1,
            z=[2] * len(s),
        )
        A.sU = s
        A.sV = s
        return A


class CodedMaskCamera(linop.AbstractLinOp):
    """docstring for CodedMask"""

    def __init__(self, in_shape, s=-1, mask=1, seed=None):
        self.nU = in_shape[1]
        self.nV = in_shape[0]
        if isinstance(mask, int):
            mask = np.random.default_rng(seed).random((mask, *in_shape[2:4]))
        self.nM = mask.shape[0]
        super(CodedMaskCamera, self).__init__(
            in_shape=in_shape, out_shape=(self.nM, *in_shape[2:])
        )

        mask = np.repeat(
            np.repeat(mask[:, None], self.nU, axis=1)[:, None], self.nV, axis=1
        )
        self.mask = np.array([refocus(s, m) for m in mask])
        if self.mask.ndim == 5 and len(self.in_shape) == 5:  # input is color
            self.mask = np.repeat(self.mask[..., None], self.in_shape[-1], axis=-1)

    def forward_project(self, x):
        return np.sum(self.mask * x, axis=(1, 2))

    def back_project(self, y):
        return np.sum(
            self.mask
            * np.repeat(
                np.repeat(y[:, None], self.nU, axis=1)[:, None], self.nV, axis=1
            ),
            axis=0,
        )

    def abs(self):
        return self

    def guess_inv(self, y):
        x = np.repeat(np.repeat(y[:, None], self.nU, axis=1)[:, None], self.nV, axis=1)
        x /= self.mask + 0.5
        return np.sum(x, axis=0) / (self.nV * self.nU * self.nM)
