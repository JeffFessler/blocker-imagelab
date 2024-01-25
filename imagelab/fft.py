import scipy.fft as _fft

try:
    import cupy.fft
except ImportError:
    pass


def ifft2(x, **kwargs):
    """The convention in this 2D FFT is that the origin
    is in the center in space, and in the center in frequency.
    This transform in Unitary.
    """
    if "cupy" in type(x).__module__:
        fftmod = cupy.fft
    else:
        fftmod = _fft
    return fftmod.fftshift(
        fftmod.ifft2(fftmod.ifftshift(x, axes=(-1, -2)), norm="ortho", **kwargs),
        axes=(-1, -2),
    )


def fft2(x, **kwargs):
    """The convention in this 2D FFT is that the origin
    is in the center in space, and in the center in frequency.
    This transform in Unitary.
    """
    if "cupy" in type(x).__module__:
        fftmod = cupy.fft
    else:
        fftmod = _fft
    return fftmod.fftshift(
        fftmod.fft2(fftmod.ifftshift(x, axes=(-1, -2)), norm="ortho", **kwargs),
        axes=(-1, -2),
    )


def ifft1(x, **kwargs):
    if "cupy" in type(x).__module__:
        fftmod = cupy.fft
    else:
        fftmod = _fft
    return fftmod.fftshift(
        fftmod.ifft(fftmod.ifftshift(x, axes=-1), norm="ortho", **kwargs), axes=-1
    )


def fft1(x, **kwargs):
    if "cupy" in type(x).__module__:
        fftmod = cupy.fft
    else:
        fftmod = _fft
    return fftmod.fftshift(
        fftmod.fft(fftmod.ifftshift(x, axes=-1), norm="ortho", **kwargs), axes=-1
    )


def get_fft2_pair(norm="ortho", device="cpu", shift_x=True, shift_f=True):
    if device in ["gpu", "cuda"]:
        fftmod = cupy.fft
    else:
        fftmod = _fft

    if shift_x and shift_f:

        def ifft2(x):
            return fftmod.fftshift(
                fftmod.ifft2(fftmod.ifftshift(x, axes=(-1, -2)), norm=norm),
                axes=(-1, -2),
            )

        def fft2(x):
            return fftmod.fftshift(
                fftmod.fft2(fftmod.ifftshift(x, axes=(-1, -2)), norm=norm),
                axes=(-1, -2),
            )

    elif not shift_x and shift_f:

        def ifft2(x):
            return fftmod.ifft2(fftmod.ifftshift(x, axes=(-1, -2)), norm=norm)

        def fft2(x):
            return fftmod.fftshift(
                fftmod.fft2(x, norm=norm),
                axes=(-1, -2),
            )

    elif shift_x and not shift_f:

        def ifft2(x):
            return fftmod.fftshift(
                fftmod.ifft2(x, norm=norm),
                axes=(-1, -2),
            )

        def fft2(x):
            return fftmod.fft2(fftmod.ifftshift(x, axes=(-1, -2)), norm=norm)

    elif not shift_x and not shift_f:

        def ifft2(x):
            return fftmod.ifft2(x, norm=norm)

        def fft2(x):
            return fftmod.fft2(x, norm=norm)

    return fft2, ifft2
