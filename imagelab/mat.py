"""imagelab/mat.py

Ports of MATLAB functions that are not directly provided
by numpy, but are useful when porting MATLAB code. Most
have a similar numpy method that performs most of the work.

In general, avoid using these methods over the similar
numpy methods.

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""

import numpy as np
from scipy.signal import convolve as _convolve


def circshift(A, shiftsize):
    """B = circshift(A,shiftsize)
    circularly shifts the values in the array, A, by shiftsize
    elements. shiftsize is a vector of integer scalars where
    the n-th element specifies the shift amount for the n-th
    dimension of array A. If an element in shiftsize is positive,
    the values of A are shifted down (or to the right). If it
    is negative, the values of A are shifted up (or to the left).
    If it is 0, the values in that dimension are not shifted.

    See np.roll

    Example
    ---
    >>> A = np.r_[1:10].reshape(3,3)
    >>> A
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> B = circshift(A, [1, -1])
    >>> B
    array([[8, 9, 7],
           [2, 3, 1],
           [5, 6, 4]])
    """
    return np.roll(A, shift=shiftsize, axis=np.r_[: len(shiftsize)])


def ndgrid(*args):
    """[X1,X2,X3,...] = ndgrid(x1,x2,x3,...)
    transforms the domain specified by vectors x1,x2,x3... into
    arrays X1,X2,X3... that can be used for the evaluation of
    functions of multiple variables and multidimensional interpolation.
    The ith dimension of the output array Xi are copies of elements of
    the vector xi.

    The ndgrid function is like meshgrid except that the order of the
    first two input arguments are switched. That is, the statement

    [X1,X2,X3] = ndgrid(x1,x2,x3)
    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)
    Because of this, ndgrid is better suited to multidimensional problems
    that aren't spatially based, while meshgrid is better suited to problems
    in two- or three-dimensional Cartesian space.

    If x1,x2,x3 are ranges, consider using np.mgrid as a replacement instead
    (or even better np.ogrid)

    Example
    ---
    >>> out1 = ndgrid(np.r_[0:3], np.r_[0:4], np.r_[0:1])
    >>> out2 = np.mgrid[0:3, 0:4, 0:1]
    >>> (out1 == out2).all()
    True
    """
    tmp0 = [args[0]]
    tmp1 = [args[1]]
    new_args = tmp1 + tmp0 + list(args[2:])
    outs = np.meshgrid(*new_args)
    tmp0 = [outs[0]]
    tmp1 = [outs[1]]
    new_outs = tmp1 + tmp0 + list(outs[2:])
    return new_outs


def _pad_to_odd(kernel):
    return np.pad(
        kernel,
        [(0, 1) if dim % 2 == 0 else (0, 0) for dim in kernel.shape],
        mode="constant",
    )


def conv(x, kernel, mode="full", *, method="auto"):
    """C = conv(A, B, shape) returns a subsection of the convolution with size
        specified by shape:
          'full'  - (default) returns the full convolution,
          'same'  - returns the central part of the convolution
                    that is the same size as A.
          'valid' - returns only those parts of the convolution
                    that are computed without the zero-padded edges.
                    len(C) is max(len(A)-max(0,len(B)-1),0).

    Note: 'same' is different than SciPy's signal.convolve when
    the kernel size is even.
    """
    if mode == "same":
        kernel = _pad_to_odd(kernel)
    return _convolve(x, kernel, mode=mode, method="method")
