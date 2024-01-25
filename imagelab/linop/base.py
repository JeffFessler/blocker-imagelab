"""imagelab/base.py
Contains some base classes that are needed through the package
"""
from abc import ABC as AbstractBaseClass
from abc import abstractmethod
import inspect
import numbers

import numpy as np
from scipy.linalg import det, inv, norm, pinv

from .. import config
from ..utils import export, isinstance_no_import

__all__ = ["I"]


@export
class AbstractLinOp(AbstractBaseClass):
    """Fake Matrix/ A linear operator with matrix notation
    A base class containing basic functionality of
    a system model so that it can act as matrix without
    actually being a matrix"""

    __array_ufunc__ = None  # we need this to override rmatmul?

    def __init__(
        self, in_shape=None, out_shape=None, vec_out=False, map_func=map, backend=np
    ):
        super(AbstractLinOp, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        if isinstance(self.in_shape, int):
            self.in_shape = (self.in_shape,)
        if isinstance(self.out_shape, int):
            self.out_shape = (self.out_shape,)
        self.vec_out = vec_out
        self.map = map_func
        self.backend = backend

        class Adjoint_View(AbstractLinOp):
            # should this be moved out into the module scope
            # and have self be passed in as an initializer?
            def __init__(self):
                pass  # stop normal LinOp initialization

            def forward_project(self, x):
                raise AssertionError("Not Possible")

            def back_project(self, y):
                raise AssertionError("Not Possible")

            def __matmul__(this, other):  # noqa: B902
                if isinstance(other, AbstractLinOp):
                    return CompositeLinOp([this, other])
                elif hasattr(other, "__iter__") and not isinstance(other, Arraylike):
                    return list(self.map(this.__matmul__, other))
                elif self.out_shape is not None:
                    other = other.reshape(self.out_shape)
                out = self.back_project(other)
                if self.vec_out:
                    out = out.reshape(-1)
                return out

            def __rmatmul__(this, other):  # noqa: B902
                if isinstance(other, AbstractLinOp):
                    raise AssertionError("Not Possible")
                    # return CompositeLinOp([other, this])
                elif hasattr(other, "__iter__") and not isinstance(other, Arraylike):
                    return list(self.map(this.__rmatmul__, other))
                elif self.in_shape is not None:
                    other = other.reshape(self.in_shape)
                out = self.forward_project(other.conj().T).conj().T
                if self.vec_out:
                    out = out.reshape(-1)
                return out

            def __call__(this, other):  # noqa: B902
                return self.back_project(other)

            @property
            def T(this):  # noqa: B902
                return self.conj()

            @property
            def H(this):  # noqa: B902
                return self

            def abs(this):  # noqa: B902
                return self.abs().H

            def inv(this):  # noqa: B902
                return self.inv().H

            def det(this):  # noqa: B902
                return self.det()

            def pow(this):  # noqa: B902
                return self.pow().H

            @property
            def shape(this):  # noqa: B902
                return tuple(reversed(self.shape))

            # these are properties instead of attributes
            # so they stay consistent with self
            in_shape = property(
                lambda this: self.out_shape,
                lambda this, new_shape: setattr(self, "out_shape", new_shape),
            )
            out_shape = property(
                lambda this: self.in_shape,
                lambda this, new_shape: setattr(self, "in_shape", new_shape),
            )
            vec_out = property(
                lambda this: self.vec_out,
                lambda this, new_vec: setattr(self, "vec_out", new_vec),
            )
            backend = property(
                lambda this: self.backend,
                lambda this, new_backend: setattr(self, "backend", new_backend),
            )

            __pow__ = pow

        self.adjoint = Adjoint_View()

    @abstractmethod
    def forward_project(self, x):
        raise NotImplementedError()

    @abstractmethod
    def back_project(self, y):
        raise NotImplementedError()

    def inv(self):
        raise NotImplementedError()

    def pinv(self):
        raise NotImplementedError()

    def abs(self):
        raise NotImplementedError()

    def det(self):
        raise NotImplementedError()

    def trace(self):
        raise NotImplementedError()

    def opnorm(self):
        raise NotImplementedError()

    def conj(self):
        return self

    def adj(self):
        return self.adjoint

    def __matmul__(self, other):
        if isinstance(other, AbstractLinOp):
            return CompositeLinOp([self, other])
        elif hasattr(other, "__iter__") and not isinstance(other, Arraylike):
            return list(self.map(self.__matmul__, other))
        elif self.in_shape is not None:
            other = other.reshape(self.in_shape)
        out = self.forward_project(other)
        if self.vec_out:
            out = out.reshape(-1)
        return out

    def __rmatmul__(self, other):
        assert not isinstance(
            other, AbstractLinOp
        ), "If `other` was LinOp we should not be here"
        if hasattr(other, "__iter__") and not isinstance(other, Arraylike):
            return list(self.map(self.__rmatmul__, other))
        elif self.out_shape is not None:
            other = other.reshape(self.out_shape)
        out = self.back_project(other.conj().T).conj().T
        if self.vec_out:
            out = out.reshape(-1)
        return out

    def guess_inv(self, other):
        try:
            return self.inv()(other)
        except NotImplementedError:
            pass
        try:
            return self.pinv()(other)
        except NotImplementedError:
            pass
        return self.adjoint(other)

    def __call__(self, other):
        return self.forward_project(other)

    def __add__(self, other):
        if isinstance(other, AbstractLinOp):
            from .stack import HStackLinOp

            return HStackLinOp([self, other])

    def __abs__(self):
        return self.abs()

    def __pos__(self):
        return self

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Scalar(other) @ self
        else:
            raise NotImplementedError

    __rmul__ = __mul__

    def __getitem__(self, indxs):
        if isinstance(indxs, slice):
            raise NotImplementedError
        idx0, idx1 = indxs
        ei = np.zeros(np.prod(self.in_shape))
        ei[idx1] = 1
        return (self @ ei)[idx0]

    def __array__(self):
        if np.prod(self.shape) > config.MAX_DENSE_SIZE:
            raise Exception(
                f"Attempted to create a dense matrix with {np.prod(self.shape)} "
                + "elements which exceeds config.MAX_DENSE_SIZE="
                + f"{config.MAX_DENSE_SIZE}. If you meant to do this, "
                + "either increase MAX_DENSE_SIZE or call full_matrix() explicitly."
            )
        return self.full_matrix()

    def full_matrix(self):
        old_vec = self.vec_out
        self.vec_out = True
        out = np.array(self @ IdentityBatch(self.shape[1])).T
        self.vec_out = old_vec
        return out

    def to_cupy(self):
        import cupy as cp

        self.backend = cp
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                self.__dict__[key] = cp.array(val)
            elif isinstance(val, AbstractLinOp) and val.backend != cp:
                val.to_cupy()
            elif isinstance(val, list):
                for ii, sval in enumerate(val):
                    if isinstance(sval, np.ndarray):
                        val[ii] = cp.array(sval)
                    if isinstance(sval, AbstractLinOp) and sval.backend != cp:
                        sval.to_cupy()

    @property
    def H(self):
        return self.adjoint

    @property
    def T(self):  # ??? do we want to handle real transpose really?
        # normal adjoint numpy code does A.conj().T which in this
        # case expands to A.conj().conj().adjoint. Keep conj cheap...
        return self.conj().adjoint

    @property
    def shape(self):
        if not self.in_shape or not self.out_shape:
            return None
        return (int(np.prod(self.out_shape)), int(np.prod(self.in_shape)))

    def pow(self, exponent):
        if exponent % 1 != 0:
            raise ValueError("Only integer values are supported")
        exponent = int(exponent)

        if exponent < -1:
            return self.__pow__(abs(exponent)).inv()
        elif exponent == -1:
            return self.inv()
        elif exponent == 0:
            return Identity()
        elif exponent == 1:
            return self
        elif exponent > 1:
            return CompositeLinOp([self] * exponent)

    __pow__ = pow


def IdentityBatch(N, copy=True):
    e_ii = np.zeros(N)
    e_ii[0] = 1.0
    if copy:
        yield e_ii.copy()
    else:
        yield e_ii
    for ii in range(1, N):
        e_ii[ii - 1] = 0
        e_ii[ii] = 1.0
        if copy:
            yield e_ii.copy()
        else:
            yield e_ii


@export
class CompositeLinOp(AbstractLinOp):
    """docstring for CompositeLinOp"""

    def __init__(self, op_list):
        super(CompositeLinOp, self).__init__()
        # prevent unnecessary compositions of compositions
        tmp_list = [
            [Op] if not isinstance(Op, CompositeLinOp) else Op.op_list for Op in op_list
        ]
        # flatten list of operations, remove identities to save the copies
        self.op_list = [
            item
            for sublist in tmp_list
            for item in sublist
            if not isinstance(item, Identity)
        ]
        # if op_list was a single element at this point,
        # it would be nice to just return the original LinOp...
        if not self.op_list:
            # ideally we'd return Identity() here, but...
            self.op_list.append(Identity())
        # ideally we would take in_shape from the right most,
        # but in case its None, backtrack. Same with out_shape
        in_shape = (
            [op.in_shape for op in self.op_list if op.in_shape is not None] or [None]
        )[-1]
        out_shape = (
            [op.out_shape for op in self.op_list if op.in_shape is not None] or [None]
        )[0]
        self.in_shape = in_shape
        self.out_shape = out_shape  # should we pass these to super?

        # how do we want to handle vec_out? should it be linked to last linop, or just
        # default to same as last linop?
        self.vec_out = self.op_list[0].vec_out

        # it would be nice to be able to reduce linops automagically, like if Q'Q=I

    def forward_project(self, x):
        y = x
        for Op in reversed(self.op_list):
            y = Op @ y
        return y

    def back_project(self, y):
        x = y
        for Op in self.op_list:
            x = Op.H @ x
        return x

    def guess_inv(self, y):
        x = y
        for Op in self.op_list:
            x = Op.guess_inv(x)
        return x

    def abs(self):
        return CompositeLinOp([Op.abs() for Op in self.op_list])

    def inv(self):
        return CompositeLinOp([Op.inv() for Op in reversed(self.op_list)])


@export
class Scalar(AbstractLinOp):
    def __init__(self, alph, *args, **kwargs):
        super(Scalar, self).__init__(*args, **kwargs)
        self.alph = alph

    def forward_project(self, x):
        return self.alph * x

    def back_project(self, y):
        return np.conj(self.alph) * y

    def inv(self):
        return Scalar(1 / self.alph)

    pinv = inv

    def abs(self):
        return Scalar(np.abs(self.alph))

    def opnorm(self):
        return np.abs(self.alph)

    def conj(self):
        return Scalar(np.conj(self.alph))


def _not_implemented(y):
    raise NotImplementedError


@export
def create_custom_linop(fproj, bproj=_not_implemented, *args, **kwargs):
    class CustomLinOp(AbstractLinOp):
        # def __init__(self, *args, **kwargs):
        #     super(CustomLinOp,self).__init__(*args, **kwargs)
        def forward_project(self, x):
            return fproj(x)

        def back_project(self, y):
            return bproj(y)

    return CustomLinOp(*args, **kwargs)


@export
class Identity(AbstractLinOp):
    """Identity System Model used in denoising
    i.e. y = x + n where n~N(0,sigma^2)"""

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__(*args, **kwargs)
        self.adjoint = self

    def forward_project(self, x):
        return x.copy()

    def back_project(self, y):
        return y.copy()

    def inv(self):
        return self

    def abs(self):
        return self

    def det(self):
        return 1.0

    def trace(self):
        shape = self.shape
        if shape is not None:
            return np.minimum(*shape)
        else:
            return None

    def opnorm(self):
        return 1.0


I = Identity()  # noqa: E741


@export
class NullIdentity(Identity):
    """Identity Operator without a copy
    i.e. a Null Operator.
    Since general LinOps involve operations,
    the return value is always not the input.
    This is why the Identity LinOp does a copy,
    to avoid an corner cases where new objects
    are expected.
    This LinOp is for foregoing that copy if
    you know you will not write to the input
    or output again.
    """

    def forward_project(self, x):
        return x

    def back_project(self, y):
        return y


@export
class Diagonal(AbstractLinOp):
    """docstring for Mask"""

    def __init__(self, diag_array):
        super(Diagonal, self).__init__(
            in_shape=diag_array.shape, out_shape=diag_array.shape
        )
        self.diag_array = diag_array

    def forward_project(self, x):
        return x * self.diag_array

    def back_project(self, y):
        return y * self.diag_array.conj()

    def abs(self):
        return Diagonal(np.abs(self.diag_array))

    def det(self):
        return np.prod(self.diag_array)

    def trace(self):
        return np.sum(self.diag_array)

    def inv(self):
        return Diagonal(1.0 / self.diag_array)

    def pinv(self):
        inv_array = self.diag_array.copy()
        inv_array[inv_array != 0] = 1.0 / inv_array[inv_array != 0]
        return Diagonal(inv_array)

    def opnorm(self):
        return np.max(np.abs(self.diag_array))


@export
class Matrix(AbstractLinOp):
    """A LinOp wrapper around a real matrix.
    This is mostly for checking AbstractLinOp functionality
    is consistent with matrices, but can also be useful to
    provide a more convenient adjoint for complex numpy arrays
    """

    def __init__(self, mtx):
        super(Matrix, self).__init__(
            in_shape=(mtx.shape[1],), out_shape=(mtx.shape[0],)
        )
        self.mtx = mtx

    def forward_project(self, x):
        return self.mtx @ x

    def back_project(self, y):
        return self.mtx.conj().T @ y

    def abs(self):
        return Matrix(np.abs(self.mtx))

    def det(self):
        return det(self.mtx)

    def trace(self):
        return np.trace(self.mtx)

    def inv(self):
        return Matrix(inv(self.mtx))

    def pinv(self):
        return Matrix(pinv(self.mtx))

    def conj(self):
        return Matrix(self.mtx.conj())

    def opnorm(self):
        return norm(self.mtx, 2)


@export
def aslinop(linop_like):
    if isinstance(linop_like, AbstractLinOp):
        return linop_like
    elif isinstance(linop_like, numbers.Number):
        return Scalar(linop_like)
    elif isinstance(linop_like, np.ndarray):
        if linop_like.ndim == 2:
            return Matrix(linop_like)
        elif linop_like.ndim == 1:
            return Diagonal(linop_like)
        else:
            raise ValueError("Multilinear operators of dimension > 2 not supported")
    if isinstance_no_import(
        linop_like, "scipy.sparse.linalg.LinearOperator"
    ):  # should cover PyLops aswell
        return linop_like  # ? lets cast it
    elif callable(linop_like):
        return create_custom_linop(linop_like)
    elif (
        hasattr(linop_like, "__len__")
        and len(linop_like) == 2
        and callable(linop_like[0])
        and callable(linop_like[1])
    ):
        return create_custom_linop(*linop_like)
    raise ValueError(f"Can't convert {type(linop_like)} to LinOp")


@export
def test_forward_adjoint_consistency(linOp, x=None, seed=None):
    rng = np.random.default_rng(seed)
    if x is None:
        if linOp.in_shape is not None:
            x = rng.standard_normal(linOp.in_shape) + 1j * rng.standard_normal(
                linOp.in_shape
            )
        else:
            x = rng.standard_normal(100) + 1j * rng.standard_normal(100)
    elif not isinstance(x, np.ndarray):
        x = rng.standard_normal(x) + 1j * rng.standard_normal(x)

    x = x.reshape(-1)
    old_vec = linOp.vec_out
    linOp.vec_out = True
    y = linOp @ x
    yp = rng.random(y.shape) + 1j * rng.random(y.shape)
    xp = linOp.H @ yp
    linOp.vec_out = old_vec
    return np.isclose(y.conj().T @ yp, x.conj().T @ xp)


@export  # should the name of this and AbstractLinOp be switched?
class LinearOperator(AbstractBaseClass):
    """LinearOperator is an abstract type/class
    that overrides `isinstance` to return true
    if an object implements matrix multiply
    shape, conj, and T"""

    def __new__(cls, *args, **kwargs):
        raise ValueError(
            "LinearOperator represents a union of types that "
            + "implement LinearOperator routines, consider AbstractLinOp instead"
        )
        print("Derive from AbstractLinOp, not LinearOperator!")
        instance = super(AbstractLinOp, cls).__new__(cls, *args, **kwargs)
        return instance

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is LinearOperator:
            if (
                any("__matmul__" in B.__dict__ for B in subclass.__mro__)
                and any("shape" in B.__dict__ for B in subclass.__mro__)
                and any("conj" in B.__dict__ for B in subclass.__mro__)
                and any("T" in B.__dict__ for B in subclass.__mro__)
            ):
                return True
        return NotImplemented


@export
class Arraylike(AbstractBaseClass):
    def __new__(cls, *args, **kwargs):
        raise ValueError(
            "Arraylike represents a union of types,"
            + " consider np.array or cp.array instead"
        )
        instance = super(Arraylike, cls).__new__(cls, *args, **kwargs)
        return instance

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is Arraylike:
            if (
                "cupy.core.core.ndarray"
                in [x.__module__ + "." + x.__name__ for x in inspect.getmro(subclass)]
                or subclass == np.ndarray
            ):
                return True
        return NotImplemented
