import functools
from inspect import signature

from ..utils import export
from .base import AbstractLinOp


@export
def ensure_vec_out(*argnames):
    """Returns a function decorator, which will require any LinOp or list
    of LinOps passed in as parameters `argnames` to have its `vec_out`
    attribute set to True. Previous value is then restored after function call.
    """

    def vec_out_decorator(func):
        @functools.wraps(func)
        def vec_out_wrapper(*args, **kwargs):
            # loop through provided arguments and set linops.vec_out=True
            arg_dict = signature(func).bind(*args, **kwargs).arguments
            linops = {}
            for argname in argnames:
                if argname in arg_dict:
                    arg = arg_dict[argname]
                    if isinstance(arg, AbstractLinOp):
                        linops[arg] = arg.vec_out
                        arg.vec_out = True
                    elif isinstance(arg, list) or isinstance(arg, tuple):
                        for a in arg:
                            if isinstance(a, AbstractLinOp):
                                linops[a] = a.vec_out
                                a.vec_out = True
            # Call orignal function
            res = func(*args, **kwargs)

            # reset to what it was
            for linop, old_vec in linops.items():
                linop.vec_out = old_vec

            return res

        return vec_out_wrapper

    return vec_out_decorator
