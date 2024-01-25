import functools
import inspect
import logging
import re
import shutil
import signal
import subprocess
import sys

from deprecation import deprecated
import numpy as np
from scipy import linalg

from . import patches

try:
    import cupy
    import cupy.linalg as cupylinalg
except ImportError:
    cupy = None
    cupylinalg = None


def get_backend(x):
    if "cupy" in type(x).__module__:
        backend = cupy
    else:
        backend = np
    return backend


def get_backend_linalg(x):
    if "cupy" in type(x).__module__:
        backendla = cupylinalg
    else:
        backendla = linalg
    return backendla


def isinstance_no_import(obj, cls):
    # https://stackoverflow.com/questions/16964467/isinstance-without-importing-candidates
    return cls in [x.__module__ + "." + x.__name__ for x in inspect.getmro(type(obj))]


def export(fn):
    # https://stackoverflow.com/a/35710527
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        if fn.__name__ not in mod.__all__:
            mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def get_pipenv_root(depth=6):
    PROJ_ROOT = (
        subprocess.run(
            [shutil.which("pipenv"), "--where"],
            stdout=subprocess.PIPE,
            env={"PIPENV_MAX_DEPTH": str(depth)},
        )
        .stdout.decode("utf-8")
        .strip()
    )
    return PROJ_ROOT


def get_frontend():
    try:
        from __main__ import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return "jupyter"  # notebook, lab, or console?
        elif shell == "TerminalInteractiveShell":
            return "ipython"
        elif shell == "Shell":
            return "google.colab"
        else:  # Fall back on module in case names changed
            mod = get_ipython().__class__.__module__
            if mod == "google.colab._shell":
                return "google.colab"
            elif mod == "ipykernel.zmqshell":
                return "jupyter"  # or spyder?
            elif mod == "IPython.terminal.interactiveshell":
                return "ipython"
            else:
                raise AssertionError(
                    "Unknown ipython shell: shell: {}, module: {}".format(shell, mod)
                )
                # return 'unknown'
    except ImportError:
        return "terminal"


def get_ipython_input():
    try:
        from __main__ import get_ipython

        return get_ipython().user_global_ns["In"]
    except ImportError:
        return [None]


def in_notebook():
    return get_frontend() in ["jupyter", "google.colab"]


# https://gist.github.com/tcwalther/ae058c64d5d9078a9f333913718bba95
# class based on: http://stackoverflow.com/a/21919644/487556


class DelayedInterrupt(object):
    def __init__(self, signals):
        if not isinstance(signals, list) and not isinstance(signals, tuple):
            signals = [signals]
        self.sigs = signals

    def __enter__(self):
        self.signal_received = {}
        self.old_handlers = {}
        for sig in self.sigs:
            self.signal_received[sig] = False
            self.old_handlers[sig] = signal.getsignal(sig)

            def handler(s, frame, sig=sig):
                self.signal_received[sig] = (s, frame)
                # Note: in Python 3.5, you can use signal.Signals(sig).name
                logging.info(
                    "Signal {} received. Delaying KeyboardInterrupt.".format(sig)
                )

            self.old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)

    def __exit__(self, type, value, traceback):
        for sig in self.sigs:
            signal.signal(sig, self.old_handlers[sig])
            if self.signal_received[sig] and self.old_handlers[sig]:
                self.old_handlers[sig](*self.signal_received[sig])


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug("SIGINT received. Delaying KeyboardInterrupt.")

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received and self.old_handler:
            self.old_handler(*self.signal_received)


@deprecated(details="just use range instead")
def irange(start, stop=None, step=1, **kwargs):
    # Deprecated
    if stop is None:
        stop = start
        start = 0
    for ii in range(start, stop, step):
        yield ii


# derived from https://stackoverflow.com/a/15586020
class Reprinter:
    def __init__(self):
        self.text = ""

    def moveup(self, lines):
        for _ in range(lines):
            sys.stdout.write("\x1b[A")

    def reprint(self, *text, end="\n"):
        text = " ".join([str(t) for t in text]) + end
        # Clear previous text by overwritig non-spaces with spaces
        self.moveup(self.text.count("\n"))
        sys.stdout.write(re.sub(r"[^\s]", " ", self.text))

        # Print new text
        lines = min(self.text.count("\n"), text.count("\n"))
        self.moveup(lines)
        sys.stdout.write(text)
        self.text = text


@deprecated(details="Use patches module instead")
def im2win(A, patch_shape, stride=1, pad="constant", no_pad_axes=()):
    return patches.im2win(A, patch_shape, stride, pad, no_pad_axes)


@deprecated(details="Use patches module instead")
def im2col(A, patch_shape, stride=1, pad="constant", no_pad_axes=()):
    return patches.im2col(A, patch_shape, stride, pad, no_pad_axes)


@deprecated(details="Use patches module instead")
def col2im(
    patch, shape, patch_shape, stride=1, pad="constant", no_pad_axes=(), out=None
):
    return patches.col2im(patch, shape, patch_shape, stride, pad, no_pad_axes, out)


@deprecated(details="Use patches module instead")
def tcol2im(
    W, patch, shape, patch_shape, stride=1, pad="constant", no_pad_axes=(), out=None
):
    return patches.tcol2im(W, patch, shape, patch_shape, stride, pad, no_pad_axes, out)


@deprecated(details="Use patches module instead")
def im2col_weights(shape, *args, backend=np, **kwargs):
    return patches.im2col_weights(shape, *args, backend=backend, **kwargs)


def kdelta(shape, *args, **kwargs):
    """Kronecker Delta Function

    Returns an array of zeros with a one in the center
    """
    delta = np.zeros(shape, *args, **kwargs)
    # is this the center we want for even?
    cnter = tuple(shp // 2 for shp in shape)
    delta[cnter] = 1
    return delta


def kdelta_like(a, *args, **kwargs):
    """Kronecker Delta Function

    Returns an array of zeros with a one in the center
    that is the same shape and type as a given array
    """
    shape = a.shape
    delta = np.zeros_like(a, *args, **kwargs)
    cnter = tuple(shp // 2 for shp in shape)
    delta[cnter] = 1
    return delta


def ignore_np_fp_err(func):
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        old_set = np.seterr(all="ignore")
        res = func(*args, **kwargs)
        np.seterr(**old_set)
        return res

    return wrapper_func


def calling_scope():
    return inspect.stack()[2][0].f_locals


def print_versions(scope=None):
    if scope is None:
        scope = calling_scope()
    v = sys.version_info
    res = [f"python {v.major}.{v.minor}.{v.micro}"]
    printed = set()
    for mod in scope.values():
        try:
            if mod.__name__ not in printed:
                res.append(f"{mod.__name__} {mod.__version__}")
                printed.add(mod.__name__)
        except Exception:
            pass
    print(", ".join(res))
