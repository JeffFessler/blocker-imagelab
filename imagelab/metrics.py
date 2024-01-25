"""imagelab/metrics.py

Tools associated with calculating quantitative metrics on an image.

"""
from abc import ABC as AbstractBaseClass
from abc import abstractmethod
import bz2
from collections import defaultdict
import contextlib
from datetime import datetime
import hashlib
import inspect
import itertools
import json
import pickle
import re
import signal
import socket
import sys
import time
import warnings

import _pickle as cPickle
import numpy as np

# from .utils import ignore_np_fp_err
# from skimage.measure import compare_ssim as ssim
import scipy.linalg
from scipy.linalg import norm
from scipy.ndimage.filters import gaussian_laplace as LoG
import scipy.stats
import toml
from tqdm.auto import tqdm

from . import patches, utils

try:
    import cupy
    import cupy.linalg as cupylinalg
except ImportError:
    cupy = None
    cupylinalg = None
import psutil

try:
    import pynvml
except ImportError:
    pass


def _hash_array(arr):
    if "cupy" in type(arr).__module__:
        arr = arr.get()
    return f"md5;{hashlib.md5(arr.reshape(-1).data).hexdigest()}"


def ssim(estim, truth, width=7, L=1.0, color=False):
    # https://en.wikipedia.org/wiki/Structural_similarity
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    if color:
        estim = np.moveaxis(estim, -1, 0)
        truth = np.moveaxis(truth, -1, 0)
    X = patches.im2win(estim, (width, width), stride=1, pad=None)
    Y = patches.im2win(truth, (width, width), stride=1, pad=None)
    muX = X.mean(axis=(0, 1))
    muY = Y.mean(axis=(0, 1))
    varX = ((X - muX) ** 2).mean(axis=(0, 1))
    varY = ((Y - muY) ** 2).mean(axis=(0, 1))
    covXY = ((X - muX) * (Y - muY).conj()).mean(axis=(0, 1))

    ssim = (
        (2 * muX * muY + c1)
        * (2 * covXY + c2)
        / ((muX ** 2 + muY ** 2 + c1) * (varX + varY + c2))
    )

    return float(ssim.mean())


def mse(estim, truth):
    """Calculates the Mean Square Error"""
    return float((abs((truth - estim)) ** 2).mean())


def mae(estim, truth):
    """Calculates the Mean Absolute Error"""
    return float((abs(truth - estim)).mean())


def maxerr(estim, truth):
    """Calculates the Max Error"""
    if "cupy" in type(estim).__module__:
        import cupy

        xp = cupy
    else:
        xp = np
    return float(xp.max(xp.abs(truth - estim)))


def rmse(estim, truth):
    """Calculates the Root Mean Square Error"""
    if "cupy" in type(estim).__module__:
        import cupy

        xp = cupy
    else:
        xp = np
    return float(xp.sqrt(mse(estim, truth)))


def nrmse(estim, truth):
    """Calculates the Normalized Root Mean Square Error"""
    if "cupy" in type(estim).__module__:
        import cupy.linalg

        xpnorm = cupy.linalg.norm
    else:
        xpnorm = norm
    return float(xpnorm((estim - truth).reshape(-1)) / xpnorm(truth.reshape(-1)))


def psnr(estim, truth):
    """Calculates the Peak-Signal-to-Noise Ratio in dB"""
    if "cupy" in type(estim).__module__:
        import cupy

        xp = cupy
    else:
        xp = np
    true_max = xp.abs(truth).max() ** 2
    return float(10 * xp.log10(true_max / mse(estim, truth)))


def snr(estim, truth):
    """Calculates the Peak-Signal-to-Noise Ratio in dB"""
    if "cupy" in type(estim).__module__:
        import cupy

        xp = cupy
    else:
        xp = np
    return float(20 * xp.log10(1 / nrmse(estim, truth)))


def hfpsnr(estim, truth):
    """Calculates the PSNR of a image high pass filtered
    with a Laplacian of Gaussian filter"""
    hf_estim = LoG(estim, 0.75)
    hf_truth = LoG(truth, 0.75)
    return psnr(hf_estim, hf_truth)


def sparsity(Z):
    return float(np.count_nonzero(Z) / Z.size)


class COLOR:
    GREEN = "\x1b[32m"
    RED = "\x1b[31m"
    YELLOW = "\x1b[33m"
    CYAN = "\x1b[36m"
    GRAY = "\x1b[37m"
    INVERT = "\x1b[7m"
    RESET = "\x1b[39m"
    RESET_BG = "\x1b[0m"


class Metric(AbstractBaseClass):
    """base Metric object"""

    requires_truth = True
    align = ">"
    dir = 0
    units = None
    convergence_tolerance = 1e-6
    convergence_iterations = 5

    def __init__(
        self,
        truth=None,
        preprocess=lambda x: x,
        name=None,
        color=True,
        backend=None,
        frozen=False,
    ):
        super(Metric, self).__init__()
        self.color = color
        self.truth = truth
        self.truth_chksum = None
        self.preprocess = preprocess
        self.name = type(self).__name__ if name is None else name
        if self.requires_truth and truth is None and not frozen:
            raise ValueError("please provide reference truth")
        elif truth is not None and np.issubdtype(truth.dtype, np.integer):
            # generally true, weird error can occur otherwise
            raise ValueError("Truth needs to be floating point")
        if truth is not None:
            truth.flags.writeable = False
            self.truth_chksum = _hash_array(truth)
        if backend is None:
            if "cupy" in type(truth).__module__:
                self.backend = cupy
                self.linalg = cupylinalg
            else:
                self.backend = np
                self.linalg = scipy.linalg

        self.log = []
        self.tab_width = max(len(self.name), len(str(self)))
        self.frozen = frozen
        if not frozen:
            self.init_cache()

    @abstractmethod
    def eval(self):
        raise NotImplementedError()

    @abstractmethod
    def format(self):
        raise NotImplementedError()

    def validate_truth(self):
        if self.truth_chksum is None:
            return
        if self.truth_chksum != _hash_array(self.truth):
            raise ValueError("the truth checksum does not match")

    def __repr__(self):
        return self._inline_str()

    def __str__(self):
        return self.format(self.last)

    def _colorize(self, data, idx=-1):
        """ Wraps `format` command in color, if enabled """
        if self.color:
            if self.dir == 0:
                return f"{COLOR.YELLOW}{data}{COLOR.RESET}"
            if (
                (len(self.log) < 2)
                or idx == 0
                or (self.dir * self.log[idx] >= self.dir * self.log[idx - 1])
            ):
                return f"{COLOR.GREEN}{data}{COLOR.RESET}"
            else:
                return f"{COLOR.RED}{data}{COLOR.RESET}"
        return data

    def _inline_str(self, idx=-1):
        data = self._colorize(self.format(self.log[idx]), idx)
        return f"{self.name} = {data}"

    def _table_str(self, idx=-1):
        data = self.format(self.log[idx])
        return self._colorize(f"{data:{self.align}{self.tab_width}s}", idx)

    def _table_hdr(self):
        return f"{self.name:^{self.tab_width}s}"

    def _identifier(self):
        class_name = type(self).__name__
        if class_name != self.name:
            return f"{class_name}({self.name})"
        else:
            return class_name

    def __call__(self, x):
        if self.frozen:
            raise ValueError("Metric is Frozen")
        x = self.preprocess(x)
        if self.truth is not None:
            x = x.reshape(self.truth.shape)
        val = self.eval(x)
        if isinstance(val, np.generic):
            val = val.item()
        self.log.append(val)
        return self

    def init_cache(self):
        pass

    def clear(self):
        self.log = []

    def __getitem__(self, index):
        return self.log[index]

    def __len__(self):
        return len(self.log)

    def __float__(self):
        return float(self.log[-1])

    @property
    def last(self):
        if self.log:
            return self.log[-1]
        else:
            return 0

    def callback_hook(self, x, *args, **kwargs):
        return self(x)

    def __lt__(self, other):
        return float(self).__lt__(float(other))

    def __le__(self, other):
        return float(self).__le__(float(other))

    def __gt__(self, other):
        return float(self).__gt__(float(other))

    def __ge__(self, other):
        return float(self).__ge__(float(other))

    def __eq__(self, other):
        return float(self).__eq__(float(other))

    def __ne__(self, other):
        return float(self).__ne__(float(other))

    def has_converged(self):
        """Converged when the average absolute difference from the
        average of the last `converge_iterations` iterations
        are less than `convergence_tolerance`.
        i.e. the Mean Absolute Deviation around the mean
        over the last few iterations."""
        if len(self.log) > self.convergence_iterations:
            recent = self.log[-self.convergence_iterations :]
            if abs(recent - recent.mean()).mean() < self.convergence_tolerance:
                return True
        return False

    def __getstate__(self):
        attributes = self.__dict__.copy()
        # pickle cupy arrays as numpy, in case cupy is not available
        cp_array_keys = []
        for key, val in attributes.items():
            if "cupy" in type(val).__module__:
                attributes[key] = val.get()
                cp_array_keys.append(key)
        attributes["__cp_array_keys"] = cp_array_keys
        del attributes["backend"]
        del attributes["linalg"]
        del attributes["preprocess"]
        for key, val in attributes.items():
            if not hasattr(val, "__getstate__") and not isinstance(
                val, (str, int, float, tuple, list, set, dict, bool)
            ):
                attributes[key] = f"Unserializable:{type(val)}"
        return attributes

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.frozen = True
        for key in state["__cp_array_keys"]:
            try:
                self.__dict__[key] = cupy.array(self.__dict__[key])
            except NameError:
                pass
        del self.__dict__["__cp_array_keys"]
        if "cupy" in type(self.truth).__module__:
            self.backend = cupy
            self.linalg = cupylinalg
        else:
            self.backend = np
            self.linalg = scipy.linalg
        self.preprocess = None

    def to_serializable(self):
        if callable(self.preprocess):
            preprocess_str = get_func_string(self.preprocess)
        elif isinstance(self.preprocess, str):
            preprocess_str = self.preprocess
        else:
            preprocess_str = None
        return {
            "name": self.name,
            "truth_chksum": self.truth_chksum,
            "log": self.log,
            "_type": type(self).__name__,
            "preprocess": preprocess_str,
        }

    @classmethod
    def from_serializable(cls, state):
        type_name = state["_type"]
        if type_name != cls.__name__:
            try:
                subcls = [
                    subcls
                    for subcls in all_subclasses(cls)
                    if subcls.__name__ == type_name
                ][0]
            except IndexError:
                raise Exception(f"Unrecognized Metric type {type_name}")
        else:
            subcls = cls
        obj = subcls(
            preprocess=state["preprocess"],
            name=state["name"],
            frozen=True,
        )
        del state["_type"]
        obj.__dict__.update(state)
        return obj


def all_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from all_subclasses(subclass)


def balance_paren(line):
    stack = []
    match = {"]": "[", ")": "(", "}": "{"}
    for ii, c in enumerate(line):
        if c in match.values():
            stack.append((ii, c))
        elif c in match.keys():
            try:
                jj, oc = stack.pop()
                if match[c] != oc:
                    return line[jj + 1 : ii]
            except IndexError:
                return line[:ii]
        elif c in ",;" and not stack:
            # end expression
            return line[:ii]

    if stack:
        jj = stack.pop()[0]
        return line[jj + 1 :]
    return line


def get_func_string(func):
    if not callable(func):
        return ValueError("Not a function")
    if func.__name__ == "<lambda>":
        func_str = str(inspect.getsourcelines(func)[0])
        func_str = "lambda" + balance_paren(
            func_str.strip("['\\n']").split("lambda")[1]
        )
        return func_str
    else:
        return f"{func.__module__}.{func.__name__}"


class PSNR(Metric):
    """ Peak Signal to Noise Ratio """

    dir = 1
    units = "dB"

    def format(self, val):
        return f"{val:6.3f}"[:6]

    def init_cache(self):
        self.true_max = self.backend.abs(self.truth).max() ** 2

    def eval(self, x):
        return float(10 * self.backend.log10(self.true_max / mse(x, self.truth)))


class SNR(Metric):
    """ Signal to Noise Ratio """

    dir = 1
    units = "dB"

    def format(self, val):
        return f"{val:6.3f}"[:6]

    def init_cache(self):
        self.truth = self.truth.reshape(-1)
        self.norm_truth = self.linalg.norm(self.truth)

    def eval(self, x):
        inv_nrmse = self.norm_truth / self.linalg.norm(x - self.truth)
        return float(20 * self.backend.log10(inv_nrmse))


class MSE(Metric):
    """ Mean Square Error """

    dir = -1

    def format(self, val):
        return f"{val:7.5f}"[:7]

    def eval(self, x):
        return mse(x, self.truth)


class MAE(Metric):
    """ Mean Absolute Error """

    dir = -1

    def format(self, val):
        return f"{val:7.5f}"[:7]

    def eval(self, x):
        return mae(x, self.truth)


class MaxErr(Metric):
    """ Max Absolute Error """

    dir = -1

    def format(self, val):
        return f"{val:7.5f}"[:7]

    def eval(self, x):
        return float(self.backend.max(self.backend.abs(self.truth - x)))


class RMSE(Metric):
    """ Root Mean Square Error """

    dir = -1

    def format(self, val):
        return f"{val:7.5f}"[:7]

    def eval(self, x):
        return rmse(x, self.truth)


class NRMSE(Metric):
    """ Normalized Root Mean Square Error """

    dir = -1

    def format(self, val):
        return f"{val:7.5f}"[:7]

    def init_cache(self):
        self.truth = self.truth.reshape(-1)
        self.norm_truth = self.linalg.norm(self.truth)

    def eval(self, x):
        return float(self.linalg.norm(x - self.truth) / self.norm_truth)


class DataErr(Metric):
    """ Data Error norm(Ax - y)**2"""

    dir = -1
    requires_truth = False

    def __init__(
        self,
        A,
        y,
        sigma2=0,
        **kwargs,
    ):
        super(DataErr, self).__init__(**kwargs)
        self.A = A  # Need to make sure this is vec_out
        self.y = y.reshape(A.shape[0], -1)
        self.sigma2 = 0
        self.extend = 1 + np.sqrt(2 / y.shape[0])

    def format(self, val):
        return f"{val:7.5f}"[:7]

    def eval(self, x):
        return mse(self.A @ x.reshape(self.A.shape[1], -1), self.y)

    def _colorize(self, data, idx=-1):
        """ Wraps `format` command in color, if enabled """
        if self.color:  # turn yellow if we are below the expected error
            if self.log[idx] < self.sigma2 * self.extend:
                return f"{COLOR.YELLOW}{data}{COLOR.RESET}"
            if (
                (len(self.log) < 2)
                or idx == 0
                or (self.dir * self.log[idx] >= self.dir * self.log[idx - 1])
            ):
                return f"{COLOR.GREEN}{data}{COLOR.RESET}"
            else:
                return f"{COLOR.RED}{data}{COLOR.RESET}"
        return data


class SSIM(Metric):
    """ Structural Similarity Index Metric """

    dir = 1

    def __init__(
        self,
        truth=None,
        width=7,
        L=1.0,
        color_img=None,
        **kwargs,
    ):
        k1 = 0.01
        k2 = 0.03
        self.c1 = (k1 * L) ** 2
        self.c2 = (k2 * L) ** 2
        self.width = width
        self.color_img = color_img
        super(SSIM, self).__init__(truth, **kwargs)

    def init_cache(self):
        if self.color_img is None:
            self.color_img = (self.truth.ndim > 2) and (self.truth.shape[-1] == 3)
        if self.color_img:
            self.truth = np.moveaxis(self.truth, -1, 0)
        self.Y = patches.im2win(
            self.truth, (self.width, self.width), stride=1, pad=None
        )
        self.muY = self.Y.mean(axis=(0, 1))
        self.varY = ((self.Y - self.muY) ** 2).mean(axis=(0, 1))

    def format(self, val):
        return f"{val:4.3f}"

    def eval(self, x):
        if self.color_img:
            x = self.backend.moveaxis(x, -1, 0)
        X = patches.im2win(x, (self.width, self.width), stride=1, pad=None)
        muX = X.mean(axis=(0, 1))
        varX = ((X - muX) ** 2).mean(axis=(0, 1))
        covXY = ((X - muX) * (self.Y - self.muY).conj()).mean(axis=(0, 1))

        ssim = (
            (2 * muX * self.muY + self.c1)
            * (2 * covXY + self.c2)
            / ((muX ** 2 + self.muY ** 2 + self.c1) * (varX + self.varY + self.c2))
        )

        return float(ssim.mean())


class Sparsity(Metric):
    dir = 0
    requires_truth = False
    units = "%"

    def format(self, val):
        return f"{val:7.2%}"

    def eval(self, x):
        return sparsity(x)


class Elapsed(Metric):
    dir = 0
    requires_truth = False
    units = "s"

    def format(self, val):
        val = int(np.round(val))
        return f"{val//60:2d}:{val%60:02d}"

    def eval(self, x):
        if self.log:
            return time.time() - self.start_time
        else:
            self.start_time = time.time()
            return 0.0


class Iteration(Metric):
    requires_truth = False
    units = "k"

    def eval(self, itr):
        if isinstance(itr, int):
            if len(self.log) > 0 and itr < self.log[-1]:
                raise ValueError(
                    "Iterations not consistent, current iter is "
                    + "{} but past was {}".format(itr, self.last)
                )
            return itr
        elif len(self.log) > 0:
            return self.log[-1] + 1
        else:
            return 0

    def format(self, val):
        return f"{val:3d}"

    def _colorize(self, data, idx=-1):
        if self.color:
            return f"{COLOR.CYAN}{data}{COLOR.RESET}"
        return data

    def _inline_str(self, idx=-1):
        data = self._colorize(self.format(self.log[idx]))
        return f"[{data}]"

    def callback_hook(self, x, itr=None, *args, **kwargs):
        return self(itr)


class Literal(Metric):
    """For when a function passes you an already computed value,
    or a function to compute that value"""

    requires_truth = False
    dir = -1

    def __init__(
        self,
        truth=None,
        func=lambda x: 0,
        format_str="{:6.5e}",
        kwarg="",
        preprocess=lambda x: x,
        name=None,
        **kwargs,
    ):
        self.format_str = format_str
        super(Literal, self).__init__(truth, preprocess, name, **kwargs)
        self.func = func
        self.kwarg = kwarg
        if name is None:
            self.name = kwarg

    def format(self, val):
        return self.format_str.format(val)

    def eval(self, x):
        return self.func(x)

    def callback_hook(self, x, *args, raw, **kwargs):
        func = kwargs.get(self.kwarg, None)
        if func is None:
            return self(raw)
        elif callable(func):
            self.func = func
            return self(raw)
        else:  # func was already computed
            self.log.append(func)
            return func


class Cost(Metric):
    requires_truth = False
    dir = -1

    def __init__(
        self, truth=None, cost_func=lambda x: 0, preprocess=lambda x: x, **kwargs
    ):
        super(Cost, self).__init__(truth, preprocess, **kwargs)
        self.eval = cost_func

    def format(self, val):
        return f"{val:6.5e}"

    def eval(self, x):
        raise NotImplementedError

    def callback_hook(self, x, itr=None, cost=None, *args, raw, **kwargs):
        if cost is None:
            return self(raw)
        elif callable(cost):
            self.eval = cost
            return self(raw)
        else:  # cost was already computed
            self.log.append(cost)
            return cost


class SystemMetric(Metric):
    requires_truth = False
    units = "%"

    def _colorize(self, data, idx=-1):
        """ Wraps `format` command in color, if enabled """
        val = self.log[idx]
        if self.color:
            if val > 70:
                if val > 90:
                    return f"{COLOR.RED}{data}{COLOR.RESET}"
                else:
                    return f"{COLOR.YELLOW}{data}{COLOR.RESET}"
            else:
                return f"{COLOR.GREEN}{data}{COLOR.RESET}"
        return data


class CPU(SystemMetric):
    def format(self, val):
        return f"{val:2.0f}%"

    def eval(self, x):
        # this returns avg cpu usage percent
        # since it was last called
        return psutil.cpu_percent()


class RAM(SystemMetric):
    def format(self, val):
        return f"{val:2.0f}%"

    def eval(self, x):
        return psutil.virtual_memory().percent


class GPU(SystemMetric):
    def __init__(self, gpu=0, *args, **kwargs):
        super(GPU, self).__init__(*args, **kwargs)

        self.gpu = gpu
        if not self.frozen:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)

    def format(self, val):
        return f"{val:3.0f}%"

    def eval(self, x):
        return pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu


class VRAM(SystemMetric):
    def __init__(self, gpu=0, *args, **kwargs):
        super(VRAM, self).__init__(*args, **kwargs)
        self.gpu = gpu
        if not self.frozen:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            self.total = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).total

    def format(self, val):
        return f"{val:2.0f}%"

    def eval(self, x):
        return 100 * pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / self.total


class MetricsList(object):
    """Metrics List"""

    def __init__(self, metrics, preprocess=lambda x: x, name="", metadata=None):
        self.metrics = metrics
        self.preprocess = preprocess

        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        self.metadata["DATE"] = str(datetime.now())
        self.metadata["host"] = socket.gethostname()
        self.metadata["argv"] = " ".join(sys.argv)
        self.name = name
        self.metadata["name"] = name

        self.sep = f" {COLOR.GRAY}|{COLOR.RESET} " if self.color else " | "

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.metrics[key]
        else:
            # give preference to user-defined names
            for metric in self.metrics:
                if key == metric.name:
                    return metric
            # if that fails use class names
            for metric in self.metrics:
                if key == type(metric).__name__:
                    return metric
        raise ValueError(f"Could not find metric {key}")

    def eval(self, estim, *args, **kwargs):
        """Evaluates the current estimate, `estim`, at each
        of the metrics specified in the object's instantiation"""

        estim_p = self.preprocess(estim)
        for metric in self.metrics:
            metric.callback_hook(estim_p, *args, **kwargs, raw=estim)

        return self

    def __call__(self, estim, *args, **kwargs):
        return self.eval(estim, *args, **kwargs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self._inline_str()

    def _inline_str(self, idx=-1):
        return self.sep.join([metric._inline_str(idx) for metric in self.metrics])

    def _table_hdr(self):
        titles = [metric._table_hdr() for metric in self.metrics]
        lines = ["-" * len(title) for title in titles]
        return f'{self.sep.join(titles)}\n{"-+-".join(lines)}'

    def _table_str(self, idx=-1):
        return self.sep.join([metric._table_str(idx) for metric in self.metrics])

    def table(self, step=10, mode="text"):
        res = [self._table_hdr()]
        res += [self._table_str(ii) for ii in range(0, len(self.metrics[0].log), step)]
        res = "\n".join(res)
        if mode in ["markdown", "md"]:
            from IPython.display import Markdown

            return Markdown(
                res.replace("+", "|")  # lazy right now
                .replace(COLOR.RED, "")
                .replace(COLOR.GREEN, "")
                .replace(COLOR.YELLOW, "")
                .replace(COLOR.CYAN, "")
                .replace(COLOR.GRAY, "")
                .replace(COLOR.RESET, "")
            )
        elif mode in ["text", "print"]:
            print(res)
        else:
            return res
        return

    def as_str(self, mode):
        if mode in ["inline"]:
            return self._inline_str()
        elif mode in ["table"]:
            return self._table_str()

    def get_display(
        self, niter=None, keep_every=10, mode="table", pbar=True, delay_interrupt=True
    ):
        try:
            from IPython.display import display
        except ImportError:

            def display(raw, *args, **kwargs):
                print(raw["text/plain"])

        hsh = np.random.rand()
        self.ih = None
        self.progbar = None

        def iter_func(x, itr=None, **kwargs):
            self(x, itr, **kwargs)
            itr = itr if itr is not None else len(self.metrics[0].log)
            if itr == 0:

                # Interrupt code begin
                if niter is not None and delay_interrupt:
                    old_handler = signal.getsignal(signal.SIGINT)

                    class int_handler(object):
                        def __init__(self):
                            self.signal_received = False

                        def handler(self, sig, frame, old_handler=old_handler):
                            if not self.signal_received:
                                self.signal_received = (sig, frame)
                                print("Interrupt Received...")
                            else:  # raise if received a second interrupt
                                signal.signal(signal.SIGINT, old_handler)
                                if old_handler:
                                    old_handler(*self.signal_received)

                    self.ih = int_handler()
                    self.ih.old_handler = old_handler
                    signal.signal(signal.SIGINT, self.ih.handler)
                # Interrupt code end

                if niter is not None and pbar:
                    self.progbar = tqdm(total=niter, leave=False)
                self.metadata["ipy"] = utils.get_ipython_input()[-1]
                if mode in "table":
                    print(self._table_hdr())
            else:
                if niter is not None and pbar:
                    if self.progbar is not None:
                        self.progbar.update(1)
                if itr == niter:
                    if self.progbar is not None:
                        self.progbar.close()
                        self.progbar = None
                    if self.ih is not None:
                        signal.signal(signal.SIGINT, self.ih.old_handler)
                        self.ih = None

            out = self.as_str(mode)
            display(
                {"text/plain": out},
                raw=True,
                display_id="{}iter{}".format(hsh, (itr - 1) // keep_every),
                update=(np.fmax(0, itr - 1) % keep_every) != 0,
            )

            # Interrupt code begin
            if self.ih is not None and self.ih.signal_received:
                print("Iteration Finished")
                signal.signal(signal.SIGINT, self.ih.old_handler)
                if delay_interrupt == "raise" and self.ih.old_handler:
                    self.ih.old_handler(*self.ih.signal_received)
                self.ih = None
                return True
            # Interrupt code end

            return False

        return iter_func

    def clear(self):
        for metric in self.metrics:
            metric.clear()

    @property
    def color(self):
        return any([metric.color for metric in self.metrics])

    def _set_color(self, val):
        for metric in self.metrics:
            metric.color = val

    @contextlib.contextmanager
    def no_color(self):
        colors = [metric.color for metric in self.metrics]
        self._set_color(False)
        yield
        for metric, color in zip(self.metrics, colors):
            metric.color = color

    def sanity_check(self):
        sizes = [
            np.prod(metric.truth.shape)
            for metric in self.metrics
            if metric.requires_truth
        ]
        if len(set(sizes)) > 1:
            raise ValueError("Not all metrics have the same size truth")

    def save(self, filename):
        if "." not in filename:  # only provided extension
            filename = self.name.replace(" ", "_") + filename
        if ".pkl" in filename.lower():
            with open(filename, "wb") as fh:
                pickle.dump(self, fh)
        elif ".pbz2" in filename.lower():
            with bz2.BZ2File(filename, "wb") as fh:
                cPickle.dump(self, fh)
        else:
            raise ValueError("Unknown file extension")

    @classmethod
    def load(cls, filename):
        if ".pkl" in filename.lower():
            with open(filename, "rb") as fh:
                return pickle.load(fh)
        elif ".pbz2" in filename.lower():
            with bz2.BZ2File(filename, "rb") as fh:
                return cPickle.load(fh)
        else:
            raise ValueError("Unknown file extension")

    def save_csv(self, filename):
        fh = open(filename, "w")
        if "tsv" in filename:
            delim = "\t"
        else:
            delim = ", "
        fh.write(delim.join([metric._identifier() for metric in self.metrics]))
        fh.write("\n")
        for results in itertools.zip_longest(
            *[metric.log for metric in self.metrics], fillvalue=0
        ):
            fh.write(delim.join(map(str, results)))
            fh.write("\n")

        fh.close()

    def load_csv(self, filename):
        if "tsv" in filename:
            delim = "\t"
        else:
            delim = ", "

        fh = open(filename, "r")
        hdr = fh.readline().strip().split(delim)
        for identifier, metric in zip(hdr, self.metrics):
            match = re.fullmatch(r"(\w+)\((\w+)\)", identifier)
            if match:
                class_name, name = match.group(1, 2)
                metric.name = name
            else:
                class_name = identifier
            if class_name != (true_type := type(metric).__name__):
                warnings.warn(
                    f"Warning: Loading column {class_name} into type {true_type}"
                )
        self.clear()
        for line in fh:
            cols = line.strip().split(delim)
            for metric, value in zip(self.metrics, cols):
                metric.log.append(eval(value))
        fh.close()

    def to_serializable(self):
        if callable(self.preprocess):
            preprocess_str = get_func_string(self.preprocess)
        elif isinstance(self.preprocess, str):
            preprocess_str = self.preprocess
        else:
            preprocess_str = None
        return {
            "metrics": [metric.to_serializable() for metric in self.metrics],
            "metadata": self.metadata,
            "name": self.name,
            "preprocess": preprocess_str,
            "_type": type(self).__name__,
        }

    @classmethod
    def from_serializable(cls, state):
        obj = cls([])
        obj.metrics = [Metric.from_serializable(metric) for metric in state["metrics"]]
        obj.metadata = state["metadata"]
        obj.name = state["name"]
        obj.preprocess = state["preprocess"]
        return obj

    @classmethod
    def default(cls, truth, color=True, **kwargs):
        try:
            gpu = truth.device.id
            gpu_mets = [GPU(gpu, color=color), VRAM(gpu, color=color)]
        except AttributeError:
            gpu_mets = []
        return cls(
            [
                Iteration(name="Iter", color=color),
                Elapsed(name="Time", color=color),
                PSNR(truth=truth, color=color),
                SSIM(truth=truth, color=color),
                NRMSE(truth=truth, color=color),
                CPU(color=color),
                RAM(color=color),
            ]
            + gpu_mets,
            **kwargs,
        )

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes["preprocess"]
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.preprocess = None


class Study(object):
    """A manage and collect metrics lists"""

    def __init__(self, name, truth=None, newlist=MetricsList.default):
        super(Study, self).__init__()

        self.name = name
        self.experiments = defaultdict(dict)

        if not isinstance(truth, dict):
            truth = {"default": truth}

        self.truth = truth
        self.newlist = newlist
        self.sa = None
        self.data_names = list(truth.keys())

    def add(self, name, data_name=None, truth=None, **kwargs):
        if data_name is None:
            data_name = self.data_names[0]
        if truth is None:
            truth = self.truth[data_name]
        self.experiments[name][data_name] = self.newlist(
            truth, name=f"{name}/{data_name}", **kwargs
        )
        return self.experiments[name][data_name]

    def __getitem__(self, key):
        if key not in self.experiments.keys():
            # DefaultDict won't do this for us
            raise KeyError(key)
        return self.experiments[key]

    def table(self, data_name=None, mode="text"):
        if data_name is None:
            data_name = self.data_names[0]
        # first remove collect Metric objects we want to show
        raw_lists = {}
        for key, val in self.experiments.items():
            raw_lists[key] = [met for met in val[data_name].metrics if met.dir != 0]
        examp = next(iter(raw_lists.values()))
        num = len(examp)  # number of metrics

        # for each column/Metric find which study had the best one
        # format the best one with color
        disp = {}
        for ii in range(num):
            best = examp[ii].dir * max(
                [examp[ii].dir * float(val[ii]) for val in raw_lists.values()]
            )
            for key, val in raw_lists.items():
                data = val[ii].format(val[ii].log[-1])
                data_str = f"{data:{val[ii].align}{val[ii].tab_width}s}"
                if ii == 0:
                    disp[key] = []
                if mode in ["markdown", "md"]:
                    disp[key].append(
                        f"  {data_str}  "
                        if float(val[ii]) != best
                        else f"**{data_str}**"
                    )
                else:
                    disp[key].append(
                        data_str
                        if float(val[ii]) != best
                        else val[ii]._colorize(data_str, 0)
                    )

        # put the table together
        name_width = max([4] + [len(key) for key in disp])
        headers = [f"{'Name':^{name_width}s}"] + [
            f"  {mets._table_hdr()}  "
            if mode in ["markdown", "md"]
            else mets._table_hdr()
            for mets in examp
        ]
        lines = ["-" * len(title) for title in headers]
        sep = " | " if mode in ["markdown", "md"] else "-+-"
        header = f'{" | ".join(headers)}\n{sep.join(lines)}'
        rows = [header]
        for key, val in disp.items():
            row = [f"{key:^{name_width}s}"] + val
            rows.append(f'{" | ".join(row)}')

        res = "\n".join(rows)
        if mode in ["markdown", "md"]:
            from IPython.display import Markdown

            return Markdown(res)
        elif mode in ["text", "print"]:
            print(res)
        else:
            return res

    def avg_plot(self, name, conf=0.95, ax=None, skip=()):
        # can't set x-axis currently
        from matplotlib import pyplot as plt

        if self.sa is None:
            self.sa = StyleArbiter()
        if ax is None:
            ax = plt.gca()
        methods = defaultdict(list)
        # First, we extract out all of the relevant logs
        for method_key, dataset_dict in self.experiments.items():
            if method_key not in skip:
                for _dataset_key, metrics in dataset_dict.items():
                    methods[method_key].append(metrics[name])
                    units = metrics[name].units
        # Then, we compute the sample mean and conf interval and plot for each algorithm/method
        for method_key in methods.keys():
            arr = np.array(methods[method_key])
            arr_mean = np.mean(arr, axis=0)
            arr_sem = scipy.stats.sem(arr, axis=0)
            arr_conf = arr_sem * scipy.stats.t.ppf((1 + conf) / 2.0, arr.shape[0] - 1)
            plt.plot(arr_mean, label=f"{method_key}", **self.sa[method_key])
            plt.fill_between(
                range(len(arr_mean)),
                arr_mean - arr_conf,
                arr_mean + arr_conf,
                color=self.sa[method_key]["color"],
                alpha=0.2,
            )
        ax.legend()
        if units is None:
            ax.set_ylabel(name)
        else:
            ax.set_ylabel(f"{name} ({units})")

        ax.set_title(f"{self.name} - Dataset Average, {conf:5.2%} CI")

    def dataset_table(self, name, skip=(), mode="print"):

        barsep = "-+-"
        if mode in ["md", "markdown"]:

            def emph(x):
                return f"**{x}**"

            barsep = "-|-"
        elif callable(mode):
            emph = mode
        else:

            def emph(x):
                return f"{COLOR.GREEN}{x}{COLOR.RESET}"

        methods = defaultdict(dict)
        avgs = {}
        # First, we extract out all of the relevant logs
        for method_key, dataset_dict in self.experiments.items():
            if method_key not in skip:
                avgs[method_key] = []
                for dataset_key, metrics in dataset_dict.items():
                    methods[method_key][dataset_key] = metrics[name]
                    avgs[method_key].append(metrics[name].last)
                avgs[method_key] = np.mean(avgs[method_key])
        # just a random Metric object for querying
        examp = methods[method_key][dataset_key]
        datakeys = sorted(list(methods[method_key].keys()))
        width = len(str(examp))
        name_width = max([len("Average")] + [len(key) for key in datakeys])

        method_hdrs = " | ".join(  # doesn't include possible large title width
            [emph(f"{name:^{name_width}s}")]
            + [f"{key:^{width}s}" for key in methods.keys()]
        )
        bar = barsep.join(["-" * name_width] + ["-" * width] * len(methods.keys()))

        # Create a line for each data point
        tab = []
        for dataset_key in datakeys:
            values = [methods[method_key][dataset_key] for method_key in methods.keys()]
            best = None
            if examp.dir:
                best = max([val.last * examp.dir for val in values]) * examp.dir

            line = [f"{dataset_key:<{name_width}s}"]
            for val in values:
                val_str = str(val) if val != best else emph(str(val))
                line += [val_str]
            tab.append(" | ".join(line))

        # Handle Average Line
        best = None
        if examp.dir:
            best = max([val * examp.dir for val in avgs.values()]) * examp.dir
        line = [f"{'Average':<{name_width}s}"]
        for method_key in methods.keys():
            val = avgs[method_key]
            val_str = examp.format(val) if val != best else emph(examp.format(val))
            line += [val_str]
        if mode not in ["md", "markdown"]:
            tab.append(bar)
        tab.append(" | ".join(line))

        # Join all of the table lines together
        result = "\n".join([method_hdrs, bar] + tab)
        if mode in ["text", "print"]:
            print(result)
        elif mode in ["md", "markdown"]:
            from IPython.display import Markdown

            return Markdown(result)
        else:
            return result

    def plot(self, name, xkey="Elapsed", data_name=None, ax=None, skip=()):
        # the ordering here assumes they all end
        # around the same time for comparison,
        # either in iteration or time (x)
        # might want to do something smarter? ask for xlim?
        from matplotlib import pyplot as plt

        if data_name is None:
            data_name = self.data_names[0]

        if self.sa is None:
            self.sa = StyleArbiter()
        if ax is None:
            ax = plt.gca()
        metrics = [
            (metriclist[data_name][name], metriclist[data_name][xkey], key)
            for key, metriclist in self.experiments.items()
            if key not in skip
        ]
        metrics.sort(reverse=True, key=lambda k: k[0])
        for metric, x, key in metrics:
            ax.plot(x.log, metric.log, label=f"{key}", **self.sa[key])
        ax.legend()
        if metrics[0][0].units is None:
            ax.set_ylabel(name)
        else:
            ax.set_ylabel(f"{name} ({metrics[0][0].units})")
        if metrics[0][1].units is None:
            ax.set_xlabel(xkey)
        else:
            ax.set_xlabel(f"{xkey} ({metrics[0][1].units})")
        ax.set_title(f"{self.name} - {data_name}")

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes["truth"]
        del attributes["newlist"]
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.truth = None
        self.newlist = None

    def save(self, filename):
        if "." not in filename:  # only provided extension
            filename = self.name.replace(" ", "_") + filename
        if ".pkl" in filename.lower():
            with open(filename, "wb") as fh:
                pickle.dump(self, fh)
        elif ".pbz2" in filename.lower():
            with bz2.BZ2File(filename, "wb") as fh:
                cPickle.dump(self, fh)
        elif ".json" in filename.lower():
            with open(filename, "w") as fh:
                json.dump(self.to_serializable(), fh, default=lambda o: o.__dict__())
        elif ".toml" in filename.lower():
            with open(filename, "w") as fh:
                toml.dump(self.to_serializable(), fh)
        else:
            raise ValueError("Unknown file extension")

    @classmethod
    def load(cls, filename):
        if ".pkl" in filename.lower():
            with open(filename, "rb") as fh:
                return pickle.load(fh)
        elif ".pbz2" in filename.lower():
            with bz2.BZ2File(filename, "rb") as fh:
                return cPickle.load(fh)
        elif ".json" in filename.lower():
            with open(filename, "r") as fh:
                return cls.from_serializable(json.load(fh))
        elif ".toml" in filename.lower():
            with open(filename, "r") as fh:
                return cls.from_serializable(toml.load(fh))
        else:
            raise ValueError("Unknown file extension")

    def to_serializable(self):
        state = {"name": self.name, "_type": type(self).__name__}
        state["experiments"] = defaultdict(dict)
        for alg_name, data_dict in self.experiments.items():
            for data_name, metriclist in data_dict.items():
                state["experiments"][alg_name][data_name] = metriclist.to_serializable()
        state["experiments"] = dict(state["experiments"])
        return state

    @classmethod
    def from_serializable(cls, state):
        obj = cls(state["name"])
        obj.experiments = state["experiments"]
        for alg_name, data_dict in obj.experiments.items():
            for data_name, metriclist in data_dict.items():
                obj.experiments[alg_name][data_name] = MetricsList.from_serializable(
                    metriclist
                )
        obj.data_names = list(data_dict.keys())
        return obj


# for when I eventually merge studies https://stackoverflow.com/q/3232943/5026175


class StyleArbiter(object):
    """docstring for StyleArbiter"""

    def __init__(self):
        super(StyleArbiter, self).__init__()
        self.index = {}
        self.styles = [
            {"color": "C0", "linestyle": (0, ())},
            {"color": "C1", "linestyle": (0, (5, 1))},
            {"color": "C2", "linestyle": (0, (6, 2, 2, 2))},
            {"color": "C3", "linestyle": (0, (1, 1))},
            {"color": "C4", "linestyle": (0, (5, 5))},
            {"color": "C5", "linestyle": (0, (3, 2, 8, 2, 3, 5))},
            {"color": "C6", "linestyle": (0, (12, 3))},
            {"color": "C7", "linestyle": (0, (8, 1, 1, 3))},
            {"color": "C8", "linestyle": (0, (3, 3, 1, 3))},
            {"color": "C9", "linestyle": (0, (14, 1, 2, 1))},
        ]

    def __getitem__(self, key):
        try:
            return self.index[key]
        except KeyError:
            style = self.allocate_style()
            self.index[key] = style
            return style

    def allocate_style(self):
        idx = len(self.index)
        return self.styles[idx]
