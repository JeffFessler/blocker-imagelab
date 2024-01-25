# flake8: noqa
import os as _os

import pooch as _pooch
import toml as _toml

try:
    _cfg = _toml.load(_os.path.expanduser("~/.imagelab.toml"))
except:
    _cfg = {}

from .__version__ import __version__


class config:
    DATABANK = (
        _os.environ.pop("IL_DATABANK", None)
        or _os.path.expanduser(_cfg.pop("DATABANK", ""))
        or _pooch.os_cache("imagelab")
    )
    MAX_DENSE_SIZE = _os.environ.pop("IL_DENSE_SIZE", None) or _cfg.pop(
        "MAX_DENSE_SIZE", 200 * 200
    )


from . import (
    fonts,
    interp,
    io,
    linop,
    mat,
    metrics,
    misc,
    pix,
    sim,
    sparsity,
    tv,
    utils,
)
from .noise import noise
from .utils import in_notebook as _in_notebook

# from . import plot

if _in_notebook():
    # convenience access in notebooks
    from .plot import *
