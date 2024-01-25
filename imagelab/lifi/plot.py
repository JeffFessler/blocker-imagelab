from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from . import extreme_views
from .. import plot as ilplot
from ..utils import export, isinstance_no_import


### Plotting
@export
def show_lf(lf, *, figsize=None, cmap="gray", constant_clim=True, labels=None):
    Vmax, Umax = lf.shape[0:2]
    cmax = lf.max()
    cmin = lf.min()
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(nrows=Vmax, ncols=Umax, wspace=0.05, hspace=0.05)
    for v in range(Vmax):
        for u in range(Umax):
            ax = fig.add_subplot(gs[v, u])
            im_art = ax.imshow(lf[v, u], cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if constant_clim:
                im_art.set_clim(cmin, cmax)
            if v == 0:
                ax.set_title(u)
            if u == 0:
                ax.set_ylabel(v)
    ax.set_xticks([0, lf.shape[3] - 1])
    ax.set_yticks([0, lf.shape[2] - 1])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    _label_lf(fig, ax, labels)
    return fig


def _label_lf(fig, ax, labels):
    if labels is not None:
        if labels is True:
            ax.set_xlabel("x")
            ax.set_ylabel("y", rotation=0)
            ilplot.sup_ylabel("v", rotation=0)
            fig.suptitle("u")
        else:
            ax.set_xlabel(labels[3])
            ax.set_ylabel(labels[2], rotation=-90)
            ilplot.sup_ylabel(labels[0])
            fig.suptitle(labels[1])


@export
def as_anaglyph(lf, u=(0, -1)):
    if lf.ndim < 5:
        lf = cm.gray(lf)
    lf_r = lf[:, u[0]].copy()
    lf_c = lf[:, u[1]].copy()
    lf_r[..., 1:] = 0
    lf_c[..., 0] = 0
    return lf_r + lf_c


@export
def animate_lf(lf, style="orbit", interval=None, **kwargs):
    if isinstance_no_import(lf, "torch.Tensor"):
        lf = lf.detach().cpu().numpy()
    if isinstance_no_import(lf, "cupy.core.core.ndarray"):
        lf = lf.get()
    if isinstance(lf, list):
        if lf[0].ndim == 5:  # color
            lf = np.concatenate(lf, axis=-2)
        elif lf[0].ndim == 4:
            lf = np.concatenate(lf, axis=-1)
    if style == "raster":
        lfvid = lf.reshape(lf.shape[0] * lf.shape[1], *lf.shape[2:])
    elif style == "snake":
        lf = lf.copy()  # don't change the user's array in the next step
        lf[1::2] = lf[1::2, ::-1]  # reverse every other row
        lfvid = lf.reshape(lf.shape[0] * lf.shape[1], *lf.shape[2:])  # raster new array
    elif style == "orbit":
        lfvid = np.concatenate(
            [lf[0, :], lf[1:, -1], lf[-1, -2::-1], lf[-2:0:-1, 0]], axis=0
        )
        assert (
            lfvid.shape[0] == lf.shape[0] * 2 + lf.shape[1] * 2 - 4
        ), "orbit vid length should equal number of border frames"
    elif style == "extremes":
        lfvid = extreme_views(lf)
    elif style == "anaglyph":
        lfvid = as_anaglyph(lf)
    else:
        raise ValueError("{} is not a valid lf animation style".format(style))

    if interval is None:
        interval = {"raster": 200, "snake": 100, "orbit": 40, "extremes": 700}.get(
            style, 200
        )

    return ilplot.play_vid(lfvid, interval=interval, **kwargs)
