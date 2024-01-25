"""imagelab/plot.py
Basic image rendering convenience tools

"""
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import numpy as np

from . import color as _color
from . import metrics
from .utils import export, isinstance_no_import


@export
def show_im(
    image,
    *,
    figsize=None,
    axsize=None,
    cmap="gray",
    colorbar=False,
    clim=True,
    axes_pad=0.2,
    color="auto",
    cdim=-1,
    titles=(),
):
    """
    a convenience function to just show images without any fuss.

    Input
    -----
    images: images to show

    Returns
    -------
    art: artists for each image
    """
    # Todo: take *image as an argument instead
    #       1-N ticks ?
    if isinstance_no_import(image, "torch.Tensor"):
        image = image.detach().cpu().numpy()
    if isinstance_no_import(image, "cupy.core.core.ndarray"):
        image = image.get()
    if isinstance(image, list):
        image = np.array(image)
    if np.iscomplexobj(image):
        image = np.abs(image)
    if isinstance(figsize, int):
        figsize = (figsize, figsize)
    if isinstance(axsize, int):
        axsize = (axsize, axsize)
    is_color = (
        color
        and (image.ndim >= 3)
        and (image.shape[cdim] == 3 or image.shape[cdim] == 4)
    )
    if is_color and cdim != -1:
        image = np.moveaxis(image, cdim, -1)
    if color in ["linear", "lrgb"]:
        image = _color.lRGB_to_sRGB(image)
    if isinstance(clim, tuple) or isinstance(clim, list):
        a, b = clim[:2]
        cmin = min(a, b)
        cmax = max(a, b)
    else:
        cmax = image.max()
        cmin = image.min()

    im_dim = 2 if not is_color else 3
    while len(image.shape[:-im_dim]) < 2:
        image = np.array([image])

    cmode = None
    if colorbar:
        if clim:
            cmode = "single"
        else:
            cmode = "each"
            axes_pad = (max(axes_pad, 0.6), axes_pad)

    Vmax, Umax = image.shape[0:2]
    fig = plt.figure(figsize=figsize)

    if axsize is not None:
        size0 = Umax * axsize[0]
        size1 = Vmax * axsize[1]
        fig.set_size_inches(size0, size1)

    gs = ImageGrid(
        fig,
        111,
        nrows_ncols=(Vmax, Umax),
        direction="row",
        axes_pad=axes_pad,  # pad between axes in inches
        # add_all=True,  # deprecated
        share_all=False,
        label_mode="L",
        cbar_mode=cmode,
        cbar_location="right",
        cbar_pad=0.1,
        cbar_size="5%",
    )
    art = []
    for v in range(Vmax):
        for u in range(Umax):
            ii = v * Umax + u
            ax = gs[ii]
            im_art = ax.imshow(image[v, u], cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if clim:  # constant clim
                im_art.set_clim(cmin, cmax)
                fig.colorbar(im_art, cax=gs.cbar_axes[ii], ticks=[cmin, cmax])
            else:
                fig.colorbar(
                    im_art,
                    cax=gs.cbar_axes[ii],
                    ticks=[image[v, u].min(), image[v, u].max()],
                )
            art.append(im_art)
    gs[0].set_yticks([0, image.shape[2] - 1])
    gs[0].set_xticks([0, image.shape[3] - 1])

    zip_title(art, titles)

    if len(art) == 1:
        art = art[0]
        plt.sca(art.axes)
    return quiet_list(art) if isinstance(art, list) else art


class quiet_list(list):
    """a list who returns an empty string
    for its __repr__ method. This is useful
    for methods that need to return something
    but you don't want it displayed in Jupyter,
    such as plotting methods."""

    def __repr__(self):
        return ""


@export
def compare_im(
    truth, old, new, clim=True, titles=None, figsize=None, axsize=None, axes_pad=0.2
):

    mets_old = (metrics.psnr(old, truth), metrics.nrmse(old, truth))
    mets_new = (metrics.psnr(new, truth), metrics.nrmse(new, truth))
    if np.iscomplexobj(truth):
        truth = np.abs(truth)
        old = np.abs(old)
        new = np.abs(new)
    if isinstance(figsize, int):
        figsize = (figsize, figsize)
    if isinstance(axsize, int):
        axsize = (axsize, axsize)
    if titles is None:
        titles = ["Truth", "Old-Way", "Proposed"]
    if isinstance(clim, tuple) or isinstance(clim, list):
        a, b = clim[:2]
        cmin = min(a, b)
        cmax = max(a, b)
    else:
        cmax = max([truth.max(), old.max(), new.max()])
        cmin = min([truth.min(), old.min(), new.min()])
    image = (np.array([truth, old, new]) - cmin) / cmax

    diff_old = old - truth
    diff_new = new - truth
    cmax = max([diff_old.max(), diff_new.max()])
    cmin = min([diff_old.min(), diff_new.min()])

    fig = plt.figure(figsize=figsize)

    if axsize is not None:
        size0 = 3 * axsize[0]
        size1 = 2 * axsize[1]
        fig.set_size_inches(size0, size1)

    gs = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 3),
        direction="column",
        axes_pad=axes_pad,  # pad between axes in inches
        add_all=True,
        share_all=False,
        label_mode="L",
        cbar_mode="edge",
        cbar_location="right",
        cbar_pad=0.1,
        cbar_size="5%",
    )

    art0 = gs[0].imshow(image[0], cmap="gray")
    gs[1].axis("off")

    art1 = gs[2].imshow(image[1], cmap="gray")
    art4 = gs[3].imshow(diff_old, cmap="viridis")
    art4.set_clim(cmin, cmax)
    gs[3].set_xlabel(f"PSNR = {mets_old[0]:6.3f} dB        NRMSE = {mets_old[1]:6.3f}")

    art2 = gs[4].imshow(image[2], cmap="gray")
    art5 = gs[5].imshow(diff_new, cmap="viridis")
    art5.set_clim(cmin, cmax)
    gs[5].set_xlabel(f"PSNR = {mets_new[0]:6.3f} dB        NRMSE = {mets_new[1]:6.3f}")

    gs.cbar_axes[1].colorbar(art5, ticks=[cmin, 0, cmax])
    gs.cbar_axes[1].yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.2f")
    )
    gs.cbar_axes[1].yaxis.get_offset_text().set_size("large")
    gs.cbar_axes[0].axis("off")

    # Only places ticks on truth
    for ii in range(1, 6):
        gs[ii].set_yticks([])
        gs[ii].set_xticks([])
    gs[0].set_yticks([0, image[0].shape[0] - 1])
    gs[0].set_xticks([0, image[0].shape[1] - 1])
    gs[0].tick_params(
        axis="both",
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
        labelsize="large",
    )
    arts = [art0, art1, art2, art4, art5]
    zip_title(arts, titles)

    return quiet_list(arts)


@export
def add_colorbar(mappable=None):
    # https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes
    if mappable is None:
        mappable = plt.gca().images[0]
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig.colorbar(mappable, cax=cax)
    plt.sca(ax)


@export
def zip_title(arts, titles):
    for ii, art in enumerate(arts[: len(titles)]):
        art.axes.set_title(titles[ii])


@export
def sup_xlabel(xlab):
    plt.gcf().text(0.5, 0.04, xlab, ha="center")


@export
def sup_ylabel(ylab, rotation="vertical"):
    plt.gcf().text(0.04, 0.5, ylab, va="center", rotation=rotation)


@export
def play_vid(
    vid,
    *,
    figsize=None,
    cmap="gray",
    interval=20,
    clim=None,
    color="auto",
    colorbar=False,
):
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance_no_import(vid, "cupy.core.core.ndarray"):
        vid = vid.get()
    if clim is None:
        cmax = vid.max()
        cmin = vid.min()
    else:
        a, b = clim[:2]
        cmin = min(a, b)
        cmax = max(a, b)
    if color in ["linear", "lrgb"]:
        vid = _color.lRGB_to_sRGB(vid)
    ax.set_axis_off()
    avid = ax.imshow(vid[0], cmap=cmap)
    avid.set_clim(cmin, cmax)
    if colorbar:
        add_colorbar()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.close()

    def animate(i):
        avid.set_data(vid[i])
        return (avid,)

    anim = animation.FuncAnimation(
        fig, animate, frames=vid.shape[0], interval=interval, blit=True
    )
    matplotlib.rc("animation", html="jshtml")  # This prob shouldn't go here
    return anim


# Convenience Methods


def set_black_mpl_theme():
    color = "white"
    matplotlib.rcParams.update(
        {
            "text.color": color,
            "axes.labelcolor": color,
            "axes.edgecolor": color,
            "xtick.color": color,
            "ytick.color": color,
            "axes.facecolor": "black",
        }
    )


def set_default_mpl_theme():
    color = "black"
    matplotlib.rcParams.update(
        {
            "text.color": color,
            "axes.labelcolor": color,
            "axes.edgecolor": color,
            "xtick.color": color,
            "ytick.color": color,
            "axes.facecolor": "white",
        }
    )


def joseph_long_theme():
    # taken from https://joseph-long.com/writing/colorbars/
    # or the cache (https://webcache.googleusercontent.com/search?q=cache:DnN1i0CZbPIJ:https://joseph-long.com/writing/colorbars/+&cd=1&hl=en&ct=clnk&gl=us) # noqa: E501
    # inspired by https://nipunbatra.github.io/blog/2014/latexify.html
    params = {
        "text.latex.preamble": ["\\usepackage{gensymb}"],
        "image.origin": "lower",
        "image.interpolation": "nearest",
        "image.cmap": "gray",
        "axes.grid": False,
        "savefig.dpi": 150,  # to adjust notebook inline plot size
        "axes.labelsize": 8,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 8,
        "font.size": 8,  # was 10
        "legend.fontsize": 6,  # was 10
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": True,
        "figure.figsize": [3.39, 2.10],
        "font.family": "serif",
    }
    matplotlib.rcParams.update(params)


def enable_retina():
    from IPython.display import set_matplotlib_formats

    set_matplotlib_formats("retina")


# I'm putting this here cause I always forget it
# left = 0.125 # the left side of the subplots of the figure
# right = 0.9  # the right side of the subplots of the figure
# bottom = 0.1 # the bottom of the subplots of the figure
# top = 0.9    # the top of the subplots of the figure
# wspace = 0.2 # the amount of width reserved for blank space between subplots,
#              # expressed as a fraction of the average axis width
# hspace = 0.2 # the amount of height reserved for white space between
#              # subplots, expressed as a fraction of the average axis height
#
# The margins are sort of like CSS-style margins, only relative to the bottom
# left corner. In other words, right=.99 means that the right margin is 1%
# away from the right edge.
