from matplotlib import cm
import numpy as np


def infer_color_format(img):
    if img.ndim > 2 and not (np.abs(img.shape[-1] - img.shape[-2]) < 10):
        if img.shape[-1] == 3:
            return "rgb"
        elif img.shape[-1] == 4:
            return "rgba"
    return "gray"


def is_color(img):
    return infer_color_format(img)[:3] == "rgb"


def infer_peak_value(img):
    immax = img.max()
    if immax > 150 and immax <= 255.5:
        return 255.0
    elif immax > 0.5 and immax <= 1.01:
        return 1.0
    else:
        return immax


def alpha_blend(array1, array2, alpha):
    # If one is not color and the other is, promote to color
    if array1.ndim == array2.ndim - 1 and infer_color_format(array2)[:3] == "rgb":
        array1 = cm.gray(array1)[..., : array2.shape[-1]]
    elif array2.ndim == array1.ndim - 1 and infer_color_format(array1)[:3] == "rgb":
        array2 = cm.gray(array2)[..., : array1.shape[-1]]

    return alpha * array1 + (1 - alpha) * array2


def histogram(img, bins=256, color=None, pval=None):
    if color is None:
        color = infer_color_format(img)
    if pval is None:
        pval = infer_peak_value(img)

    if color[:3] == "rgb":
        hist = np.zeros((3, bins))
        hist[0] = histogram(img.T[0], bins, color="gray", pval=pval)
        hist[1] = histogram(img.T[1], bins, color="gray", pval=pval)
        hist[2] = histogram(img.T[2], bins, color="gray", pval=pval)
    elif color == "gray":
        hist = np.zeros((bins,))
        img = img.astype(float)
        img /= pval
        img *= bins - 1
        img = img.round().astype(int).ravel(order="K")
        for val in img:
            hist[val] += 1
    return hist


def show_histogram(img, ax=None, style="bar", pval=None, **kwargs):
    from matplotlib import pyplot as plt

    if pval is None:
        pval = infer_peak_value(img)

    hist = histogram(img, pval=pval, **kwargs)

    if ax is None:
        ax = plt.gca()
    bins = hist.shape[-1]
    bin_width = pval / (bins - 1)
    x_axis = np.r_[0:bins] * bin_width
    if style == "plot":
        if hist.ndim > 1:
            ax.plot(x_axis, hist[0], "r", alpha=0.2)
            ax.plot(x_axis, hist[1], "g", alpha=0.2)
            ax.plot(x_axis, hist[2], "b", alpha=0.2)
        else:
            ax.plot(x_axis, hist)
    elif style == "bar":
        if hist.ndim > 1:
            hist_mean = [
                (img.T[0] / bin_width).mean().round().astype(int),
                (img.T[1] / bin_width).mean().round().astype(int),
                (img.T[2] / bin_width).mean().round().astype(int),
            ]
            rbars = ax.bar(x_axis, hist[0], alpha=0.33, width=bin_width, color="r")
            gbars = ax.bar(x_axis, hist[1], alpha=0.33, width=bin_width, color="g")
            bbars = ax.bar(x_axis, hist[2], alpha=0.33, width=bin_width, color="b")
            rbars[hist_mean[0]].set_alpha(0.7)
            gbars[hist_mean[1]].set_alpha(0.7)
            bbars[hist_mean[2]].set_alpha(0.7)
        else:
            hist_mean = (img.mean() / bin_width).round().astype(int)
            bars = ax.bar(x_axis, hist, width=bin_width)
            bars[hist_mean].set_color("b")

    xticks = list(x_axis)
    while len(xticks) > 20:
        xticks = xticks[::2]
    if pval not in xticks:
        xticks.append(pval)
    ax.set_xlim([0, pval])
    if pval > 100:
        xticks = [np.round(x) for x in xticks]
    else:
        xticks = [np.round(x, decimals=1) for x in xticks]
    ax.set_xticks(xticks)


def show_imhist(img, figsize=None, cmap="gray"):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.1)
    ax[0].imshow(img, cmap=cmap)
    ax[0].set_xticks([0, img.shape[0] - 1])
    ax[0].set_yticks([0, img.shape[1] - 1])
    show_histogram(img, ax=ax[1])
    show_histogram(img, ax=ax[1], style="plot")
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
