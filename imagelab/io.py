"""imagelab/io.py

basic image loading and test images
"""

import os

from PIL import ExifTags, Image, ImageSequence
import matplotlib.image as mpimg
import numpy as np
import pooch
import scipy.io as sio

from . import config
from .utils import export


@export
def load(img_path, *args, **kwargs):
    if ".tif" in img_path.lower():
        return load_tiff(img_path)
    elif ".mat" in img_path.lower():
        return load_mat(img_path)
    # This is using Pillow under the hood
    return mpimg.imread(img_path, *args, **kwargs)


@export
def load_exif(img_path):
    img = Image.open(img_path)
    try:
        exif = {
            ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS
        }
        return exif
    except AttributeError:
        return {}


@export
def load_metadata(img_path):
    return Image.open(img_path).info


def load_tiff(img_path):
    """
    img_path - Path to the multipage-tiff file
    """
    img = Image.open(img_path)
    images = []
    for im in ImageSequence.Iterator(img):
        images.append(np.array(im))

    img_arr = np.array(images)
    img.close()
    try:
        return img_arr.squeeze(axis=0)
    except ValueError:
        return img_arr


def load_mat(mat_path):
    data = sio.loadmat(mat_path)
    data.pop("__header__", None)
    data.pop("__version__", None)
    data.pop("__globals__", None)
    keys = list(data.keys())
    if len(keys) == 1:
        return data[keys[0]]
    else:
        return data


@export
def load_csv(file_path, delim=",", skip_first_line=False):
    # TODO: make delimeter dynamically chosen?
    result = []
    with open(file_path) as fh:
        if skip_first_line:
            fh.readline()
        for line in fh:
            result.append([_convert_string(s) for s in line.strip().split(delim)])
    return result


@export
def load_tsv(file_path, *args, **kwargs):
    return load_csv(file_path, "\t", *args, **kwargs)


def _convert_string(strng):
    try:
        return int(strng)
    except ValueError:
        pass
    try:
        return float(strng)
    except ValueError:
        pass
    return strng


def save(img, img_path):
    raise NotImplementedError


# Common Test images
pup_common = pooch.create(
    path=os.path.join(config.DATABANK, "common"),
    base_url="",
    registry=None,
)
pup_common.load_registry(
    os.path.join(os.path.dirname(__file__), "io/data_registry.txt")
)

pup_kodak = pooch.create(
    path=os.path.join(config.DATABANK, "kodak"),
    base_url="",
    registry=None,
)
pup_kodak.load_registry(
    os.path.join(os.path.dirname(__file__), "io/kodak_registry.txt")
)

pup_usc = pooch.create(
    path=os.path.join(config.DATABANK, "usc_image_db"),
    base_url="",
    registry=None,
)
pup_usc.load_registry(os.path.join(os.path.dirname(__file__), "io/usc_registry.txt"))


@export
def goldhill():
    return load(pup_common.fetch("goldhill.png"))


@export
def monarch():
    return load(pup_common.fetch("monarch.png"))


@export
def cat():
    return load(pup_common.fetch("cat.png"))


@export
def sails():
    return load(pup_common.fetch("sails.png"))


@export
def tulips():
    return load(pup_common.fetch("tulips.png"))


@export
def mountain():
    return load(pup_common.fetch("mountain.png"))


@export
def barbara():
    return load(pup_common.fetch("barbara.jpg"))


@export
def kodak(num):
    return load(pup_kodak.fetch(f"kodim{num:02}.png"))


@export
def cameraman():
    return load(pup_common.fetch("cameraman.tif"))


def lena():  # no longer with us
    try:
        return load(pup_usc.fetch("4.2.04.tiff"))
    except ValueError:
        raise ValueError("This image is no longer provided by USC")


@export
def hubble():
    return load(pup_common.fetch("large_web.jpg"))


@export
def peppers():
    return load(pup_usc.fetch("4.2.07.tiff"))


@export
def mandrill():
    return load(pup_usc.fetch("4.2.03.tiff"))


@export
def house():
    return load(pup_usc.fetch("4.2.05.tiff"))


@export
def house2():
    return load(pup_usc.fetch("7.1.07.tiff"))


@export
def boat():
    return load(pup_usc.fetch("boat.512.tiff"))


def elaine():
    try:
        return load(pup_usc.fetch("elaine.512.tiff"))
    except ValueError:
        raise ValueError("This image is no longer provided by USC")
