from glob import glob
import os

import numpy as np

from .. import config
from ..io import load as imread

# import pooch


def get_lf_directories():
    dirs = glob(
        os.path.join(config.DATABANK, "lightfield/**/config.toml"), recursive=True
    )
    res = {}
    for d in dirs:
        dlist = d.split(os.sep)
        res[dlist[-2]] = os.sep.join(dlist[:-1])
    return res


def _expand_wildcard(filepath):
    return glob(filepath)[0]


def load(lf_directory, ext, v=None, u=None, prefix=""):
    # find the image names
    fnames = glob(os.path.join(lf_directory, f"{prefix}*{ext}"))
    if len(fnames) == 0:
        raise ValueError(f"No images found matching {prefix}*{ext} at {lf_directory}")
    fnames.sort()

    # load all of the image into memory
    lf = []
    for filename in fnames:
        # print(filename)
        lf.append(imread(filename))
    lf = np.array(lf)

    # determine proper v,u reshape
    num = lf.shape[0]
    if v is None and u is None:
        u = int(np.sqrt(num))
        v = u
    elif v is not None:
        u = num // v
    else:
        v = num // u
    lf = lf.reshape([v, u, *lf.shape[1:]])
    return lf


def stanford_lf(name):
    """http://lightfield.stanford.edu/lfs.html"""
    name = name.lower()
    if name in ["jellybeans", "jelly_beans"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/jelly_beans"), ".png"
        )[::-1]
    elif name in ["lego_truck", "truck"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/lego_truck"), ".png"
        )[::-1]
    elif name in ["lego_bulldozer", "bulldozer"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/lego_bulldozer"),
            ".png",
        )[::-1]
    elif name in ["lego_knights", "knights"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/lego_knights"), ".png"
        )[::-1, ::-1]
    elif name in ["stanford_bunny", "bunny"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/stanford_bunny"),
            ".png",
        )[::-1]
    elif name in ["bracelet"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/bracelet"), ".png"
        )[::-1, ::-1]
    elif name in ["chess"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/chess"), ".png"
        )[::-1]
    elif name in ["amethyst", "crystal", "rock"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/amethyst"), ".png"
        )[::-1]
    elif name in ["treasure_chest", "treasure"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/treasure_chest"),
            ".png",
        )[::-1]
    elif name in ["eucalyptus", "flower"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/eucalyptus"), ".png"
        )[::-1]
    elif name in ["crystal_ball", "tarot"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/stanford_lf/crystal_ball_small"),
            ".png",
        )[::-1, ::-1]
    else:
        raise ValueError("Unrecognized light field identifier")


def mit_lf(name):
    """http://web.media.mit.edu/~gordonw/SyntheticLightFields/index.php"""
    name = name.lower()
    if name in ["dice"]:
        return load(
            os.path.join(
                config.DATABANK, "lightfield/mit_synthetic_lf/dice_camera/dice_5x5_ap50"
            ),
            ".png",
        )[::-1]
    elif name in ["dragonandbunnies", "dragon"]:
        return load(
            os.path.join(
                config.DATABANK,
                "lightfield/mit_synthetic_lf/DragonAndBunnies/DragonsAndBunnies_5x5_ap6.6",
            ),
            ".png",
        )[::-1]
    elif name in ["fish"]:
        return load(
            os.path.join(
                config.DATABANK, "lightfield/mit_synthetic_lf/fishi/fishi_5x5_ap50"
            ),
            ".png",
        )[::-1]
    elif name in ["messerschmitt", "car"]:
        return load(
            os.path.join(
                config.DATABANK,
                "lightfield/mit_synthetic_lf/messerschmitt_camera/messerschmitt_5x5_ap50",
            ),
            ".png",
        )[::-1]
    elif name in ["shrubbery"]:
        return load(
            os.path.join(
                config.DATABANK,
                "lightfield/mit_synthetic_lf/shrubbery_camera/shrubbery_5x5_ap50",
            ),
            ".png",
        )[::-1]
    else:
        raise ValueError("Unrecognized light field identifier")


heidelberg_lf_all = [
    "backgammon",
    "dots",
    "pyramids",
    "stripes",
    "bedroom",
    "bicycle",
    "herbs",
    "origami",
    "boxes",
    "cotton",
    "dino",
    "sideboard",
    "antinous",
    "boardgames",
    "dishes",
    "greek",
    "kitchen",
    "museum",
    "pens",
    "pillows",
    "platonic",
    "rosemary",
    "table",
    "tomb",
    "tower",
    "town",
    "vinyl",
]
heidelberg_lf_stratified = heidelberg_lf_all[:4]
heidelberg_lf_test = heidelberg_lf_all[4:8]
heidelberg_lf_train = heidelberg_lf_all[8:12]
heidelberg_lf_additional = heidelberg_lf_all[12:]


def heidelberg_lf(name):
    """https://lightfield-analysis.uni-konstanz.de/"""
    name = name.lower()
    # Stratified
    if name in ["backgammon"]:
        return load(
            os.path.join(
                config.DATABANK, "lightfield/heidelberg/stratified/backgammon"
            ),
            ".png",
            prefix="input",
        )
    elif name in ["dots"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/stratified/dots"),
            ".png",
            prefix="input",
        )
    elif name in ["pyramids"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/stratified/pyramids"),
            ".png",
            prefix="input",
        )
    elif name in ["stripes"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/stratified/stripes"),
            ".png",
            prefix="input",
        )
    # Test
    elif name in ["bedroom"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/test/bedroom"),
            ".png",
            prefix="input",
        )
    elif name in ["bicycle"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/test/bicycle"),
            ".png",
            prefix="input",
        )
    elif name in ["herbs"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/test/herbs"),
            ".png",
            prefix="input",
        )
    elif name in ["origami"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/test/origami"),
            ".png",
            prefix="input",
        )
    # Training
    elif name in ["boxes"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/training/boxes"),
            ".png",
            prefix="input",
        )
    elif name in ["cotton"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/training/cotton"),
            ".png",
            prefix="input",
        )
    elif name in ["dino", "trex", "shadow"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/training/dino"),
            ".png",
            prefix="input",
        )
    elif name in ["sideboard", "basketball", "shelf"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/training/sideboard"),
            ".png",
            prefix="input",
        )
    # Additional
    elif name in ["antinous"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/antinous"),
            ".png",
            prefix="input",
        )
    elif name in ["boardgames"]:
        return load(
            os.path.join(
                config.DATABANK, "lightfield/heidelberg/additional/boardgames"
            ),
            ".png",
            prefix="input",
        )
    elif name in ["dishes"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/dishes"),
            ".png",
            prefix="input",
        )
    elif name in ["greek"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/greek"),
            ".png",
            prefix="input",
        )
    elif name in ["kitchen"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/kitchen"),
            ".png",
            prefix="input",
        )
    elif name in ["medieval2", "medieval"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/medieval2"),
            ".png",
            prefix="input",
        )
    elif name in ["museum"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/museum"),
            ".png",
            prefix="input",
        )
    elif name in ["pens"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/pens"),
            ".png",
            prefix="input",
        )
    elif name in ["pillows"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/pillows"),
            ".png",
            prefix="input",
        )
    elif name in ["platonic"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/platonic"),
            ".png",
            prefix="input",
        )
    elif name in ["rosemary"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/rosemary"),
            ".png",
            prefix="input",
        )
    elif name in ["table"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/table"),
            ".png",
            prefix="input",
        )
    elif name in ["tomb"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/tomb"),
            ".png",
            prefix="input",
        )
    elif name in ["tower"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/tower"),
            ".png",
            prefix="input",
        )
    elif name in ["town"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/town"),
            ".png",
            prefix="input",
        )
    elif name in ["vinyl"]:
        return load(
            os.path.join(config.DATABANK, "lightfield/heidelberg/additional/vinyl"),
            ".png",
            prefix="input",
        )
    else:
        raise ValueError("Unrecognized light field identifier")
