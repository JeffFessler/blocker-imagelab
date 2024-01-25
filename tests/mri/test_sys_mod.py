# import numpy as np
# import pytest
from imagelab.mri import system_model as sysmod

# try:
#     import cupy as cp

#     cuda_en = True
# except ImportError:
#     cuda_en = False


def test_phase_encode_mask():
    mask = sysmod.create_phase_encode_mask((512, 333))
    assert mask is not None
