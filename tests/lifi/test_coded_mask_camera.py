import numpy as np

import imagelab as il
from imagelab import lifi


def test_forward_adjoint_consistency_CodedMaskCameraColor():
    s = -0.5
    A = lifi.system_model.CodedMaskCamera((5, 5, 10, 10, 3), s, mask=2)
    assert il.linop.test_forward_adjoint_consistency(A)


def test_forward_adjoint_consistency_CodedMaskCameraColorMask():
    s = -0.25
    mask = np.random.rand(2, 10, 10, 3)
    A = lifi.system_model.CodedMaskCamera((5, 5, 10, 10, 3), s, mask=mask)
    assert il.linop.test_forward_adjoint_consistency(A)


def test_forward_adjoint_consistency_CodedMaskCamera():
    s = -0.75
    A = lifi.system_model.CodedMaskCamera((3, 5, 17, 11), s)
    assert il.linop.test_forward_adjoint_consistency(A)


def test_mask_shifting():
    mask = np.zeros((5, 5, 5, 5))
    for ii in range(5):
        for jj in range(5):
            mask[ii, jj, ii, jj] = 1

    s = -1
    A = lifi.system_model.CodedMaskCamera((5, 5, 5, 5), s, mask=mask[None, 2, 2])
    assert (mask == A.mask[0]).all()
