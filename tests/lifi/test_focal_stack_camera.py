import os

import numpy as np
import scipy.io as sio

import imagelab as il
from imagelab import lifi

here = os.path.abspath(os.path.dirname(__file__))


def test_forward_adjoint_consistency_FocalStackCamera():
    A = lifi.system_model.FocalStackCamera(
        51, 51, 0.02, 0.02, 5, 5, 0.3, 0.3, 50, [2000, 800, 400]
    )
    assert il.linop.test_forward_adjoint_consistency(A)


def test_forward_adjoint_consistency_ShiftSumFocalStackCamera():
    A = lifi.system_model.ShiftSumFocalStackCamera(
        51, 51, 0.02, 0.02, 5, 5, 0.3, 0.3, 50, [2000, 800, 400]
    )
    assert il.linop.test_forward_adjoint_consistency(A)


def test_forward_adjoint_consistency_ShiftSumFocalStackCamera2():
    s = [-1, -0.5, 0, 0.75, 1.5]
    A = il.lifi.system_model.ShiftSumFocalStackCamera(
        nX=134, nY=50, nU=5, nV=5, nC=3, dX=1, dY=1, dU=1, dV=1, f=1, z=[2] * len(s)
    )
    A.sU = s
    A.sV = s
    assert il.linop.test_forward_adjoint_consistency(A)


# def test_FocalStackCamera_forward_speed(benchmark):
#     A = lifi.system_model.FocalStackCamera(51,51,0.02,0.02,5,5,0.3,0.3, 50, [2000, 800, 400])
#     lf = np.random.randn(*A.in_shape)
#     res = benchmark(A.forward_project, lf)

# def test_FocalStackCamera_backward_speed(benchmark):
#     A = lifi.system_model.FocalStackCamera(51,51,0.02,0.02,5,5,0.3,0.3, 50, [2000, 800, 400])
#     lf = np.random.randn(*A.out_shape)
#     res = benchmark(A.back_project, lf)

# def test_ShiftSumFocalStackCamera_forward_speed(benchmark):
#     A = lifi.system_model.ShiftSumFocalStackCamera(51,51,0.02,0.02,5,5,0.3,0.3, 50, [2000, 800, 400])
#     lf = np.random.randn(*A.in_shape)
#     res = benchmark(A.forward_project, lf)

# def test_ShiftSumFocalStackCamera_backward_speed(benchmark):
#     A = lifi.system_model.ShiftSumFocalStackCamera(51,51,0.02,0.02,5,5,0.3,0.3, 50, [2000, 800, 400])
#     lf = np.random.randn(*A.out_shape)
#     res = benchmark(A.back_project, lf)


def test_focal_stack_system_model():
    A = lifi.system_model.FocalStackCamera(
        151, 151, 0.02, 0.02, 5, 5, 0.3, 0.3, 50, [2000, 1600, 1200, 800, 400]
    )
    mat_dict = sio.loadmat(here + "/matlab_system_outputs.mat")
    lf = mat_dict["lf_t"].T[:, :, ::-1, ::-1]
    bp = mat_dict["lf_bp"].T[:, :, ::-1, ::-1]
    fs = mat_dict["fs"].T[:, ::-1, ::-1]

    fs0 = A @ lf
    assert np.allclose(fs * 1e6, fs0 * 1e6), "Forward Model Broken"

    bp0 = A.T @ fs0
    bp0 /= bp0.max()
    assert np.allclose(bp, bp0), "Backward Model Broken"


def test_focal_stack_kernel():
    mat_g = sio.loadmat(here + "/g.mat")["g"].T
    A = lifi.system_model.FocalStackCamera(
        151, 151, 0.02, 0.02, 5, 5, 0.3, 0.3, 50, [2000, 1600, 1200, 800, 400]
    )
    g0 = A._generate_g()[:, :, :, ::-1, ::-1]
    # pad estimate to full size
    g0big = np.zeros_like(mat_g)
    pad = (g0big.shape[-1] - g0.shape[-1]) // 2
    g0big[:, :, :, pad:-pad, pad:-pad] = g0
    assert np.allclose(
        mat_g * 1e6, g0big * 1e6
    ), "FocalStackCamera g Kernel has changed!"


def test_continuous_lightfield():
    mat_lf = sio.loadmat(here + "/matlab_system_outputs.mat")["lf_t"].T[
        :, :, ::-1, ::-1
    ]

    disk1 = il.sim.continuous_disk(R=20)
    disk2 = disk1
    obj_list = [(1000, -5, -5, disk1), (2000, 18, 18, disk2)]
    contLF = lifi.sim.generate_continuous_lightfield(
        obj_list, f=50, F=1 / (1 / 50 - 1 / 400)
    )
    lf = lifi.sim.discretize((5, 5, 151, 151), d=(0.3, 0.3, 0.02, 0.02), clf=contLF)
    lf /= lf.max()

    assert np.allclose(mat_lf, lf), "Simulation Results differ from MATLAB"


def test_focal_stack_rotation_inv():
    # Setup
    # Generate a Light-Field
    disk1 = il.sim.continuous_disk(R=20)
    disk2 = disk1
    obj_list = [(1000, -5, -5, disk1), (2000, 18, 18, disk2)]
    contLF = lifi.sim.generate_continuous_lightfield(
        obj_list, f=50, F=1 / (1 / 50 - 1 / 400)
    )
    lf = lifi.sim.discretize((5, 5, 51, 51), d=(0.3, 0.3, 0.02, 0.02), clf=contLF)
    lf /= lf.max()
    # Generate system model
    A_orig = lifi.system_model.FocalStackCamera(
        51, 51, 0.02, 0.02, 5, 5, 0.3, 0.3, 50, [2000, 1600, 1200, 800, 400]
    )
    A_back = lifi.system_model.FocalStackCamera(
        51, 51, 0.02, 0.02, 5, 5, 0.3, 0.3, 50, [2000, 1600, 1200, 800, 400]
    )
    A_back._g = A_back._g[:, :, :, ::-1, ::-1]

    # Run Test
    lf_orig = lf
    lf_back = lf[:, :, ::-1, ::-1]
    fs_orig = A_orig @ lf_orig
    fs_back = A_back @ lf_back
    assert np.allclose(
        fs_orig[:, ::-1, ::-1] * 1e6, fs_back * 1e6
    ), "Forward Model Failed Rotatation Invariance"
    bp_orig = A_orig.T @ fs_orig
    bp_back = A_back.T @ fs_back
    assert np.allclose(
        bp_orig[:, :, ::-1, ::-1] * 1e6, bp_back * 1e6
    ), "Backward Model Failed Rotation Invariance"


def test_contiguity():
    disk1 = il.sim.continuous_disk(R=20)
    disk2 = disk1
    obj_list = [(1000, -5, -5, disk1), (2000, 18, 18, disk2)]
    contLF = lifi.sim.generate_continuous_lightfield(
        obj_list, f=50, F=1 / (1 / 50 - 1 / 400)
    )
    lf = lifi.sim.discretize((3, 3, 51, 51), d=(0.3, 0.3, 0.02, 0.02), clf=contLF)
    lf /= lf.max()
    assert lf.flags.c_contiguous, "Simulation LightFields are not Contiguous"

    A = lifi.system_model.FocalStackCamera(
        51, 51, 0.02, 0.02, 3, 3, 0.3, 0.3, 50, [2000, 1600, 1200, 800, 400]
    )
    assert A._g.flags.c_contiguous, "FocalStackCamera Kernels are not Contiguous"
