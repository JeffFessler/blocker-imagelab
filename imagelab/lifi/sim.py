"""imagelab/lifi/sim.py

Simulate light-fields, based on Hung Nien's matlab code

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""

import functools

import numpy as np

import imagelab as il


def generate_continuous_lightfield(obj_list, f, F):
    F = abs(F)
    obj_list = sorted(obj_list)

    def contLF(v, u, y, x):
        return 0.0

    def window(v, u, y, x):
        return 1.0

    for cz, cy, cx, obj in obj_list:
        T1 = cz / F  # removed minus (-) sign to flip image
        T2 = 1 - cz / f + cz / F

        def objLF(v, u, y, x, T1=T1, T2=T2, cy=cy, cx=cx, obj=obj):
            return obj(T1 * y + T2 * v - cy, T1 * x + T2 * u - cx)

        def occ_objLF(v, u, y, x, objLF=objLF, window=window):
            return objLF(v, u, y, x) * window(v, u, y, x)

        def window(v, u, y, x, window=window, occ_objLF=occ_objLF):
            return window(v, u, y, x) * (occ_objLF(v, u, y, x) == 0)

        def contLF(v, u, y, x, contLF=contLF, occ_objLF=occ_objLF):
            return contLF(v, u, y, x) + occ_objLF(v, u, y, x)

    return contLF


@functools.singledispatch
def discretize(clf, nV, nU, nY, nX, dV, dU, dY, dX, oversample=1, f=None, F=None):
    if isinstance(oversample, int):
        if oversample < 1:
            raise ValueError("oversample must be greater than or equal to 1")
        elif oversample != 1:
            nY, nX = nY * oversample, nX * oversample
            dY, dX = dY / oversample, dX / oversample
    elif isinstance(oversample, tuple) or isinstance(oversample, list):
        while len(oversample) < 4:
            oversample = [1] + list(oversample)
        nV, nU, nY, nX = (n * o for o, n in zip(oversample, (nV, nU, nY, nX)))
        dV, dU, dY, dX = (n / o for o, n in zip(oversample, (dV, dU, dY, dX)))
    v = dV * np.r_[-((nV - 1) // 2) : (nV - 1) // 2 + 1]
    u = dU * np.r_[-((nU - 1) // 2) : (nU - 1) // 2 + 1]
    y = dY * np.r_[-((nY - 1) // 2) : (nY - 1) // 2 + 1]
    x = dX * np.r_[-((nX - 1) // 2) : (nX - 1) // 2 + 1]
    # simulate ndgrid with meshgrid
    [U, V, Y, X] = np.meshgrid(u, v, y, x)
    lf = clf(V, U, Y, X)
    if oversample != 1:
        if isinstance(oversample, int):
            lf = il.interp.downsample(lf, (oversample, oversample))
        else:
            lf = il.interp.downsample(lf, oversample)
    return np.ascontiguousarray(lf)


@discretize.register
def deprecated_discretize(N: tuple, d: tuple, clf):
    print("Deprecated")
    nV, nU, nY, nX = N
    dV, dU, dY, dX = d
    v = dV * np.r_[-((nV - 1) // 2) : (nV - 1) // 2 + 1]
    u = dU * np.r_[-((nU - 1) // 2) : (nU - 1) // 2 + 1]
    y = dY * np.r_[-((nY - 1) // 2) : (nY - 1) // 2 + 1]
    x = dX * np.r_[-((nX - 1) // 2) : (nX - 1) // 2 + 1]
    # simulate ndgrid with meshgrid
    [U, V, Y, X] = np.meshgrid(u, v, y, x)

    return np.ascontiguousarray(clf(V, U, Y, X))


def ruler(nums, offset, scale):
    obj = []
    for r in nums:
        dist = scale * r
        obj.append((dist, *offset, il.fonts.continuous_letter(str(r))))
    return obj


def ruler_lf(nums, offset, scale, f, F):
    obj = ruler(nums, offset, scale)
    return generate_continuous_lightfield(obj, f, F)


def ivmsp18_lf():
    """aka the two disks. A light field I did a lot of work with in MATLAB
    and is my main test case for porting to Python. It appears
    in my IVMSP 2018 paper (inverted since I've changed axis conventions).
    """
    disk = il.sim.continuous_disk(R=20)
    obj_list = [(1000, -5, -5, disk), (2000, 18, 18, disk)]
    contLF = generate_continuous_lightfield(obj_list, f=50, F=1 / (1 / 50 - 1 / 400))
    lf = discretize(contLF, *(5, 5, 151, 151), *(0.3, 0.3, 0.02, 0.02))
    lf /= lf.max()
    return lf


def generate_continuous_lightfield_depthmap(obj_list, f, F):
    depth_obj_list = []
    for obj in obj_list:

        def new_func(y, x, obj=obj):
            return obj[0] * (obj[3](y, x) > 0)

        depth_obj_list.append((*obj[:3], new_func))
    depth_obj_list.append(
        (np.finfo("d").max, 0, 0, eval(f'lambda y,x: {np.finfo("d").max}'))
    )  # add infinite backdrop
    return generate_continuous_lightfield(depth_obj_list, f, F)
