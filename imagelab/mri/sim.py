"""imagelab/mri/sim.py
This code is mostly lifted from MIRT.jl
Copyright 2019-03-05, Jeff Fessler, University of Michigan

It needs to be redone to be more streamline/pythonic and to use
my image axis conventions.
"""

import numpy as np

from .. import interp


def shepp_logan_emis(nx=256, ny=None, dx=1, dy=None, offset_x=0, offset_y=0, rot=0):
    if ny is None:
        ny = nx
    if dy is None:
        dy = dx
    fovx = nx * dx
    fovy = ny * dy
    params = shepp_logan_parameters(fovx, fovy, "emis")
    return ellipse_im_fast(
        nx, ny, params, dx, dy, offset_x, offset_y, rot, replace=False
    )


def eecs598_sl():
    return shepp_logan_emis()[:, 127 - (192 // 2) : 127 + (192 // 2)]


def ellipse_im(
    ig, params, rot=0, oversample=1, hu_scale=1, replace=False, return_params=False
):

    params[:, 5] *= hu_scale

    if oversample > 1:
        ig = ig.over(oversample)

    phantom = ellipse_im_fast(
        ig.nx,
        ig.ny,
        params,
        ig.dx,
        ig.dy,
        ig.offset_x,
        ig.offset_y,
        rot,
        oversample,
        replace,
    )

    if oversample > 1:
        phantom = interp.downsample(phantom, oversample)

    if return_params:
        return (phantom, params)
    return phantom


def ellipse_im_fast(nx, ny, params_in, dx, dy, offset_x, offset_y, rot, replace):

    params = params_in.copy()
    if params.shape[1] != 6:
        raise ValueError("bad ellipse parameter vector size")

    # optional rotation
    if rot != 0:
        th = rot * 180 / np.pi
        cx = params[:, 0]
        cy = params[:, 1]
        params[:, 0] = cx * np.cos(th) + cy * np.sin(th)
        params[:, 1] = -cx * np.sin(th) + cy * np.cos(th)
        params[:, 4] += rot

    phantom = np.zeros((nx, ny), np.float32)

    wx = (nx - 1) / 2 + offset_x
    wy = (ny - 1) / 2 + offset_y
    x1 = ((np.r_[0:nx]) - wx) * dx
    y1 = ((np.r_[0:ny]) - wy) * dy
    yy, xx = np.meshgrid(y1, x1)

    hx = np.abs(dx) / 2
    hy = np.abs(dy) / 2

    for ie in range(params.shape[0]):

        ell = params[ie, :]
        cx = ell[0]
        rx = ell[2]
        cy = ell[1]
        ry = ell[3]
        theta = ell[4] * np.pi / 180

        xs = xx - cx  # shift per ellipse center
        ys = yy - cy

        # coordinates of "outer" corner of each pixel, relative to ellipse center
        xo = xs + np.sign(xs) * hx
        yo = ys + np.sign(ys) * hy

        # voxels that are entirely inside the ellipse:
        (xr, yr) = rot2(xo, yo, theta)
        is_inside = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1

        if replace:
            phantom[is_inside] = ell[5]
        else:
            phantom += ell[5] * is_inside.astype(np.float32)

    return phantom.T[::-1]


def rot2(x, y, theta):
    xr = np.cos(theta) * x + np.sin(theta) * y
    yr = -np.sin(theta) * x + np.cos(theta) * y
    return (xr, yr)


def shepp_logan_parameters(xfov, yfov, case="kak"):
    """
    `params = shepp_logan_parameters(xfov, yfov)`
    parameters from Kak and Slaney text, p. 255
    the first four columns are unitless "fractions of field of view"
    """
    params = np.array(
        [
            [0, 0, 0.92, 0.69, 90, 2],
            [0, -0.0184, 0.874, 0.6624, 90, -0.98],
            [0.22, 0, 0.31, 0.11, 72, -0.02],
            [-0.22, 0, 0.41, 0.16, 108, -0.02],
            [0, 0.35, 0.25, 0.21, 90, 0.01],
            [0, 0.1, 0.046, 0.046, 0, 0.01],
            [0, -0.1, 0.046, 0.046, 0, 0.01],
            [-0.08, -0.605, 0.046, 0.023, 0, 0.01],
            [0, -0.605, 0.023, 0.023, 0, 0.01],
            [0.06, -0.605, 0.046, 0.023, 90, 0.01],
        ]
    )

    params[:, [0, 2]] *= xfov / 2
    params[:, [1, 3]] *= yfov / 2

    if case == "emis":
        params[:, 5] = np.array([1, 1, -2, 2, 3, 4, 5, 6, 1, 1])
    elif case == "brainweb":
        params[:, 5] = np.array(
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]
        )  # brainweb uses index 1-10
    elif case != "kak":
        raise ValueError("bad phantom case")

    return params


def blochsim(Mi, bx, by, bz, T1, T2, dt):
    """
    Evolve magnetization field using time-discretized Bloch equation
    Input
        Mi [3,dim]  initial X,Y,Z magnetization
                    Note: normalized to magnitude <= Mo=1
        bx,by,bz [ntime,dim]    effective X,Y,Z applied magnetic field
                    (Tesla), for rotating frame (no Bo)
        T1  [dim]       spin-lattice relaxation time (msec)
        T2  [dim]       spin-spin relaxation time (msec)
        dt  scalar      time interval (msec)
    Output
        mx,my,mz [ntime,dim]  X,Y,Z magnetization as a function of time
    """

    # constants
    gambar = 42.57e3  # gamma/2pi in kHz/T
    gam = gambar * 2 * np.pi
    # Put Beff into units rotations, T1 and T2 into losses
    bx = bx * (dt * gam)
    # rotation angle/step
    by = by * (dt * gam)
    # rotation angle/step
    bz = bz * (dt * gam)
    # rotation angle/step
    # Put relaxations into losses/recovery per step
    T1 = dt / T1
    T2 = 1 - dt / T2

    # size checks
    # if ~isreal(bx) | ~isreal(by) | ~isreal(bz)
    #     bx = real(bx);
    #     by = real(by);
    #     bz = real(bz);
    #     disp('Warning: B field must be real valued - using only the real part');
    # end
    # if (size(bx) ~= size(by)) | (size(bx) ~= size(bz))
    #     disp('Error: B vectors not the same length')
    #     return;
    # end
    # if (size(Mi,2) ~= size(bx,2))
    #     disp('Error: Initial magnetization not right size')
    #     return;
    # end
    # if (size(Mi,2) ~= length(T1))
    #     disp('Error: T1 vector not right size')
    #     return;
    # end
    # if (size(Mi,2) ~= length(T2))
    #     disp('Error: T2 vector not right size')
    #     return;
    # end

    nstep = bx.shape[0]
    #
    # Initialize outputs
    mx = np.zeros(bx.shape)
    my = np.zeros(bx.shape)
    mz = np.zeros(bx.shape)
    mx[0, :] = Mi[0, :]
    my[0, :] = Mi[1, :]
    mz[0, :] = Mi[2, :]

    # stable bloch equation simulator: rotations are explicitly
    # calculated and carried out on the magnetization vector
    for lp in range(1, nstep):
        B = np.vstack((bx[lp - 1, :], by[lp - 1, :], bz[lp - 1, :])).T
        # Compute sines & cosines of field angles:
        # Theta = angle w.r.t positive z axis
        # Phi   = angle w.r.t positive x axis
        # Psi   = angle w.r.t transformed positive x axis
        #
        Bmag = np.sqrt(np.sum(B ** 2, axis=1))  # Magnitude of applied field
        Btrans = np.sqrt(B[:, 0] ** 2 + B[:, 1] ** 2)
        # Magnitude of transverse applied field
        ct = np.ones(B.shape[0])
        good = Bmag != 0
        if np.any(good):
            ct[good] = B[good, 2] / Bmag[good]  # cos(theta)

        st = np.sqrt(1 - ct ** 2)  # sin(theta) > 0

        cphi = np.ones(B.shape[0])
        good = Btrans != 0
        if np.any(good):
            cphi[good] = B[good, 0] / Btrans[good]  # cos(phi)

        sphi = np.sqrt(1 - cphi ** 2) * np.sign(B[:, 1])  # sin(phi)

        cpsi = np.cos(Bmag)
        # cos(psi)
        spsi = np.sin(Bmag)
        # sin(psi)

        #
        # Evolve
        #
        if np.any(Bmag != 0):
            Mx0 = mx[lp - 1, :].T
            My0 = my[lp - 1, :].T
            Mz0 = mz[lp - 1, :].T

            Mx1 = cphi * (
                ct
                * (
                    cpsi * (ct * (sphi * My0 + cphi * Mx0) - st * Mz0)
                    + spsi * (cphi * My0 - sphi * Mx0)
                )
                + st * (ct * Mz0 + st * (sphi * My0 + cphi * Mx0))
            ) - sphi * (
                -spsi * (ct * (sphi * My0 + cphi * Mx0) - st * Mz0)
                + cpsi * (cphi * My0 - sphi * Mx0)
            )
            My1 = sphi * (
                ct
                * (
                    cpsi * (ct * (sphi * My0 + cphi * Mx0) - st * Mz0)
                    + spsi * (cphi * My0 - sphi * Mx0)
                )
                + st * (ct * Mz0 + st * (sphi * My0 + cphi * Mx0))
            ) + cphi * (
                -spsi * (ct * (sphi * My0 + cphi * Mx0) - st * Mz0)
                + cpsi * (cphi * My0 - sphi * Mx0)
            )
            Mz1 = ct * (ct * Mz0 + st * (sphi * My0 + cphi * Mx0)) - st * (
                cpsi * (ct * (sphi * My0 + cphi * Mx0) - st * Mz0)
                + spsi * (cphi * My0 - sphi * Mx0)
            )
        else:
            Mx1 = mx[lp - 1, :].T
            My1 = my[lp - 1, :].T
            Mz1 = mz[lp - 1, :].T

        # relaxation effects: "1" in Mz since Mo=1 by assumption
        mx[lp, :] = (Mx1 * T2).T
        my[lp, :] = (My1 * T2).T
        mz[lp, :] = (Mz1 + (1 - Mz1) * T1).T

    return mx, my, mz
