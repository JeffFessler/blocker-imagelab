"""imagelab/mri/system_model.py
Module containing model of the physics involved in creating
MR images.

:author: Cameron Blocker <cameronjblocker@gmail.com>
"""
import numpy as np

from imagelab import fft, linop


class CartesianSingleCoilMRI2D(linop.AbstractLinOp):
    """System Model for Single Coil MRI, i.e.
    a FFT with optional mask for sampling pattern"""

    def __init__(self, mask=1, in_shape=None, out_shape=None, **kwargs):
        if in_shape is None:
            if out_shape is None:
                if hasattr(mask, "shape") and len(np.squeeze(mask).shape) > 1:
                    in_shape = mask.shape
            else:
                in_shape = out_shape
        out_shape = in_shape
        super(CartesianSingleCoilMRI2D, self).__init__(
            in_shape=in_shape, out_shape=out_shape
        )
        self.mask = np.real(mask)
        self.device = "gpu" if "cupy" in type(mask).__module__ else "cpu"
        self.kwargs = kwargs
        self.fft2, self.ifft2 = fft.get_fft2_pair(device=self.device, **kwargs)

    def forward_project(self, x):
        return self.mask * self.fft2(x)

    def back_project(self, y):
        return self.ifft2(self.mask * y)

    def to_cupy(self):
        if self.device == "gpu":
            return
        super(CartesianSingleCoilMRI2D, self).to_cupy()
        self.device = "gpu"
        self.fft2, self.ifft2 = fft.get_fft2_pair(device=self.device, **self.kwargs)


def create_phase_encode_mask(
    shape,
    center_frac=0.08,
    accel=4,
    same_shape=False,
    phase_dim=-1,
    seed=None,
):
    """
    The mask selects a subset of columns or rows from the input k-space data.
    E.g. if the k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_frac) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal
    to (N / acceleration).

    Parameters
    ----------
    center_frac (List[float]): Fraction of low-frequency columns to be retained.
        If multiple values are provided, then one of these numbers is chosen uniformly
        each time.
    accel (List[int]): Amount of under-sampling. This should have the same length
        as center_fractions. If multiple values are provided, then one of these is chosen
        uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
        not be spaced evenly.
    same_shape (bool): whether to broadcast the mask to be the same shape as `shape`.
        If false, only the 1D mask along the phase encode dimension is returned.
    phase_dim: which dimension to mask along. default is columns (-1)
        (i.e. mask is constant along the columns). For rows pass -2.
    seed: seed for random number generator passed to `np.random.default_rng`

    Returns
    -------
    mask

    """
    rng = np.random.default_rng(seed)
    shape = list(shape)
    if len(shape) < 2:
        raise ValueError("Shape should have 2 or more dimensions")

    num_cols = shape[phase_dim]

    # Create the mask
    num_low_freqs = int(round(num_cols * center_frac))
    prob = (num_cols / accel - num_low_freqs) / (num_cols - num_low_freqs)
    mask = rng.uniform(size=num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad : pad + num_low_freqs] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[phase_dim] = num_cols
    mask = mask.reshape(*mask_shape)
    if same_shape:
        shape[phase_dim] = 1
        mask = mask * np.ones(shape, dtype=bool)

    return mask
