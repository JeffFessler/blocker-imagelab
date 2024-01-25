from numbers import Number

from .. import linop, optim, utils
from ..prox import soft_thresholding_inplace as soft
from ..tv import Cdiff1_npdiff


@linop.utils.ensure_vec_out("A")
def anisotropic_tv(
    y,
    A,
    beta,
    mu=0.1,
    axes=None,
    x0=None,
    niter=100,
    ninner=10,
    callback=lambda x, itr: False,
):
    """Anisotropic (l1) Total Variation
    minimizes the cost:
        1/2 * norm(Ax - y, 2) + beta*norm(Tx, 1)
    using ADMM on the augmented lagrangian after a variable split z=Tx
        1/2 * norm(Ax - y, 2) = beta*norm(z, 1) + mu/2 * (norm(Tx - z + eta) - norm(eta))


    Parameters
    ----------
    y:
        measurements
    A:
        forward system model
    beta:
        l1 norm regularization parameter
    mu:
        ADMM parameter
    axes:
        which axes to apply finite difference regularization to
    x0:
        initial iterate. Default A.T@y
    niter:
        number of iterations to run. Default = 100
    ninner:
        number of conjugate gradient steps for x update. Default = 10
    callback(x, itr):
        user provided function that is called with initial iterate and
        at the end of every iteration with image iterate and iteration number.
        If return object is Truth-y, iterations break.

    Example
    -------
    >>> import tv
    >>> xtrue  = imread(...)
    >>> xrue = xtrue/xtrue.max()
    >>> y  = xtrue + 0.1*np.random.randn(*xtrue.shape)
    >>> A = imagelab.linop.Identity(y.shape) # denoising
    >>> xhat = tv.anisotropic_tv(y, A, beta=0.1)

    """
    x = x0 if x0 is not None else A.T @ y
    backend = utils.get_backend(x)
    axes = axes if axes is not None else list(range(x.ndim))
    if isinstance(beta, Number):
        beta = [beta] * len(axes)
    # axes = axes[-len(beta):]
    C = [
        Cdiff1_npdiff(n, in_shape=x.shape, vec_out=True, backend=backend) for n in axes
    ]
    eta = [backend.zeros(C1.shape[0]) for C1 in C]
    z = [soft(C1 @ x, beta1 / mu, backend) for eta1, beta1, C1 in zip(eta, beta, C)]

    # x update
    B = [A] + C
    Lg = [1] + [mu] * len(C)
    y = y.reshape(-1)
    g = [lambda Ax: Ax - y] + [
        lambda C1x: mu * (C1x - z1 + eta1) for z1, eta1 in zip(z, eta)
    ]

    if callback(x, 0):
        return x
    for itr in range(1, niter + 1):

        Cx = [C1 @ x for C1 in C]
        eta = [eta1 + C1x - z1 for eta1, C1x, z1 in zip(eta, Cx, z)]
        z = [
            soft(C1x + eta1, beta1 / mu, backend)
            for eta1, beta1, C1x in zip(eta, beta, Cx)
        ]
        x = optim.ncg_inv(
            B,
            g,
            Lg,
            x0=x,
            niter=ninner,
            ninner=1,  # quadratic
        )

        if callback(x, itr):
            break

    return x


def edge_preserving(
    y,
    A,
    beta,
    delta,
    axes=None,
    x0=None,
    niter=100,
    ninner=10,
    potential="huber",
    callback=lambda x, itr, cost: False,
):
    x = x0 if x0 is not None else A.T @ y
    backend = utils.get_backend(x)
    axes = axes if axes is not None else list(range(x.ndim))
    if isinstance(beta, Number):
        beta = [beta] * len(axes)
    if isinstance(delta, Number):
        delta = [delta] * len(axes)
    # axes = axes[-len(beta):]
    C = [  # finite difference operators
        Cdiff1_npdiff(n, in_shape=x.shape, vec_out=True, backend=backend) for n in axes
    ]
    potential = optim.pot.get_pot_func_by_name(potential)
    psi = [  # potential functions
        potential(reg=beta1, delta=delta1) for beta1, delta1 in zip(beta, delta)
    ]

    # x update
    B = [A] + C
    Lg = [1] + [psi1.L for psi1 in psi]
    y = y.reshape(-1)
    g = [lambda x: x - y] + [psi1.grad for psi1 in psi]

    def cost_func(x):
        return float(
            backend.sum((A @ x - y) ** 2)
            + sum(backend.sum(psi1(C1 @ x)) for psi1, C1 in zip(psi, C))
        )

    return optim.ncg_inv_mm(
        B,
        g,
        Lg,
        x0=x0,
        callback=lambda x, itr: callback(x, itr, cost=cost_func),
        niter=niter,
        ninner=ninner,
    )


def isotropic_tv(
    y,
    A,
    beta,
    axes,
    x0=None,
    niter=100,
    ninner=10,
    callback=lambda x, itr, cost: False,
):
    return edge_preserving(
        y=y,
        A=A,
        beta=beta,
        delta=0,
        axes=axes,
        x0=x0,
        niter=niter,
        ninner=ninner,
        potential="L2",
        callback=callback,
    )
