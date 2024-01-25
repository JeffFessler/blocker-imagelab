import numpy as np
from scipy.linalg import norm

DEFAULT_SCALES = np.logspace(-2, 1, num=21)


def lipcheck(g, L, x, *, scales=DEFAULT_SCALES, sig_scale=0.01, npert=20, seed=None):
    """
    Syntax: true_or_false = lipcheck(g, L, x)

    Description: performs some numerical tests to see if L could be the
    Lipschitz constant of the function g(). This routine cannot prove
    that L is correct, but should have a decent probability of discovering
    and reporting that L is incorrect if it is so. (Provides a sanity check.)

    Input
    g:    function from R^N into R^M (such as gradient of cost function)
    L:    Lipschitz constant of cost function gradient
    x:    a "typical" input argument of g

    Option

    Output
    returns true if the g,L pair passes the tests and false otherwise.
    return type is Bool
    """
    # principles:
    # often the highest slope is near 0
    # small perturbations approximate derivative
    rng = np.random.default_rng(seed)
    for scale in scales:
        xs = scale * x
        gs = g(xs)
        for _ii in range(npert):
            pert = rng.standard_normal(x.shape)
            sig = sig_scale * norm(pert) / norm(x)
            pert *= sig
            xp = xs + pert
            gp = g(xp)
            measured = norm(gs - gp) / norm(pert)
            if measured > L:
                return False
    return True
