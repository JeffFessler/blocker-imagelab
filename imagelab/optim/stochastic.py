"""imagelab/optim/stochastic.py

Stochastic Methods of optimization

"""
from numpy import np

# from ..utils import export

# http://ruder.io/optimizing-gradient-descent/index.html


def sgd(grad, w0, lr=0.01, nepoch=100):
    w = w0
    for _itr in range(1, nepoch + 1):
        w -= lr * grad(w)
    return w


def adagrad(grad, w0, lr=0.01, nepoch=100):
    """AdaGrad
    Like SGD but adjusts learning rate so features
    not seen often have greater weight.

    Problem: The sum of square gradients G gets
    larger and larger until we stop moving at all.
    """
    w = w0
    eps = 1e-8
    G = 0
    for _itr in range(1, nepoch + 1):
        g = grad(w)
        G += g ** 2
        w -= lr * g / np.sqrt(G + eps)
    return w


def rmsprop(grad, w0, gamma=0.9, lr=0.01, nepoch=100):
    """RMSprop
    Like Adagrad, but has a decaying memory so that
    G doesn't diverge and stop movement.

    Problem: It was later noted, with the introduction
    of AMSgrad, that these forgetting methods tend
    to forget to quickly.
    """
    w = w0
    eps = 1e-8
    G = 0
    for _itr in range(1, nepoch + 1):
        g = grad(w)
        G += gamma * G + (1 - gamma) * g ** 2
        w -= lr * g / np.sqrt(G + eps)
    return w


def adadelta(grad, w0, gamma=0.9, nepoch=100):
    """AdaDelta
    Like RMSprop, with Gw added to try to fix units.

    "you don't have to set a learning rate!"
    yes, but you have to initialize Gw...
    """
    w = w0
    Gw = 0.001
    eps = 1e-8
    G = 0
    for _itr in range(1, nepoch + 1):
        g = grad(w)
        G = gamma * G + (1 - gamma) * g ** 2
        del_w = np.sqrt((Gw + eps) / (G + eps)) * g
        w -= del_w
        Gw = gamma * Gw + (1 - gamma) * del_w ** 2
    return w


def adam(grad, w0, lr=0.01, beta1=0.9, beta2=0.999, nepoch=100):
    """ADAM
    Like RMSprop, but adds an averaging of the gradients
    instead of just the square gradients. This is like momentum
    """
    w = w0
    eps = 1e-8
    m = 0
    v = 0
    for itr in range(1, nepoch + 1):
        g = grad(w)
        m += beta1 * m + (1 - beta1) * g
        v += beta2 * v + (1 - beta2) * g ** 2

        mh = m / (1 + beta1 ** itr)
        vh = v / (1 + beta2 ** itr)
        w -= lr * mh / (np.sqrt(vh) + eps)
    return w


def adamax(grad, w0, lr=0.002, beta1=0.9, beta2=0.999, nepoch=100):
    """AdaMax
    Like Adam, but replaces squares with max for v
    """
    w = w0
    m = 0
    v = 0
    for itr in range(1, nepoch + 1):
        g = grad(w)
        m += beta1 * m + (1 - beta1) * g
        v += max(beta2 * v, g)

        mh = m / (1 + beta1 ** itr)
        w -= lr * mh / v
    return w


def nadam(grad, w0, lr=0.01, beta1=0.9, beta2=0.999, nepoch=100):
    """Nestorav Accelerated Adaptive Moment Estimation
    Like ADAM but with nesterov accelerated gradient, i.e. take step then
    compute gradient, instead of add gradient to momentum and take step.
    Order is reorganized so you take step with last iteration.
    """
    w = w0
    eps = 1e-8
    m = 0
    v = 0
    for itr in range(1, nepoch + 1):
        g = grad(w)
        m += beta1 * m + (1 - beta1) * g
        v += beta2 * v + (1 - beta2) * g ** 2

        mh = m / (1 + beta1 ** itr)
        vh = v / (1 + beta2 ** itr)
        w -= (
            lr
            * (beta1 * mh + (1 - beta1) * g / (1 - beta1 ** itr))
            / (np.sqrt(vh) + eps)
        )
    return w


def amsgrad(grad, w0, lr=0.01, beta1=0.9, beta2=0.999, nepoch=100):
    """AMSgrad
    Like Adam, but keeps step sizes from getting bigger,
    and removes debiasing.
    Solves the problem of ADAM forgetting things to quickly
    and converging suboptimally.

    "It remains to be seen whether AMSGrad is able to
    consistently outperform Adam in practice"
    """
    w = w0
    eps = 1e-8
    m = 0
    v = 0
    vh = 0
    mh = 0
    for _itr in range(1, nepoch + 1):
        g = grad(w)
        m += beta1 * m + (1 - beta1) * g
        v += beta2 * v + (1 - beta2) * g ** 2

        vh = max(vh, v)
        w -= lr * mh / (np.sqrt(vh) + eps)
    return w
