import numpy as np


def csign(B, backend=np):
    """Complex sign function
    returns exp(1j*B) if B != 0
    otherwise 0.
    np.sign is equivalent to np.sign(B.real)
    """
    if backend.isrealobj(B):
        return backend.sign(B)
    else:
        return backend.exp(1j * backend.angle(B)) * (B != 0)


def soft_thresholding(B, gamma, backend=np):
    return (backend.abs(B) >= gamma) * ((backend.abs(B) - gamma) * csign(B))


def soft_thresholding_inplace(B, gamma, backend=np):
    signB = csign(B)
    B = backend.absolute(B, out=B)  # B is real now
    B -= gamma
    B[B < 0] = 0  # ok bc B is real
    B *= signB  # now B is possibly complex
    return B


def hard_thresholding(B, gamma, backend=np):
    return B * (backend.abs(B) >= gamma)


def hard_thresholding_inplace(B, gamma, backend=np):
    # in the real case, we can get away
    # with not allocating memory for abs(B)
    # but in the complex case we have to.
    # mask is allocated in both cases
    # (but two masks are generated in real
    # case, so we only save the diff of bool
    # over dtype B. Maybe not worth branching over...)
    # todo: there might be np.where options that
    # are more memory efficient... (being a bit pedantic)
    if backend.isrealobj(B):
        mask = B < gamma
        mask[B <= -gamma] = False
        B[mask] = 0
        return B
    else:
        B[backend.abs(B) < gamma] = 0
        return B


def get_weighted_prox(prox_op, D):
    return lambda B, gamma, backend=np: prox_op(B, gamma * D[:, None], backend)


def get_prox_op_by_name(prox, inplace=True):
    if callable(prox):
        return prox
    if prox.lower() == "l0":
        if inplace:
            prox_op = hard_thresholding_inplace
        else:
            prox_op = hard_thresholding
    elif prox.lower() == "l1":
        if inplace:
            prox_op = soft_thresholding_inplace
        else:
            prox_op = soft_thresholding
    else:
        raise ValueError("Unknown Prox Operator {}".format(prox))
    return prox_op
