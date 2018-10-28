import numpy as np
import numpy.linalg as la


def s_dot(A, B, shift):
    """
    Returns matrix multiplication between A and B, shifting columns of B.

    shift < 0, shifts columns to left.
    shift > 0, shifts columns to right.
    """
    if shift < 0:
        AB = np.dot(A, B[:, (-shift):])
        Z = np.zeros((A.shape[0], -shift))
        return np.column_stack((AB, Z))
    elif shift > 0:
        AB = np.dot(A, B[:, :(-shift)])
        Z = np.zeros((A.shape[0], shift))
        return np.column_stack((Z, AB))
    else:
        return np.dot(A, B)   # shift == 0


def s_T_dot(A, B, shift):
    """
    Returns matrix multiplication between A and transpose(B), shifting
    columns of B.

    shift < 0, shifts columns of B to left. Rows of B.T up.
    shift > 0, shifts columns of B to right. Rows of B.T down.
    """
    if shift < 0:
        return np.dot(A[:, :shift], B.T[(-shift):])

    elif shift > 0:
        return np.dot(A[:, shift:], B.T[:(-shift)])

    else:
        return np.dot(A, B.T)   # shift == 0


def cmf_predict(W, H):
    """
    Compute estimate from a CMF model.
    """
    n, t = W.shape[1], H.shape[1]
    pred = np.zeros((n, t))
    for lag, Wl in enumerate(W):
        pred += s_dot(Wl, H, lag)
    return pred


def cmf_loss(data, W, H):
    """
    Compute the loss of a CMF model. Note that this is un-normalized
    metric unlike the CMF.score(...) function. The un-normalized loss
    should be used for gradient checking.
    """
    # resid = data.copy()
    # for lag, Wl in enumerate(W):
    #     resid -= s_dot(Wl, H, lag)
    est = np.zeros_like(data)
    for lag, Wl in enumerate(W):
        est += s_dot(Wl, H, lag)
    return 0.5 * (la.norm(data - est) ** 2)


def grad_W(data, H, gW):
    """Gradient of W. Stores result in `gW`."""
    for lag in range(gW.shape[0]):
        gW[lag] = s_T_dot(data, H, lag)


def grad_H(data, W, gH):
    """Gradient of H. Stores result in `gH`."""
    gH.fill(0.)
    for lag, Wl in enumerate(W):
        gH += s_dot(Wl.T, data, lag)


def proj_step(X, dX, step_size, linesearch=False):
    """
    Updates parameters along a specified direction and projects onto positive
    orthant.

    Parameters
    ----------
    X : ndarray
        Parameters to be updated.
    dX : ndarray
        Search direction (e.g. negative gradient of X)
    step_size : float
        Size of parameter update.
    linesearch : bool
        If True, performs backtracking linesearch to optimize step_size.

    Returns
    -------
    newX : ndarray
        Updated parameters.
    """
    if linesearch:
        return NotImplementedError("linesearch has not been implemented yet.")

    else:
        return np.maximum(X - step_size * dX, 0.)
