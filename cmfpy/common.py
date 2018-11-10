"""
Common numeric routines.
"""
import numpy as np
import numpy.linalg as la
import numba


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
