"""
Common numeric routines.
"""
import numpy as np
import numpy.linalg as la
import numba


EPSILON = np.finfo(np.float).eps
FACTOR_MIN = 0


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


def tensor_transconv(W, X):
    """
    Transpose tensor convolution of tensor `W` and matrix `X`.
    Parameters
    ----------
    W : ndarray, shape (n_lag, n_neurons, n_components)
        Tensor of neural sequences.
    X : ndarray, shape (n_components, n_time)
        Matrix of time componenents.
    Returns
    -------
    X : ndarray, shape (n_neurons, n_time)
        Transpose tensor convolution of W and X, i.e. the tensor convolution
        but shifted the opposite direction.
    Notes
    -----
    # TODO: Consider speed up
    """
    L, N, K = W.shape
    T = X.shape[1]

    result = np.zeros((K, T))
    for lag, w in enumerate(W):
        result[:, :T-lag] += np.dot(w.T, shift_cols(X, -lag))

    return result


def shift_cols(X, lag):
    """
    Shifts the columns of a matrix `X` right by `l` lags. Drops columns that
    are shifted beyond the dimension of the matrix.
    # TODO: remove cases
    """
    if (lag <= 0):
        return X[:, -lag:]
    else:  # lag > 0
        return X[:, :-lag]


def shift_and_stack(H, L):
    """
    Vertically stack several shifted copies of H.
    """
    K, T = H.shape

    H_stacked = np.zeros((L*K, T))
    for lag in range(L):
        H_stacked[K*lag:K*(lag+1), lag:] = shift_cols(H, lag)

    return H_stacked
