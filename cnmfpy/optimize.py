import numpy as np


from numpy.linalg import norm
from cnmfpy.conv import ShiftMatrix, tensor_conv, tensor_transconv


def compute_gH(data, W, H, shifts):
    """
    Compute the gradient of H.
    """
    # compute estimate
    est = tensor_conv(W, H, shifts)

    # compute residual and loss
    resid = est - data.shift(0)
    loss = norm(resid)

    # wrap residual in ShiftMatrix
    maxlag = int((len(shifts) - 1) / 2)
    resid = ShiftMatrix(resid, maxlag)

    # compute grad
    Hgrad = tensor_transconv(W, resid, shifts)
    
    return loss, Hgrad


def compute_gW(data, W, H, shifts):
    """
    Compute the gradient of W.
    """
    # compute estimate
    est = tensor_conv(W, H, shifts)

    # compute residual and loss
    resid = est - data.shift(0)
    loss = norm(resid)

    # TODO: replace with broadcasting
    Wgrad = np.empty(W.shape)
    for l, t in enumerate(shifts):
        Wgrad[l] = np.dot(resid, H.shift(t).T)

    return loss, Wgrad


def soft_thresh(X, l):
    """
    Soft thresholding function for sparsity.
    """
    return np.maximum(X-l, 0) - np.maximum(-X-l, 0)


def compute_loss(data, W, H, shifts):
    """
    Compute the loss of a CNMF factorization.
    """
    resid = tensor_conv(W, H, shifts) - data.shift(0)
    return norm(resid)





def _rms(X):
    # DEPRECATED
    """
    Compute root mean square error.
    """
    return norm(X) / np.sqrt(np.size(X))
    #return np.sqrt(np.dot(X_rav, X_rav)) / np.sqrt(X.size)