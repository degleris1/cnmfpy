import numpy as np

from numpy.linalg import norm
from .conv import shift_cols, tensor_conv, tensor_transconv


EPSILON = np.finfo(np.float).eps


def compute_gH(data, W, H):
    """
    Compute the gradient of H.
    """
    # compute estimate
    est = tensor_conv(W, H)

    # compute residual and loss
    resid = est - data

    # compute grad
    Hgrad = tensor_transconv(W, resid)

    return Hgrad


def compute_gW(data, W, H):
    """
    Compute the gradient of W.
    """
    # compute estimate
    est = tensor_conv(W, H)

    # compute residual and loss
    resid = est - data

    # TODO: replace with broadcasting
    Wgrad = np.empty(W.shape)

    for l in np.arange(W.shape[0]):
        # TODO verify calc
        Wgrad[l] = np.dot(resid[:, l:], shift_cols(H, l).T)

    return Wgrad


def soft_thresh(X, l):
    """
    Soft thresholding function for sparsity.
    """
    return np.maximum(X-l, 0) - np.maximum(-X-l, 0)


def compute_loss(data, W, H):
    """
    Compute the loss of a CNMF factorization.
    """
    resid = tensor_conv(W, H) - data
    return 0.5 * (norm(resid) ** 2)


def compute_loadings(data, W, H):
    """
    Compute the power explained by each factor.
    """
    loadings = []
    K, T = H.shape

    data_mag = norm(data)

    for i in range(K):
        Wi = W[:, :, i:i+1]
        Hi = H[i:i+1, :]
        est = tensor_conv(Wi, Hi)
        loadings += [norm(est - data) / (data_mag + EPSILON)]

    return loadings


def renormalize(W, H):
    """
    Renormalizes the rows of H to have constant energy.
    Updates passed parameters.
    """
    # TODO choice of norm??
    L, N, K = W.shape

    row_norms = norm(H, axis=1) + EPSILON

    H = np.diag(np.divide(1, row_norms)).dot(H)

    for l in range(L):
        W[l] = W[l].dot(np.diag(row_norms))

    return W, H


def shift_factors(W, H):
    """
    UNDER CONSTRUCTION: do not use

    Shift factors by their center of mass.
    """
    L, N, K = W.shape

    if (L == 1):
        raise IndexError('No room to shift. Disable shifting.')

    center = int(np.floor(L / 2))

    # TODO broadcast
    shifted_H = np.zeros(H.shift(0).shape)
    for k in range(K):
        masses = np.sum(W[:, k, :], axis=1)

        if (np.sum(masses) > EPSILON):
            ind = np.arange(0, L)
            cmass = int(np.floor(np.dot(masses, ind) / np.sum(masses)))+1
            loc = center - cmass

            Wpad = np.pad(W[:, :, k], ((L, L), (0, 0)), mode='constant')
            W[:, :, k] = Wpad[L - loc: 2*L - loc, :]

            shifted_H[k, :] = shift_cols(H, -loc)[k, :]

    H.assign(shifted_H)

    return W, H
