import numpy as np

from ..conv import tensor_transconv, tensor_conv, shift_cols

# TODO float or float32?
EPSILON = np.finfo(np.float).eps


class MultUpdate:
    def __init__(self, data, model):
        self.data = data
        self.model = model

    def step(self):
        # TODO is this bad?
        model = self.model

        # Compute multiplier for W
        num_W, denom_W = _compute_mult_W(self.data, model.W, model.H,
                                         model.maxlag)
        # Update W
        model.W = np.divide(np.multiply(model.W, num_W), denom_W + EPSILON)

        # Compute multiplier for H
        num_H, denom_H = _compute_mult_H(self.data, model.W, model.H)
        # Update H
        model.H = np.divide(np.multiply(model.H, num_H), denom_H + EPSILON)


"""
Shared functions
NOTE: these two functions have been left out of the MultUpdate class, so they
can be shared by other multiplicative update rules.
"""


def _compute_mult_W(data, W, H, L):
    # preallocate
    num = np.zeros(W.shape)
    denom = np.zeros(W.shape)

    est = tensor_conv(W, H)

    # TODO: broadcast
    for l in np.arange(L):
        num[l] = np.dot(data[:, l:], shift_cols(H, l).T)
        denom[l] = np.dot(est[:, l:], shift_cols(H, l).T)

    return num, denom


def _compute_mult_H(data, W, H):
    est = tensor_conv(W, H)

    num = tensor_transconv(W, data)
    denom = tensor_transconv(W, est)

    return num, denom
