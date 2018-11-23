import numpy as np

from ..common import cmf_predict, tensor_transconv, shift_cols, EPSILON
from .base import AbstractOptimizer


class MultUpdate(AbstractOptimizer):
    """
    Multiplicative update rule.
    """

    def __init__(self, data, dims, patience=3, tol=1e-5, **kwargs):
        super().__init__(data, dims, patience=patience, tol=tol, **kwargs)

    def update(self):
        # update W
        num_W, denom_W = self._compute_mult_W()
        self.W = np.divide(np.multiply(self.W, num_W), denom_W + EPSILON)

        # update H
        num_H, denom_H = self._compute_mult_H()
        self.H = np.divide(np.multiply(self.H, num_H), denom_H + EPSILON)

        self.cache_resids()
        return self.loss

    def _compute_mult_W(self):
        # preallocate
        num = np.zeros(self.W.shape)
        denom = np.zeros(self.W.shape)

        est = cmf_predict(self.W, self.H)

        # TODO: broadcast
        for l in np.arange(self.W.shape[0]):
            # TODO switch to sdot
            num[l] = np.dot(self.X[:, l:], shift_cols(self.H, l).T)
            denom[l] = np.dot(est[:, l:], shift_cols(self.H, l).T)

        return num, denom

    def _compute_mult_H(self):
        est = cmf_predict(self.W, self.H)

        num = tensor_transconv(self.W, self.X)
        denom = tensor_transconv(self.W, est)

        return num, denom
