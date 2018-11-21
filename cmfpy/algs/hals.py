import numpy as np
import numpy.linalg as la

from .base import AbstractOptimizer, EPSILON
from ..common import shift_and_stack


class SimpleHALSUpdate(AbstractOptimizer):
    """
    HALS update rule.
    """

    def __init__(self, data, dims, patience=3, tol=1e-5, **kwargs):
        super().__init__(data, dims, patience=patience, tol=tol, **kwargs)

    def update(self):
        self.update_W()
        self.update_H()

        self.cache_resids()
        return self.loss

    def update_W(self):
        L, N, K = self.W.shape

        # Unfold matrices
        H_unfold = shift_and_stack(self.H, L)

        # Compute norms
        H_norms = la.norm(H_unfold, axis=1)

        for k in range(K):
            for l in range(L):
                ind = l*K + k

                self.resids -= np.outer(self.W[l, :, k], H_unfold[ind, :])
                self.W[l, :, k] = self.new_W_col(H_unfold[ind, :],
                                                 H_norms[ind],
                                                 self.resids)
                self.resids += np.outer(self.W[l, :, k], H_unfold[ind, :])

    def update_H(self):
        pass

    def new_W_col(self, Hkl, norm_Hkl, resid):
        """
        """
        # TODO reconsider transpose
        return np.maximum(np.dot(-resid, Hkl) / (norm_Hkl**2 + EPSILON), 0)
