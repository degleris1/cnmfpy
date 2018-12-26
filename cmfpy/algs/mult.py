import numpy as np

from ..common import cmf_predict, tensor_transconv, shift_cols
from .base import AbstractOptimizer, EPSILON


class MultUpdate(AbstractOptimizer):
    """
    Multiplicative update rule.
    """
    def __init__(self, data, dims, patience=3, tol=1e-5):
        super().__init__(data, dims, patience=patience, tol=tol)

    def update(self):
        # update W
        num_W, denom_W = self._compute_mult_W()
        self.W = np.divide(np.multiply(self.W, num_W), denom_W + EPSILON)

        # update H
        num_H, denom_H = self._compute_mult_H()
        self.H = np.divide(np.multiply(self.H, num_H), denom_H + EPSILON)

        self.cache_resids()
        return self.loss

    def _compute_mult_W(self, X=None, W=None, H=None):
        X = X if X is not None else self.X
        W = W if W is not None else self.W
        H = H if H is not None else self.H

        # preallocate
        num = np.zeros(W.shape)
        denom = np.zeros(W.shape)

        est = cmf_predict(W, H)

        # TODO: broadcast
        for l in np.arange(W.shape[0]):
            # TODO switch to sdot
            num[l] = np.dot(X[:, l:], shift_cols(H, l).T)
            denom[l] = np.dot(est[:, l:], shift_cols(H, l).T)

        return num, denom

    def _compute_mult_H(self, X=None, W=None, H=None):
        X = X if X is not None else self.X
        W = W if W is not None else self.W
        H = H if H is not None else self.H

        est = cmf_predict(W, H)

        num = tensor_transconv(W, X)
        denom = tensor_transconv(W, est)

        return num, denom


class StochasticMultUpdate(MultUpdate):
    def __init__(self, data, dims, patience=3, tol=1e-5, window_size=5000):
        """Initialize Stochastic Update Rule."""

        # Invoke super constructor, but additionally stores data window size.
        super().__init__(data, dims, patience=patience, tol=tol)
        if (window_size > self.n_timepoints):
            raise ValueError("Window Size must be less than "
                             "or equal to the width of the data matrix. "
                             "window_size = %d, n_timepoints=%d" % (window_size, self.n_timepoints))
        self.window_size = window_size

    def update(self):
        """
        Overrides update rule to work on portions of the data matrix.
        """

        # Pick random window of data
        start_idx = np.random.randint(self.n_timepoints - self.window_size + 1)
        t_idx = range(start_idx, start_idx + self.window_size)

        # Select time window.
        X = self.X[:, t_idx]
        H = self.H[:, t_idx]

        # The stochastic W update is unchanged – we still update every entry
        num_W, denom_W = self._compute_mult_W(X, self.W, H)
        self.W = (self.W * num_W) / (denom_W + EPSILON)

        # For the stochastic H update we only update a subset of entries
        num_H, denom_H = self._compute_mult_H(X, self.W, H)
        self.H[:, t_idx] = (H * num_H) / (denom_H + EPSILON)

        # TODO: cache_resids needs to be changed so that it's limited to the
        # time window (t_idx)!!! Otherwise this step is going to be unfeasibly
        # expensive...
        self.cache_resids(t_idx, H)
        return self.loss

    def cache_resids(self, t_idx=None, H=None):
        """
        Overrides the standard cache_resids function so that we only
        compute residuals using a subset of the data matrix.
        """
        if ((self.W is None) or (self.H is None)):
            raise ValueError("W or H not initalized.")

        t_idx = t_idx if t_idx is not None else range(0, self.n_timepoints)
        H = H if H is not None else self.H

        # When super().__init__ is called, self.est and self.resids
        # have not been initialized, so we initialize them if they are None.
        if self.est is None:
            self.est = cmf_predict(self.W, H)
        else:
            self.est[:, t_idx] = cmf_predict(self.W, H)

        if self.resids is None:
            self.resids = self.est - self.X
        else:
            self.resids[:, t_idx] = self.est[:, t_idx] - self.X[:, t_idx]