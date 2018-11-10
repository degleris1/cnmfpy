import itertools

import numba

import numpy as np
import numpy.linalg as la
from numbers import Integral

from ..common import cmf_predict, s_dot, s_T_dot
from .base import AbstractOptimizer

from scipy.linalg import eigh


class GradDescent(AbstractOptimizer):
    """
    Gradient Descent update rules for CMF.
    """

    def __init__(self, data, dims, tol=1e-5, patience=3,
                 step_decrement=5., **kwargs):

        # Check inputs,
        if patience < 1 or not isinstance(patience, Integral):
            raise ValueError("Patience must be a positive integer.")

        # Invoke super class initialization procedures.
        super().__init__(data, dims, **kwargs)

        # TODO: results seem sensitive to step size so let's see if we
        # can get a rough lipshitz constant working....
        self.step_size = 1e-4

        # Hyperparameters.
        self.tol = tol
        self.patience = patience
        self.step_decrement = step_decrement

        # Initialize W and H randomly.
        if (self.W is None) or (self.H is None):
            self.W, self.H = self.rand_init()

        # Preallocate space for gradients.
        self.gW = np.empty((self.maxlag, self.n_features, self.n_components))
        self.gH = np.empty((self.n_components, self.n_timepoints))

        # Cache useful intermediate results. Norm of data matrix,
        # residuals, and gradients.
        self.normX = la.norm(data)
        self.cache_resids()
        self.cache_gW()
        self.cache_gH()

    def cache_resids(self):
        """Updates intermediate computations."""
        self.est = cmf_predict(self.W, self.H)
        self.resids = self.est - self.X

    def cache_gW(self):
        """Gradient of W. Stores result in `gW`."""
        for lag in range(self.W.shape[0]):
            self.gW[lag] = s_T_dot(self.resids, self.H, lag)
        return self.gW

    def cache_gH(self):
        """Gradient of H. Stores result in `gH`."""
        self.gH.fill(0.)
        for lag, Wl in enumerate(self.W):
            self.gH += s_dot(Wl.T, self.resids, -lag)
        return self.gH

    def lipschitz_W(self):
        L, K = self.W.shape[0], self.W.shape[-1]
        first_row = [s_T_dot(self.H, self.H, l) for l in range(L)]

        hW = np.empty((K * L, K * L))
        for i, j in itertools.product(range(L), range(L)):
            i0 = i * K
            i1 = (i * K) + K
            j0 = j * K
            j1 = (j * K) + K
            if i <= j:
                hW[i0:i1, j0:j1] = first_row[j - i]
            else:
                hW[i0:i1, j0:j1] = first_row[i - j].T

        return eigh(hW, eigvals=(K * L - 1, K * L - 1), eigvals_only=True)[0]

    def lipschitz_H(self):
        raise NotImplementedError()
        # max_eigval = -1.0  # all eigvals will be positive.
        # L, K = self.W.shape[0], self.W.shape[-1]
        # block = np.empty((K, K))
        # for k in range(K):
        #     for l in range(L):
        #         self.W[l, k]
        # return eigh(hW, eigvals=(K * L - 1, K * L - 1), eigvals_only=True)[0]

    def update(self):
        """Update parameters."""

        # Update parameters.
        _projected_step(self.W, self.gW, 1.0 / self.lipschitz_W())
        _projected_step(self.H, self.gH, self.step_size)

        # Update residuals and gradients and return loss
        self.cache_resids()
        self.cache_gW()
        self.cache_gH()
        return self.loss

    def converged(self, loss_hist):
        """Check if converged, decrease step size if needed."""

        # Improvement in loss function over iteration.
        d_loss = np.diff(loss_hist[-self.patience:])

        # Objective went up
        if d_loss[-1] > 0:
            self.step_size /= self.step_decrement
            return False

        # Objective converged
        elif np.all(np.abs(d_loss) < self.tol):
            return True

        # Objective went down, but not yet converged.
        else:
            return False

    @property
    def unnormalized_loss(self):
        """Compute unnormalized loss (for testing gradients)."""
        return 0.5 * la.norm(self.resids) ** 2

    @property
    def loss(self):
        """Compute normalized loss (for user-facing functions)."""
        return la.norm(self.resids) / self.normX


class BlockDescent(GradDescent):
    """
    Block Coordinate Descent update rules for CMF.
    """

    def update(self):
        """Overrides gradient update rule. Same idea, but different order."""

        # Update W (gradient should be up-to-date)
        _projected_step(self.W, self.gW, 1.0 / self.lipschitz_W())

        # Update H (need to recompute residuals since W was updated).
        self.cache_resids()
        self.cache_gH()
        _projected_step(self.H, self.gH, self.step_size)

        # Update residuals and gradient computation for W (for next iteration).
        self.cache_resids()
        self.cache_gW()

        # Return loss
        return self.loss


def _projected_step(X, dX, ss):
    """Thin wrapper, vectorizes parameters for easy update."""
    return _raveled_step(X.ravel(), dX.ravel(), ss)


@numba.jit(nopython=True)
def _raveled_step(x, dx, ss):
    """
    Adds dx to x and projects onto nonegative orthant. Modifies x in-place.
    """
    for i in range(len(x)):
        x[i] = np.maximum(x[i] - ss * dx[i], 0.0)
