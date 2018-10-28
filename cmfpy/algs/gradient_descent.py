import numba

import numpy as np
import numpy.linalg as la
from numbers import Integral

from ..common import cmf_predict, s_dot, s_T_dot
from .base import AbstractOptimizer


class GradDescent(AbstractOptimizer):
    """
    Gradient Descent update rules for CMF.
    """

    def __init__(self, data, initW, initH, tol=1e-5, patience=3,
                 step_decrement=5.):

        # Check inputs,
        if patience < 1 or not isinstance(patience, Integral):
            raise ValueError("Patience must be a positive integer.")

        # Invoke super class initialization procedures.
        super().__init__(data, initW, initH)

        # TODO: Figure out whether this is a reasonable guess.
        self.step_size = 1 / la.norm(data)

        # Hyperparameters.
        self.tol = tol
        self.patience = patience
        self.step_decrement = step_decrement

        # Preallocate space for gradients.
        self.gW = np.empty_like(self.W)
        self.gH = np.empty_like(self.H)

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

    def update(self):
        """Update parameters."""

        # Update parameters.
        _projected_step(self.W, -self.step_size * self.gW)
        _projected_step(self.H, -self.step_size * self.gH)

        # Update residuals and gradients and return loss
        self.cache_resids()
        self.cache_gW()
        self.cache_gH()
        return self.loss()

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
        _projected_step(self.W, -self.step_size * self.gW)

        # Update H (need to recompute residuals since W was updated).
        self.cache_resids()
        self.cache_gH()
        _projected_step(self.H, -self.step_size * self.gH)

        # Update residuals and gradient computation for W (for next iteration).
        self.cache_resids()
        self.cache_gW()

        # Return loss
        return self.loss


# TODO: parallel loop here may be overkill
@numba.jit(nopython=True, parallel=True)
def _projected_step(X, dX):
    """
    Adds dX to X and projects onto nonegative orthant. Modifies X in-place.
    """
    x, dx = X.ravel(), dX.ravel()
    for i in numba.prange(len(x)):
        x[i] = np.maximum(x[i] + dx[i], 0.0)
