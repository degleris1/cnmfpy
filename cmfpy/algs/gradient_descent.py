import numpy as np
import numpy.linalg as la
from numbers import Integral

from ..common import grad_W, grad_H, proj_step


class GradDescent:
    """
    Gradient Descent update rules for CMF.
    """

    def __init__(self, model, data, tol=1e-5, patience=3, step_decrement=5.):
        """Initialize Algorithm."""

        # Check inputs,
        if patience < 1 or not isinstance(patience, Integral):
            raise ValueError("Patience must be a positive integer.")

        # Save data and model parameters.
        self.X = data
        self.W = model.W
        self.H = model.H

        # TODO: Figure out whether this is a reasonable guess.
        self.step_size = 1 / la.norm(data)

        # Hyperparameters.
        self.tol = tol
        self.patience = patience
        self.step_decrement = step_decrement

        # Preallocate space for gradients.
        self.gW = np.empty_like(self.W)
        self.gH = np.empty_like(self.H)

    def update(self):
        """Update parameters."""
        grad_W(self.X, model.H, self.gW)
        grad_H(self.X, model.W, self.gH)
        self.W = proj_step(model.W, self.gW, self.step_size)
        self.H = proj_step(model.W, self.gH, self.step_size)

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


class BlockDescent(GradDescent):
    """
    Block Coordinate Descent update rules for CMF.
    """

    def update(self):
        """Overrides gradient update rule. Same idea, but different order."""
        grad_W(data, model.H, self.gW)
        self.W = proj_step(model.W, self.gW, self.step_size)
        grad_H(data, model.W, self.gH)
        self.H = proj_step(model.W, self.gH, self.step_size)
