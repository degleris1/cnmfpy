"""
Base functionality and interface for algorithm classes.
"""
import numpy.random as npr
import numpy.linalg as la
import numpy as np

from ..common import cmf_predict
from numbers import Integral


class AbstractOptimizer:
    """Defines common API for optimizer objects."""

    def __init__(self, data, model_dimensions, initW=None, initH=None,
                 tol=1e-5, patience=3):
        """Initialize algorithm."""

        # Check inputs,
        if (patience < 1 or not isinstance(patience, Integral)):
            raise ValueError("Patience must be a positive integer.")

        # Store data
        self.X = data
        self.normX = la.norm(data)
        self.resids = None

        # Hyperparameters
        self.patience = patience
        self.tol = tol

        # Store model dimensions
        for k, v in model_dimensions:
            setattr(self, k, v)

        # Don't initialize model parameters
        if ((initW is None) or (initH is None)):
            self.W, self.H = self.initialize()
        else:
            self.W = initW
            self.H = initH

        self.cache_resids()

    def update(self):
        """Update model parameters."""
        raise NotImplementedError("Base class must override update(...)")

    def initialize(self):
        """
        Initialize W and H.

        Individual optimizers can override this with specific initialization.
        """
        return self.rand_init()

    def cache_resids(self):
        """Updates intermediate computations."""
        if ((self.W is None) or (self.H is None)):
            raise ValueError("W or H not initalized.")
        self.est = cmf_predict(self.W, self.H)
        self.resids = self.est - self.X

    def converged(self, loss_hist):
        """
        Check model parameters for convergence.
        """

        # Improvement in loss function over iteration.
        d_loss = np.diff(loss_hist[-self.patience:])

        # Objective converged
        if (np.all(np.abs(d_loss) < self.tol)):
            return True
        else:
            return False

    def rand_init(self):
        """Default initialization procedure for model parameters."""

        # Randomize nonnegative parameters.
        W = npr.rand(self.maxlag, self.n_features, self.n_components)
        H = npr.rand(self.n_components, self.n_timepoints)

        # Correct scale of parameters.
        est = cmf_predict(W, H)
        alpha = (self.X * est).sum() / la.norm(est)**2
        return alpha * W, alpha * H

    @property
    def loss(self):
        """
        Compute normalized loss (for user-facing functions)
        """
        if (self.resids is None):
            raise ValueError("Residuals not initialized.")
        return la.norm(self.resids) / self.normX
