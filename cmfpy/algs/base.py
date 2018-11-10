"""
Base functionality and interface for algorithm classes.
"""
import numpy.random as npr
import numpy.linalg as la
from ..common import cmf_predict


class AbstractOptimizer:
    """Defines common API for optimizer objects."""

    def __init__(self, data, model_dimensions, initW=None, initH=None):
        """Initialize algorithm."""

        # Store data
        self.X = data

        # Store model dimensions
        for k, v in model_dimensions:
            setattr(self, k, v)

        # Don't initialize model parameters
        self.W = initW
        self.H = initH

    def update(self):
        """Update model parameters."""
        raise NotImplementedError("Base class must override update(...)")

    def converged(self, loss_hist):
        """Check model parameters for convergence."""
        raise NotImplementedError("Base class must override convergence check.")

    def rand_init(self):
        """Default initialization procedure for model parameters."""

        # Randomize nonnegative parameters.
        W = npr.rand(self.maxlag, self.n_features, self.n_components)
        H = npr.rand(self.n_components, self.n_timepoints)

        # Correct scale of parameters.
        est = cmf_predict(W, H)
        alpha = (self.X * est).sum() / la.norm(est)**2
        return alpha * W, alpha * H
