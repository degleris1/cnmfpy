"""
Base functionality and interface for algorithm classes.
"""


class AbstractOptimizer:

    def __init__(self, data, initW, initH):
        """Initialize algorithm."""
        self.X = data
        self.W = initW
        self.H = initH

    def update(self, data):
        raise NotImplementedError("Base class must override update(...)")

    def converged(self, loss_hist):
        raise NotImplementedError("Base class must override convergence check.")
