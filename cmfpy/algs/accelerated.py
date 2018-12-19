import numpy.linalg as la
from .base import AbstractOptimizer


class AcceleratedOptimizer(AbstractOptimizer):
    """
    Accelerated Update Rules.

    From Gillis, https://arxiv.org/abs/1107.5194v2.
    """

    def __init__(self, data, dims, patience=3, tol=1-5,
                 max_iter=1, stop_thresh=0, weightW=1, weightH=1,
                 **kwargs):
        """
        Parameters
        ----------
        data : ndarray
            The data to be fit
        dims : ModelDimensions
            A dimensions object containing all relevant dimensions.
        patience : int
            Patience for stopping early.
        tol : float
            Tolerance for stopping early.
        max_iter : int
            Maximum number of inner iterations, i.e. number of times to update
            `W` for before switching to `H`, and vice versa.
        stop_thresh : float
            Threshold for stop inner iterations early, i.e. break if the change
            in `W` is too small.
        weightW : float
            Actual maximum number of inner iterations is `weightW` times
            `max_iter`. This way, `W` and `H` can update different numbers of
            times.
        weightH : float
            See `weightW`.
        """
        super().__init__(data, dims, patience=patience, tol=tol, **kwargs)

        self.max_iter = max_iter
        self.stop_thresh = stop_thresh
        self.weightW = weightW
        self.weightH = weightH

        if (max_iter * min(1, weightH, weightW) < 1):
            raise ValueError("Requires at least 1 iteration for both W and H.")

    def _accelerated_update(self, var, weight, update_rule):
        hist = [var.copy()]

        # Update once
        update_rule()
        hist.append(var.copy())
        init_diff = la.norm(hist[-1] - hist[-2])
        diff = init_diff

        # Update several more times
        # for itr in range(int(self.max_inner * self.weightW)):
        itr = 1
        while ((itr < self.max_iter * weight) and
               (diff > self.stop_thresh * init_diff)):
            itr += 1
            update_rule()
            hist.append(var.copy())
            diff = la.norm(hist[-1] - hist[-2])

    def update(self):
        """
        Updates `H` and `W` several times, stopping either when the max
        number of iterations has been reached, or when the change in the `W`
        or `H` is below a threshold.
        """
        self.setup_W_update()
        self._accelerated_update(self.W, self.weightW, self.update_W)

        self.setup_H_update()
        self._accelerated_update(self.H, self.weightH, self.update_H)

        self.cache_resids()
        return self.loss

    def setup_W_update(self):
        raise NotImplementedError("Accelerated class must be overridden.")

    def setup_H_update(self):
        raise NotImplementedError("Accelerated class must be overridden.")

    def update_W(self):
        raise NotImplementedError("Accelerated class must be overridden.")

    def update_H(self):
        raise NotImplementedError("Accelerated class must be overridden.")
