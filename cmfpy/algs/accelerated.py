import numpy.linalg as la
from .base import AbstractOptimizer


class AcceleratedOptimizer(AbstractOptimizer):
    """
    Accelerated Update Rules.

    From Gillis, https://arxiv.org/abs/1107.5194v2.
    """

    def __init__(self, data, dims, patience=3, tol=1-5,
                 max_inner=0, inner_thresh=0, weightW=1, weightH=1,
                 **kwargs):
        super().__init__(data, dims, patience=patience, tol=tol, **kwargs)

        self.max_inner = max_inner
        self.inner_thresh = inner_thresh
        self.weightW = weightW
        self.weightH = weightH

    def update(self):
        """
        Updates `H` and `W` several times, stopping either when the max
        number of iterations has been reached, or when the change in the `W`
        or `H` is below a threshold.
        """
        W_hist = [self.W.copy()]
        self.setup_W_update()

        # Update once
        self.update_W()
        W_hist.append(self.W.copy())
        W_init_change = la.norm(W_hist[-1] - W_hist[-2])

        # Update several more times
        for itr in range(int(self.max_inner * self.weightW)):
            self.update_W()
            W_hist.append(self.W.copy())
            if (self.inner_thresh > 0 and la.norm(W_hist[-1] - W_hist[-2])
                    < self.inner_thresh * W_init_change):
                break

        H_hist = [self.H.copy()]
        self.setup_H_update()

        # Update once
        self.update_H()
        H_hist.append(self.H.copy())
        H_init_change = la.norm(H_hist[-1] - H_hist[-2])

        # Update several more times
        for itr in range(int(self.max_inner * self.weightH)):
            self.update_H()
            H_hist.append(self.H.copy())
            if (self.inner_thresh > 0 and la.norm(H_hist[-1] - H_hist[-2])
                    < self.inner_thresh * H_init_change):
                break

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
