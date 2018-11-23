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
        # TODO add dynamic stopping criterion
        self.setup_W_update()
        self.update_W()
        for itr in range(int(self.max_inner * self.weightW)):
            self.update_W()

        self.setup_H_update()
        self.update_H()
        for itr in range(int(self.max_inner * self.weightH)):
            self.update_H()

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
