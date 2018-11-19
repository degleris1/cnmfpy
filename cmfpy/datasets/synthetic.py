import numpy as np
from ..common import cmf_predict


class Synthetic:
    """Synthetic data."""
    def __init__(self,
                 n_components=3,
                 n_features=100,
                 n_lags=10,
                 n_timebins=1000,
                 W_sparsity=0.5,
                 H_sparsity=0.9,
                 noise_scale=1.0,
                 seed=None):

        # Set data name and random state.
        self.name = "synthetic"
        self.rs = np.random.RandomState(seed)

        # Generate random convolutional parameters
        W = self.rs.rand(n_lags, n_features, n_components)
        H = self.rs.rand(n_components, n_timebins)

        # Add sparsity to factors
        self.W = W * self.rs.binomial(1, 1 - W_sparsity, size=W.shape)
        self.H = H * self.rs.binomial(1, 1 - H_sparsity, size=H.shape)

        # Determine noise
        self.noise = noise_scale * self.rs.rand(n_features, n_timebins)

        # Add noise to model prediction
        self.data = cmf_predict(self.W, self.H) + self.noise

    def generate(self):
        return self.data
