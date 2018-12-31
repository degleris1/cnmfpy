import numpy as np
from ..common import cmf_predict


class Synthetic:
    """Synthetic data."""
    def __init__(self,
                 n_components=3,
                 n_features=100,
                 n_lags=100,
                 n_timebins=10000,
                 H_sparsity=0.9,
                 noise_scale=1.0,
                 seed=None):

        # Set data name and random state.
        self.name = "synthetic"
        self.rs = np.random.RandomState(seed)

        # Generate random convolutional parameters
        W = np.zeros((n_lags, n_features, n_components))
        H = self.rs.rand(n_components, n_timebins)

        # Add sparsity to factors
        self.H = H * self.rs.binomial(1, 1 - H_sparsity, size=H.shape)

        # Add structure to motifs
        for i, j in enumerate(np.random.choice(n_components, size=n_features)):
            W[:, i, j] += _gauss_plus_delay(n_lags)
        self.W = W

        # Determine noise
        self.noise = noise_scale * self.rs.rand(n_features, n_timebins)

        # Add noise to model prediction
        self.data = cmf_predict(self.W, self.H) + self.noise

    def generate(self):
        return self.data + self.noise


def _gauss_plus_delay(n_steps):
    tau = np.random.uniform(-1.5, 1.5)
    x = np.linspace(-3-tau, 3-tau, n_steps)
    y = np.exp(-x**2)
    return y / y.max()
