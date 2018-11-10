"""
Extension of XRAY algorithm for convolutive NMF model.
"""
import numpy.linalg as la


class XRAY(AbstractOptimizer):
    """
    Class implementing XRAY algorithm for convolutive NMF.

    Reference
    ---------
    Kumar A, Sindhwani V, Kambadur P (2013). Fast Conical Hull Algorithms for
        Near-separable Non-negative Matrix Factorization. ICML.
    """

    def __init__(data, dims):
        super().__init__(data, dims)  # save data and dimensions as attributes.

    def init_params(self):
        """Override initialization, self.update(...) takes care of it."""
        pass

    def update(self):

        # Initialize storage for W and H
        self.W = np.empty((self.maxlag, self.n_features, self.n_components))
        self.H = np.empty((self.n_components, self.n_timepoints))

        # Initialize residual matrix
        residual = self.data.copy()

        # Iteratively solve each component in the model.
        for k in range(self.n_components):

            # Assign next W component to extreme ray.
            colnorms = np.linalg.norm(residual, axis=0) ** 2
            t = _conv_argmax(colnorms, self.maxlag)
            self.W[:, :, k] = self.data[:, t:(t + maxlag)].T

            # Fit best H for last W component
            _chals_H(residual, self.W[:, :, k], self.H[k])

            # Update residual.
            residual -= cmf_predict(self.W[:, :, k], self.H[k])

            # TODO: update all W and H components <k ??

        # need to compute loss manually
        return np.linalg.norm(residual) / np.linalg.norm(data)

    def converged(self):
        """Algorithm converges after one iteration."""
        return self.W is not None


@numba.jit(nopython=True)
def _conv_argmax(x, maxlag):
    # Index interating over elements of x
    i = 0

    # Compute running sum over maxlag elements
    rs = 0.0
    while i < maxlag:
        mx += x[i]
        i += 1

    # Store largest running sum
    mx = rs   # stores max
    amx = 0   # stores argmax

    # Iterate over remaining elements in x
    while i < len(x):
        # Recompute moving sum within window.
        rs += x[i] - x[i - maxlag]

        # Save largest running sum.
        if rs > mx:
            mx = rs
            amx = i - maxlag

        # Move to next timepoint.
        i += 1

    return amx
