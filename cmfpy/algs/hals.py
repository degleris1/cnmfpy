import numpy as np
import numpy.linalg as la

from .base import AbstractOptimizer, EPSILON
from ..common import shift_and_stack, cmf_predict


class SimpleHALSUpdate(AbstractOptimizer):
    """
    HALS update rule.
    """

    def __init__(self, data, dims, patience=3, tol=1e-5, **kwargs):
        super().__init__(data, dims, patience=patience, tol=tol, **kwargs)

    def update(self):
        self.update_W()
        self.update_H()

        self.cache_resids()
        return self.loss

    def update_W(self):
        L, N, K = self.W.shape

        # Unfold matrices
        H_unfold = shift_and_stack(self.H, L)

        # Compute norms
        H_norms = la.norm(H_unfold, axis=1)

        for k in range(K):
            for l in range(L):
                self.update_W_col(k, l, H_unfold, H_norms)

    def update_W_col(self, k, l, H_unfold, H_norms):
        L, N, K = self.W.shape
        ind = l*K + k

        self.resids -= np.outer(self.W[l, :, k], H_unfold[ind, :])
        self.W[l, :, k] = self.next_W_col(H_unfold[ind, :],
                                          H_norms[ind],
                                          self.resids)
        self.resids += np.outer(self.W[l, :, k], H_unfold[ind, :])

    def next_W_col(self, Hkl, norm_Hkl, resid):
        """
        """
        # TODO reconsider transpose
        return np.maximum(np.dot(-resid, Hkl) / (norm_Hkl**2 + EPSILON), 0)

    def update_H(self):
        L, N, K = self.W.shape
        T = self.H.shape[1]

        # Set up residual and norms of each column
        W_norms = la.norm(self.W, axis=1).T  # K * L, norm along N

        # Update each component
        for k in range(K):
            Wk = self.W[:, :, k].T  # TODO: will this slow things down?

            # Update each timebin
            for t in range(T):
                self.update_H_entry(k, t, Wk, W_norms)

    def update_H_entry(self, k, t, Wk, W_norms):
        """
        Update a single entry of `H`.
        """
        L, N, K = self.W.shape
        T = self.H.shape[1]

        # Remove factor from residual
        remainder = self.resids[:, t:t+L] - self.H[k, t] * Wk[:, :T-t]

        norm_Wkt = np.sqrt(np.sum(W_norms[k, :T-t]**2))
        self.H[k, t] = self.next_H_entry(Wk[:, :T-t], norm_Wkt, remainder)

        # Add factor back to residual
        self.resids[:, t:t+L] = remainder + self.H[k, t] * Wk[:, :T-t]

    def next_H_entry(self, Wkt, norm_Wkt, remainder):
        """
        """
        trace = np.dot(np.ravel(Wkt), np.ravel(-remainder))

        return np.maximum(trace / (norm_Wkt**2 + EPSILON), 0)


class AdvancedHALSUpdate(SimpleHALSUpdate):
    """
    Advanced version of HALS update, updating T/L entries of `H` at a time.
    """

    def __init__(self, data, dims, patience=3, tol=1e-5, **kwargs):
        # Create zero padded residuals
        self.resids_pad = np.zeros((dims.n_features,
                                    dims.n_timepoints + dims.maxlag))

        super().__init__(data, dims, patience, tol, **kwargs)

    def cache_resids(self):
        """
        Updates residual (zero padded)
        """
        super(AdvancedHALSUpdate, self).cache_resids()
        self.resids_pad[:, :self.n_timepoints] = self.resids

    def update_H(self):
        L, N, K = self.W.shape
        T = self.H.shape[1]

        # Update each component
        for k in range(K):
            Wk = self.W[:, :, k].ravel()
            norm_Wk = la.norm(Wk)
            # TODO update the last few entries in H

            # Update each lag
            for l in range(L):
                batch = self.H[k, l:T:L]

                n_batch = len(batch)
                end_batch = l + L*n_batch
                last_slice_crop = T - L*(n_batch - 1)

                # Create residual tensor and factor tensor
                resid_tens = self.fold_resids(l, n_batch)
                factors_tens = self.fold_factor(Wk, batch)

                # TODO this is hacky... fix!
                last_slice = self.W[:, :, k]
                last_slice[last_slice_crop:, :] = 0
                factors_tens[-1] = batch[-1] * last_slice.ravel()

                # Create norms for the batch
                # TODO only the last entry differs; is there a more efficient
                # way to do this?
                batch_norms = np.ones(n_batch) * norm_Wk
                batch_norms[-1] = la.norm(self.W[:last_slice_crop, :, k])

                # Clone Wk several times
                Wk_clones = self.clone_Wk(Wk, n_batch)

                # DEBUG
                assert resid_tens.shape == factors_tens.shape
                assert Wk_clones.shape == resid_tens.shape

                # Generate remainder (residual - factor) tensor and remove
                # factor contribution from residual
                remainder = resid_tens - factors_tens

                self.resids_pad[:, l:end_batch] -= self.unfold_factor(
                    factors_tens, n_batch
                )

                # Update H
                self.H[k, l:T:L] = self.new_H_batch(Wk_clones,
                                                    batch_norms,
                                                    remainder)

                # Add factor contribution back to residual
                updated_batch = self.H[k, l:T:L]
                new_factors_tens = self.fold_factor(Wk, updated_batch)
                self.resids_pad[:, l:end_batch] += self.unfold_factor(
                    new_factors_tens, n_batch
                )

    def new_H_batch(self, Wk_clones, batch_norms, remainder):
        traces = np.inner(Wk_clones, -remainder)[0]
        return np.maximum(np.divide(traces, np.square(batch_norms) + EPSILON),
                          0)

    def fold_resids(self, start, n_batch):
        """
        Select the appropriate part of the residual matrix, and fold into
        a tensor.
        """
        cropped = self.resids_pad[:, start:(start + self.maxlag * n_batch)]
        return cropped.T.reshape(n_batch, self.maxlag*self.n_features)

    def fold_factor(self, Wk, batch):
        """
        Generate factor prediction for a given component and lag. Then fold
        into a tensor.
        """
        return np.outer(batch, Wk)

    def unfold_factor(self, factors_tens, n_batch):
        """
        Expand the factor tensor into a matrix.
        """
        return factors_tens.reshape(self.maxlag*n_batch, self.n_features).T

    def clone_Wk(self, Wk, n_batch):
        """
        Clone Wk several times and place into a tensor.
        """
        return np.outer(np.ones(n_batch), Wk)