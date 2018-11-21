import numpy as np
import numpy.linalg as la

from .base import AbstractOptimizer, EPSILON
from ..common import shift_and_stack


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
                ind = l*K + k

                self.resids -= np.outer(self.W[l, :, k], H_unfold[ind, :])
                self.W[l, :, k] = self.new_W_col(H_unfold[ind, :],
                                                 H_norms[ind],
                                                 self.resids)
                self.resids += np.outer(self.W[l, :, k], H_unfold[ind, :])

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
                # TODO could use some help with indentation here
                resid_slice = (self.resids[:, t:t+L]
                               - self.H[k, t] * Wk[:, :T-t])

                norm_Wkt = np.sqrt(np.sum(W_norms[k, :T-t]**2))
                self.H[k, t] = self.new_H_entry(Wk[:, :T-t],
                                                norm_Wkt,
                                                resid_slice)

                self.resids[:, t:t+L] = (resid_slice
                                         + self.H[k, t] * Wk[:, :T-t])

    def new_W_col(self, Hkl, norm_Hkl, resid):
        """
        """
        # TODO reconsider transpose
        return np.maximum(np.dot(-resid, Hkl) / (norm_Hkl**2 + EPSILON), 0)

    def new_H_entry(self, Wkt, norm_Wkt, resid_slice):
        """
        """
        trace = np.dot(np.ravel(Wkt), np.ravel(-resid_slice))

        return np.maximum(trace / (norm_Wkt**2 + EPSILON), 0)


class AdvancedHALSUpdate(SimpleHALSUpdate):
    """
    Advanced version of HALS update, updating T/L entries of `H` at a time.
    """

    def gen_resids_tens(self, resid, L, n_lay, N):
        T_crop = L * n_lay

        return resid[:, :T_crop].T.reshape(n_lay, L*N)

    def gen_factors_tens(self, Wk, entries):
        return np.outer(entries, Wk)

    def expand_factors_tens(self, factors_tens, n_lay, L, N):
        return factors_tens.reshape(L*n_lay, N).T

    def clone_Wk(self, Wk, n_lay):
        return np.outer(np.ones(n_lay), Wk)

    def update_H(self):
        L, N, K = self.W.shape
        T = self.H.shape[1]

        self.H[T-L:L] = 0
        self.cache_resids()

        # Set up norms of each column
        # W_norms = la.norm(model.W, axis=1).T  # K * L, norm along N

        # Update each component
        for k in range(K):
            Wk = self.W[:, :, k].ravel()
            norm_Wk = la.norm(Wk)
            # TODO update the last few entries in H

            # Update each lag
            for l in range(L):
                entries = self.H[k, l:T-L:L]
                n_lay = len(entries)

                # Create residual and factor tensors
                resid_tens = self.gen_resids_tens(self.resids, L, n_lay, N)
                factors_tens = self.gen_factors_tens(Wk, entries)
                # DEBUG
                assert resid_tens.shape == factors_tens.shape

                # Generate remainder
                remainder = resid_tens - factors_tens

                # Remove factor contribution to residual
                self.resids[:, l:l+n_lay*L] += self.expand_factors_tens(
                    factors_tens, n_lay, L, N)

                # Clone Wk several times
                Wk_clones = self.clone_Wk(Wk, n_lay)
                assert Wk_clones.shape == remainder.shape

                # Update H
                self.H[k, l:T-L:L] = self.update_Hkl(Wk_clones,
                                                     norm_Wk,
                                                     remainder)

                # Add factor contribution back to residual
                new_entries = self.H[k, l:T-L:L]
                new_factors_tens = self.gen_factors_tens(Wk, new_entries)
                self.resids[:, l:l+n_lay*L] += self.expand_factors_tens(
                    new_factors_tens, n_lay, L, N)

                # TODO clean
                # TODO update last entries

    def update_Hkl(self, Wk_clones, norm_Wk, remainder):
        traces = np.inner(Wk_clones, -remainder)[0]
        return np.maximum(traces / (norm_Wk**2 + EPSILON), 0)
