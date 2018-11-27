import numpy as np
import numpy.linalg as la

from .accelerated import AcceleratedOptimizer
from ..common import shift_and_stack, EPSILON, FACTOR_MIN


class SimpleHALSUpdate(AcceleratedOptimizer):
    """
    HALS update rule.
    """

    def __init__(self, data, dims, patience=3, tol=1e-5,
                 max_inner=4, weightW=1, weightH=0, inner_thresh=0, **kwargs):
        super().__init__(data, dims, patience=patience, tol=tol,
                         max_inner=max_inner, weightW=weightW, weightH=weightH,
                         inner_thresh=inner_thresh, **kwargs)

    """
    W update
    """

    def setup_W_update(self):
        L, N, K = self.W.shape

        self.H_unfold = shift_and_stack(self.H, L)  # Unfold matrices
        self.H_norms = la.norm(self.H_unfold, axis=1)  # Compute norms

    def update_W(self):
        L, N, K = self.W.shape
        for k in range(K):
            for l in range(L):
                self.update_W_col(k, l)

    def update_W_col(self, k, l):
        L, N, K = self.W.shape
        ind = l*K + k

        self.resids -= np.outer(self.W[l, :, k], self.H_unfold[ind, :])
        self.W[l, :, k] = self.next_W_col(self.H_unfold[ind, :],
                                          self.H_norms[ind],
                                          self.resids)
        self.resids += np.outer(self.W[l, :, k], self.H_unfold[ind, :])

    def next_W_col(self, Hkl, norm_Hkl, resid):
        """
        """
        # TODO reconsider transpose
        return np.maximum(np.dot(-resid, Hkl) / (norm_Hkl**2 + EPSILON),
                          FACTOR_MIN)

    """
    H update
    """

    def setup_H_update(self):
        self.W_norms = la.norm(self.W, axis=1).T  # K * L, norm along N

    def update_H(self):
        K = self.W.shape[2]
        T = self.H.shape[1]

        for k in range(K):  # Update each component
            for t in range(T):  # Update each timebin
                self.update_H_entry(k, t)

    def update_H_entry(self, k, t):
        """
        Update a single entry of `H`.
        """
        L, N, K = self.W.shape
        T = self.H.shape[1]

        # Collect cached data
        Wk = self.W[:, :, k].T
        # TODO is this valid?
        # norm_Wkt = np.sqrt(np.sum(W_norms[k, :T-t]**2))
        norm_Wkt = la.norm(self.W_norms[k, :T-t])

        # Remove factor from residual
        remainder = self.resids[:, t:t+L] - self.H[k, t] * Wk[:, :T-t]

        # Update
        self.H[k, t] = self.next_H_entry(Wk[:, :T-t], norm_Wkt, remainder)

        # Add factor back to residual
        self.resids[:, t:t+L] = remainder + self.H[k, t] * Wk[:, :T-t]

    def next_H_entry(self, Wkt, norm_Wkt, remainder):
        """
        """
        trace = np.dot(np.ravel(Wkt), np.ravel(-remainder))

        return np.maximum(trace / (norm_Wkt**2 + EPSILON), FACTOR_MIN)


"""
"""


class HALSUpdate(SimpleHALSUpdate):
    """
    Advanced version of HALS update, updating T/L entries of `H` at a time.
    """

    def __init__(self, data, dims, patience=3, tol=1e-5, max_inner=0,
                 weightW=1, weightH=0, inner_thresh=0, **kwargs):
        super().__init__(data, dims, patience, tol,
                         max_inner=max_inner, weightW=weightW, weightH=weightH,
                         inner_thresh=inner_thresh, **kwargs)

        # Set up batches
        self.batch_inds = []
        self.batch_sizes = []
        for k in range(self.n_components):
            self.batch_sizes.append([])
            self.batch_inds.append([])
            for l in range(self.maxlag):
                batch = range(l, self.n_timepoints-self.maxlag, self.maxlag)
                self.batch_inds[k].append(batch)
                self.batch_sizes[k].append(len(batch))

    def setup_H_update(self):
        L, N, K = self.W.shape

        # Set up norms and cloned tensors
        self.W_norms = la.norm(self.W, axis=1).T  # K * L, norm along N
        self.W_raveled = []
        self.W_clones = []
        for k in range(K):
            self.W_raveled.append(self.W[:, :, k].ravel())
            self.W_clones.append([])
            for l in range(L):
                self.W_clones[k].append(self.clone_Wk(self.W_raveled[k],
                                                      k, l))

    def update_H(self):
        L, N, K = self.W.shape
        T = self.H.shape[1]

        for k in range(K):  # Update each component
            for l in range(L):  # Update each lag
                self.update_H_batch(k, l)  # Update batch
                self.update_H_entry(k, T-L+l)   # Update the last entry

    def update_H_batch(self, k, l):
        L, N, K = self.W.shape

        # Collect cached data
        Wk = self.W_raveled[k]
        Wk_clones = self.W_clones[k][l]
        batch_ind = self.batch_inds[k][l]
        n_batch = self.batch_sizes[k][l]
        norm_Wk = la.norm(self.W_norms[k, :])

        # Set up batch
        batch = self.H[k, batch_ind]
        end_batch = l + L*n_batch

        # Create residual tensor and factor tensor
        resid_tens = self.fold_resids(l, n_batch)
        factors_tens = self.fold_factor(Wk, batch)

        # Generate remainder (residual - factor) tensor and remove factor
        # contribution from residual
        remainder = resid_tens - factors_tens

        # Subtract out factor from residual
        self.resids[:, l:end_batch] -= self.unfold_factor(
            factors_tens, n_batch)

        # Update H
        self.H[k, batch_ind] = self.next_H_batch(Wk_clones,
                                                 norm_Wk,
                                                 remainder)

        # Add factor contribution back to residual
        updated_batch = self.H[k, batch_ind]
        new_factors_tens = self.fold_factor(Wk, updated_batch)
        self.resids[:, l:end_batch] += self.unfold_factor(
            new_factors_tens, n_batch)

    def next_H_batch(self, Wk_clones, norm_Wk, remainder):
        traces = np.inner(Wk_clones, -remainder)[0]
        return np.maximum(np.divide(traces, norm_Wk**2 + EPSILON), FACTOR_MIN)

    def fold_resids(self, start, n_batch):
        """
        Select the appropriate part of the residual matrix, and fold into
        a tensor.
        """
        cropped = self.resids[:, start:(start + self.maxlag * n_batch)]
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

    def clone_Wk(self, Wk, k, l):
        """
        Clone Wk several times and place into a tensor.
        """
        n_batch = self.batch_sizes[k][l]
        return np.outer(np.ones(n_batch), Wk)
