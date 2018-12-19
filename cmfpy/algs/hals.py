import numpy as np
import numpy.linalg as la
import numba

from .accelerated import AcceleratedOptimizer
from ..common import shift_and_stack, EPSILON, FACTOR_MIN


class SimpleHALSUpdate(AcceleratedOptimizer):
    """
    HALS update rule.
    """

    def __init__(self, data, dims, patience=3, tol=1e-5,
                 max_iter=1, weightW=1, weightH=1, stop_thresh=0, **kwargs):
        super().__init__(data, dims, patience=patience, tol=tol,
                         max_iter=max_iter, weightW=weightW, weightH=weightH,
                         stop_thresh=stop_thresh, **kwargs)

    """
    W update
    """

    def setup_W_update(self):
        L, N, K = self.W.shape

        self.H_unfold = shift_and_stack(self.H, L)  # Unfold matrices
        self.H_norms = la.norm(self.H_unfold, axis=1)  # Compute norms

    def update_W(self):
        _update_W(self.W, self.H_unfold, self.H_norms, self.resids)

    """
    H update
    """

    def setup_H_update(self):
        self.W_norms = la.norm(self.W, axis=1).T  # K * L, norm along N

    def update_H(self):
        raise NotImplementedError("Must use full HALS optimzer.")


class HALSUpdate(SimpleHALSUpdate):
    """
    Advanced version of HALS update, updating T/L entries of `H` at a time.
    """

    def __init__(self, data, dims, patience=3, tol=1e-5, max_iter=1,
                 weightW=1, weightH=1, stop_thresh=0, **kwargs):
        super().__init__(data, dims, patience, tol,
                         max_iter=max_iter, weightW=weightW, weightH=weightH,
                         stop_thresh=stop_thresh, **kwargs)

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
                self.W_clones[k].append(_clone_Wk(self.W_raveled[k],
                                                  k, l, self.batch_sizes))

    def update_H(self):
        _update_H(self.W, self.H, self.resids,
                  self.W_norms, self.W_raveled, self.W_clones,
                  self.batch_inds, self.batch_sizes)


"""
Internal methods
"""


@numba.jit(nopython=True)
def _update_W(W, H_unfold, H_norms, resids):
    L, N, K = W.shape
    for k in range(K):
        for l in range(L):
            _update_W_col(k, l, W, H_unfold, H_norms, resids)


@numba.jit(nopython=True)
def _update_W_col(k, l, W, H_unfold, H_norms, resids):
    L, N, K = W.shape
    ind = l*K + k

    resids -= np.outer(W[l, :, k], H_unfold[ind, :])
    W[l, :, k] = _next_W_col(H_unfold[ind, :], H_norms[ind], resids)
    resids += np.outer(W[l, :, k], H_unfold[ind, :])


@numba.jit(nopython=True)
def _next_W_col(Hkl, norm_Hkl, resid):
    """
    """
    # TODO reconsider transpose
    return np.maximum(np.dot(-resid, Hkl) / (norm_Hkl**2 + EPSILON),
                      FACTOR_MIN)


def _clone_Wk(Wk, k, l, batch_sizes):
    """
    Clone Wk several times and place into a tensor.
    """
    n_batch = batch_sizes[k][l]
    return np.outer(np.ones(n_batch), Wk)


@numba.jit()
def _update_H(W, H, resids, W_norms, W_raveled, W_clones,
              batch_inds, batch_sizes):
    L, N, K = W.shape
    T = H.shape[1]

    for k in range(K):  # Update each component
        for l in range(L):  # Update each lag
            _update_H_batch(k, l, W, H, resids,
                            W_raveled[k], W_clones[k][l],
                            batch_inds[k][l], batch_sizes[k][l],
                            la.norm(W_norms[k, :]))
            _update_H_entry(k, T-L+l, W, H, resids, W_norms)


@numba.jit()
def _update_H_batch(k, l, W, H, resids, Wk, Wk_clones, batch_ind, n_batch,
                    norm_Wk):
    L, N, K = W.shape

    # Set up batch
    batch = H[k, batch_ind]
    end_batch = l + L*n_batch

    # Create residual tensor and factor tensor
    resid_tens = _fold_resids(l, n_batch, resids, L, N)
    factors_tens = _fold_factor(Wk, batch)

    # Generate remainder (residual - factor) tensor and remove factor
    # contribution from residual
    remainder = resid_tens - factors_tens

    # Subtract out factor from residual
    resids[:, l:end_batch] -= _unfold_factor(factors_tens, n_batch, L, N)

    # Update H
    H[k, batch_ind] = _next_H_batch(Wk_clones, norm_Wk, remainder)

    # Add factor contribution back to residual
    updated_batch = H[k, batch_ind]
    new_factors_tens = _fold_factor(Wk, updated_batch)
    resids[:, l:end_batch] += _unfold_factor(new_factors_tens, n_batch, L, N)


@numba.jit(nopython=True)
def _update_H_entry(k, t, W, H, resids, W_norms):
        """
        Update a single entry of `H`.
        """
        L, N, K = W.shape
        T = H.shape[1]

        # Collect cached data
        Wk = W[:, :, k].T
        # TODO is this valid?
        # norm_Wkt = np.sqrt(np.sum(W_norms[k, :T-t]**2))
        norm_Wkt = la.norm(W_norms[k, :T-t])

        # Remove factor from residual
        remainder = resids[:, t:t+L] - H[k, t] * Wk[:, :T-t]

        # Update
        H[k, t] = _next_H_entry(Wk[:, :T-t], norm_Wkt, remainder)

        # Add factor back to residual
        resids[:, t:t+L] = remainder + H[k, t] * Wk[:, :T-t]


@numba.jit(nopython=True)
def _next_H_entry(Wkt, norm_Wkt, remainder):
    trace = np.dot(np.ravel(Wkt), np.ravel(-remainder))
    return np.maximum(trace / (norm_Wkt**2 + EPSILON), FACTOR_MIN)


@numba.jit(nopython=True)
def _unfold_factor(factors_tens, n_batch, L, N):
        """
        Expand the factor tensor into a matrix.
        """
        return factors_tens.reshape(L*n_batch, N).T


@numba.jit(nopython=True)
def _fold_factor(Wk, batch):
    """
    Generate factor prediction for a given component and lag. Then fold
    into a tensor.
    """
    return np.outer(batch, Wk)


@numba.jit()
def _fold_resids(start, n_batch, resids, L, N):
    """
    Select the appropriate part of the residual matrix, and fold into
    a tensor.
    """
    cropped = resids[:, start:(start + L * n_batch)]
    return cropped.T.reshape(n_batch, L*N)


@numba.jit()
def _next_H_batch(Wk_clones, norm_Wk, remainder):
        traces = np.inner(Wk_clones, -remainder)[0]
        return np.maximum(np.divide(traces, norm_Wk**2 + EPSILON), FACTOR_MIN)
