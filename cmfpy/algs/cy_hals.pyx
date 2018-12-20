import numpy as np
import numpy.linalg as la

EPSILON = 1e-6
FACTOR_MIN = 0


cpdef void _update_W(double[:, :, :] W,
                     double [:, :] H_unfold,
                     double [:] H_norms,
                     double [:, :] resids):
    cdef int L = W.shape[0]
    cdef int K = W.shape[2]
    for k in range(K):
        for l in range(L):
            _update_W_col(k, l, W, H_unfold, H_norms, resids)


cpdef void _update_W_col(int k, int l, 
                         double [:, :, :] W,
                         double[:, :] H_unfold,
                         double[:] H_norms,
                         double[:, :] resids):
    cdef int N = W.shape[1]
    cdef int K = W.shape[2]
    cdef int T = resids.shape[1]
    cdef int ind = l*K + k

    cdef double [:, :] factor = np.outer(W[l, :, k], H_unfold[ind, :])
    with nogil:
        for n in range(N):
            for t in range(T):
                resids[n, t] -= factor[n, t]

    W[l, :, k] =  _next_W_col(H_unfold[ind, :],
                              H_norms[ind],
                              resids)

    factor = np.outer(W[l, :, k], H_unfold[ind, :])
    with nogil:
        for n in range(N):
            for t in range(T):
                resids[n, t] += factor[n, t]


cpdef double [:] _next_W_col(double[:] Hkl,
                            double norm_Hkl,
                            double[:, :] resid):
    return np.maximum(-np.dot(resid, Hkl) / (norm_Hkl**2 + EPSILON),
                      FACTOR_MIN)


# cdef void _next_W_col_fast(double[:] Hkl,
#                            double norm_Hkl,
#                            double[:, :] resid,
#                            double[:] result):
#     cdef int N = resid.shape[0]
#     cdef int T = resid.shape[1]
#     cdef double dot
    
#     for n in range(N):
#         dot = 0
#         with nogil:
#             for t in range(T):
#                 dot += resid[n, t] * Hkl[t]
#         result[n] = max(-dot / (norm_Hkl**2 + EPSILON), FACTOR_MIN)


"""
H update
"""


def _update_H(W, H, resids, W_norms, W_raveled, W_clones,
              batch_inds, batch_sizes):
    L, N, K = W.shape
    T = H.shape[1]

    for k in range(K):  # Update each component
        for l in range(L):  # Update each lag
            norm_Wk = la.norm(W_norms[k, :])
            _update_H_batch(k, l, W, H, resids,
                            W_raveled[k], W_clones[k][l],
                            batch_inds[k][l], batch_sizes[k][l], norm_Wk)
            _update_H_entry(k, T-L+l, W, H, resids, W_norms)


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


def _next_H_entry(Wkt, norm_Wkt, remainder):
    trace = np.dot(np.ravel(Wkt), np.ravel(-remainder))
    return np.maximum(trace / (norm_Wkt**2 + EPSILON), FACTOR_MIN)


def _unfold_factor(factors_tens, n_batch, L, N):
        """
        Expand the factor tensor into a matrix.
        """
        return factors_tens.reshape(L*n_batch, N).T


def _fold_factor(Wk, batch):
    """
    Generate factor prediction for a given component and lag. Then fold
    into a tensor.
    """
    return np.outer(batch, Wk)


def _fold_resids(start, n_batch, resids, L, N):
    """
    Select the appropriate part of the residual matrix, and fold into
    a tensor.
    """
    cropped = resids[:, start:(start + L * n_batch)]
    return cropped.T.reshape(n_batch, L*N)


def _next_H_batch(Wk_clones, norm_Wk, remainder):
        traces = np.inner(Wk_clones, -remainder)[0]
        return np.maximum(np.divide(traces, norm_Wk**2 + EPSILON), FACTOR_MIN)


def _clone_Wk(Wk, k, l, batch_sizes):
    """
    Clone Wk several times and place into a tensor.
    """
    n_batch = batch_sizes[k][l]
    return np.outer(np.ones(n_batch), Wk)
