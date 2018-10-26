import numpy as np  # Linear algebra
import numpy.linalg as la

from ..conv import tensor_conv, shift_and_stack, hunfold, pad_shift_cols

# TODO make EPSILON universal
EPSILON = np.finfo(np.float).eps


def chals_step(data, model):
    update_W(model.W, model.H, data)

    assert model.score(data) >= model.loss_hist[-1] - EPSILON, "W prob " + \
                                str(model.score(data)) + " " + \
                                str(model.loss_hist[-1])

    update_H(model.W, model.H, data)

    assert model.score(data) >= model.loss_hist[-1] - EPSILON, "H prob"


def update_W(W, H, X):
    L, N, K = W.shape

    # Unfold matrices
    H_unfold = shift_and_stack(H, L)

    # Set up residual and norms
    resid = X - tensor_conv(W, H)
    H_norms = la.norm(H_unfold, axis=1)

    for k in range(K):
        for l in range(L):
            ind = l*K + k

            assert np.array_equal(H_unfold[ind, :], pad_shift_cols(H, l)[k])

            # DEBUG
            r1 = la.norm(resid)

            resid += np.outer(W[l, :, k], H_unfold[ind, :])
            W[l, :, k] = new_W_col(H_unfold[ind, :], H_norms[ind], resid)
            resid -= np.outer(W[l, :, k], H_unfold[ind, :])

            # DEBUG
            r2 = la.norm(resid)
            assert r2 <= r1


def update_H(W, H, X):
    L, N, K = W.shape
    T = H.shape[1]

    # Set up residual and norms of each column
    W_norms = la.norm(W, axis=1).T  # K * L, norm along N

    resid = X - tensor_conv(W, H)

    # Update each component
    for k in range(K):
        Wk = W[:, :, k].T  # TODO: will this slow things down?

        # Update each timebin
        for t in range(T):
            # DEBUG
            r1 = la.norm(resid)

            resid_slice = resid[:, t:t+L] + H[k, t] * Wk[:, :T-t]
            norm_Wkt = np.sqrt(np.sum(W_norms[k, :T-t]**2))

            H[k, t] = new_H_entry(Wk[:, :T-t], norm_Wkt, resid_slice)
            resid[:, t:t+L] = resid_slice - H[k, t] * Wk[:, :T-t]

            # DEBUG
            r2 = la.norm(resid)
            assert r2 <= r1, "update H fails" + str(r2) + " " + str(r1)


def new_W_col(Hkl, norm_Hkl, resid):
    """
    """
    # TODO reconsider transpose
    return np.maximum(np.dot(resid, Hkl) / (norm_Hkl**2 + EPSILON), 0)


def new_H_entry(Wkt, norm_Wkt, resid_slice):
    """
    """
    trace = np.dot(np.ravel(Wkt), np.ravel(resid_slice))

    return np.maximum(trace / (norm_Wkt**2 + EPSILON), 0)


"""
DEPRECATED FUNCTIONS BELOW
"""


def update_H_component(W, H, X, k):
    """
    """
    L, N, K = W.shape
    T = H.shape[1]

    # Set up Wk and residual
    Wk = W[:, :, k].T  # TODO: will this slow things down?
    norm_Wk = la.norm(Wk)
    resid = X - tensor_conv(W, H)

    # Update each entry in the row, from left to right
    for t in range(T):
        resid_slice = resid[:, t:t+L] + H[k, t] * Wk[:, :T-t]
        H[k, t] = new_H_entry(Wk[:, :T-t], norm_Wk, resid_slice)
        resid[:, t:t+L] = resid_slice - H[k, t] * Wk[:, :T-t]


def update_W_component(W, H, X, k):
    """
    """
    L, N, K = W.shape

    W_unfold = hunfold(W)
    H_unfold = shift_and_stack(H, L)

    for l in range(L):
        ind = k*L + l
        W[l, :, k] = W_col_update(W_unfold, H_unfold, X, ind)


def W_col_update(W_unfold, H_unfold, X, ind):
    """
    Updates `W[l, :, k]` using the HALS update rule
    """

    W_col = np.dot(X, H_unfold[ind, :].T)
    for j in range(W_unfold.shape[1]):
        if (j != ind):
            W_col -= W_unfold[:, j] * np.dot(H_unfold[j, :], H_unfold[ind, :])

    W_col /= (la.norm(H_unfold[ind, :])**2 + EPSILON)

    return np.maximum(0, W_col)
