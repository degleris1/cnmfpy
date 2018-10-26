import pytest
import numpy as np
from cmfpy.utils import ModelDimensions
from cmfpy.optimize import compute_gH, compute_gW, compute_loss
from cmfpy.datasets import Synthetic, SongbirdHVC
from scipy.optimize import check_grad, approx_fprime

TOL = 1e-4
SEED = 1234
# DATASETS = [Data().generate() for Data in (Synthetic, SongbirdHVC)]
DATASETS = [Synthetic(sparsity=.2).generate()]


# TODO - add this to the codebase and import it instead.
def init_rand(rs, dims, scale):
    W = rs.rand(dims.n_lags, dims.n_features, dims.n_components)
    H = rs.rand(dims.n_components, dims.n_timepoints)
    return W * scale, H * scale


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("L", [5, 10])
@pytest.mark.parametrize("K", [1, 5])
def test_gradients(data, L, K):

    # Get model dimensions.
    dims = ModelDimensions(data=data, n_lags=L, n_components=K)

    # Initialize model parameters.
    rs = np.random.RandomState(SEED)
    W, H = init_rand(rs, dims, np.max(data))

    # Check gradient for W. Thinly wrap internal functions to deal with
    # raveled parameter vectors.
    def loss_W(w_vec):
        w = w_vec.reshape(W.shape)
        return compute_loss(data, w, H)

    def grad_W(w_vec):
        w = w_vec.reshape(W.shape)
        return compute_gW(data, w, H).ravel()

    eps = np.sqrt(np.finfo(float).eps)

    approx_grad = approx_fprime(W.ravel(), loss_W, eps)
    approx_grad /= np.linalg.norm(approx_grad)

    grad = compute_gW(data, W, H).ravel()
    grad /= np.linalg.norm(grad)

    assert np.linalg.norm(grad - approx_grad) < TOL, "W failed for " + str(eps)

    # err = check_grad(f_W, g_W, W.ravel())
    # assert err < (TOL * np.sqrt(W.size))

    # Check gradient for H.
    def loss_H(h_vec):
        h = h_vec.reshape(H.shape)
        return compute_loss(data, W, h)

    def grad_H(h_vec):
        h = h_vec.reshape(H.shape)
        return compute_gW(data, W, h).ravel()

    approx_grad = approx_fprime(H.ravel(), loss_H, eps)
    approx_grad /= np.linalg.norm(approx_grad)

    grad = compute_gH(data, W, H).ravel()
    grad /= np.linalg.norm(grad)

    assert np.linalg.norm(grad - approx_grad) < TOL, "H failed"

    # err = check_grad(f_H, g_H, H.ravel())
    # assert err < (TOL * np.sqrt(H.size))
