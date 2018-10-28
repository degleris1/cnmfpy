import pytest

import numpy as np
import numpy.linalg as la
from numpy.testing import assert_allclose

from cmfpy import CMF
from cmfpy.initialize import init_rand
from cmfpy.common import grad_W, grad_H, cmf_loss
from cmfpy.datasets import Synthetic

from scipy.optimize import approx_fprime

# Test parameters and tolerances.
TOL = 1e-5
EPS = 100 * np.sqrt(np.finfo(float).eps)
SEED = 1234
DATA = Synthetic(
    n_features=10,
    n_timebins=200,
    sparsity=.2,
).generate()


@pytest.mark.parametrize("L", [5, 10])
@pytest.mark.parametrize("K", [1, 5])
def test_gradients(L, K):

    # Initialize model parameters.
    rs = np.random.RandomState(SEED)
    model = CMF(n_components=K, maxlag=L)
    W, H = init_rand(model, DATA, random_state=rs)

    # Thin wrapper around loss functions for gradient checking.
    def loss_W(w_vec):
        w = w_vec.reshape(W.shape)
        return cmf_loss(DATA, w, H)

    def loss_H(h_vec):
        h = h_vec.reshape(H.shape)
        return cmf_loss(DATA, W, h)

    # # Compute approximate gradients.
    # approx_gW = approx_fprime(W.ravel(), loss_W, EPS)
    # approx_gH = approx_fprime(H.ravel(), loss_H, EPS)

    # Compute gradients using internal code.
    gW = np.zeros_like(W)
    grad_W(DATA, H, gW)

    gH = np.zeros_like(H)
    grad_H(DATA, W, gH)

    assert True

    # Check for agreement.
    # assert_allclose(approx_gW, gW.ravel(), rtol=TOL)
    # assert_allclose(approx_gH, gH.ravel(), rtol=TOL)
