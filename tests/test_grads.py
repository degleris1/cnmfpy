import pytest

import numpy as np
import numpy.linalg as la
from numpy.testing import assert_allclose

from cmfpy import CMF
from cmfpy.initialize import init_rand
from cmfpy.algs.gradient_descent import GradDescent, BlockDescent
from cmfpy.datasets import Synthetic

from scipy.optimize import approx_fprime

# Test parameters and tolerances.
TOL = 1e-4
EPS = 1e-7
SEED = 1234
N, T = 10, 200
DATA = Synthetic(
    n_features=N,
    n_timebins=T,
    sparsity=.2,
).generate()


@pytest.mark.parametrize("algclass", [GradDescent, BlockDescent])
@pytest.mark.parametrize("L", [5, 10])
@pytest.mark.parametrize("K", [1, 5])
def test_gradients(algclass, L, K):

    # Initialize parameters.
    rs = np.random.RandomState(SEED)
    W = rs.rand(L, N, K)
    H = rs.rand(K, T)

    # Initialize algorithm.
    alg = algclass(DATA, W.copy(), H.copy())

    # Computed gradients.
    gW = alg.gW.copy()
    gH = alg.gH.copy()

    # Do a gradient check by finite differencing.
    def loss_W(w_vec):
        _W = w_vec.reshape((L, N, K))
        _alg = algclass(DATA, _W, H.copy())
        return _alg.unnormalized_loss

    def loss_H(h_vec):
        _H = h_vec.reshape((K, T))
        _alg = algclass(DATA, W.copy(), _H)
        return _alg.unnormalized_loss

    # Compute approximate gradients.
    approx_gW = approx_fprime(W.ravel(), loss_W, EPS)
    approx_gH = approx_fprime(H.ravel(), loss_H, EPS)

    # Check for agreement.
    assert_allclose(approx_gW, gW.ravel(), rtol=TOL)
    assert_allclose(approx_gH, gH.ravel(), rtol=TOL)
