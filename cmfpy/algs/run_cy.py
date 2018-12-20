import pyximport
pyximport.install()

import numpy as np
import numpy.linalg as la
from cmfpy.common import shift_and_stack
import cy_hals


def setup_W_update(W, H):
    L, N, K = W.shape

    H_unfold = shift_and_stack(H, L)  # Unfold matrices
    H_norms = la.norm(H_unfold, axis=1)  # Compute norms

    return H_unfold, H_norms


N, T, L, K = 100, 250, 10, 3
W = np.random.randn(L, N, K)
H = np.random.randn(K, T)
resids = np.random.randn(N, T)

H_unfold, H_norms = setup_W_update(W, H)


print(
    cy_hals._update_W(W, H_unfold, H_norms, resids)
)
