"""
Initialization procedures for CMF.
"""

import numpy as np
import numpy.random as npr
from .common import cmf_predict


def init_rand(model, data, random_state=None, rescale=True):
    """
    Random initialization with appropriate scaling.
    """

    if isinstance(random_state, npr.RandomState):
        rs = random_state
    else:
        rs = np.RandomState(random_state)

    n_features, n_time = data.shape

    W = rs.rand(model.maxlag, n_features, model.n_components)
    H = rs.rand(model.n_components, n_time)

    if rescale:
        # TODO add brief note/reference here.
        est = cmf_predict(W, H)
        alpha = np.dot(data.ravel(), est.ravel()) / np.linalg.norm(est)**2
        W *= alpha
        H *= alpha

    return W, H
