import numpy as np
import time
from tqdm import trange

from ..optimize import renormalize

from .mult import MultUpdate
from .chals import CHALSUpdate, FastCHALSUpdate


ALGORITHMS = {'mult': MultUpdate,
              'chals': FastCHALSUpdate}


def fit_alg(data, model, UpdateRule):
    m, n = data.shape

    # Initial loss
    model.loss_hist = [model.score(data)]
    model.time_hist = [0.0]
    t0 = time.time()

    update_rule = UpdateRule(data, model)

    itr = 0
    for itr in trange(model.n_iter_max):

        if (np.isnan(model.W).any()):
            raise Exception('W has Nans!!')

        if (np.isnan(model.H).any()):
            raise Exception('H has NANs!!')

        update_rule.step()

        model.loss_hist.append(model.score(data))
        model.time_hist.append(time.time() - t0)

        # renormalize H
        model.W, model.H = renormalize(model.W, model.H)

        # check convergence
        prev_loss, new_loss = model.loss_hist[-2:]
        if (np.abs(prev_loss - new_loss) < model.tol):
            break
