import numpy as np
import numpy.linalg as la

from ..optimize import compute_gW, compute_gH, compute_loss
from ..regularize import compute_scfo_reg

"""
TODO add backtracking back in
TODO add regularizers back in
"""


def bcd_step(data, model):
    step_size = 1 / la.norm(data)

    # assert model.W.sum() > 0.1, "initialization test " + str(model.W.sum())

    W_grad = compute_gW(data, model.W, model.H)
    model.W = np.maximum(model.W - step_size * W_grad, 0)

    # print("W", np.sum(np.abs(W_grad)), model.W.sum())

    # assert model.W.sum() > 0.1, str(model.W.sum())

    H_grad = compute_gH(data, model.W, model.H)
    model.H -= np.maximum(model.H - step_size * H_grad, 0)

    # print("H", np.sum(np.abs(H_grad)), model.H.sum())


def _scale_gW(data, grad_W, model, step_type='constant'):
    if (step_type == 'backtrack'):
        step_W = _backtrack(data, grad_W, 0, model)

    elif (step_type == 'constant'):
        step_W = 1e-3

    else:
        raise ValueError('Invalid BCD step type.')

    return step_W


def _scale_gH(data, grad_H, model, step_type='backtrack'):
    if (step_type == 'backtrack'):
        step_H = _backtrack(data, 0, grad_H, model)

    elif (step_type == 'constant'):
        step_H = 1e-3

    else:
        raise ValueError('Invalid BCD step type.')

    return step_H


def _backtrack(data, grad_W, grad_H, model, beta=0.8, alpha=0.00001,
               max_iters=500):
    """
    Backtracking line search to find a step length.
    """
    # compute initial loss and gradient magnitude
    past_loss = compute_loss(data, model.W, model.H)
    if (model.l2_scfo != 0):  # regularizer
        past_loss += model.l2_scfo * compute_scfo_reg(data, model.W, model.H,
                                                      model._kernel)

    grad_mag = la.norm(grad_W)**2 + la.norm(grad_H)**2

    new_loss = past_loss
    t = 1.0
    iters = 0
    new_H = model.H.copy()
    # backtracking line search
    while ((new_loss > past_loss - alpha*t*grad_mag) and (iters < max_iters)):
        t = beta * t

        new_H = np.maximum(model.H - t*grad_H, 0)
        new_W = np.maximum(model.W - t*grad_W, 0)
        new_loss = compute_loss(data, new_W, new_H)
        if (model.l2_scfo != 0):  # regularizer
            new_loss += model.l2_scfo * compute_scfo_reg(data, new_W, new_H,
                                                         model._kernel)

        iters += 1

    return t


# TODO: compute the lipschitz constant for optimal learning rate
