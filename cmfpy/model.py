"""
A Python implementation of CNMF.

Written by Alex Williams and Anthony Degleris.
"""
import numpy as np
import numpy.random as npr
import numpy.linalg as la

import time
from tqdm import trange

from .algs import ALGORITHMS
from . import initialize

from .common import cmf_predict

NOT_FITTED_ERROR = ValueError(
    "This CMF instance is not fitted yet. Call 'fit' with appropriate"
    "arguments before using this method."
)


class CMF(object):
    """
    Convolutive Matrix Factorization (CNMF)

    Factors a matrix into a convolution between a tensor `W` and a
    matrix `H`.
    """

    def __init__(self, n_components, maxlag, n_iter_max=100,
                 l1_W=0.0, l1_H=0.0, verbose=True, alg_name='mult',
                 **alg_opts):
        """
        Parameters
        ----------
        n_components : int
            Number of components to fit.
        maxlag : int
            Maximum time lag in each sequence. A single sequence can lag up to
            `maxlag` entries right of the first time point.
        tol : float, optional
            Tolerance for convergence. If the change in cost is less than the
            `tol`, the algorithm will terminate early.
        n_iter_max : int, optional
            Maximum number of iterations during algorithm fitting.
        l2_scfo : float, optional
            Weight of the soft cross-factors orthogonality regularizer. See
            references for details.
        l1_W : float, optional
            Weight of the L1 regularizer for the entries of `W`.
        l1_H : float, optional
            Weight of the L1 regularizer for the entries of `H`.
        alg_name : str
            String specifying algorithm.
        **alg_opts : dict
            Additional keyword arguments are passed as algorithm-specific
            options when, such as learning rate and convergence tolerance.
            See algorithm classes for details.
        """
        self.n_components = n_components
        self.maxlag = maxlag

        self.n_iter_max = n_iter_max

        self.l1_W = l1_W
        self.l1_H = l1_H

        self.alg_name = alg_name
        self.alg_opts = alg_opts
        self.verbose = verbose

    def fit(self, data):
        """
        Fits model to the data.

        Parameters
        ----------
        data : array-like, shape (n_neurons, n_time)
            Training data to fit.

        Returns
        -------
        self : object
            Returns the model instance.
        """

        # Check input
        if (data < 0).any():
            raise ValueError('Negative values in data to fit')

        # Initialize W and H.
        n_features, n_time = data.shape

        self._W = npr.rand(self.maxlag, n_features, self.n_components)
        self._H = npr.rand(self.n_components, n_time)

        est = self.predict()
        alpha = (data * est).sum() / la.norm(est)**2

        # Create algorithm helper class.
        alg_class = ALGORITHMS[self.alg_name]
        algorithm = alg_class(
            data, alpha * self._W, alpha * self._H, **self.alg_opts)

        # Set up optimization tracking.
        self.loss_hist = [algorithm.loss]
        self.time_hist = [0.0]
        if self.verbose:
            iterations = trange(self.n_iter_max)
        else:
            iterations = range(self.n_iter_max)

        # Run optimization.
        for itr in iterations:

            # Time each parameter update.
            t0 = time.time()

            # Update model parameters.
            loss = algorithm.update()

            # Record time of parameter update.
            dur = time.time() - t0
            self.time_hist.append(self.time_hist[-1] + dur)
            self.loss_hist.append(loss)

            # Check convergence.
            if algorithm.converged(self.loss_hist):
                break

        # Extract model parameters from algorithm class.
        self._W = algorithm.W
        self._H = algorithm.H

    def sort_components(self):
        """
        Sorts model components by explanatory power.
        """

        # Compute explanatory power of each factor.
        loadings = compute_loadings(data, self.motifs, self.factors)

        # Sort components by power
        ind = np.argsort(loadings)
        self._W = self._W[:, :, ind]  # motifs
        self._H = self._H[ind, :]     # factors

    def predict(self):
        """
        Computes low-rank reconstruction of data.

        Returns
        -------
        est : array-like, shape (n_time, n_features)
            Model estimate of the data.
        """
        return cmf_predict(self.motifs, self.factors)

    def score(self, data):
        """
        Return the R^2 score, defined as the one minus the squared norm of the
        error divided by the square norm of the data.

        Parameters
        ----------
        data : array-like, shape (n_time, n_features)
            Data to fit when scoring.

        Returns
        -------
        score : float
            A score for how well the model fits the data. A score of 1 is
            perfect.

        TODO: demean the data
        """
        error = self.predict() - data
        return 1 - (la.norm(error)**2 / la.norm(data)**2)

    @property
    def motifs(self):
        """Returns `W` (lags x features x components) parameters."""
        try:
            return self._W
        except AttributeError:
            raise NOT_FITTED_ERROR

    @property
    def factors(self):
        """Returns `H` (components x timebins) parameters."""
        try:
            return self._H
        except AttributeError:
            raise NOT_FITTED_ERROR


def compute_loadings(data, W, H):
    """
    Compute the power explained by each factor.
    """
    loadings = []
    K, T = H.shape

    data_mag = norm(data)

    for i in range(K):
        Wi = W[:, :, i:i+1]
        Hi = H[i:i+1, :]
        est = tensor_conv(Wi, Hi)
        loadings += [norm(est - data) / (data_mag + EPSILON)]

    return loadings


def renormalize(W, H):
    """
    Renormalizes the rows of H to have constant energy.
    Updates passed parameters.
    """
    # TODO choice of norm??
    L, N, K = W.shape

    row_norms = norm(H, axis=1) + EPSILON

    H = np.diag(np.divide(1, row_norms)).dot(H)

    for l in range(L):
        W[l] = W[l].dot(np.diag(row_norms))

    return W, H


def shift_factors(W, H):
    """
    UNDER CONSTRUCTION: do not use

    Shift factors by their center of mass.
    """
    L, N, K = W.shape

    if (L == 1):
        raise IndexError('No room to shift. Disable shifting.')

    center = int(np.floor(L / 2))

    # TODO broadcast
    shifted_H = np.zeros(H.shift(0).shape)
    for k in range(K):
        masses = np.sum(W[:, k, :], axis=1)

        if (np.sum(masses) > EPSILON):
            ind = np.arange(0, L)
            cmass = int(np.floor(np.dot(masses, ind) / np.sum(masses)))+1
            loc = center - cmass

            Wpad = np.pad(W[:, :, k], ((L, L), (0, 0)), mode='constant')
            W[:, :, k] = Wpad[L - loc: 2*L - loc, :]

            shifted_H[k, :] = shift_cols(H, -loc)[k, :]

    H.assign(shifted_H)

    return W, H
