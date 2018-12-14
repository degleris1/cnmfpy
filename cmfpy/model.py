"""
A Python implementation of CNMF.
"""
import numpy as np
import numpy.random as npr
import numpy.linalg as la

import time
from tqdm import trange

from .algs import ALGORITHMS
from . import initialize
from . import visualize

from .common import cmf_predict

NOT_FITTED_ERROR = ValueError(
    "This CMF instance is not fitted yet. Call 'fit' with appropriate"
    "arguments before using this method."
)


class ModelDimensions:
    """Holds dimensions of CMF model."""

    def __init__(self, data=None, n_features=None, n_timepoints=None,
                 maxlag=None, n_components=None):
        """
        Parameters
        ----------
            data : ndarray
                Matrix holding time series, shape: (n_features, n_timepoints).
                If not specified, must provide both `n_features` and
                `n_timepoints`.
            n_features : int
                Number of features/measurements recorded in time series.
                Must be specified if `data` is not given.
            n_timepoints : int
                Length of time series. Must be specified if `data` is not
                given.
            maxlag : int
                Number of time steps in each temporal motifs/sequence. Must be
                specified.
            n_components : int
                Number of temporal motifs/sequences. Must be specified.
        """

        if data is None:
            if (n_features is None) or (n_timepoints is None):
                raise ValueError("Must either specify 'data' or ('n_features' "
                                 "and 'n_timepoints').")
        else:
            n_features, n_timepoints = data.shape

        if maxlag is None:
            raise ValueError("Must specify 'n_lags'.")

        if n_components is None:
            raise ValueError("Must specify 'n_components'.")

        self.n_features = n_features
        self.n_timepoints = n_timepoints
        self.maxlag = maxlag
        self.n_components = n_components

    def __iter__(self):
        """Iterator over model dimensions."""
        yield from self.__dict__.items()


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

        # Determine dimensions of model.
        dims = ModelDimensions(
            data, maxlag=self.maxlag, n_components=self.n_components)

        # Initialize algorithm class.
        algorithm = ALGORITHMS[self.alg_name](data, dims, **self.alg_opts)

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
    def n_features(self):
        return self.motifs.shape[1]

    @property
    def n_timesteps(self):
        return self.factors.shape[1]

    @property
    def factors(self):
        """Returns `H` (components x timebins) parameters."""
        try:
            return self._H
        except AttributeError:
            raise NOT_FITTED_ERROR

    def argsort_units(self):
        # Retrieve motifs (throws error if model not fitted).
        W = self.motifs

        # For each unit, find the component with which its loading is largest.
        max_component_per_unit = np.argmax(np.sum(W, axis=0), axis=-1)

        # Find the time of peak activity for each feature on its max component.
        peak_activity = np.argmax(
            W[:, np.arange(self.n_features), max_component_per_unit], axis=0)

        # Lexographically sort units by component and time lag.
        return np.lexsort((peak_activity, max_component_per_unit))

    def plot(self, data=None, sort=True, **kwargs):
        """
        Visualizes data and model components.
        """
        # Get motifs and factors (throws error if model is not fitted).
        W, H = self.motifs, self.factors

        # Plot model prediction is data is not provided
        data = self.predict() if data is None else data

        # Reorder units by model factors
        idx = self.argsort_units() if sort else np.arange(self.n_features)

        # Call plotting function.
        return visualize.plot_result(data[idx], W[:, idx, :], H, **kwargs)


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
