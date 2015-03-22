"""
A simple adaptation of the RBM implementation from scikit-learn, to allow training given a visible+hidden state.

Adapted from:
https://github.com/scikit-learn/scikit-learn/blob/38104ff4e8f1c9c39a6f272eff7818b75a27da46/sklearn/neural_network/rbm.py

Also simplified initialisation and added batch-functionality to partial_fit, see issue #4211 (https://github.com/scikit-learn/scikit-learn/issues/4211).
"""

import numpy as np
from sklearn.neural_network.rbm import *


class BernoulliRBM_plus(BernoulliRBM):
    def __init__(self, *args, **kwargs):
        super(BernoulliRBM_plus, self).__init__(*args, **kwargs)
        self.initialised = False

    def _fit(self, v_pos, rng, h_pos=None):
        if h_pos is None:
            h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (np.asarray(
                                         v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0))

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def partial_fit(self, X, h=None):
        X, = check_arrays(X, sparse_format='csr', dtype=np.float)
        n_samples = X.shape[0]
        n_inputs = X.shape[1]
        if not self.initialised:
            self._init_components(n_inputs)
        rng = check_random_state(self.random_state)

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                if h is not None:
                    h_slice = h[batch_slice]
                else:
                    h_slice = None
                self._fit(X[batch_slice], rng, h_pos=h_slice)

            if verbose:
                end = time.time()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(), end - begin))
                begin = end

        return self

    def _init_components(self, n_inputs):
        # Initialise RBM once, rather than on each fit
        rng = check_random_state(self.random_state)
        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_components, n_inputs)),
            order='fortran')
        self.intercept_hidden_ = np.zeros(self.n_components, )
        self.intercept_visible_ = np.zeros(n_inputs, )
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))
        self.initialised = True
