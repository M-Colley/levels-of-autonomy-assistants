"""Coherent alternative to binary relevance: predict a contiguous LoA interval [lo, hi].

Binary relevance (MultiOutputClassifier / OneVsRest) decides each level independently, so
it can return empty or non-contiguous sets (e.g. {L0, L1, L4}).
Here every observed label set is mapped to its interval hull [min, max]
(15 possible intervals for 5 levels) and a single multiclass model is trained on it.
``predict`` returns the most probable interval (never empty, always contiguous);
``predict_proba`` returns P(level k is in the interval) = sum of the probabilities of all
intervals that contain k, i.e. per-level scores comparable to the binary-relevance models.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

N_LEVELS = 5
INTERVALS = [(lo, hi) for lo in range(N_LEVELS) for hi in range(lo, N_LEVELS)]
MEMBERSHIP = np.array([[lo <= k <= hi for k in range(N_LEVELS)] for lo, hi in INTERVALS], dtype=float)


def to_interval_index(Y):
    Y = np.asarray(Y)
    lo = np.argmax(Y, 1)
    hi = Y.shape[1] - 1 - np.argmax(Y[:, ::-1], 1)
    lookup = {iv: i for i, iv in enumerate(INTERVALS)}
    return np.array([lookup[(a, b)] for a, b in zip(lo, hi)])


class IntervalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, Y):
        z = to_interval_index(Y)
        self.estimator_ = clone(self.estimator).fit(X, z)
        self.seen_ = np.asarray(self.estimator_.classes_)
        return self

    def _interval_proba(self, X):
        p = np.zeros((len(X), len(INTERVALS)))
        p[:, self.seen_] = self.estimator_.predict_proba(X)
        return p

    def predict(self, X):
        return MEMBERSHIP[self._interval_proba(X).argmax(1)].astype(int)

    def predict_proba(self, X):
        return self._interval_proba(X) @ MEMBERSHIP  # (n, 5) marginal inclusion probabilities
