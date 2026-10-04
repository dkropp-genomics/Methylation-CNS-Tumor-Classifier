"""In-fold feature steps: everything here is FIT on a set of samples.

The test for leakage: "would the result change if other samples were added?"
Every class below answers yes, so each one learns its numbers in `fit`
(training-fold samples only) and applies them unchanged in `transform`
(to anything: the same training fold, a test fold, one new sample).

Order inside a fold:
    1. MissingnessFilter   drop probes missing in > 5% of the training fold
    2. MedianImputer       fill what is left with training-fold medians
    3. (material correction: to be added, a setting chosen in the inner loop)
    4. TopVarianceSelector keep the k most variable probes of the training fold

Put them in a scikit-learn Pipeline (`make_feature_pipeline`). A Pipeline
calls `fit` only on what it is given to fit, so as long as the Pipeline is
handed training-fold rows, no step can see a test row.

All steps take and return float32 arrays, samples x probes.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

_BLOCK = 4096  # columns handled at a time, to keep temporary arrays small


def _fail(msg: str):
    raise ValueError(f"ERROR: {msg}")


def _check_X(X, n_features=None, who=""):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        _fail(f"{who}: expected a non-empty samples x probes array, got shape {X.shape}")
    if n_features is not None and X.shape[1] != n_features:
        _fail(f"{who}: fitted on {n_features} probes, but given {X.shape[1]}")
    return X


def _names_out(input_features, n_in, index):
    if input_features is None:
        _fail("pass the probe IDs: get_feature_names_out(probe_ids)")
    names = np.asarray(input_features, dtype=object)
    if names.shape[0] != n_in:
        _fail(f"expected {n_in} probe IDs, got {names.shape[0]}")
    return names if index is None else names[index]


class MissingnessFilter(TransformerMixin, BaseEstimator):
    """Drop probes missing (NaN) in more than `max_missing` of the fit samples."""

    def __init__(self, max_missing: float = 0.05):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        X = _check_X(X, who="MissingnessFilter.fit")
        if not 0.0 <= self.max_missing < 1.0:
            _fail(f"max_missing must be in [0, 1), got {self.max_missing}")
        n, p = X.shape
        n_missing = np.zeros(p, dtype=np.int64)
        for j in range(0, p, _BLOCK):
            n_missing[j:j + _BLOCK] = np.isnan(X[:, j:j + _BLOCK]).sum(axis=0)
        self.frac_missing_ = n_missing / n
        self.keep_ = np.flatnonzero(self.frac_missing_ <= self.max_missing)
        self.n_features_in_ = p
        if self.keep_.size == 0:
            _fail(f"no probe has <= {self.max_missing:.0%} missing in the fit samples")
        return self

    def transform(self, X):
        X = _check_X(X, self.n_features_in_, "MissingnessFilter.transform")
        return X[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return _names_out(input_features, self.n_features_in_, self.keep_)


class MedianImputer(TransformerMixin, BaseEstimator):
    """Replace NaN with the probe's median over the fit samples.

    copy=False fills the given array in place (saves one full copy of the
    matrix; safe after MissingnessFilter, whose output is already a new array).
    """

    def __init__(self, copy: bool = True):
        self.copy = copy

    def fit(self, X, y=None):
        X = _check_X(X, who="MedianImputer.fit")
        p = X.shape[1]
        med = np.empty(p, dtype=np.float32)
        for j in range(0, p, _BLOCK):
            s = np.sort(X[:, j:j + _BLOCK], axis=0)  # NaN sorts to the end
            n_obs = (~np.isnan(s)).sum(axis=0)
            if (n_obs == 0).any():
                bad = j + int(np.flatnonzero(n_obs == 0)[0])
                _fail(f"MedianImputer.fit: probe column {bad} is missing in every fit "
                      f"sample; run MissingnessFilter first")
            c = np.arange(s.shape[1])
            lo, hi = s[(n_obs - 1) // 2, c], s[n_obs // 2, c]
            med[j:j + _BLOCK] = (lo.astype(np.float64) + hi) / 2
        self.medians_ = med
        self.n_features_in_ = p
        return self

    def transform(self, X):
        X = _check_X(X, self.n_features_in_, "MedianImputer.transform")
        if self.copy:
            X = X.copy()
        for j in range(0, X.shape[1], _BLOCK):
            block = X[:, j:j + _BLOCK]  # a view: writing to it writes to X
            r, c = np.nonzero(np.isnan(block))
            block[r, c] = self.medians_[j + c]
        return X

    def get_feature_names_out(self, input_features=None):
        return _names_out(input_features, self.n_features_in_, None)


class TopVarianceSelector(TransformerMixin, BaseEstimator):
    """Keep the `k` probes with the highest variance over the fit samples.

    Kept probes stay in their original column order. Ties are broken by
    column position, so the result is the same on every run.
    """

    def __init__(self, k: int = 10000):
        self.k = k

    def fit(self, X, y=None):
        X = _check_X(X, who="TopVarianceSelector.fit")
        n, p = X.shape
        if not 1 <= self.k <= p:
            _fail(f"TopVarianceSelector: k={self.k}, but {p} probes are available")
        if n < 2:
            _fail("TopVarianceSelector: need at least 2 samples to compute variance")
        var = np.empty(p, dtype=np.float64)
        for j in range(0, p, _BLOCK):
            var[j:j + _BLOCK] = X[:, j:j + _BLOCK].var(axis=0, ddof=1, dtype=np.float64)
        if np.isnan(var).any():
            _fail("TopVarianceSelector.fit: NaN in the input; run MedianImputer first")
        order = np.lexsort((np.arange(p), -var))  # by variance (high first), then column
        self.variances_ = var
        self.keep_ = np.sort(order[: self.k])
        self.n_features_in_ = p
        return self

    def transform(self, X):
        X = _check_X(X, self.n_features_in_, "TopVarianceSelector.transform")
        return X[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return _names_out(input_features, self.n_features_in_, self.keep_)


def make_feature_pipeline(n_probes: int = 10000, max_missing: float = 0.05) -> Pipeline:
    """The in-fold feature steps, in order. Fit it on training-fold rows only."""
    return Pipeline([
        ("missing", MissingnessFilter(max_missing=max_missing)),
        ("impute", MedianImputer(copy=False)),
        ("select", TopVarianceSelector(k=n_probes)),
    ])
