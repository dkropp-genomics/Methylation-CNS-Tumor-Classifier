"""In-fold feature steps: everything here is FIT on a set of samples.

The test for leakage: "would the result change if other samples were added?"
Every class below answers yes, so each one learns its numbers in `fit`
(training-fold samples only) and applies them unchanged in `transform`
(to anything: the same training fold, a test fold, one new sample).

Order inside a fold:
    1. MissingnessFilter   drop probes missing in > 5% of the training fold
    2. MedianImputer       fill what is left with training-fold medians
    3. MaterialCorrector   optional: remove the FFPE-vs-frozen shift, estimated
                           within classes (on/off is chosen in the inner loop)
    4. TopVarianceSelector keep the k most variable probes of the training fold

Use them through `FeaturePipeline`, which runs the steps in order. It learns
only from what `fit` is given, so as long as it is handed training-fold rows,
no step can see a test row. (It is our own small class, not sklearn's
Pipeline, because the material step needs each sample's material at transform
time, and passing that through sklearn's Pipeline depends on the version.)

All steps take and return float32 arrays, samples x probes.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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


class MaterialCorrector(TransformerMixin, BaseEstimator):
    """Remove the FFPE-vs-frozen shift from each probe, estimated WITHIN classes.

    Why within classes: 34 classes are entirely FFPE, so "mean of all FFPE
    minus mean of all frozen" is partly a difference between tumor classes.
    Subtracting it would remove class signal. Instead, for every class that
    has both materials in the fit samples, take (FFPE mean - frozen mean) for
    each probe, and average those differences over classes with weight
    n_ffpe * n_frozen / (n_ffpe + n_frozen). That is exactly the material
    coefficient of the per-probe linear model  beta ~ class + material.

    fit needs the class labels (training fold only). transform needs only each
    sample's material, never its class, so it works for a new, unlabeled tumor.
    FFPE samples are shifted onto the frozen scale and clipped to [0, 1];
    frozen samples are returned unchanged.

    Assumption that cannot be tested: the shift is the same in classes that
    have only one material.
    """

    def __init__(self, ffpe_label: str = "FFPE", min_per_material: int = 2, copy: bool = True):
        self.ffpe_label = ffpe_label
        self.min_per_material = min_per_material
        self.copy = copy

    def _is_ffpe(self, material, n, who):
        if material is None:
            _fail(f"{who}: material is required (one value per sample)")
        material = np.asarray(material, dtype=object)
        if material.shape != (n,):
            _fail(f"{who}: {n} samples but {material.shape[0]} material values")
        return material, material == self.ffpe_label

    def fit(self, X, y=None, material=None):
        X = _check_X(X, who="MaterialCorrector.fit")
        n, p = X.shape
        if y is None or len(y) != n:
            _fail("MaterialCorrector.fit: class labels y are required, one per sample")
        y = np.asarray(y, dtype=object)
        material, ffpe = self._is_ffpe(material, n, "MaterialCorrector.fit")
        values = sorted(set(material))
        if len(values) != 2 or self.ffpe_label not in values:
            _fail(f"MaterialCorrector.fit: expected two materials including "
                  f"'{self.ffpe_label}', found {values}")
        num = np.zeros(p, dtype=np.float64)
        total_w, used = 0.0, []
        for c in sorted(set(y)):
            f = np.flatnonzero((y == c) & ffpe)
            z = np.flatnonzero((y == c) & ~ffpe)
            if min(f.size, z.size) < self.min_per_material:
                continue
            w = f.size * z.size / (f.size + z.size)
            num += w * (X[f].mean(axis=0, dtype=np.float64) - X[z].mean(axis=0, dtype=np.float64))
            total_w += w
            used.append(c)
        if not used:
            _fail(f"MaterialCorrector.fit: no class has >= {self.min_per_material} samples "
                  f"of each material; the shift cannot be estimated")
        if np.isnan(num).any():
            _fail("MaterialCorrector.fit: NaN in the input; run MedianImputer first")
        self.shift_ = (num / total_w).astype(np.float32)
        self.classes_used_ = used
        self.materials_ = values
        self.n_features_in_ = p
        return self

    def transform(self, X, material=None):
        X = _check_X(X, self.n_features_in_, "MaterialCorrector.transform")
        material, ffpe = self._is_ffpe(material, X.shape[0], "MaterialCorrector.transform")
        unknown = sorted(set(material) - set(self.materials_))
        if unknown:
            _fail(f"MaterialCorrector.transform: unknown material {unknown}; "
                  f"fitted on {self.materials_}")
        if self.copy:
            X = X.copy()
        rows = np.flatnonzero(ffpe)
        for i in range(0, rows.size, 256):  # in row blocks, to keep temporaries small
            r = rows[i:i + 256]
            X[r] = np.clip(X[r] - self.shift_, 0.0, 1.0)
        return X

    def get_feature_names_out(self, input_features=None):
        return _names_out(input_features, self.n_features_in_, None)


class FeaturePipeline(BaseEstimator):
    """The in-fold feature steps, in order. Fit it on training-fold rows only.

        pipe = FeaturePipeline(n_probes=10000, correct_material=True)
        pipe.fit(X_train, y_train, material_train)
        Z_train = pipe.transform(X_train, material_train)
        Z_test = pipe.transform(X_test, material_test)     # no labels needed

    With correct_material=False, y and material are ignored.
    Fitted steps: missing_, impute_, correct_ (None when off), select_.
    """

    def __init__(self, n_probes: int = 10000, max_missing: float = 0.05,
                 correct_material: bool = False, ffpe_label: str = "FFPE"):
        self.n_probes = n_probes
        self.max_missing = max_missing
        self.correct_material = correct_material
        self.ffpe_label = ffpe_label

    def fit(self, X, y=None, material=None):
        self.missing_ = MissingnessFilter(self.max_missing).fit(X)
        Z = self.missing_.transform(X)  # a new array, so later steps may write to it
        self.impute_ = MedianImputer(copy=False).fit(Z)
        Z = self.impute_.transform(Z)
        self.correct_ = None
        if self.correct_material:
            self.correct_ = MaterialCorrector(self.ffpe_label, copy=False).fit(Z, y, material)
            Z = self.correct_.transform(Z, material)
        self.select_ = TopVarianceSelector(self.n_probes).fit(Z)
        self.n_features_in_ = self.missing_.n_features_in_
        return self

    def transform(self, X, material=None):
        Z = self.impute_.transform(self.missing_.transform(X))
        if self.correct_ is not None:
            Z = self.correct_.transform(Z, material)
        return self.select_.transform(Z)

    def get_feature_names_out(self, input_features=None):
        names = self.missing_.get_feature_names_out(input_features)
        return self.select_.get_feature_names_out(names)
