"""Scoring: one place that turns (true labels, predicted probabilities) into
the project's numbers.

Metrics (all computed from the same probability matrix):
  macro_f1            mean of per-class F1; every class counts equally
  balanced_accuracy   mean of per-class recall
  accuracy            share of samples correct (dominated by the big classes)
  brier               mean over samples of sum_k (p_k - y_k)^2; 0 is perfect, 2 is worst
  ece                 expected calibration error of the top score (15 equal-width bins)
  confident_share     share of samples whose top score is >= threshold (default 0.9)
  confident_accuracy  accuracy among those samples

Two conventions that matter:
  * Macro averages run over the classes PRESENT in y_true. A class with no
    true samples has no recall, so it is left out of the average (this is why
    a 69-class validation figure is not comparable with a 91-class CV figure).
  * Family level: a sample's family score is the SUM of its class scores in
    that family (as in Capper et al.), then everything is recomputed.

`summarize` adds bootstrap confidence intervals: resample the scored samples
with replacement, recompute, take percentiles. This measures how much the
number would move with a different draw of test samples; it does not cover
variation from retraining the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = [
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "brier",
    "ece",
    "confident_share",
    "confident_accuracy",
]


def _fail(msg: str):
    raise ValueError(f"ERROR: {msg}")


def _encode(labels, classes, what):
    index = {c: i for i, c in enumerate(classes)}
    if len(index) != len(classes):
        _fail(f"{what}: duplicated class names")
    unknown = sorted({x for x in labels if x not in index})
    if unknown:
        _fail(
            f"{what}: {len(unknown)} label(s) not among the model's classes, "
            f"first: {unknown[0]!r}"
        )
    return np.array([index[x] for x in labels], dtype=np.int64)


def _check(y_true, proba, classes):
    proba = np.asarray(proba, dtype=np.float64)
    classes = list(classes)
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        _fail(f"proba has shape {proba.shape}, expected (n_samples, {len(classes)})")
    if len(y_true) != proba.shape[0]:
        _fail(f"{len(y_true)} true labels but {proba.shape[0]} rows of proba")
    if proba.shape[0] == 0:
        _fail("no samples to score")
    if np.isnan(proba).any() or (proba < -1e-9).any():
        _fail("proba contains NaN or negative values")
    if np.abs(proba.sum(axis=1) - 1.0).max() > 1e-3:
        _fail("rows of proba do not sum to 1")
    return _encode(list(y_true), classes, "y_true"), proba, classes


def confusion_counts(t, p, k):
    """k x k counts, rows = true class, columns = predicted class."""
    return np.bincount(t * k + p, minlength=k * k).reshape(k, k)


def _from_confusion(cm):
    support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    present = support > 0
    recall = tp[present] / support[present]
    f1 = 2 * tp[present] / (support[present] + predicted[present])
    return f1.mean(), recall.mean(), tp.sum() / cm.sum()


def _ece(conf, correct, n_bins):
    bins = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    acc = np.bincount(bins, weights=correct, minlength=n_bins)
    cf = np.bincount(bins, weights=conf, minlength=n_bins)
    return np.abs(acc - cf).sum() / conf.size  # = sum_b (n_b / n) * |acc_b - conf_b|


class _Scored:
    """Per-sample quantities, computed once so resampling is cheap."""

    def __init__(self, t, proba, threshold, n_bins):
        self.t, self.k = t, proba.shape[1]
        self.pred = proba.argmax(axis=1)
        self.conf = proba.max(axis=1)
        self.correct = (self.pred == t).astype(np.float64)
        onehot_p = proba[np.arange(t.size), t]
        self.brier = (proba**2).sum(axis=1) - 2 * onehot_p + 1.0
        self.threshold, self.n_bins = threshold, n_bins

    def metrics(self, idx=None) -> dict:
        s = slice(None) if idx is None else idx
        t, pred, conf, correct = self.t[s], self.pred[s], self.conf[s], self.correct[s]
        f1, bal, acc = _from_confusion(confusion_counts(t, pred, self.k))
        sure = conf >= self.threshold
        return {
            "macro_f1": f1,
            "balanced_accuracy": bal,
            "accuracy": acc,
            "brier": self.brier[s].mean(),
            "ece": _ece(conf, correct, self.n_bins),
            "confident_share": sure.mean(),
            "confident_accuracy": correct[sure].mean() if sure.any() else np.nan,
        }


def to_family(proba, classes, family_of):
    """Sum class scores within each family. Returns (family_proba, families)."""
    classes = list(classes)
    missing = [c for c in classes if c not in family_of]
    if missing:
        _fail(f"{len(missing)} class(es) have no family, first: {missing[0]!r}")
    families = sorted({family_of[c] for c in classes})
    col = {f: j for j, f in enumerate(families)}
    member = np.zeros((len(classes), len(families)))
    for i, c in enumerate(classes):
        member[i, col[family_of[c]]] = 1.0
    return np.asarray(proba, dtype=np.float64) @ member, families


def score(y_true, proba, classes, threshold=0.9, n_bins=15) -> dict:
    """Point values of every metric."""
    t, proba, _ = _check(y_true, proba, classes)
    return _Scored(t, proba, threshold, n_bins).metrics()


def confusion_table(y_true, proba, classes) -> pd.DataFrame:
    """Confusion matrix as a table: rows = true class, columns = predicted."""
    t, proba, classes = _check(y_true, proba, classes)
    cm = confusion_counts(t, proba.argmax(axis=1), len(classes))
    return pd.DataFrame(
        cm, index=pd.Index(classes, name="true"), columns=pd.Index(classes, name="predicted")
    )


def reliability_table(y_true, proba, classes, n_bins=15) -> pd.DataFrame:
    """Per confidence bin: n, mean top score, accuracy. The data behind a
    reliability diagram and behind ECE."""
    t, proba, _ = _check(y_true, proba, classes)
    conf, correct = proba.max(axis=1), proba.argmax(axis=1) == t
    bins = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append(
            {
                "bin_low": b / n_bins,
                "bin_high": (b + 1) / n_bins,
                "n": int(m.sum()),
                "mean_confidence": conf[m].mean() if m.any() else np.nan,
                "accuracy": correct[m].mean() if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize(
    y_true,
    proba,
    classes,
    family_of=None,
    groups=None,
    n_boot=1000,
    seed=0,
    ci=0.95,
    threshold=0.9,
    n_bins=15,
) -> pd.DataFrame:
    """Every metric with a bootstrap confidence interval.

    family_of : dict class -> family; adds level "family" rows.
    groups    : one value per sample (for example material); adds the same
                rows computed within each group, beside group "all".
    Returns columns: group, level, metric, value, ci_low, ci_high, n, n_classes.
    """
    t, proba, classes = _check(y_true, proba, classes)
    levels = [("class", t, proba)]
    if family_of is not None:
        fam_proba, families = to_family(proba, classes, family_of)
        fam_t = _encode([family_of[classes[i]] for i in t], families, "family")
        levels.append(("family", fam_t, fam_proba))
    subsets = [("all", np.arange(t.size))]
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        if groups.shape != (t.size,):
            _fail(f"{t.size} samples but {groups.shape[0]} group values")
        subsets += [(str(g), np.flatnonzero(groups == g)) for g in sorted(set(groups))]

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    rows = []
    for level, lt, lp in levels:
        scored = _Scored(lt, lp, threshold, n_bins)
        for g, (group, members) in enumerate(subsets):
            point = scored.metrics(members)
            rng = np.random.default_rng([seed, g])  # same resamples at class and family level
            boots = {m: np.empty(n_boot) for m in METRICS}
            for b in range(n_boot):
                res = scored.metrics(members[rng.integers(0, members.size, members.size)])
                for m in METRICS:
                    boots[m][b] = res[m]
            for m in METRICS:
                ok = boots[m][~np.isnan(boots[m])]
                lo, hi = np.percentile(ok, [lo_q, hi_q]) if ok.size else (np.nan, np.nan)
                rows.append(
                    {
                        "group": group,
                        "level": level,
                        "metric": m,
                        "value": point[m],
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": members.size,
                        "n_classes": int(np.unique(lt[members]).size),
                    }
                )
    return pd.DataFrame(rows)
