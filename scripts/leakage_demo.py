#!/usr/bin/env python
"""Leakage demo: the same model scored with feature selection done the wrong
way (on all samples, before cross-validation) and the right way (inside each
training fold).

Four experiments, each with both arms:
  1. variance, real labels      top-k most variable probes (uses no labels)
  2. F-statistic, real labels   top-k probes that best separate the classes
  3. F-statistic, shuffled      the same, after shuffling the class labels, so
                                there is NOTHING to learn and any score above
                                chance is produced by the leak alone

  4. F-statistic, small noise   150 random samples given 3 made-up classes at
                                random: few samples, many probes, no signal.
                                This is the setting where the leak is worst.

Arms:
  leaky   missingness filter, medians and probe selection fit on ALL samples
          of the demo set, then cross-validated
  inside  the same steps fit on each training fold only

The demo set is the TRAINING side of outer fold 0 (about 2,213 samples), split
by its 3 inner folds. The outer test folds are never read, so nothing here
previews the project's final scores. Model: an untuned random forest.

Run:  python scripts/leakage_demo.py            (about 5-10 minutes)
Out:  results/leakage_demo/leakage_demo.tsv     one row per experiment, arm, fold
      results/leakage_demo/leakage_summary.tsv  means and the leaky-minus-inside gap
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from methylclf.features import MedianImputer, MissingnessFilter  # noqa: E402
from methylclf.metrics import score  # noqa: E402

BLOCK = 4096
EXPERIMENTS = [  # name, selection score, which labels
    ("variance, real labels", "variance", "real"),
    ("F-statistic, real labels", "f", "real"),
    ("F-statistic, shuffled labels", "f", "shuffled"),
]


def f_statistic(Z, codes, n_classes):
    """One-way ANOVA F per column: variation between class means over
    variation within classes. High F = the probe separates the classes."""
    n, p = Z.shape
    total = np.zeros(p)
    sq = np.zeros(p)
    for j in range(0, p, BLOCK):
        b = Z[:, j:j + BLOCK].astype(np.float64)
        total[j:j + BLOCK] = b.sum(axis=0)
        sq[j:j + BLOCK] = (b * b).sum(axis=0)
    between = -total**2 / n
    for c in range(n_classes):
        idx = np.flatnonzero(codes == c)
        if idx.size:
            between += Z[idx].sum(axis=0, dtype=np.float64) ** 2 / idx.size
    within = sq - total**2 / n - between
    k = np.unique(codes).size
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (between / (k - 1)) / (within / (n - k))
    return np.where(within > 1e-12, f, 0.0)


def top_k(values, k):
    order = np.lexsort((np.arange(values.size), -values))
    return np.sort(order[:k])


def fit_selection(X, rows, labelings, k):
    """Fit filter, medians and all selections on X[rows] only.
    Returns {experiment name: (columns of X, their medians)}."""
    Xs = X if rows is None else X[rows]
    filt = MissingnessFilter().fit(Xs)
    Z = filt.transform(Xs)
    del Xs
    imp = MedianImputer(copy=False).fit(Z)
    Z = imp.transform(Z)
    var = np.empty(Z.shape[1])
    for j in range(0, Z.shape[1], BLOCK):
        var[j:j + BLOCK] = Z[:, j:j + BLOCK].var(axis=0, ddof=1, dtype=np.float64)
    out = {}
    for name, kind, which in EXPERIMENTS:
        codes, n_classes = labelings[which]
        codes = codes if rows is None else codes[rows]
        values = var if kind == "variance" else f_statistic(Z, codes, n_classes)
        pos = top_k(values, min(k, values.size))
        out[name] = (filt.keep_[pos], imp.medians_[pos])
    return out


def take(X, rows, cols, medians):
    """X[rows][:, cols] with NaN replaced by the given medians."""
    A = X[rows][:, cols] if cols.size * 20 > X.shape[1] else X[np.ix_(rows, cols)]
    r, c = np.nonzero(np.isnan(A))
    A[r, c] = medians[c]
    return A


def shuffle_within_folds(y, fold, seed):
    """Shuffle labels among the samples of each fold, so every fold keeps its
    class counts but no label belongs to its sample any more."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for f in np.unique(fold):
        idx = np.flatnonzero(fold == f)
        out[idx] = y[rng.permutation(idx)]
    return out


def run_demo(X, y, fold, k=1000, n_trees=300, n_jobs=10, seed=42, log=print):
    """X: samples x probes (NaN allowed); y: class labels; fold: fold id per
    sample. Returns (per-fold table, summary table)."""
    y = np.asarray(y, dtype=object)
    fold = np.asarray(fold)
    classes = sorted(set(y))
    code = {c: i for i, c in enumerate(classes)}
    labels = {"real": y, "shuffled": shuffle_within_folds(y, fold, seed)}
    labelings = {w: (np.array([code[v] for v in lab]), len(classes)) for w, lab in labels.items()}

    t0 = time.time()
    leaky = fit_selection(X, None, labelings, k)
    log(f"selection fit on all {X.shape[0]} samples ({time.time() - t0:.0f} s)")
    rows = []
    for f in np.unique(fold):
        train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        t0 = time.time()
        inside = fit_selection(X, train, labelings, k)
        for name, _, which in EXPERIMENTS:
            lab = labels[which]
            for arm, sel in (("leaky", leaky[name]), ("inside", inside[name])):
                cols, med = sel
                rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_jobs, random_state=seed)
                rf.fit(take(X, train, cols, med), lab[train])
                s = score(lab[test], rf.predict_proba(take(X, test, cols, med)), list(rf.classes_))
                shared = np.intersect1d(cols, leaky[name][0]).size
                rows.append({"experiment": name, "arm": arm, "fold": int(f),
                             "n_train": train.size, "n_test": test.size, "n_probes": cols.size,
                             "probes_shared_with_leaky": shared,
                             "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                             "balanced_accuracy": s["balanced_accuracy"]})
        log(f"fold {f}: {train.size} train, {test.size} test ({time.time() - t0:.0f} s)")
    per_fold = pd.DataFrame(rows)
    mean = per_fold.groupby(["experiment", "arm"], sort=False)[
        ["accuracy", "macro_f1", "balanced_accuracy", "probes_shared_with_leaky"]].mean()
    wide = mean.unstack("arm")
    summary = pd.DataFrame({
        "accuracy_leaky": wide[("accuracy", "leaky")], "accuracy_inside": wide[("accuracy", "inside")],
        "macro_f1_leaky": wide[("macro_f1", "leaky")], "macro_f1_inside": wide[("macro_f1", "inside")],
        "probes_shared": wide[("probes_shared_with_leaky", "inside")],
    })
    summary.insert(2, "accuracy_gap", summary["accuracy_leaky"] - summary["accuracy_inside"])
    summary["macro_f1_gap"] = summary["macro_f1_leaky"] - summary["macro_f1_inside"]
    return per_fold, summary.reset_index()


def small_noise_case(n_total, n=150, n_classes=3, seed=42):
    """Rows of a random subset, made-up balanced labels, and 3 folds that
    each contain every made-up class."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))
    order = rng.permutation(rows.size)
    y = np.empty(rows.size, dtype=object)
    fold = np.empty(rows.size, dtype=np.int64)
    y[order] = [f"made-up {i % n_classes}" for i in range(rows.size)]
    fold[order] = (np.arange(rows.size) // n_classes) % 3
    return rows, y, fold


SMALL = "F-statistic, 150 samples, 3 made-up classes"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", default="data/betas/zarr/GSE90496.zarr")
    ap.add_argument("--folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--probes", default="results/probes/probes_kept.tsv")
    ap.add_argument("--out", default="results/leakage_demo")
    ap.add_argument("--k", type=int, default=1000, help="probes selected")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from methylclf.data import BetaStore
    st = BetaStore.open(a.store, a.folds, a.probes)
    s = st.samples
    demo = s[s["outer_fold"].astype(int) != 0]
    inner = demo["inner_fold_o0"].astype(int).to_numpy()
    if (inner < 0).any():
        sys.exit("ERROR: an outer-fold-0 training sample has no inner fold")
    print(f"demo set: {len(demo)} samples (training side of outer fold 0), "
          f"{demo['mc_class'].nunique()} classes, inner folds {np.bincount(inner).tolist()}")
    t0 = time.time()
    X = st.load(demo["geo_accession"].to_numpy())
    print(f"loaded {X.shape}, {X.nbytes / 1e9:.2f} GB ({time.time() - t0:.0f} s)")

    per_fold, summary = run_demo(X, demo["mc_class"].to_numpy(), inner, a.k, a.trees, a.jobs, a.seed)
    rows, fake_y, fake_fold = small_noise_case(X.shape[0], seed=a.seed)
    pf2, sm2 = run_demo(X[rows], fake_y, fake_fold, a.k, a.trees, a.jobs, a.seed, log=lambda *_: None)
    keep = "F-statistic, real labels"   # "real" here means the made-up labels as given
    per_fold = pd.concat([per_fold, pf2[pf2.experiment == keep].assign(experiment=SMALL)])
    summary = pd.concat([summary, sm2[sm2.experiment == keep].assign(experiment=SMALL)])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out / "leakage_demo.tsv", sep="\t", index=False, float_format="%.4f")
    summary.to_csv(out / "leakage_summary.tsv", sep="\t", index=False, float_format="%.4f")
    counts = demo["mc_class"].value_counts()
    print(f"\nchance, experiments 1-3: always guessing the largest class = "
          f"{counts.iloc[0] / len(demo):.3f} accuracy; experiment 4: 0.333")
    print(f"k = {a.k} probes, {a.trees} trees, mean over 3 folds\n")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"\nwrote {out}/leakage_demo.tsv and {out}/leakage_summary.tsv")


if __name__ == "__main__":
    main()
