#!/usr/bin/env python
"""The cross-validation figure that the validation result is compared with.

Only some of the 91 training classes occur in the validation cohort, and a
macro average runs over the classes that are present. So the CV figure is
recomputed here on the training samples whose class does occur in validation.
A prediction of an absent class still counts as an error.

Reads the per-sample outer-fold predictions written in Phase 3 and the
validation LABELS. No beta values, no model, nothing from validation scores.

  python scripts/cv_reference.py rf_v1 lgbm_v1

Output: results/final_v1/cv_reference.tsv
  run, level, metric, value, ci_low, ci_high, n, n_groups
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def fail(msg: str):
    sys.exit(f"ERROR: {msg}")


def macro_f1(t, p, k):
    """Mean F1 over the groups present in the true labels (integer codes)."""
    tp = np.bincount(t[t == p], minlength=k).astype(float)
    n_true = np.bincount(t, minlength=k).astype(float)
    n_pred = np.bincount(p, minlength=k).astype(float)
    present = n_true > 0
    return float((2 * tp[present] / (n_true[present] + n_pred[present])).mean())


def reference(preds: pd.DataFrame, val_classes, n_boot=1000, seed=0) -> pd.DataFrame:
    """Macro-F1 and accuracy at class and family level on the shared classes."""
    val_classes = set(val_classes)
    absent = sorted(val_classes - set(preds["mc_class"]))
    if absent:
        fail(f"{len(absent)} validation class(es) have no CV samples, first: {absent[0]!r}")
    sub = preds[preds["mc_class"].isin(val_classes)]
    rows = []
    for level, true_col, pred_col in (("class", "mc_class", "predicted_class"),
                                      ("family", "family", "predicted_family")):
        names = sorted(set(preds[true_col]) | set(preds[pred_col]))
        code = {c: i for i, c in enumerate(names)}
        t = sub[true_col].map(code).to_numpy()
        p = sub[pred_col].map(code).to_numpy()
        rng = np.random.default_rng(seed)             # same resamples at both levels
        point = {"macro_f1": macro_f1(t, p, len(names)), "accuracy": float((t == p).mean())}
        boots = {m: np.empty(n_boot) for m in point}
        for b in range(n_boot):
            i = rng.integers(0, t.size, t.size)
            boots["macro_f1"][b] = macro_f1(t[i], p[i], len(names))
            boots["accuracy"][b] = (t[i] == p[i]).mean()
        for m, v in point.items():
            lo, hi = np.percentile(boots[m], [2.5, 97.5]) if n_boot else (np.nan, np.nan)
            rows.append({"level": level, "metric": m, "value": v, "ci_low": lo, "ci_high": hi,
                         "n": int(t.size), "n_groups": int(np.unique(t).size)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--kept", default="results/splits/GSE109379_kept.tsv")
    ap.add_argument("--res-root", default="results/cv")
    ap.add_argument("--out", default="results/final_v1/cv_reference.tsv")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    if not Path(args.kept).is_file():
        fail(f"{args.kept}: not found; run scripts/make_validation_table.py first")
    val_classes = set(pd.read_csv(args.kept, sep="\t", dtype=str,
                                  keep_default_na=False)["mc_class"])
    need = ["mc_class", "predicted_class", "family", "predicted_family"]
    tables = []
    for run in args.runs:
        path = Path(args.res_root) / run / "outer_predictions.tsv"
        if not path.is_file():
            fail(f"{path}: not found")
        preds = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if [c for c in need if c not in preds.columns]:
            fail(f"{path}: expected columns {need}")
        t = reference(preds, val_classes, args.n_boot)
        t.insert(0, "run", run)
        tables.append(t)
    out = pd.concat(tables, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False, float_format="%.5f")
    print(f"validation classes: {len(val_classes)}; wrote {args.out}")
    print(out.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
