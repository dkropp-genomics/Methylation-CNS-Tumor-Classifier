#!/usr/bin/env python
"""Score calibration, fitted and checked on inner out-of-fold scores only.

A classifier's raw scores are not probabilities: a random forest that is right
95% of the time may give its top class a score of 0.4. Calibration is a second,
small model that maps the 91 raw scores of a sample to 91 probabilities. As in
Capper et al. it is an L2-penalized multinomial logistic regression.

It is a fitted step, so it gets the same treatment as every other fitted step:
it never sees the outer test fold. For outer fold k, the selected setting has
one saved prediction per training-side sample, each made by a model that did
not train on that sample (the 3 inner folds). This script reads only those
files. No beta values are loaded and no model is refit.

  For each outer fold k and each candidate (score transform, penalty C):
    fit the calibrator on the scores of two inner folds, apply it to the third,
    for each of the three; record the held-out log loss.
  Pick the candidate with the lowest mean held-out log loss (per outer fold).
  Report every metric before and after calibration on those held-out scores.

One caveat, stated so the numbers are read correctly: the scores used to FIT a
calibrator for inner fold j came from models whose training data included fold
j. The held-out scores themselves are clean, so this is a second-order effect,
but the calibration numbers that count are the ones from the final outer-fold
scoring, where the calibrator is fit on all three inner folds and applied to an
outer test fold that nothing has seen.

  python scripts/calibrate_inner.py rf_v1 lgbm_v1

Outputs, per run, in results/cv/<run>/:
  calibration_candidates.tsv   held-out log loss of every candidate
  calibration_selected.tsv     the candidate chosen for each outer fold
  calibration_inner_metrics.tsv   all metrics before/after, class and family level
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nested_cv as cv  # noqa: E402

TRANSFORMS = ("raw", "log")
CS = (0.01, 0.1, 1.0, 10.0, 100.0)  # smaller C = stronger penalty
MAX_ITER = 2000


def transform_scores(P, how: str):
    P = np.asarray(P, dtype=np.float64)
    if how == "raw":
        return P
    if how == "log":  # a forest gives exact zeros; floor them
        return np.log(np.clip(P, 1e-4, 1.0))
    cv.fail(f"unknown score transform '{how}' (known: {TRANSFORMS})")


def fit_calibrator(P, y, how, C, classes):
    """L2-penalized multinomial logistic regression on (transformed) scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=MAX_ITER))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence is reported below
        model.fit(transform_scores(P, how), y)
    if list(model[-1].classes_) != list(classes):
        cv.fail(
            f"calibrator: {len(model[-1].classes_)} classes in the fitting samples, "
            f"expected {len(classes)}"
        )
    model.converged_ = bool(np.max(model[-1].n_iter_) < MAX_ITER)
    return model


def apply_calibrator(model, P, how):
    return model.predict_proba(transform_scores(P, how))


def load_inner(pred_dir: Path, k: int, setting: str, folds: pd.DataFrame):
    """The three inner prediction files of one outer fold, with truth and material."""
    truth = dict(zip(folds["geo_accession"], folds["mc_class"]))
    material = dict(zip(folds["geo_accession"], folds["material"]))
    inner = dict(zip(folds["geo_accession"], folds[f"inner_fold_o{k}"].astype(int)))
    test_ids = set(folds.loc[folds["outer_fold"].astype(int) == k, "geo_accession"])
    parts, classes = [], None
    for j in range(cv.N_INNER):
        path = cv.pred_path(pred_dir, k, j, setting)
        if not path.exists():
            cv.fail(f"{path}: prediction file not found")
        P, ids, cls = cv.load_pred(path)
        if set(ids) & test_ids:
            cv.fail(f"{path.name}: contains samples of outer test fold {k}")
        if any(inner[i] != j for i in ids):
            cv.fail(f"{path.name}: samples are not inner fold {j} of outer fold {k}")
        if classes is not None and list(cls) != classes:
            cv.fail(f"{path.name}: class order differs between inner folds")
        classes = list(cls)
        parts.append(
            {
                "ids": ids,
                "y": np.array([truth[i] for i in ids], dtype=object),
                "material": np.array([material[i] for i in ids], dtype=object),
                "P": P.astype(np.float64),
            }
        )
    return parts, classes


def cross_calibrate(parts, how, C, classes):
    """For each inner fold: calibrator fit on the other two, applied to it."""
    out, converged = [], True
    for j in range(len(parts)):
        others = [p for i, p in enumerate(parts) if i != j]
        model = fit_calibrator(
            np.vstack([p["P"] for p in others]),
            np.concatenate([p["y"] for p in others]),
            how,
            C,
            classes,
        )
        converged &= model.converged_
        out.append(apply_calibrator(model, parts[j]["P"], how))
    return out, converged


def run(run_name, folds, family_of, pred_root, res_root, summarize, transforms=TRANSFORMS, Cs=CS):
    pred_dir, res_dir = Path(pred_root) / run_name, Path(res_root) / run_name
    sel_path = res_dir / "selected.tsv"
    if not sel_path.exists():
        cv.fail(f"{sel_path} not found; finish the inner search of '{run_name}' first")
    chosen = pd.read_csv(sel_path, sep="\t", dtype={"setting": str}).set_index("outer")["setting"]
    missing = sorted(set(folds["mc_class"]) - set(family_of))
    if missing:
        cv.fail(f"class-to-family table lacks {missing}")
    cand_rows, sel_rows, metric_rows = [], [], []
    for k in range(cv.N_OUTER):
        if k not in chosen.index:
            cv.fail(f"{sel_path}: no selected setting for outer fold {k}")
        parts, classes = load_inner(pred_dir, k, chosen[k], folds)
        y_all = np.concatenate([p["y"] for p in parts])
        mat_all = np.concatenate([p["material"] for p in parts])
        P_raw = np.vstack([p["P"] for p in parts])
        before = cv.score_pred(y_all, P_raw, classes)["log_loss"]
        best = None
        for how in transforms:
            for C in Cs:
                cal, ok = cross_calibrate(parts, how, C, classes)
                ll = float(
                    np.mean(
                        [cv.score_pred(p["y"], c, classes)["log_loss"] for p, c in zip(parts, cal)]
                    )
                )
                cand_rows.append(
                    {
                        "outer": k,
                        "setting": chosen[k],
                        "transform": how,
                        "C": C,
                        "log_loss_heldout": ll,
                        "converged": ok,
                    }
                )
                if best is None or ll < best[0]:  # ties keep the earlier candidate
                    best = (ll, how, C, cal)
        ll, how, C, cal = best
        sel_rows.append(
            {
                "outer": k,
                "setting": chosen[k],
                "transform": how,
                "C": C,
                "log_loss_before": before,
                "log_loss_after": ll,
            }
        )
        for stage, P in (("before", P_raw), ("after", np.vstack(cal))):
            t = summarize(y_all, P, classes, family_of=family_of, groups=mat_all, n_boot=0)
            t = t[["group", "level", "metric", "value", "n", "n_classes"]].copy()
            t.insert(0, "stage", stage)
            t.insert(0, "outer", k)
            metric_rows.append(t)
        print(
            f"{run_name} outer {k}: chose transform={how} C={C}; "
            f"log loss {before:.3f} -> {ll:.3f}",
            flush=True,
        )
    cands, sel = pd.DataFrame(cand_rows), pd.DataFrame(sel_rows)
    metrics = pd.concat(metric_rows, ignore_index=True)
    cands.to_csv(res_dir / "calibration_candidates.tsv", sep="\t", index=False, float_format="%.5f")
    sel.to_csv(res_dir / "calibration_selected.tsv", sep="\t", index=False, float_format="%.5f")
    metrics.to_csv(
        res_dir / "calibration_inner_metrics.tsv", sep="\t", index=False, float_format="%.5f"
    )
    return cands, sel, metrics


def overview(metrics: pd.DataFrame, group="all") -> pd.DataFrame:
    """Mean over the 5 outer folds: one row per metric, before/after at each level."""
    m = metrics[metrics["group"] == group]
    t = (
        m.groupby(["metric", "level", "stage"], sort=False)["value"]
        .mean()
        .unstack(["level", "stage"])
    )
    cols = [
        (lv, st)
        for lv in ("class", "family")
        for st in ("before", "after")
        if (lv, st) in t.columns
    ]
    return t[cols]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run names, e.g. rf_v1 lgbm_v1")
    ap.add_argument("--folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--families", default="results/meta/class_to_family.tsv")
    ap.add_argument("--pred-root", default="data/predictions")
    ap.add_argument("--res-root", default="results/cv")
    args = ap.parse_args(argv)

    from methylclf.metrics import summarize

    for path in (args.folds, args.families):
        if not Path(path).exists():
            cv.fail(f"{path}: not found")
    folds = pd.read_csv(args.folds, sep="\t", dtype=str, keep_default_na=False)
    fam = pd.read_csv(args.families, sep="\t", dtype=str, keep_default_na=False)
    family_of = dict(zip(fam["mc_class"], fam["family"]))
    pd.set_option(
        "display.width",
        250,
        "display.max_columns",
        30,
        "display.max_rows",
        200,
        "display.float_format",
        "{:.4f}".format,
    )
    for name in args.runs:
        cands, sel, metrics = run(name, folds, family_of, args.pred_root, args.res_root, summarize)
        print(f"\n== {name}: chosen calibrator per outer fold ==")
        print(sel.to_string(index=False))
        if not cands["converged"].all():
            print(
                f"note: {int((~cands['converged']).sum())} of {len(cands)} candidate fits "
                f"hit the iteration limit (see calibration_candidates.tsv)"
            )
        for group in ["all"] + sorted(set(metrics["group"]) - {"all"}):
            print(
                f"\n== {name}: inner held-out metrics, mean of 5 outer folds, "
                f"samples: {group} =="
            )
            print(overview(metrics, group).to_string())
        print()


if __name__ == "__main__":
    main()
