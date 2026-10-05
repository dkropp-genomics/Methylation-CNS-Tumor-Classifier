#!/usr/bin/env python
"""Final scoring: calibrate the outer-test-fold scores and compute every metric.

Run after `run_nested_cv.py --stage outer --final-scoring` has saved, for each
outer fold k, the selected model's raw scores on test fold k. For each fold:

  1. fit the calibrator chosen in calibrate_inner.py on the inner out-of-fold
     scores of all three inner folds (training-side samples only);
  2. apply it to the raw scores of outer test fold k.

Every sample is in exactly one outer test fold, so pooling the five folds gives
one out-of-sample prediction per sample. Metrics are computed on that pool,
with bootstrap confidence intervals, for raw and calibrated scores, at class
and family level, for all samples and by material.

No beta values are read and no model is refit. The test-fold labels are used
only to compute the metrics.

  python scripts/score_outer.py rf_v1 lgbm_v1

Outputs, per run, in results/cv/<run>/:
  outer_metrics.tsv       every metric with its 95% interval
  outer_per_class.tsv     per class: n, sensitivity, precision, F1 (calibrated)
  outer_predictions.tsv   per sample: truth, predicted class and family, scores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_inner as cal  # noqa: E402

cv = cal.cv


def score_run(run_name, folds, family_of, pred_root, res_root, summarize, to_family, n_boot=1000):
    pred_dir, res_dir = Path(pred_root) / run_name, Path(res_root) / run_name
    for name in ("selected.tsv", "calibration_selected.tsv"):
        if not (res_dir / name).exists():
            cv.fail(f"{res_dir / name} not found; run the search and calibrate_inner.py first")
    chosen = pd.read_csv(res_dir / "selected.tsv", sep="\t", dtype={"setting": str}).set_index(
        "outer"
    )["setting"]
    calib = pd.read_csv(
        res_dir / "calibration_selected.tsv", sep="\t", dtype={"setting": str}
    ).set_index("outer")
    truth = dict(zip(folds["geo_accession"], folds["mc_class"]))
    material = dict(zip(folds["geo_accession"], folds["material"]))
    outer_of = dict(zip(folds["geo_accession"], folds["outer_fold"].astype(int)))

    ids_all, raw_all, cal_all, fold_all, classes = [], [], [], [], None
    for k in range(cv.N_OUTER):
        if calib.loc[k, "setting"] != chosen[k]:
            cv.fail(
                f"outer {k}: calibration was chosen for setting "
                f"{calib.loc[k, 'setting']}, but the selected setting is {chosen[k]}"
            )
        path = cv.pred_path(pred_dir, k, None, chosen[k])
        if not path.exists():
            cv.fail(f"{path} not found; run run_nested_cv.py --stage outer --final-scoring")
        P_test, ids, cls = cv.load_pred(path)
        want = {i for i, o in outer_of.items() if o == k}
        if set(ids) != want or len(ids) != len(want):
            cv.fail(f"{path.name}: samples are not exactly outer test fold {k}")
        parts, inner_classes = cal.load_inner(pred_dir, k, chosen[k], folds)
        if list(cls) != inner_classes or (classes is not None and list(cls) != classes):
            cv.fail(f"{path.name}: class order differs from the other prediction files")
        classes = list(cls)
        how, C = calib.loc[k, "transform"], float(calib.loc[k, "C"])
        model = cal.fit_calibrator(
            np.vstack([p["P"] for p in parts]),  # training side
            np.concatenate([p["y"] for p in parts]),
            how,
            C,
            classes,
        )
        ids_all.append(ids)
        raw_all.append(P_test.astype(np.float64))
        cal_all.append(cal.apply_calibrator(model, P_test, how))  # test fold k
        fold_all.append(np.full(len(ids), k))
        print(
            f"{run_name} outer {k}: {len(ids)} test samples, calibrator {how} C={C:g} "
            f"fit on {sum(len(p['y']) for p in parts)} training-side scores",
            flush=True,
        )

    ids = np.concatenate(ids_all)
    if sorted(ids) != sorted(folds["geo_accession"]):
        cv.fail(f"{run_name}: the five test folds do not cover every sample exactly once")
    raw, calibrated, fold = np.vstack(raw_all), np.vstack(cal_all), np.concatenate(fold_all)
    y = np.array([truth[i] for i in ids], dtype=object)
    mat = np.array([material[i] for i in ids], dtype=object)

    tables = []
    for stage, P in (("raw", raw), ("calibrated", calibrated)):
        t = summarize(y, P, classes, family_of=family_of, groups=mat, n_boot=n_boot)
        t.insert(0, "scores", stage)
        tables.append(t)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.insert(0, "run", run_name)

    cls_arr = np.asarray(classes, dtype=object)
    pred = cls_arr[np.argmax(calibrated, axis=1)]
    fam_proba, families = to_family(calibrated, classes, family_of)
    fam_pred = np.asarray(families, dtype=object)[np.argmax(fam_proba, axis=1)]
    preds = pd.DataFrame(
        {
            "geo_accession": ids,
            "outer_fold": fold,
            "material": mat,
            "mc_class": y,
            "predicted_class": pred,
            "class_score": calibrated.max(axis=1),
            "raw_class_score": raw[np.arange(len(ids)), np.argmax(calibrated, axis=1)],
            "family": [family_of[c] for c in y],
            "predicted_family": fam_pred,
            "family_score": fam_proba.max(axis=1),
        }
    )
    rows = []
    for c in classes:
        tp = int(((y == c) & (pred == c)).sum())
        n, n_pred = int((y == c).sum()), int((pred == c).sum())
        sens = tp / n if n else np.nan
        prec = tp / n_pred if n_pred else np.nan
        f1 = 2 * tp / (n + n_pred) if (n + n_pred) else np.nan
        rows.append(
            {
                "mc_class": c,
                "family": family_of[c],
                "n": n,
                "n_predicted": n_pred,
                "sensitivity": sens,
                "precision": prec,
                "f1": f1,
                "n_family_correct": int(
                    ((y == c) & (preds["family"] == preds["predicted_family"])).sum()
                ),
            }
        )
    per_class = pd.DataFrame(rows)

    metrics.to_csv(res_dir / "outer_metrics.tsv", sep="\t", index=False, float_format="%.5f")
    per_class.to_csv(res_dir / "outer_per_class.tsv", sep="\t", index=False, float_format="%.4f")
    preds.to_csv(res_dir / "outer_predictions.tsv", sep="\t", index=False, float_format="%.5f")
    return metrics, per_class, preds


def show(metrics: pd.DataFrame, group: str) -> pd.DataFrame:
    m = metrics[metrics["group"] == group].copy()
    m["text"] = [
        f"{v:.3f} ({lo:.3f}-{hi:.3f})" for v, lo, hi in zip(m["value"], m["ci_low"], m["ci_high"])
    ]
    t = m.pivot_table(
        index="metric", columns=["level", "scores"], values="text", aggfunc="first", sort=False
    )
    cols = [
        (lv, sc)
        for lv in ("class", "family")
        for sc in ("raw", "calibrated")
        if (lv, sc) in t.columns
    ]
    return t[cols]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--families", default="results/meta/class_to_family.tsv")
    ap.add_argument("--pred-root", default="data/predictions")
    ap.add_argument("--res-root", default="results/cv")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    from methylclf.metrics import summarize, to_family

    for path in (args.folds, args.families):
        if not Path(path).exists():
            cv.fail(f"{path}: not found")
    folds = pd.read_csv(args.folds, sep="\t", dtype=str, keep_default_na=False)
    fam = pd.read_csv(args.families, sep="\t", dtype=str, keep_default_na=False)
    family_of = dict(zip(fam["mc_class"], fam["family"]))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    for name in args.runs:
        metrics, per_class, preds = score_run(
            name, folds, family_of, args.pred_root, args.res_root, summarize, to_family, args.n_boot
        )
        for group in ["all"] + sorted(set(metrics["group"]) - {"all"}):
            n = int(metrics.loc[metrics["group"] == group, "n"].iloc[0])
            print(
                f"\n== {name}: outer test folds pooled, samples: {group} (n={n}); "
                f"value (95% interval) =="
            )
            print(show(metrics, group).to_string())
        worst = per_class.sort_values("sensitivity").head(10)
        print(f"\n== {name}: the 10 classes with the lowest sensitivity (calibrated) ==")
        print(worst.to_string(index=False, float_format="{:.3f}".format))
        wrong = preds[preds["mc_class"] != preds["predicted_class"]]
        print(
            f"\n{name}: {len(wrong)} of {len(preds)} samples misclassified at class level; "
            f"{int((preds['family'] != preds['predicted_family']).sum())} at family level\n"
        )


if __name__ == "__main__":
    main()
