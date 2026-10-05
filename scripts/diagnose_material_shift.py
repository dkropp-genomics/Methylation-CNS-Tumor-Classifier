#!/usr/bin/env python
"""Diagnostic: where does the fitted FFPE-vs-frozen shift come from?

The material correction (methylclf.features.MaterialCorrector) estimates ONE
shift per probe and subtracts it from every FFPE sample, including samples of
the all-FFPE classes where the shift cannot be measured. That is only sound if
the classes that have both materials roughly agree on the shift.

This script checks that, on the TRAINING SIDE OF OUTER FOLD 0 ONLY (no outer
test fold is read, no class prediction is made, nothing here is a model score):

  Part 1  per-class FFPE-minus-frozen differences
          - class_table.tsv: one row per usable class: sizes, weight, and how
            well that class's differences agree with the shift estimated from
            all the OTHER classes (leave-one-class-out correlation).
          - top_probes.tsv: the probes with the largest |shift|: how many
            classes share the sign, which class contributes most, and what the
            shift would be without that class.
          - top_probes_by_class.tsv: the raw per-class differences behind it.

  Part 2  does the shift carry over to classes it was not estimated from?
          A logistic regression is trained to tell FFPE from frozen on some
          classes and tested on OTHER classes (grouped by class), once on
          uncorrected betas and with the correction fitted on the training
          classes. AUC is computed inside each held-out class, so class
          differences cannot help. If the correction's assumption holds, the
          AUC falls toward 0.5 in every held-out class when the correction is
          on. A class far above 0.5 was corrected too little; far below, too much.

Run:  python scripts/diagnose_material_shift.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTER_FOLD = 0  # diagnostics use the training side of outer fold 0 only


def fail(msg: str):
    sys.exit(f"ERROR: {msg}")


# --------------------------------------------------------------------------
# Part 1: per-class differences
# --------------------------------------------------------------------------
def per_class_differences(Z, y, material, ffpe_label="FFPE", min_per_material=2):
    """FFPE mean minus frozen mean per probe, for each class with both materials.

    Returns (classes, D): a table with mc_class, n_ffpe, n_frozen, weight, and
    a float32 array D of shape (classes, probes). The weights are the ones
    MaterialCorrector uses: n_ffpe * n_frozen / (n_ffpe + n_frozen).
    """
    y = np.asarray(y, dtype=object)
    material = np.asarray(material, dtype=object)
    n = Z.shape[0]
    if y.shape != (n,) or material.shape != (n,):
        fail(
            f"per_class_differences: {n} samples but {y.shape[0]} labels and "
            f"{material.shape[0]} material values"
        )
    if ffpe_label not in set(material):
        fail(
            f"per_class_differences: no sample has material '{ffpe_label}'; "
            f"found {sorted(set(material))}"
        )
    ffpe = material == ffpe_label
    rows, diffs = [], []
    for c in sorted(set(y)):
        f = np.flatnonzero((y == c) & ffpe)
        z = np.flatnonzero((y == c) & ~ffpe)
        if min(f.size, z.size) < min_per_material:
            continue
        d = Z[f].mean(axis=0, dtype=np.float64) - Z[z].mean(axis=0, dtype=np.float64)
        diffs.append(d.astype(np.float32))
        rows.append((c, f.size, z.size, f.size * z.size / (f.size + z.size)))
    if not rows:
        fail(
            f"per_class_differences: no class has >= {min_per_material} samples "
            f"of each material"
        )
    D = np.vstack(diffs)
    if np.isnan(D).any():
        fail("per_class_differences: NaN in the input; impute first")
    classes = pd.DataFrame(rows, columns=["mc_class", "n_ffpe", "n_frozen", "weight"])
    return classes, D


def pooled_shift(D, weights):
    """Weighted average of the per-class differences (what the corrector fits)."""
    w = np.asarray(weights, dtype=np.float64)
    return (w[:, None] * D).sum(axis=0) / w.sum()


def class_table(classes, D):
    """Per class: weight share and agreement with the shift from all other classes."""
    w = classes["weight"].to_numpy(dtype=np.float64)
    W = w.sum()
    shift = pooled_shift(D, w)
    out = classes.copy()
    out["weight_share"] = w / W
    med, med_abs, n_big, corr = [], [], [], []
    for i in range(len(w)):
        d = D[i].astype(np.float64)
        med.append(np.median(d))
        med_abs.append(np.median(np.abs(d)))
        n_big.append(int((np.abs(d) > 0.3).sum()))
        if len(w) > 1 and d.std() > 0:
            others = (shift * W - w[i] * d) / (W - w[i])  # shift without this class
            corr.append(np.corrcoef(d, others)[0, 1] if others.std() > 0 else np.nan)
        else:
            corr.append(np.nan)
    out["median_diff"] = med
    out["median_abs_diff"] = med_abs
    out["n_probes_abs_diff_gt_0.3"] = n_big
    out["corr_with_other_classes"] = corr
    return out.sort_values("weight", ascending=False).reset_index(drop=True)


def probe_table(classes, D, probe_ids, top=300):
    """The `top` probes by |shift|, and the per-class differences behind them."""
    w = classes["weight"].to_numpy(dtype=np.float64)
    W = w.sum()
    names = classes["mc_class"].to_numpy()
    shift = pooled_shift(D, w)
    top = min(top, shift.size)
    order = np.argsort(-np.abs(shift), kind="stable")[:top]
    sub = D[:, order].astype(np.float64)  # classes x top
    sgn = np.sign(shift[order])
    signed = (w[:, None] * sub / W) * sgn  # each class's part of |shift|
    k = np.argmax(signed, axis=0)  # class contributing most
    cols = np.arange(top)
    abs_shift = np.abs(shift[order])
    safe = np.where(abs_shift > 0, abs_shift, np.nan)
    if len(w) > 1:
        without = (shift[order] * W - w[k] * sub[k, cols]) / (W - w[k])
    else:
        without = np.full(top, np.nan)
    table = pd.DataFrame(
        {
            "rank": cols + 1,
            "Probe_ID": np.asarray(probe_ids, dtype=object)[order],
            "shift": shift[order],
            "n_classes": len(w),
            "n_same_sign": ((sub * sgn) > 0).sum(axis=0),
            "n_same_sign_gt_0.05": ((sub * sgn) > 0.05).sum(axis=0),
            "median_class_diff": np.median(sub, axis=0),
            "top_class": names[k],
            "top_class_diff": sub[k, cols],
            "top_class_n_ffpe": classes["n_ffpe"].to_numpy()[k],
            "top_class_n_frozen": classes["n_frozen"].to_numpy()[k],
            "top_class_share": signed[k, cols] / safe,
            "shift_without_top_class": without,
        }
    )
    wide = pd.DataFrame(sub.T, columns=names)
    wide.insert(0, "Probe_ID", table["Probe_ID"].to_numpy())
    return table, wide


# --------------------------------------------------------------------------
# Part 2: material model on held-out classes
# --------------------------------------------------------------------------
def column_variance(Z, rows, block=64):
    """Per-column variance over the given rows, without copying them all."""
    p = Z.shape[1]
    s, ss = np.zeros(p), np.zeros(p)
    for i in range(0, len(rows), block):
        b = Z[rows[i : i + block]].astype(np.float64)
        s += b.sum(axis=0)
        ss += np.einsum("ij,ij->j", b, b)
    m = s / len(rows)
    return ss / len(rows) - m * m


def material_model(
    Z, y, material, classes_used, ffpe_label="FFPE", min_per_material=2, n_probes=5000, n_splits=5
):
    """Predict material on classes the model (and the correction) never saw.

    Returns one row per held-out class, with two within-class AUCs:
      auc_off  model trained and tested on uncorrected betas
      auc_on   that same model, tested on corrected betas
               (0.5 = right amount removed; near 1 = too little; near 0 = too
               much: FFPE now looks "more frozen than frozen")
    The model is NOT retrained on corrected betas: the correction sets the
    weighted within-class FFPE-minus-frozen mean to zero on the training
    classes, which is exactly what a linear model looks for, so a retrained
    linear model sees nothing by construction and would report 0.5 whether or
    not the correction fits. The same probes are used in both columns (the most
    variable on the training classes, uncorrected).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from methylclf.features import MaterialCorrector

    y = np.asarray(y, dtype=object)
    material = np.asarray(material, dtype=object)
    rows = np.flatnonzero(np.isin(y, list(classes_used)))
    n_splits = min(n_splits, len(set(classes_used)))
    if n_splits < 2:
        fail("material_model: needs at least 2 classes with both materials")
    out = []
    splits = GroupKFold(n_splits=n_splits).split(rows, groups=y[rows])
    for fold, (a, b) in enumerate(splits):
        tr, te = np.sort(rows[a]), np.sort(rows[b])
        var = column_variance(Z, tr)
        cols = np.sort(np.argsort(-var, kind="stable")[: min(n_probes, Z.shape[1])])
        A_tr = np.ascontiguousarray(Z[tr][:, cols], dtype=np.float32)
        A_te = np.ascontiguousarray(Z[te][:, cols], dtype=np.float32)
        t_tr = (material[tr] == ffpe_label).astype(int)
        t_te = (material[te] == ffpe_label).astype(int)
        corr = MaterialCorrector(ffpe_label, min_per_material=min_per_material, copy=True).fit(
            A_tr, y[tr], material[tr]
        )
        C_te = corr.transform(A_te, material[te])

        def new_model():
            return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))

        raw_model = new_model().fit(A_tr, t_tr)  # learns what FFPE looks like
        scores = {
            "off": raw_model.decision_function(A_te),
            # the same detector, shown corrected samples: 0.5 = the correction
            # fits this class; >0.5 = too little removed; <0.5 = too much
            "on": raw_model.decision_function(C_te),
        }
        for c in sorted(set(y[te])):
            m = y[te] == c
            nf, nz = int(t_te[m].sum()), int((1 - t_te[m]).sum())
            out.append(
                {
                    "fold": fold,
                    "mc_class": c,
                    "n_ffpe": nf,
                    "n_frozen": nz,
                    "weight": nf * nz / (nf + nz),
                    "auc_off": roc_auc_score(t_te[m], scores["off"][m]),
                    "auc_on": roc_auc_score(t_te[m], scores["on"][m]),
                }
            )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Everything on an in-memory matrix (this is what the tests call)
# --------------------------------------------------------------------------
def run(
    Z,
    y,
    material,
    probe_ids,
    out_dir,
    top=300,
    min_per_material=2,
    ffpe_label="FFPE",
    model_probes=5000,
    skip_model=False,
):
    """Z: imputed betas (samples x probes, no NaN), training-fold rows only."""
    from methylclf.features import MaterialCorrector

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(probe_ids) != Z.shape[1]:
        fail(f"run: {Z.shape[1]} probes but {len(probe_ids)} probe IDs")

    classes, D = per_class_differences(Z, y, material, ffpe_label, min_per_material)
    w = classes["weight"].to_numpy(dtype=np.float64)
    shift = pooled_shift(D, w)

    # The diagnostic must describe the real code: compare with MaterialCorrector.
    fitted = MaterialCorrector(ffpe_label, min_per_material=min_per_material, copy=False).fit(
        Z, y, material
    )
    gap = float(np.max(np.abs(fitted.shift_.astype(np.float64) - shift)))
    if gap > 1e-4 or list(fitted.classes_used_) != list(classes["mc_class"]):
        fail(
            f"run: this script's shift differs from MaterialCorrector's "
            f"(max difference {gap:.2e}); the diagnostic would not describe the code"
        )

    ctab = class_table(classes, D)
    ptab, wide = probe_table(classes, D, probe_ids, top)
    ctab.to_csv(out_dir / "class_table.tsv", sep="\t", index=False, float_format="%.5g")
    ptab.to_csv(out_dir / "top_probes.tsv", sep="\t", index=False, float_format="%.5g")
    wide.to_csv(out_dir / "top_probes_by_class.tsv", sep="\t", index=False, float_format="%.4f")

    # Robust alternative for comparison: plain median of the class differences.
    med = np.median(D, axis=0)
    a = np.abs(shift)
    summary = [
        ("n_samples", Z.shape[0]),
        ("n_probes", Z.shape[1]),
        ("n_classes_used", len(classes)),
        ("n_ffpe_in_used_classes", int(classes["n_ffpe"].sum())),
        ("n_frozen_in_used_classes", int(classes["n_frozen"].sum())),
        ("max_abs_diff_vs_MaterialCorrector", gap),
        ("shift_median", float(np.median(shift))),
        ("shift_abs_median", float(np.median(a))),
        ("shift_abs_max", float(a.max())),
        ("n_probes_abs_shift_gt_0.05", int((a > 0.05).sum())),
        ("n_probes_abs_shift_gt_0.10", int((a > 0.10).sum())),
        ("median_of_classes_abs_max", float(np.abs(med).max())),
        ("n_probes_abs_median_of_classes_gt_0.05", int((np.abs(med) > 0.05).sum())),
        ("n_probes_abs_median_of_classes_gt_0.10", int((np.abs(med) > 0.10).sum())),
        (
            "corr_shift_vs_median_of_classes",
            (
                float(np.corrcoef(shift, med)[0, 1])
                if shift.std() > 0 and med.std() > 0
                else float("nan")
            ),
        ),
        ("top_n", len(ptab)),
        ("top_min_abs_shift", float(ptab["shift"].abs().min())),
        ("top_n_same_sign_min", int(ptab["n_same_sign"].min())),
        ("top_n_same_sign_median", float(ptab["n_same_sign"].median())),
        ("top_n_same_sign_max", int(ptab["n_same_sign"].max())),
        ("top_n_one_class_gives_over_half", int((ptab["top_class_share"] > 0.5).sum())),
        (
            "top_n_shift_halves_without_top_class",
            int((ptab["shift_without_top_class"].abs() < 0.5 * ptab["shift"].abs()).sum()),
        ),
    ]

    mtab = None
    if not skip_model:
        mtab = material_model(
            Z, y, material, list(classes["mc_class"]), ffpe_label, min_per_material, model_probes
        )
        mtab.to_csv(out_dir / "material_model.tsv", sep="\t", index=False, float_format="%.4f")
        mw = mtab["weight"].to_numpy()
        summary += [
            ("model_classes_scored", len(mtab)),
            ("model_auc_off_weighted", float(np.average(mtab["auc_off"], weights=mw))),
            ("model_auc_on_weighted", float(np.average(mtab["auc_on"], weights=mw))),
            ("model_n_classes_under_corrected_auc_on_gt_0.8", int((mtab["auc_on"] > 0.8).sum())),
            ("model_n_classes_over_corrected_auc_on_lt_0.2", int((mtab["auc_on"] < 0.2).sum())),
            ("model_n_classes_auc_on_0.35_to_0.65", int(mtab["auc_on"].between(0.35, 0.65).sum())),
        ]
    stab = pd.DataFrame(summary, columns=["item", "value"])
    stab.to_csv(out_dir / "summary.tsv", sep="\t", index=False, float_format="%.5g")
    return {"classes": ctab, "probes": ptab, "wide": wide, "model": mtab, "summary": stab}


def report(res, show=15):
    pd.set_option(
        "display.width",
        250,
        "display.max_columns",
        50,
        "display.max_rows",
        500,
        "display.float_format",
        "{:.4g}".format,
    )
    print("\n== summary ==")
    print(res["summary"].to_string(index=False))
    print("\n== classes used (sorted by weight) ==")
    print(res["classes"].to_string(index=False))
    p = res["probes"]
    print(f"\n== top {show} probes by |shift| ==")
    print(p.head(show).to_string(index=False))
    print(f"\n== which class contributes most, over the top {len(p)} probes ==")
    print(p["top_class"].value_counts().head(10).to_string())
    if res["model"] is not None:
        print("\n== material model, per held-out class ==")
        print(res["model"].to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default="data/betas/zarr/GSE90496.zarr")
    ap.add_argument("--folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--probes", default="results/probes/probes_kept.tsv")
    ap.add_argument("--out", default="results/material_shift")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--min-per-material", type=int, default=2)
    ap.add_argument("--max-missing", type=float, default=0.05)
    ap.add_argument("--model-probes", type=int, default=5000)
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args(argv)

    from methylclf.data import BetaStore
    from methylclf.features import MedianImputer, MissingnessFilter

    t0 = time.time()
    st = BetaStore.open(args.store, args.folds, args.probes)
    s = st.samples
    train = (s["outer_fold"].astype(int) != OUTER_FOLD).to_numpy()
    ids, y, mat = (s[c].to_numpy() for c in ("geo_accession", "mc_class", "material"))
    print(
        f"training side of outer fold {OUTER_FOLD}: {train.sum()} samples "
        f"(the {(~train).sum()} test-fold samples are not read)"
    )

    # Same first two steps as FeaturePipeline.fit, on the same rows.
    X = st.load(ids[train])
    missing = MissingnessFilter(args.max_missing).fit(X)
    Z = missing.transform(X)
    del X
    Z = MedianImputer(copy=False).fit(Z).transform(Z)
    probe_ids = np.asarray(missing.get_feature_names_out(st.probe_ids), dtype=object)
    print(
        f"after missingness filter and imputation: {Z.shape[0]} x {Z.shape[1]} "
        f"({time.time() - t0:.0f} s)"
    )

    res = run(
        Z,
        y[train],
        mat[train],
        probe_ids,
        args.out,
        args.top,
        args.min_per_material,
        "FFPE",
        args.model_probes,
        args.skip_model,
    )
    report(res)
    print(f"\nwrote {args.out}/ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
