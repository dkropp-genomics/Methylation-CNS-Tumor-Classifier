#!/usr/bin/env python
"""External validation on GSE109379: score once, then report.

  score   FINAL SCORING, refused without --final-scoring. Loads the saved final
          models (scripts/fit_final.py) and predicts every QC-passed validation
          sample at each coverage level. Only probabilities are saved. This
          stage is given sample IDs and materials, never labels, and a file
          that exists is never overwritten: the cohort is scored once.
  report  Reads the saved probabilities and the labels and writes every table
          and figure agreed in configs/final_v1.yaml. Fits nothing and can be
          rerun freely.

Nothing here is fit to the validation cohort. Each sample goes through the
feature pipeline learned on GSE90496 (probe filter, medians, material shift,
probe selection), so one sample's result does not depend on any other.

What the numbers mean: the validation labels are the published classifier's
own calls (see configs/final_v1.yaml). Every figure is AGREEMENT with those
calls, not accuracy against an independent truth.

  python scripts/score_validation.py configs/final_v1.yaml --stage score --final-scoring
  python scripts/score_validation.py configs/final_v1.yaml --stage report

Outputs:
  data/predictions/final_v1/val_<model>.npz        probabilities (git-ignored)
  results/final_v1/validation_scored.json          when and what was scored
  results/final_v1/validation_metrics.tsv          every metric with intervals
  results/final_v1/validation_target.tsv           the pre-registered target
  results/final_v1/validation_top_class.tsv        agreement by model and coverage
  results/final_v1/validation_differences.tsv      paired model differences
  results/final_v1/validation_per_class.tsv, validation_predictions.tsv
  results/final_v1/validation_sparsity_curve.png, validation_sparsity_curve_family.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_final as ff  # noqa: E402
import sparsity_outer as so  # noqa: E402

cv = ff.cv
ORDER_METRICS = ["macro_f1", "balanced_accuracy", "accuracy"]  # need only the ranking


def val_path(pred_dir: Path, name: str) -> Path:
    return Path(pred_dir) / f"val_{name}.npz"


# --------------------------------------------------------------------------
# score: probabilities only, no labels
# --------------------------------------------------------------------------
def stage_score(cfg, ids, material, load, probe_ids, pred_dir, model_dir, res_dir):
    from methylclf.masking import observed_uniform, probe_positions

    levels = [float(v) for v in cfg["analyses"]["sparsity"]["levels"]]
    if 1.0 not in levels:
        cv.fail("config: analyses.sparsity.levels must include 1.0")
    seed = int(cfg["analyses"]["sparsity"]["mask_seed"])
    todo = [n for n in cfg["analyses"]["full_coverage"] if not val_path(pred_dir, n).exists()]
    for name in cfg["analyses"]["full_coverage"]:
        if name not in todo:
            print(f"{name}: already scored; the saved file is kept", flush=True)
    if not todo:
        return 0
    arts = {n: ff.load_model(ff.model_path(model_dir, n)) for n in todo}
    digest = ff.probes_digest(probe_ids)
    for name, art in arts.items():  # every check before any data is read
        if art["probes_sha256"] != digest or art["n_array_probes"] != len(probe_ids):
            cv.fail(
                f"{name}: the validation store's probe list or order differs from the "
                f"one the model was fitted on"
            )
        if art["run_name"] != cfg["run_name"]:
            cv.fail(f"{name}: the saved model belongs to run {art['run_name']}")
    t0 = time.time()
    X = load(ids)
    print(
        f"loaded {X.shape[0]} x {X.shape[1]} validation samples in {time.time() - t0:.0f} s",
        flush=True,
    )
    done = []
    for name, art in arts.items():
        t1 = time.time()
        Z = art["pipe"].transform(X, material)
        if np.isnan(Z).any():
            cv.fail(f"{name}: NaN after the feature pipeline")
        u = observed_uniform(
            ids, len(probe_ids), probe_positions(art["feature_names"], probe_ids), seed
        )
        proba = ff.predict_levels(art, Z, u, levels).astype(np.float32)
        extra = {}
        if art["calibrator"] is not None:  # tree models, full coverage only
            extra["calibrated"] = ff.calibrated(art, proba[levels.index(1.0)])
        path = val_path(pred_dir, name)
        tmp = path.with_name(path.name[:-4] + ".tmp.npz")
        np.savez_compressed(
            tmp,
            proba=proba,
            sample_ids=np.asarray(ids, dtype=str),
            classes=np.asarray(art["classes"], dtype=str),
            levels=np.asarray(levels, dtype=np.float64),
            **extra,
        )
        os.replace(tmp, path)
        done.append(name)
        print(
            f"{name}: scored {len(ids)} samples at {len(levels)} coverage levels in "
            f"{time.time() - t1:.0f} s",
            flush=True,
        )
        del Z, u, proba
    record = res_dir / "validation_scored.json"
    old = json.loads(record.read_text()) if record.exists() else []
    old.append(
        {
            "scored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": done,
            "n_samples": int(len(ids)),
            "levels": levels,
            "mask_seed": seed,
        }
    )
    record.write_text(json.dumps(old, indent=2) + "\n")
    return len(done)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def load_val(pred_dir, name, ids):
    path = val_path(pred_dir, name)
    if not path.exists():
        cv.fail(f"{path} not found; run --stage score --final-scoring first")
    with np.load(path) as z:
        if list(z["sample_ids"].astype(object)) != list(ids):
            cv.fail(f"{path.name}: samples differ from the kept-sample table")
        out = {
            "proba": z["proba"].astype(np.float64),
            "classes": list(z["classes"].astype(object)),
            "levels": [float(v) for v in z["levels"]],
        }
        if "calibrated" in z.files:
            out["calibrated"] = z["calibrated"].astype(np.float64)
    return out


def subsets_of(kept: pd.DataFrame, cfg) -> dict:
    """Name -> row positions. "all" first, then the subsets fixed before scoring."""
    out = {"all": np.arange(len(kept))}
    for name in cfg["labels"]["subsets"]:
        if name not in kept.columns or not set(kept[name]) <= {"True", "False"}:
            cv.fail(f"kept-sample table: subset column '{name}' must hold True/False")
        out[name] = np.flatnonzero((kept[name] == "True").to_numpy())
        if out[name].size == 0:
            cv.fail(f"subset '{name}' is empty")
    return out


def boot_share(ok, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    if not n_boot:
        return np.nan, np.nan
    b = [ok[rng.integers(0, ok.size, ok.size)].mean() for _ in range(n_boot)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def plot_curve(top, path, label_level, n):
    """Agreement against the share of CpGs observed (same look as the CV figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e6e5df"
    t = top[(top["label_level"] == label_level) & (top["subset"] == "all")]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    order = [m for m in so.LABELS if m in set(t["model"])] + sorted(
        set(t["model"]) - set(so.LABELS)
    )
    for name in order:
        d = t[t["model"] == name].sort_values("coverage", ascending=False)
        color, marker = so.STYLE.get(name, ("#6b6a63", "x"))
        x, v = d["coverage"].to_numpy() * 100, d["agreement"].to_numpy()
        ax.errorbar(
            x,
            v,
            yerr=[v - d["ci_low"].to_numpy(), d["ci_high"].to_numpy() - v],
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            capsize=3,
            markeredgecolor="#fcfcfb",
            markeredgewidth=1.2,
            label=so.LABELS.get(name, name),
            zorder=3,
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    levels = sorted(set(t["coverage"] * 100), reverse=True)
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{v:g}%" for v in levels])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Share of array CpGs observed (fewer to the right)", color=muted)
    what = "class" if label_level == "class" else "family of the predicted class"
    ax.set_ylabel(f"Agreement with published calls ({what})", color=muted)
    ax.set_title(
        "External cohort: agreement as fewer CpGs are observed", color=ink, loc="left", fontsize=12
    )
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, length=0)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=ink)
    fig.text(
        0.01,
        0.01,
        f"GSE109379, n = {n:,}, models fit on GSE90496; labels are the "
        "published classifier's calls; bars are 95% bootstrap intervals.",
        color=muted,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def stage_report(
    cfg, kept, family_of, summarize, to_family, pred_dir, res_dir, reference=None, n_boot=None
):
    n_boot = int(cfg["analyses"]["n_boot"]) if n_boot is None else n_boot
    ids = kept["geo_accession"].to_numpy(dtype=object)
    y = kept["mc_class"].to_numpy(dtype=object)
    mat = kept["material"].to_numpy(dtype=object)
    subsets = subsets_of(kept, cfg)
    models = list(cfg["analyses"]["full_coverage"])
    preds = {n: load_val(pred_dir, n, ids) for n in models}
    classes, levels = preds[models[0]]["classes"], preds[models[0]]["levels"]
    for n, p in preds.items():
        if p["classes"] != classes or p["levels"] != levels:
            cv.fail(f"{n}: classes or coverage levels differ from the other models")
    unknown = sorted(set(y) - set(classes))
    if unknown:
        cv.fail(f"{len(unknown)} validation label(s) are not model classes, first: {unknown[0]!r}")
    full = levels.index(1.0)
    cls_arr = np.asarray(classes, dtype=object)
    index = {c: i for i, c in enumerate(classes)}
    t_idx = np.array([index[c] for c in y])
    fam_of_class = np.array([family_of[c] for c in classes], dtype=object)
    fam_true = np.array([family_of[c] for c in y], dtype=object)

    # ---- 1. every metric at full coverage, by subset; by material for "all"
    tables = []
    for name in models:
        kinds = [("raw", preds[name]["proba"][full])]
        if "calibrated" in preds[name]:
            kinds.append(("calibrated", preds[name]["calibrated"]))
        for scores, P in kinds:
            for sub, rows in subsets.items():
                t = summarize(
                    y[rows],
                    P[rows],
                    classes,
                    family_of=family_of,
                    groups=mat[rows] if sub == "all" else None,
                    n_boot=n_boot,
                )
                # a family score is a sum of class scores: only meaningful when calibrated
                t = t[(t["level"] == "class") | (scores == "calibrated")]
                if cfg["models"][name]["kind"] == "centroid":  # its scores are distances
                    t = t[t["metric"].isin(ORDER_METRICS)]
                t = t.copy()
                t.insert(0, "subset", sub)
                t.insert(0, "scores", scores)
                t.insert(0, "model", name)
                tables.append(t)
        print(f"{name}: metrics done", flush=True)
    metrics = pd.concat(tables, ignore_index=True)

    # ---- 2. top-class agreement at every coverage level (raw scores, all models)
    rows, top = [], {}
    for name in models:
        for a, lv in enumerate(levels):
            pred = preds[name]["proba"][a].argmax(axis=1)
            top[(name, lv)] = pred
            for sub, r in subsets.items():
                for label, ok in (
                    ("class", pred == t_idx),
                    ("family_of_predicted_class", fam_of_class[pred] == fam_true),
                ):
                    lo, hi = boot_share(ok[r], n_boot)
                    rows.append(
                        {
                            "model": name,
                            "coverage": lv,
                            "subset": sub,
                            "label_level": label,
                            "agreement": float(ok[r].mean()),
                            "ci_low": lo,
                            "ci_high": hi,
                            "n": int(r.size),
                        }
                    )
    top_acc = pd.DataFrame(rows)

    # ---- 3. paired differences between models, as in Phase 4
    rows = []
    for a, b in so.PAIRS:
        if a in models and b in models:
            for lv in levels:
                for r in so.paired_difference(
                    t_idx, top[(a, lv)], top[(b, lv)], len(classes), n_boot, seed=0
                ):
                    rows.append({"model_a": a, "model_b": b, "coverage": lv, **r})
    diffs = pd.DataFrame(rows)

    # ---- 4. the pre-registered target
    tg = cfg["target"]
    hit = metrics[
        (metrics["model"] == tg["model"])
        & (metrics["scores"] == tg["scores"])
        & (metrics["subset"] == "all")
        & (metrics["group"] == "all")
        & (metrics["level"] == tg["level"])
        & (metrics["metric"] == tg["metric"])
    ]
    if len(hit) != 1:
        cv.fail("target: the target metric is not among the computed metrics")
    target = None
    if reference is not None:
        ref = reference[
            (reference["run"] == cfg["models"][tg["model"]]["run"])
            & (reference["level"] == tg["level"])
            & (reference["metric"] == tg["metric"])
        ]
        if len(ref) != 1:
            cv.fail("target: the CV reference has no row for the target metric")
        cv_value, value = float(ref["value"].iloc[0]), float(hit["value"].iloc[0])
        target = pd.DataFrame(
            [
                {
                    "model": tg["model"],
                    "scores": tg["scores"],
                    "level": tg["level"],
                    "metric": tg["metric"],
                    "cv_reference": cv_value,
                    "cv_n_groups": int(ref["n_groups"].iloc[0]),
                    "max_drop": float(tg["max_drop"]),
                    "threshold": cv_value - float(tg["max_drop"]),
                    "validation": value,
                    "ci_low": float(hit["ci_low"].iloc[0]),
                    "ci_high": float(hit["ci_high"].iloc[0]),
                    "n": int(hit["n"].iloc[0]),
                    "n_groups": int(hit["n_classes"].iloc[0]),
                    "difference": value - cv_value,
                    "met": bool(value >= cv_value - float(tg["max_drop"])),
                }
            ]
        )

    # ---- 5. per sample and per class, for the headline model's calibrated scores
    head = cfg["analyses"]["headline"]
    P = preds[head].get("calibrated", preds[head]["proba"][full])
    pred = cls_arr[P.argmax(axis=1)]
    fam_proba, families = to_family(P, classes, family_of)
    fam_pred = np.asarray(families, dtype=object)[fam_proba.argmax(axis=1)]
    per_sample = pd.DataFrame(
        {
            "geo_accession": ids,
            "material": mat,
            "mc_class": y,
            "predicted_class": pred,
            "class_score": P.max(axis=1),
            "family": fam_true,
            "predicted_family": fam_pred,
            "family_score": fam_proba.max(axis=1),
        }
    )
    for c in ["capper_score"] + list(cfg["labels"]["subsets"]):
        if c in kept.columns:
            per_sample[c] = kept[c].to_numpy()
    for name in models:
        if name != head:
            per_sample[f"{name}_class"] = cls_arr[top[(name, 1.0)]]
    rows = []
    for c in sorted(set(y)):
        n, n_pred = int((y == c).sum()), int((pred == c).sum())
        tp = int(((y == c) & (pred == c)).sum())
        rows.append(
            {
                "mc_class": c,
                "family": family_of[c],
                "n": n,
                "n_predicted": n_pred,
                "sensitivity": tp / n,
                "precision": tp / n_pred if n_pred else np.nan,
                "f1": 2 * tp / (n + n_pred),
                "n_family_agree": int(((y == c) & (fam_pred == fam_true)).sum()),
            }
        )
    per_class = pd.DataFrame(rows)

    res_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "validation_metrics.tsv": metrics,
        "validation_top_class.tsv": top_acc,
        "validation_differences.tsv": diffs,
        "validation_per_class.tsv": per_class,
        "validation_predictions.tsv": per_sample,
    }
    if target is not None:
        out["validation_target.tsv"] = target
    for fname, t in out.items():
        t.to_csv(res_dir / fname, sep="\t", index=False, float_format="%.5f")
    for label, stem in (
        ("class", "validation_sparsity_curve"),
        ("family_of_predicted_class", "validation_sparsity_curve_family"),
    ):
        plot_curve(top_acc, res_dir / f"{stem}.png", label, len(ids))
    return metrics, top_acc, diffs, target, per_class, per_sample


# --------------------------------------------------------------------------
def run(
    stage,
    cfg,
    kept,
    load=None,
    probe_ids=None,
    pred_root="data/predictions",
    res_root="results",
    model_root="data/models",
    final_scoring=False,
    family_of=None,
    summarize=None,
    to_family=None,
    reference=None,
    n_boot=None,
):
    for key in ("run_name", "models", "labels", "target", "analyses"):
        if key not in cfg:
            cv.fail(f"config: missing '{key}'")
    pred_dir = Path(pred_root) / cfg["run_name"]
    res_dir = Path(res_root) / cfg["run_name"]
    if stage == "score":
        if not final_scoring:
            cv.fail(
                "score: this stage scores the validation cohort, which is done once. "
                "Add --final-scoring to confirm."
            )
        pred_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)
        return stage_score(
            cfg,
            kept["geo_accession"].to_numpy(dtype=object),
            kept["material"].to_numpy(dtype=object),
            load,
            probe_ids,
            pred_dir,
            Path(model_root) / cfg["run_name"],
            res_dir,
        )
    if stage == "report":
        return stage_report(
            cfg, kept, family_of, summarize, to_family, pred_dir, res_dir, reference, n_boot
        )
    cv.fail(f"unknown stage '{stage}'")


def text(v, lo, hi):
    return f"{v:.3f} ({lo:.3f}-{hi:.3f})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--stage", required=True, choices=["score", "report"])
    ap.add_argument("--final-scoring", action="store_true")
    ap.add_argument("--store", default="data/betas/zarr/GSE109379.zarr")
    ap.add_argument("--kept", default="results/splits/GSE109379_kept.tsv")
    ap.add_argument("--probes", default="results/probes/probes_kept.tsv")
    ap.add_argument("--families", default="results/meta/class_to_family.tsv")
    ap.add_argument("--pred-root", default="data/predictions")
    ap.add_argument("--res-root", default="results")
    ap.add_argument("--model-root", default="data/models")
    args = ap.parse_args(argv)

    import yaml

    for path in (args.config, args.kept, args.families):
        if not Path(path).is_file():
            cv.fail(f"{path}: not found")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.stage == "score":
        if not args.final_scoring:  # refuse before the store is opened
            cv.fail(
                "score: this stage scores the validation cohort, which is done once. "
                "Add --final-scoring to confirm."
            )
        if cfg["validation_cohort"] not in Path(args.store).name:
            cv.fail(f"--store must be the {cfg['validation_cohort']} store")
        from methylclf.data import BetaStore

        st = BetaStore.open(args.store, args.kept, args.probes)
        n = run(
            "score",
            cfg,
            st.samples,
            st.load,
            np.asarray(st.probe_ids, dtype=object),
            args.pred_root,
            args.res_root,
            args.model_root,
            final_scoring=True,
        )
        print(f"score stage: {n} models scored now. Next: --stage report")
        return

    from methylclf.metrics import summarize, to_family

    kept = pd.read_csv(args.kept, sep="\t", dtype=str, keep_default_na=False)
    fam = pd.read_csv(args.families, sep="\t", dtype=str, keep_default_na=False)
    ref_path = Path(cfg["target"]["reference"])
    if not ref_path.is_file():
        cv.fail(f"{ref_path}: not found; run scripts/cv_reference.py first")
    metrics, top, diffs, target, per_class, per_sample = run(
        "report",
        cfg,
        kept,
        pred_root=args.pred_root,
        res_root=args.res_root,
        family_of=dict(zip(fam["mc_class"], fam["family"])),
        summarize=summarize,
        to_family=to_family,
        reference=pd.read_csv(ref_path, sep="\t"),
    )
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    head = cfg["analyses"]["headline"]
    print(
        f"\nAll figures are AGREEMENT with the published classifier's calls " f"(n = {len(kept)})."
    )
    r = target.iloc[0]
    print(
        f"\n== pre-registered target: {r['model']} {r['scores']}, {r['level']}-level "
        f"{r['metric']} =="
    )
    print(
        f"CV reference {r['cv_reference']:.3f}; threshold {r['threshold']:.3f}; "
        f"validation {text(r['validation'], r['ci_low'], r['ci_high'])}; "
        f"difference {r['difference']:+.3f}; target {'MET' if r['met'] else 'NOT MET'}"
    )
    m = metrics[
        (metrics["model"] == head)
        & (metrics["scores"] == "calibrated")
        & (metrics["group"] == "all")
    ].copy()
    m["text"] = [text(*v) for v in zip(m["value"], m["ci_low"], m["ci_high"])]
    for level in ("class", "family"):
        t = m[m["level"] == level].pivot_table(
            index="metric", columns="subset", values="text", aggfunc="first", sort=False
        )
        print(
            f"\n== {head}, calibrated, {level} level; columns = subset; " f"value (95% interval) =="
        )
        print(t[list(dict.fromkeys(m["subset"]))].to_string())
    n_sub = m.drop_duplicates("subset").set_index("subset")["n"]
    print("samples per subset:", n_sub.to_dict())
    g = metrics[
        (metrics["model"] == head)
        & (metrics["scores"] == "calibrated")
        & (metrics["subset"] == "all")
        & (metrics["group"] != "all")
        & (metrics["metric"].isin(["accuracy", "macro_f1", "ece"]))
    ]
    print(f"\n== {head}, calibrated, by material ==")
    print(
        g[["group", "level", "metric", "value", "ci_low", "ci_high", "n"]].to_string(
            index=False, float_format="{:.3f}".format
        )
    )
    cs = cfg["analyses"]["confident_share"]
    share = m[
        (m["level"] == "class") & (m["metric"] == "confident_share") & (m["subset"] == "all")
    ].iloc[0]
    print(
        f"\nshare scoring >= {cs['threshold']}: validation {share['text']}; "
        f"CV {cs['cv_value']:.3f}"
    )
    a = top[top["subset"] == "all"].copy()
    a["text"] = [text(*v) for v in zip(a["agreement"], a["ci_low"], a["ci_high"])]
    for label in ("class", "family_of_predicted_class"):
        t = a[a["label_level"] == label].pivot_table(
            index="model", columns="coverage", values="text", aggfunc="first", sort=False
        )
        print(f"\n== agreement (raw scores), {label}; columns = share of CpGs observed ==")
        print(t[sorted(t.columns, reverse=True)].to_string())
    print("\n== paired differences in class agreement, model_a minus model_b ==")
    print(
        diffs[diffs["metric"] == "accuracy"]
        .drop(columns="metric")
        .to_string(index=False, float_format="{:.4f}".format)
    )
    wrong = per_sample[per_sample["mc_class"] != per_sample["predicted_class"]]
    print(
        f"\n{head}: {len(wrong)} of {len(per_sample)} samples differ from the published "
        f"call at class level; "
        f"{int((per_sample['family'] != per_sample['predicted_family']).sum())} at family level"
    )
    print(f"tables and figures are in {Path(args.res_root) / cfg['run_name']}/")


if __name__ == "__main__":
    main()
