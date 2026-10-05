#!/usr/bin/env python3
"""Sample-level QC in two separate steps.

  summarize  Look at the TRAINING cohort's QC numbers only (no classes, no
             material, no validation cohort). Writes a count-below-threshold
             grid and histograms, so a threshold can be chosen.

  apply      Read the committed threshold from configs/sample_qc.yaml and apply
             it, unchanged, to both cohorts. Only now are classes and material
             joined in, to report what the threshold removed.

The split exists so the threshold is fixed before anything about classes is
seen. `summarize` never opens the fold or label tables.

Usage:
  python scripts/sample_qc.py summarize
  python scripts/sample_qc.py apply
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN = "GSE90496"
VALID = "GSE109379"
METRICS = ("frac_detected", "bisulfite_gct", "mean_intensity")
CONFIG_KEYS = ("metric", "min_value", "decided_on", "decided_from", "reason")
MIN_CLASS_SIZE = 5  # 5 outer folds need at least 5 samples per class


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def read_qc(path):
    """Read one cohort's SeSAMe QC table and check what we rely on."""
    path = Path(path)
    if not path.exists():
        die(f"QC table not found: {path}")
    q = pd.read_csv(path, sep="\t")
    missing = {"cohort", "geo_accession", "status", *METRICS} - set(q.columns)
    if missing:
        die(f"{path}: missing columns {sorted(missing)}")
    if q["geo_accession"].duplicated().any():
        dup = q.loc[q["geo_accession"].duplicated(), "geo_accession"].iloc[0]
        die(f"{path}: duplicated sample {dup}")
    if (q["status"] != "ok").any():
        bad = q.loc[q["status"] != "ok", "geo_accession"].iloc[0]
        die(f"{path}: sample {bad} has status other than 'ok'")
    if q[list(METRICS)].isna().any().any():
        die(f"{path}: missing values in a QC metric column")
    return q


# --------------------------------------------------------------------------
# summarize: training cohort only, no labels
# --------------------------------------------------------------------------
def summarize(args):
    qc_path = Path(args.qc_dir) / f"{TRAIN}_sesame_qc.tsv"
    q = read_qc(qc_path)
    cohorts = sorted(q["cohort"].unique())
    if cohorts != [TRAIN]:
        die(
            f"{qc_path}: expected only cohort {TRAIN}, found {cohorts}. "
            "The threshold must be chosen from training samples only."
        )

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    n = len(q)
    fd = q["frac_detected"].to_numpy()

    # How many samples fall below each candidate cut.
    grid = np.round(np.arange(0.30, 0.9501, 0.05), 2)
    tab = pd.DataFrame(
        {
            "threshold": grid,
            "n_below": [(fd < t).sum() for t in grid],
        }
    )
    tab["pct_below"] = (100 * tab["n_below"] / n).round(2)
    grid_path = out_dir / f"{TRAIN}_qc_grid.tsv"
    tab.to_csv(grid_path, sep="\t", index=False)

    print(f"{n} training samples. Samples with frac_detected below each cut:")
    print(tab.to_string(index=False))

    # The lowest values, sorted: a natural cut shows up as a gap.
    low = np.sort(fd)[: min(60, n)]
    print("\nLowest frac_detected values, sorted (look for a gap):")
    for i in range(0, len(low), 10):
        print("  " + "  ".join(f"{v:.3f}" for v in low[i : i + 10]))

    # Do the other metrics flag the same samples or different ones?
    print(
        "\nAmong samples below each cut, medians of the other metrics "
        "(all-sample medians: "
        f"bisulfite_gct {q['bisulfite_gct'].median():.2f}, "
        f"mean_intensity {q['mean_intensity'].median():.0f}):"
    )
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        sub = q[q["frac_detected"] < t]
        if len(sub):
            print(
                f"  < {t}: n={len(sub):4d}  bisulfite_gct "
                f"{sub['bisulfite_gct'].median():.2f}  mean_intensity "
                f"{sub['mean_intensity'].median():.0f}"
            )

    # Histograms. Log counts so a thin tail is visible.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, m in zip(axes, METRICS):
        ax.hist(q[m], bins=80, color="#4477aa")
        ax.set_yscale("log")
        ax.set_xlabel(m)
        ax.set_ylabel("samples (log scale)")
    axes[3].scatter(q["mean_intensity"], q["frac_detected"], s=4, alpha=0.4, color="#4477aa")
    axes[3].set_xlabel("mean_intensity")
    axes[3].set_ylabel("frac_detected")
    fig.suptitle(f"{TRAIN} sample QC (n={n}), pooled, no labels")
    fig.tight_layout()
    fig_path = out_dir / "figures" / f"{TRAIN}_qc_hist.png"
    fig.savefig(fig_path, dpi=130)
    print(f"\nWrote {grid_path}\nWrote {fig_path}")


# --------------------------------------------------------------------------
# apply: committed threshold, both cohorts, labels joined only for reporting
# --------------------------------------------------------------------------
def read_config(path):
    import yaml

    path = Path(path)
    if not path.exists():
        die(f"threshold file not found: {path}. Write and commit it before " "running 'apply'.")
    cfg = yaml.safe_load(path.read_text()) or {}
    missing = [k for k in CONFIG_KEYS if not cfg.get(k) and cfg.get(k) != 0]
    if missing:
        die(f"{path}: missing or empty keys {missing}")
    if cfg["metric"] != "frac_detected":
        die(f"{path}: metric must be 'frac_detected', got {cfg['metric']!r}")
    try:
        cfg["min_value"] = float(cfg["min_value"])
    except (TypeError, ValueError):
        die(f"{path}: min_value is not a number: {cfg['min_value']!r}")
    if not 0 < cfg["min_value"] < 1:
        die(f"{path}: min_value must be between 0 and 1, got {cfg['min_value']}")
    return cfg


def join_labels(q, table_path, cohort):
    """Attach class and material. Sample sets must match exactly."""
    table_path = Path(table_path)
    if not table_path.exists():
        die(f"{cohort}: table not found: {table_path}")
    t = pd.read_csv(table_path, sep="\t")
    for col in ("geo_accession", "mc_class", "material"):
        if col not in t.columns:
            die(f"{table_path}: missing column {col}")
    a, b = set(q["geo_accession"]), set(t["geo_accession"])
    if a != b:
        example = sorted(a ^ b)[0]
        die(
            f"{cohort}: QC table and {table_path} do not hold the same "
            f"samples ({len(a - b)} only in QC, {len(b - a)} only in table; "
            f"e.g. {example})"
        )
    return q.merge(
        t[["geo_accession", "mc_class", "material"]],
        on="geo_accession",
        how="left",
        validate="one_to_one",
    )


def apply(args):
    cfg = read_config(args.config)
    cut = cfg["min_value"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Threshold: keep samples with frac_detected >= {cut} " f"(decided {cfg['decided_on']})")

    small = None
    for cohort, table in ((TRAIN, args.folds), (VALID, args.labels)):
        q = read_qc(Path(args.qc_dir) / f"{cohort}_sesame_qc.tsv")
        q = join_labels(q, table, cohort)
        q["keep"] = q["frac_detected"] >= cut
        n, n_drop = len(q), int((~q["keep"]).sum())
        print(
            f"\n{cohort}: {n} samples, {n_drop} dropped "
            f"({100 * n_drop / n:.1f}%), {n - n_drop} kept"
        )

        status_path = out_dir / f"{cohort}_sample_qc_status.tsv"
        q[["geo_accession", "frac_detected", "keep"]].to_csv(status_path, sep="\t", index=False)

        mat = q.groupby("material")["keep"].agg(n="size", kept="sum")
        mat["dropped"] = mat["n"] - mat["kept"]
        mat["pct_dropped"] = (100 * mat["dropped"] / mat["n"]).round(1)
        print("By material:")
        print(mat.to_string())

        cls = q.groupby("mc_class")["keep"].agg(n_before="size", n_after="sum")
        cls["n_dropped"] = cls["n_before"] - cls["n_after"]
        cls = cls.sort_values(["n_after", "n_before"]).reset_index()
        impact_path = out_dir / f"{cohort}_qc_class_impact.tsv"
        cls.to_csv(impact_path, sep="\t", index=False)
        print(
            f"Classes losing at least one sample: "
            f"{(cls['n_dropped'] > 0).sum()} of {len(cls)}; "
            f"smallest class after QC: {cls['n_after'].min()}"
        )
        print(f"Wrote {status_path}\nWrote {impact_path}")
        if cohort == TRAIN:
            small = cls[cls["n_after"] < MIN_CLASS_SIZE]

    if len(small):
        print(
            f"\nWARNING: {len(small)} training class(es) fall below "
            f"{MIN_CLASS_SIZE} samples, too few for 5 outer folds:"
        )
        print(small.to_string(index=False))
        print(
            "Do not change the threshold in response. Decide how to handle "
            "these classes and record it."
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize", help="training QC distributions, no labels")
    s.add_argument("--qc-dir", default="results/qc")
    s.add_argument("--out-dir", default="results/qc")
    s.set_defaults(func=summarize)

    a = sub.add_parser("apply", help="apply the committed threshold")
    a.add_argument("--qc-dir", default="results/qc")
    a.add_argument("--out-dir", default="results/qc")
    a.add_argument("--config", default="configs/sample_qc.yaml")
    a.add_argument("--folds", default="results/splits/folds_seed42.tsv")
    a.add_argument("--labels", default="results/meta/GSE109379_labels.tsv")
    a.set_defaults(func=apply)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
