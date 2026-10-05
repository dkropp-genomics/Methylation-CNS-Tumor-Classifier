#!/usr/bin/env python3
"""Assign every GSE90496 sample to nested cross-validation folds.

Why this script exists
----------------------
To measure how well a classifier works on samples it has never seen, we
split the data into K "folds" (groups). Each fold takes one turn as the
held-out test set while the model trains on the other K-1 folds. Averaging
the K test scores gives an honest estimate of real-world performance.

We use *nested* cross-validation:
  * OUTER loop (5 folds): produces the performance numbers we report.
  * INNER loop (3 folds, run inside each outer training set): used only to
    choose settings (hyperparameters, number of CpGs). The outer test fold
    is never seen while those choices are made.

Folds are *stratified* by methylation class, so each fold holds roughly the
same class proportions as the full dataset. That matters here because the
rarest classes have only 8 samples.

The fold table is written once, with a fixed random seed, and every later
step (feature selection, FFPE correction, model training) reads it. That
guarantees every model is compared on identical splits.

Usage (from the repo root):
    python scripts/make_folds.py
    python scripts/make_folds.py --seed 7 --outer 5 --inner 3

After sample QC, rebuild the folds on the samples that passed:
    python scripts/make_folds.py \
        --qc-status results/qc/GSE90496_sample_qc_status.tsv \
        --store-index data/betas/zarr/GSE90496.samples.tsv
This writes folds_seed{SEED}_qc.tsv and leaves the pre-QC table untouched
(other scripts read that one as the full sample list). Folds are redrawn
rather than filtered: deleting rows from an existing table can leave a rare
class absent from a test fold.

Output:
    results/splits/folds_seed{SEED}.tsv     -- one row per sample (no QC)
    results/splits/folds_seed{SEED}_qc.tsv  -- QC-passed samples only, with
        zarr_row = that sample's row in the beta store (if --store-index)
    columns: geo_accession, mc_class, mc_family, material,
             outer_fold, inner_fold_o0 ... inner_fold_o{K-1}
    inner_fold_o{k} is -1 for samples in outer test fold k (they are
    excluded from that outer round's inner loop).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

DEFAULT_META = Path("data/meta/GSE90496_samples.tsv")
DEFAULT_OUTDIR = Path("results/splits")


def load_labels(path: Path) -> pd.DataFrame:
    """Read the parsed GEO metadata and return clean label columns."""
    meta = pd.read_csv(path, sep="\t")
    meta = meta.rename(columns={"methylation.class": "mc_class"})

    # Collapse repeated whitespace: GEO has "PIN T,  PB A" with two spaces.
    meta["mc_class"] = meta["mc_class"].str.replace(r"\s+", " ", regex=True).str.strip()
    # Rough family = text before the first comma. A placeholder until we
    # add the curated family mapping from Capper et al.
    meta["mc_family"] = meta["mc_class"].str.split(",").str[0].str.strip()

    cols = ["geo_accession", "mc_class", "mc_family", "material"]
    meta = meta[cols].copy()

    assert meta["geo_accession"].is_unique, "duplicate sample IDs"
    assert meta.notna().all().all(), "missing values in label columns"
    return meta


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def apply_qc(meta: pd.DataFrame, status_path: Path) -> pd.DataFrame:
    """Keep only samples that passed sample QC. Sample sets must match."""
    if not status_path.exists():
        die(f"QC status file not found: {status_path}")
    st = pd.read_csv(status_path, sep="\t")
    for col in ("geo_accession", "keep"):
        if col not in st.columns:
            die(f"{status_path}: missing column {col}")
    if st["keep"].dtype != bool:
        die(f"{status_path}: column 'keep' must be True/False")
    a, b = set(meta["geo_accession"]), set(st["geo_accession"])
    if a != b or st["geo_accession"].duplicated().any():
        example = sorted(a ^ b)[0] if a != b else "duplicate IDs"
        die(
            f"metadata and {status_path} do not hold the same samples "
            f"({len(a - b)} only in metadata, {len(b - a)} only in QC; "
            f"e.g. {example})"
        )
    kept_ids = set(st.loc[st["keep"], "geo_accession"])
    kept = meta[meta["geo_accession"].isin(kept_ids)].reset_index(drop=True)
    print(
        f"Sample QC: {len(meta) - len(kept)} of {len(meta)} samples " f"dropped, {len(kept)} kept"
    )
    return kept


def add_store_rows(folds: pd.DataFrame, index_path: Path) -> pd.DataFrame:
    """Add zarr_row: each sample's row number in the beta store."""
    if not index_path.exists():
        die(f"store index not found: {index_path}")
    idx = pd.read_csv(index_path, sep="\t")
    for col in ("row", "geo_accession"):
        if col not in idx.columns:
            die(f"{index_path}: missing column {col}")
    if idx["geo_accession"].duplicated().any():
        die(f"{index_path}: duplicated sample IDs")
    rows = folds["geo_accession"].map(idx.set_index("geo_accession")["row"])
    if rows.isna().any():
        bad = folds.loc[rows.isna(), "geo_accession"].iloc[0]
        die(f"sample {bad} is not in the store index {index_path}")
    folds = folds.copy()
    folds.insert(1, "zarr_row", rows.astype(int))
    return folds


def assign_folds(meta: pd.DataFrame, n_outer: int, n_inner: int, seed: int) -> pd.DataFrame:
    """Add outer and inner stratified fold columns to `meta`."""
    folds = meta.reset_index(drop=True).copy()
    y = folds["mc_class"].to_numpy()

    # Outer loop: each sample lands in exactly one outer test fold.
    folds["outer_fold"] = -1
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(outer.split(folds, y)):
        folds.loc[test_idx, "outer_fold"] = k

    # Inner loop: split each outer TRAINING set again, for tuning only.
    for k in range(n_outer):
        col = f"inner_fold_o{k}"
        folds[col] = -1
        train_idx = folds.index[folds["outer_fold"] != k].to_numpy()
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + k + 1)
        for j, (_, val_idx) in enumerate(inner.split(train_idx, y[train_idx])):
            folds.loc[train_idx[val_idx], col] = j

    return folds


def check_folds(folds: pd.DataFrame, n_outer: int, n_inner: int) -> None:
    """Fail loudly if any split is unusable; print a short summary."""
    n_classes = folds["mc_class"].nunique()

    # 1. Every sample has an outer fold.
    assert (folds["outer_fold"] >= 0).all()

    # 2. Every class appears in every outer test fold and every outer
    #    training set (otherwise that class can't be scored or learned).
    per_fold = folds.groupby("outer_fold")["mc_class"].nunique()
    assert (per_fold == n_classes).all(), f"class missing from a test fold:\n{per_fold}"

    min_train = []
    for k in range(n_outer):
        col = f"inner_fold_o{k}"
        train = folds[folds["outer_fold"] != k]
        assert (train[col] >= 0).all()
        assert (folds.loc[folds["outer_fold"] == k, col] == -1).all()
        # Each inner training set (inner folds except j) must contain every class.
        for j in range(n_inner):
            inner_train = train[train[col] != j]
            counts = inner_train["mc_class"].value_counts()
            assert len(counts) == n_classes, f"class missing: outer {k}, inner {j}"
            min_train.append(counts.min())

    print(f"Samples: {len(folds)}  Classes: {n_classes}")
    print("\nSamples per outer test fold:")
    print(folds["outer_fold"].value_counts().sort_index().to_string())
    print(
        "\nFFPE fraction per outer test fold (overall "
        f"{(folds['material'] == 'FFPE').mean():.3f}):"
    )
    print(
        folds.groupby("outer_fold")["material"]
        .apply(lambda s: round((s == "FFPE").mean(), 3))
        .to_string()
    )
    print(f"\nSmallest class count in any inner training set: {min(min_train)}")
    rare = folds["mc_class"].value_counts().idxmin()
    print(f"\nRarest class '{rare}' across outer test folds:")
    print(folds[folds["mc_class"] == rare]["outer_fold"].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer", type=int, default=5)
    parser.add_argument("--inner", type=int, default=3)
    parser.add_argument(
        "--qc-status",
        type=Path,
        default=None,
        help="sample QC status table; keep only passed samples",
    )
    parser.add_argument(
        "--store-index", type=Path, default=None, help="<GSE>.samples.tsv; adds the zarr_row column"
    )
    args = parser.parse_args()

    meta = load_labels(args.meta)
    if args.qc_status is not None:
        meta = apply_qc(meta, args.qc_status)
    sizes = meta["mc_class"].value_counts()
    if sizes.min() < args.outer:
        die(
            f"class '{sizes.idxmin()}' has {sizes.min()} samples, fewer than "
            f"the {args.outer} outer folds"
        )
    folds = assign_folds(meta, args.outer, args.inner, args.seed)
    check_folds(folds, args.outer, args.inner)
    if args.store_index is not None:
        folds = add_store_rows(folds, args.store_index)

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = "_qc" if args.qc_status is not None else ""
    out = args.outdir / f"folds_seed{args.seed}{suffix}.tsv"
    folds.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
