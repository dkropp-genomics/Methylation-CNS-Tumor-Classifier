#!/usr/bin/env python
"""Kept-sample table for the validation cohort (GSE109379). Reads no beta values.

The training cohort has a fold table that says which store row each modeling
sample is in. The validation cohort needs the same thing, without folds: one
row per sample that passed QC, with its store row, label, family and material,
plus the subset flags that were fixed before scoring.

BetaStore.open takes this table in place of the fold table, so a validation
sample that failed QC cannot be loaded.

  python scripts/make_validation_table.py

Output: results/splits/GSE109379_kept.tsv
  geo_accession, zarr_row, mc_class, family, material,
  capper_score, capper_matched, path_concordant, capper_no_match
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MATERIALS = {"FFPE", "Frozen"}                 # the spellings the fold table uses
FLAGS = ["capper_matched", "path_concordant", "capper_no_match"]


def fail(msg: str):
    sys.exit(f"ERROR: {msg}")


def read(path, required):
    if not Path(path).is_file():
        fail(f"{path}: not found")
    t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in required if c not in t.columns]
    if missing:
        fail(f"{path}: missing column(s) {missing}")
    if t["geo_accession"].duplicated().any():
        fail(f"{path}: a sample appears more than once")
    return t


def build(status, index, labels, families, subsets, train_classes=None):
    """All inputs are string tables. Returns the kept-sample table."""
    for name, t in (("QC status", status), ("labels", labels), ("subsets", subsets)):
        if set(t["geo_accession"]) != set(index["geo_accession"]):
            fail(f"the {name} table and the store index do not hold the same samples")
    if not set(status["keep"]) <= {"True", "False"}:
        fail(f"QC status: keep must be True or False, found {sorted(set(status['keep']))}")
    keep = set(status.loc[status["keep"] == "True", "geo_accession"])
    if not keep:
        fail("QC status: no sample is kept")

    t = index.loc[index["geo_accession"].isin(keep), ["geo_accession", "row"]]
    t = t.rename(columns={"row": "zarr_row"})
    t = t.merge(labels[["geo_accession", "mc_class", "material"]], on="geo_accession")
    t = t.merge(subsets[["geo_accession", "capper_score"] + FLAGS], on="geo_accession")
    t["mc_class"] = t["mc_class"].str.split().str.join(" ")      # GEO has double spaces

    family_of = dict(zip(families["mc_class"], families["family"]))
    unknown = sorted(set(t["mc_class"]) - set(family_of))
    if unknown:
        fail(f"{len(unknown)} validation class(es) have no family, first: {unknown[0]!r}")
    if train_classes is not None:
        new = sorted(set(t["mc_class"]) - set(train_classes))
        if new:
            fail(f"{len(new)} validation class(es) are not training classes, first: {new[0]!r}")
    bad = sorted(set(t["material"]) - MATERIALS)
    if bad:
        fail(f"material must be one of {sorted(MATERIALS)}, found {bad}")
    for c in FLAGS:
        if not set(t[c]) <= {"True", "False"}:
            fail(f"subsets: {c} must be True or False")
    both = (t["capper_matched"] == "True") == (t["capper_no_match"] == "True")
    if both.any():
        fail("subsets: every sample must be exactly one of capper_matched, capper_no_match")
    if len(t) != len(keep):
        fail(f"{len(keep)} samples pass QC but {len(t)} could be joined")

    t.insert(3, "family", t["mc_class"].map(family_of))
    t["zarr_row"] = t["zarr_row"].astype(int)
    return t.sort_values("zarr_row").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", default="results/qc/GSE109379_sample_qc_status.tsv")
    ap.add_argument("--index", default="data/betas/zarr/GSE109379.samples.tsv")
    ap.add_argument("--labels", default="results/meta/GSE109379_labels.tsv")
    ap.add_argument("--subsets", default="results/meta/GSE109379_subsets.tsv")
    ap.add_argument("--families", default="results/meta/class_to_family.tsv")
    ap.add_argument("--train-folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--out", default="results/splits/GSE109379_kept.tsv")
    args = ap.parse_args(argv)

    status = read(args.status, ["geo_accession", "keep"])
    index = read(args.index, ["geo_accession", "row"])
    labels = read(args.labels, ["geo_accession", "mc_class", "material"])
    subsets = read(args.subsets, ["geo_accession", "capper_score"] + FLAGS)
    if not Path(args.families).is_file():
        fail(f"{args.families}: not found")
    families = pd.read_csv(args.families, sep="\t", dtype=str, keep_default_na=False)
    train = read(args.train_folds, ["geo_accession", "mc_class"])

    t = build(status, index, labels, families, subsets, set(train["mc_class"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}: {len(t)} samples, {t['mc_class'].nunique()} classes, "
          f"{t['family'].nunique()} families")
    print("material:", t["material"].value_counts().to_dict())
    for c in FLAGS:
        print(f"  {c:16s} {int((t[c] == 'True').sum())}")


if __name__ == "__main__":
    main()
