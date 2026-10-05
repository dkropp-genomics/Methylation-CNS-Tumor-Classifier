#!/usr/bin/env python
"""The list of arrays to preprocess: data/meta/all_samples.tsv.

One row per array, columns cohort and geo_accession: the reference cohort in
the order of the pre-QC fold table, then the external cohort in the order of
its label table. Preprocessing writes batches in this order, and the beta
stores are checked against the same two tables, so the order is fixed here once.

  python scripts/make_sample_list.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def fail(msg: str):
    sys.exit(f"ERROR: {msg}")


def build(parts) -> pd.DataFrame:
    """parts: (cohort, table with a geo_accession column), in output order."""
    rows = []
    for cohort, table in parts:
        if "geo_accession" not in table.columns:
            fail(f"{cohort}: the table has no geo_accession column")
        if len(table) == 0:
            fail(f"{cohort}: the table has no samples")
        rows.append(pd.DataFrame({"cohort": cohort, "geo_accession": table["geo_accession"]}))
    out = pd.concat(rows, ignore_index=True)
    dup = out.loc[out["geo_accession"].duplicated(), "geo_accession"]
    if len(dup):
        fail(f"sample {dup.iloc[0]} is listed more than once")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", default="results/splits/folds_seed42.tsv",
                    help="pre-QC fold table of GSE90496 (all 2,801 arrays)")
    ap.add_argument("--external", default="results/meta/GSE109379_labels.tsv")
    ap.add_argument("--out", default="data/meta/all_samples.tsv")
    args = ap.parse_args(argv)
    parts = []
    for cohort, path in (("GSE90496", args.reference), ("GSE109379", args.external)):
        if not Path(path).is_file():
            fail(f"{path}: not found")
        parts.append((cohort, pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)))
    out = build(parts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}: {len(out)} arrays "
          f"({out['cohort'].value_counts(sort=False).to_dict()})")


if __name__ == "__main__":
    main()
