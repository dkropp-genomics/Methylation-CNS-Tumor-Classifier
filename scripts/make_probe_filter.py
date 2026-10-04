#!/usr/bin/env python3
"""Build the fixed probe filter from public annotation only.

Every rule here is a fact about the array design, not about our samples, so
applying it to all data at once cannot leak. Rules that depend on the data
(for example "missing in more than 5% of samples") are fit inside training
folds in Phase 2, not here.

Filters, applied in order (each count is what that step removed from what
was left):
  1. not a CpG probe   ID does not start with "cg" (ch, rs, control probes)
  2. chrX / chrY       sex chromosomes
  3. SeSAMe mask       "recommended" quality mask: SNP-affected and poorly or
                       non-uniquely mapping probes. Already NaN in the store.
  4. not on EPIC       probe absent from the EPIC array

Input:  results/probes/hm450_annotation.tsv.gz (from R/export_probe_annotation.R)
Output: results/probes/probes_kept.tsv          col, Probe_ID of kept probes
        results/probes/probe_filter_summary.tsv step, removed, remaining

Usage:
  python scripts/make_probe_filter.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load(annotation, probes):
    for path in (annotation, probes):
        if not Path(path).exists():
            die(f"file not found: {path}")
    a = pd.read_csv(annotation, sep="\t", dtype={"chr": "string"})
    need = {"col", "Probe_ID", "chr", "on_epic", "sesame_recommended"}
    if need - set(a.columns):
        die(f"{annotation}: missing columns {sorted(need - set(a.columns))}")
    for flag in ("on_epic", "sesame_recommended"):
        if a[flag].dtype != bool:
            die(f"{annotation}: column {flag} must be TRUE/FALSE")
    p = pd.read_csv(probes, sep="\t")
    # The annotation must describe exactly the store's columns, in order.
    if len(a) != len(p):
        die(f"annotation has {len(a)} probes, store index has {len(p)}")
    if not (a["col"].to_numpy() == range(len(a))).all():
        die(f"{annotation}: col is not 0..{len(a) - 1} in order")
    diff = a["Probe_ID"].to_numpy() != p["Probe_ID"].to_numpy()
    if diff.any() or (a["col"].to_numpy() != p["col"].to_numpy()).any():
        where = int(diff.argmax()) if diff.any() else 0
        die(f"annotation and store index disagree at column {where}")
    return a


def build_filter(a):
    """Return (kept rows, summary table)."""
    is_cg = a["Probe_ID"].str.startswith("cg")
    no_chr = is_cg & (a["chr"].isna() | (a["chr"] == ""))
    if no_chr.any():
        die(f"CpG probe {a.loc[no_chr, 'Probe_ID'].iloc[0]} has no chromosome")
    steps = [
        ("not_cpg_probe", ~is_cg),
        ("chrX_chrY", a["chr"].isin(["chrX", "chrY"])),
        ("sesame_recommended_mask", a["sesame_recommended"]),
        ("not_on_epic", ~a["on_epic"]),
    ]
    keep = pd.Series(True, index=a.index)
    rows = [("start", 0, int(keep.sum()))]
    for name, remove in steps:
        n = int((keep & remove).sum())
        keep &= ~remove
        rows.append((name, n, int(keep.sum())))
    summary = pd.DataFrame(rows, columns=["step", "removed", "remaining"])
    return a.loc[keep, ["col", "Probe_ID"]], summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotation", default="results/probes/hm450_annotation.tsv.gz")
    ap.add_argument("--probes", default="data/betas/zarr/GSE90496.probes.tsv")
    ap.add_argument("--out-dir", default="results/probes")
    args = ap.parse_args(argv)

    kept, summary = build_filter(load(args.annotation, args.probes))
    if len(kept) == 0:
        die("the filter removed every probe")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out / "probes_kept.tsv", sep="\t", index=False)
    summary.to_csv(out / "probe_filter_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"\nKept {len(kept)} probes (Capper et al. kept 428,799)")
    print(f"Wrote {out / 'probes_kept.tsv'}\nWrote {out / 'probe_filter_summary.tsv'}")


if __name__ == "__main__":
    main()
