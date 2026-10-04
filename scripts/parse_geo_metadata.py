#!/usr/bin/env python3
"""Parse a GEO series matrix into a tidy one-row-per-sample metadata table.

Replaces R/01_parse_metadata.R, generalized to any GEO series. Standard
library only, so it runs in any Python 3.9+ environment (e.g. methyl-py).

What it does
------------
1. Downloads <GSE>_series_matrix.txt.gz into data/meta/ if it isn't there.
2. Reads only the "!Sample_..." header lines and stops at the data table,
   so a series matrix that also holds beta values is never loaded.
3. Builds data/meta/<GSE>_samples.tsv with columns:
       geo_accession, title, source, <one column per characteristic key>
   Characteristic keys are matched PER CELL ("material: FFPE" -> column
   "material"), not per row. GEO shifts a sample's fields left when that
   sample lacks one, so taking "the most common key in the row" (what the
   R script did) can silently put a value under the wrong column.
   Column names follow R's make.names() ("methylation class" ->
   "methylation.class"), so output for GSE90496 matches the R script's.
4. Prints, and writes to results/meta/<GSE>_fields.tsv, a summary of every
   field: how many samples have it, how many distinct values, the top
   values. That is how we find out what labels a series actually carries.

Usage (from the repo root):
    python scripts/parse_geo_metadata.py GSE109379
    python scripts/parse_geo_metadata.py GSE90496 --outdir /tmp/check   # compare with old output

Fails loudly on: duplicate sample IDs, a characteristic key appearing twice
for one sample, rows whose length doesn't match the sample count, or a
truncated gzip file.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

GEO_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo"
UNLABELED = "unlabeled"  # characteristic cells without "key: value"


def matrix_url(acc: str, base: str = GEO_BASE_URL) -> str:
    """GEO groups series in folders like GSE109nnn."""
    return f"{base}/series/{acc[:-3]}nnn/{acc}/matrix/{acc}_series_matrix.txt.gz"


def download(url: str, dest: Path, tries: int = 3) -> None:
    """Download to a .part file; rename only after gzip reads it fully."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        try:
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            with gzip.open(part, "rb") as g:  # full read = integrity check
                while g.read(1 << 20):
                    pass
            part.replace(dest)
            return
        except Exception as e:  # network error or truncated file
            print(f"  attempt {attempt}/{tries} failed: {e}", file=sys.stderr)
    sys.exit(f"ERROR: could not download {url}")


def r_make_names(key: str) -> str:
    """Mimic R's make.names(): invalid characters -> '.', leading digit -> 'X'."""
    name = re.sub(r"[^A-Za-z0-9._]", ".", key.strip())
    if not name or not re.match(r"[A-Za-z.]", name) or re.match(r"\.[0-9]", name):
        name = "X" + name
    return name


def unquote(cell: str) -> str:
    cell = cell.strip()
    if len(cell) >= 2 and cell[0] == cell[-1] == '"':
        cell = cell[1:-1]
    return cell


def read_sample_lines(path: Path) -> list[tuple[str, list[str]]]:
    """Return [(key, values)] for every !Sample_ line before the data table."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_"):
                key, *vals = line.split("\t")
                rows.append((key, [unquote(v) for v in vals]))
    if not rows:
        sys.exit(f"ERROR: no !Sample_ lines in {path}")
    return rows


def parse(rows: list[tuple[str, list[str]]]) -> tuple[list[str], list[dict], dict]:
    """Build (columns, records, other_fields) from the !Sample_ lines."""
    first = dict(rows)  # first occurrence wins for single-valued fields
    ids = first.get("!Sample_geo_accession")
    if ids is None:
        sys.exit("ERROR: no !Sample_geo_accession line")
    n = len(ids)
    dup = [k for k, c in Counter(ids).items() if c > 1]
    if dup:
        sys.exit(f"ERROR: duplicate sample IDs: {dup[:5]}")
    for key, vals in rows:
        if len(vals) != n:
            sys.exit(f"ERROR: {key} has {len(vals)} values for {n} samples")

    records = [
        {"geo_accession": ids[i],
         "title": first.get("!Sample_title", [""] * n)[i],
         "source": first.get("!Sample_source_name_ch1", [""] * n)[i]}
        for i in range(n)
    ]
    columns = ["geo_accession", "title", "source"]

    # Characteristics: every cell is "key: value"; match keys cell by cell.
    for key, vals in rows:
        if key != "!Sample_characteristics_ch1":
            continue
        for i, cell in enumerate(vals):
            if not cell:
                continue  # GEO pads short samples with empty cells
            k, sep, v = cell.partition(":")
            col = r_make_names(k) if sep else UNLABELED
            v = v.strip() if sep else cell
            if col in ("geo_accession", "title", "source"):
                col = f"characteristic.{col}"
            if col in records[i]:
                sys.exit(f"ERROR: {ids[i]} has characteristic '{col}' twice")
            records[i][col] = v
            if col not in columns:
                columns.append(col)

    # Other single-valued !Sample_ fields, reported (not stored) so we can
    # see whether a diagnosis hides in e.g. !Sample_description.
    skip = {"!Sample_geo_accession", "!Sample_title", "!Sample_source_name_ch1",
            "!Sample_characteristics_ch1"}
    other = {k: v for k, v in first.items() if k not in skip}
    return columns, records, other


def summarize(columns, records, other, top: int = 8) -> list[list[str]]:
    """One row per field: field, n_samples_with_value, n_distinct, top values."""
    n = len(records)
    out = []
    for col in columns:
        vals = [r[col] for r in records if r.get(col, "")]
        counts = Counter(vals)
        head = "; ".join(f"{v} ({c})" for v, c in counts.most_common(top))
        out.append([col, "characteristic" if col not in ("geo_accession", "title", "source")
                    else "core", f"{len(vals)}/{n}", str(len(counts)), head])
    for key, vals in other.items():
        counts = Counter(v for v in vals if v)
        if len(counts) <= 1 and key.startswith(("!Sample_contact", "!Sample_status",
                                                "!Sample_submission", "!Sample_last_update")):
            continue  # boilerplate identical across samples
        head = "; ".join(f"{v[:60]} ({c})" for v, c in counts.most_common(3))
        out.append([key, "other (not stored)", f"{sum(counts.values())}/{n}",
                    str(len(counts)), head])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("accession", help="GEO series, e.g. GSE109379")
    p.add_argument("--meta-dir", type=Path, default=Path("data/meta"),
                   help="where the series matrix is kept (default data/meta)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="where <GSE>_samples.tsv goes (default: --meta-dir)")
    p.add_argument("--fields-dir", type=Path, default=Path("results/meta"),
                   help="where the field summary goes (default results/meta)")
    p.add_argument("--base-url", default=GEO_BASE_URL, help=argparse.SUPPRESS)
    a = p.parse_args()

    acc = a.accession
    if not re.fullmatch(r"GSE\d{4,}", acc):
        sys.exit(f"ERROR: not a GEO series accession: {acc}")
    matrix = a.meta_dir / f"{acc}_series_matrix.txt.gz"
    if not matrix.exists():
        download(matrix_url(acc, a.base_url), matrix)

    columns, records, other = parse(read_sample_lines(matrix))

    outdir = a.outdir or a.meta_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{acc}_samples.tsv"
    with open(out, "w", newline="") as f:
        # Written like R's write.table(quote = FALSE): raw values, tab-separated.
        for row in [columns] + [[r.get(c, "NA") for c in columns] for r in records]:
            bad = [v for v in row if "\t" in v or "\n" in v]
            if bad:
                sys.exit(f"ERROR: value contains a tab or newline: {bad[0]!r}")
            f.write("\t".join(row) + "\n")

    summary = summarize(columns, records, other)
    a.fields_dir.mkdir(parents=True, exist_ok=True)
    fields = a.fields_dir / f"{acc}_fields.tsv"
    with open(fields, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["field", "kind", "samples_with_value", "n_distinct", "top_values"])
        w.writerows(summary)

    print(f"\n{acc}: {len(records)} samples, {len(columns) - 3} characteristic field(s)\n")
    for field, kind, have, nd, head in summary:
        print(f"[{kind}] {field}: {have} samples, {nd} distinct")
        if head:
            print(f"    {head[:300]}")
    missing = [(c, sum(1 for r in records if c not in r)) for c in columns]
    for c, m in missing:
        if m:
            print(f"NOTE: '{c}' is missing for {m} sample(s); written as NA")
    print(f"\nWrote {out}\nWrote {fields}")


if __name__ == "__main__":
    main()
