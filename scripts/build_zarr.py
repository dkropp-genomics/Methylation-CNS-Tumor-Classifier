#!/usr/bin/env python
"""Assemble the per-batch Parquet files from R/preprocess_sesame.R into one
float32 Zarr store per cohort, with explicit sample and probe index files.

Outputs for cohort C:
  <out>/C.zarr          group with array "betas", shape (samples, probes), NaN = missing
  <out>/C.samples.tsv   row number, geo_accession, status and QC columns
  <out>/C.probes.tsv    column number, Probe_ID
  <qc-out>/C_sesame_qc.tsv   the per-sample QC table (small; committed)

Stops with ERROR if a batch is missing, if sample or probe order differs from
the plan, or (with --order-from) if rows are not in the reference table's order.

Usage:
  python scripts/build_zarr.py GSE90496 --order-from results/splits/folds_seed42.tsv
  python scripts/build_zarr.py GSE109379 --order-from results/meta/GSE109379_labels.tsv
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build(cohort, batch_dir, out_dir, qc_out, order_from=None):
    batch_dir, out_dir, qc_out = Path(batch_dir), Path(out_dir), Path(qc_out)
    plan_path = batch_dir / "batches.tsv"
    if not plan_path.exists():
        fail(f"batch plan not found: {plan_path}")
    plan = [r for r in read_tsv(plan_path) if r["cohort"] == cohort]
    if not plan:
        fail(f"cohort {cohort} is not in {plan_path}")
    ids = [r["geo_accession"] for r in plan]
    if order_from:
        ref = [r["geo_accession"] for r in read_tsv(order_from)]
        if ref != ids:
            fail(f"sample order in {plan_path} differs from {order_from}")

    out_dir.mkdir(parents=True, exist_ok=True)
    qc_out.mkdir(parents=True, exist_ok=True)
    final, part = out_dir / f"{cohort}.zarr", out_dir / f"{cohort}.zarr.part"
    shutil.rmtree(part, ignore_errors=True)

    probes, arr, row, qc_rows, block = None, None, 0, [], None
    for b in sorted({int(r["batch"]) for r in plan}):
        expect = [r["geo_accession"] for r in plan if int(r["batch"]) == b]
        stem = batch_dir / cohort / f"batch_{b:04d}"
        pq_path, qc_path = Path(f"{stem}.parquet"), Path(f"{stem}.qc.tsv")
        if not (pq_path.exists() and qc_path.exists()):
            fail(
                f"{cohort} batch {b} is missing or incomplete ({stem}.*); rerun R/preprocess_sesame.R"
            )
        tab = pq.read_table(pq_path)
        if tab.column_names != ["Probe_ID"] + expect:
            fail(f"{pq_path}: sample columns differ from the batch plan")
        p = tab["Probe_ID"].to_pylist()
        if probes is None:
            probes = p
            if len(set(probes)) != len(probes):
                fail(f"{pq_path}: duplicated Probe_ID")
            root = zarr.open_group(str(part), mode="w")
            arr = root.create_array(
                "betas",
                shape=(len(ids), len(probes)),
                dtype="float32",
                chunks=(min(len(ids), 256), min(len(probes), 8192)),
                fill_value=float("nan"),
            )
            root.attrs.update(
                {"cohort": cohort, "prep": "QCDPB", "rows": "samples", "columns": "probes"}
            )
        elif p != probes:
            fail(f"{pq_path}: probe order differs from batch 1")
        for c in expect:
            if str(tab.schema.field(c).type) != "float":
                fail(f"{pq_path}: column {c} is {tab.schema.field(c).type}, expected float32")
        # Parquet nulls become NaN here.
        block = np.stack([np.asarray(tab[c].to_numpy(), dtype=np.float32) for c in expect])
        arr[row : row + len(expect), :] = block
        qc = read_tsv(qc_path)
        if [r["geo_accession"] for r in qc] != expect:
            fail(f"{qc_path}: samples differ from the batch plan")
        qc_rows += qc
        row += len(expect)
        print(f"wrote {cohort} batch {b}: rows {row - len(expect)}-{row - 1}")

    # Read back the last block from disk before accepting the store.
    back = zarr.open_group(str(part), mode="r")["betas"][row - block.shape[0] : row, :]
    if not np.array_equal(back, block, equal_nan=True):
        fail(f"{cohort}: read-back of the last batch does not match what was written")

    for i, r in enumerate(qc_rows):
        r["row"] = i
    qc_fields = [k for k in qc_rows[0] if k != "row"]
    write_tsv(out_dir / f"{cohort}.samples.tsv", qc_rows, ["row"] + qc_fields)
    write_tsv(
        out_dir / f"{cohort}.probes.tsv",
        [{"col": j, "Probe_ID": q} for j, q in enumerate(probes)],
        ["col", "Probe_ID"],
    )
    write_tsv(qc_out / f"{cohort}_sesame_qc.tsv", qc_rows, qc_fields)
    shutil.rmtree(final, ignore_errors=True)
    os.rename(part, final)

    failed = [r["geo_accession"] for r in qc_rows if r["status"] != "ok"]
    print(f"{cohort}: {len(ids)} samples x {len(probes)} probes -> {final}")
    print(
        f"{cohort}: {len(failed)} failed samples (all-NaN rows){': ' + ', '.join(failed) if failed else ''}"
    )
    return final


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cohort")
    ap.add_argument("--batch-dir", default="data/betas/sesame_qcdpb")
    ap.add_argument("--out", default="data/betas/zarr")
    ap.add_argument("--qc-out", default="results/qc")
    ap.add_argument(
        "--order-from", help="TSV whose geo_accession column gives the required row order"
    )
    a = ap.parse_args()
    build(a.cohort, a.batch_dir, a.out, a.qc_out, a.order_from)


if __name__ == "__main__":
    main()
