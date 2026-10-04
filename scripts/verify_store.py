#!/usr/bin/env python3
"""Compare the Zarr stores with the independent 20-sample pilot run.

For every pilot sample: same probe order, same NaN positions, same values.
Exits with an error unless every sample matches exactly.
"""
import glob
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr

rows = []
for gse in ("GSE90496", "GSE109379"):
    z = zarr.open_group(f"data/betas/zarr/{gse}.zarr", mode="r")["betas"]
    idx = pd.read_csv(f"data/betas/zarr/{gse}.samples.tsv", sep="\t")
    row_of = idx.set_index("geo_accession")["row"]
    probes = pd.read_csv(f"data/betas/zarr/{gse}.probes.tsv", sep="\t")["Probe_ID"].to_numpy()
    files = sorted(glob.glob(f"data/betas/pilot3/{gse}/*.parquet"))
    if not files:
        sys.exit(f"ERROR: no pilot Parquet files for {gse}")
    for f in files:
        t = pq.read_table(f).to_pandas()
        if not (t["Probe_ID"].to_numpy() == probes).all():
            sys.exit(f"ERROR: probe order differs between {f} and the store")
        for gsm in t.columns[1:]:
            if gsm not in row_of.index:
                sys.exit(f"ERROR: pilot sample {gsm} is not in the {gse} store index")
            a = t[gsm].to_numpy(dtype="float32")
            b = np.asarray(z[int(row_of[gsm])])
            same_nan = bool((np.isnan(a) == np.isnan(b)).all())
            both = ~np.isnan(a) & ~np.isnan(b)
            max_diff = float(np.abs(a[both] - b[both]).max())
            rows.append((gse, gsm, int(row_of[gsm]), int(np.isnan(b).sum()), same_nan, max_diff))

out = pd.DataFrame(rows, columns=["cohort", "geo_accession", "zarr_row", "n_nan",
                                  "nan_positions_match", "max_abs_diff"])
out.to_csv("results/qc/store_vs_pilot.tsv", sep="\t", index=False)
print(out.to_string(index=False))
ok = out["nan_positions_match"].all() and (out["max_abs_diff"] == 0).all()
print(f"\n{len(out)} samples compared:", "PASS, all identical" if ok else "FAIL")
sys.exit(0 if ok else 1)
