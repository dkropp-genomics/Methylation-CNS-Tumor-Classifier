#!/usr/bin/env python
"""Smoke check of methylclf.data on the real training store (read-only).

Opens the store (which checks the fold and probe tables against the store's
index files), loads 200 samples x all kept probes, and reports shape, timing
and missingness. Touches GSE90496 only.
"""
import time

import numpy as np

from methylclf.data import BetaStore

t0 = time.time()
st = BetaStore.open("data/betas/zarr/GSE90496.zarr",
                    "results/splits/folds_seed42_qc.tsv",
                    "results/probes/probes_kept.tsv")
print(f"opened: {len(st.sample_ids)} samples, {len(st.probe_ids)} probes "
      f"({time.time() - t0:.1f} s)")
ids = np.random.default_rng(42).choice(st.sample_ids, 200, replace=False)
t0 = time.time()
X = st.load(ids)
dt = time.time() - t0
print(f"loaded: {X.shape} {X.dtype}, {X.nbytes / 1e9:.2f} GB, {dt:.1f} s")
print(f"values: min {np.nanmin(X):.4f}, max {np.nanmax(X):.4f}, "
      f"missing {np.isnan(X).mean():.4%}")
print(f"probes all-missing in these 200: {int(np.isnan(X).all(axis=0).sum())}")
