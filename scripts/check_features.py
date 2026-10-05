#!/usr/bin/env python
"""Run the in-fold feature pipeline once on real data (outer fold 0), with
material correction off and on.

Fits on the outer-fold-0 TRAINING samples, transforms that fold's test
samples, and reports counts, timing and memory. No model is trained and no
score is computed. Touches GSE90496 only.
"""

import resource
import time

import numpy as np

from methylclf.data import BetaStore
from methylclf.features import FeaturePipeline

st = BetaStore.open(
    "data/betas/zarr/GSE90496.zarr",
    "results/splits/folds_seed42_qc.tsv",
    "results/probes/probes_kept.tsv",
)
s = st.samples
is_train = (s["outer_fold"].astype(int) != 0).to_numpy()
ids, y, mat = (s[c].to_numpy() for c in ("geo_accession", "mc_class", "material"))
print(f"outer fold 0: {is_train.sum()} training, {(~is_train).sum()} test samples")
print(f"material values (training fold): {s.loc[is_train, 'material'].value_counts().to_dict()}")

t0 = time.time()
X_train, X_test = st.load(ids[is_train]), st.load(ids[~is_train])
print(f"loaded training {X_train.shape}, {X_train.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s")

selected = {}
for on in (False, True):
    t0 = time.time()
    pipe = FeaturePipeline(n_probes=10000, correct_material=on)
    pipe.fit(X_train, y[is_train], mat[is_train])
    Z_test = pipe.transform(X_test, mat[~is_train])
    selected[on] = set(pipe.get_feature_names_out(st.probe_ids))
    print(f"\ncorrection {'ON' if on else 'OFF'}: fit + transform {time.time() - t0:.0f} s")
    print(f"  kept by missingness: {len(pipe.missing_.keep_)} of {pipe.n_features_in_}")
    print(f"  test fold: {Z_test.shape} {Z_test.dtype}, NaN left: {int(np.isnan(Z_test).sum())}")
    if on:
        d = pipe.correct_.shift_
        a = np.abs(d)
        print(
            f"  classes with both materials used: {len(pipe.correct_.classes_used_)} of "
            f"{len(set(y[is_train]))}"
        )
        print(
            f"  FFPE - frozen shift per probe: median {np.median(d):+.4f}, "
            f"median |shift| {np.median(a):.4f}, 99th pct |shift| {np.percentile(a, 99):.4f}, "
            f"max |shift| {a.max():.4f}"
        )
        print(
            f"  probes with |shift| > 0.05: {int((a > 0.05).sum())}; > 0.10: {int((a > 0.10).sum())}"
        )
print(f"\nselected probes shared by OFF and ON: {len(selected[False] & selected[True])} of 10000")
print(f"peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
