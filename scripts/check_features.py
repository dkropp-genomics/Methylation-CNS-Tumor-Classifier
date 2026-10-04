#!/usr/bin/env python
"""Run the in-fold feature pipeline once on real data (outer fold 0).

Fits on the outer-fold-0 TRAINING samples, transforms that fold's test
samples, and reports counts, timing and memory. No model is trained and no
score is computed. Touches GSE90496 only.
"""
import resource
import time

from methylclf.data import BetaStore
from methylclf.features import make_feature_pipeline

st = BetaStore.open("data/betas/zarr/GSE90496.zarr",
                    "results/splits/folds_seed42_qc.tsv",
                    "results/probes/probes_kept.tsv")
fold = st.samples["outer_fold"].astype(int)
train_ids = st.samples.loc[fold != 0, "geo_accession"].to_numpy()
test_ids = st.samples.loc[fold == 0, "geo_accession"].to_numpy()
print(f"outer fold 0: {len(train_ids)} training, {len(test_ids)} test samples")

t0 = time.time()
X_train = st.load(train_ids)
print(f"loaded training {X_train.shape}, {X_train.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s")

t0 = time.time()
pipe = make_feature_pipeline(n_probes=10000).fit(X_train)
print(f"fit: {time.time() - t0:.0f} s")
miss, sel = pipe.named_steps["missing"], pipe.named_steps["select"]
print(f"probes in: {miss.n_features_in_}")
print(f"  kept by missingness (<= 5% missing in training fold): {len(miss.keep_)}")
print(f"  missing in every training sample: {int((miss.frac_missing_ == 1).sum())}")
print(f"  selected by variance: {len(sel.keep_)}")
del X_train

X_test = pipe.transform(st.load(test_ids))
probes = pipe.get_feature_names_out(st.probe_ids)
print(f"test fold transformed: {X_test.shape} {X_test.dtype}, NaN left: {int((X_test != X_test).sum())}")
print(f"first selected probes: {list(probes[:3])}")
print(f"peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
