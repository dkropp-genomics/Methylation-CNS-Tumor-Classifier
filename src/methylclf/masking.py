"""Simulated sparse coverage: which CpGs of a sample count as "observed".

A low-coverage sequencing run reads a random subset of CpGs. Here every sample
gets one random number per array probe, drawn from a seed and the sample's ID.
A probe is observed at coverage level p when its number is below p. So:

  * the same sample always gets the same subset, whichever model asks;
  * subsets are nested: observed at 0.1% implies observed at 1%, 10%, 100%;
  * the subset is drawn over the whole array, and a model sees whichever of its
    own input probes fall inside it.

Nothing here is fitted to data, so it is applied to any sample without leakage.
"""
from __future__ import annotations

import zlib

import numpy as np


def probe_positions(feature_names, array_probe_ids) -> np.ndarray:
    """Position of each model feature in the array's probe list."""
    where = {p: i for i, p in enumerate(array_probe_ids)}
    missing = [p for p in feature_names if p not in where]
    if missing:
        raise SystemExit(f"ERROR: masking: {len(missing)} feature probes are not in the "
                         f"array probe list (first: {missing[0]})")
    return np.fromiter((where[p] for p in feature_names), dtype=np.int64,
                       count=len(feature_names))


def observed_uniform(sample_ids, n_array_probes: int, cols, seed: int) -> np.ndarray:
    """One number in [0, 1) per (sample, feature); observed at level p when below p.

    Row i depends only on `seed` and `sample_ids[i]`, never on the other samples
    or on which features were asked for.
    """
    cols = np.asarray(cols, dtype=np.int64)
    out = np.empty((len(sample_ids), cols.size), dtype=np.float32)
    for i, sid in enumerate(sample_ids):
        rng = np.random.default_rng([int(seed), zlib.crc32(str(sid).encode())])
        out[i] = rng.random(n_array_probes, dtype=np.float32)[cols]
    return out


def observed(u: np.ndarray, level: float) -> np.ndarray:
    """Boolean mask of observed values at coverage `level` (1.0 = everything)."""
    if not 0 < level <= 1:
        raise SystemExit(f"ERROR: masking: level must be in (0, 1], got {level}")
    return u < np.float32(level) if level < 1 else np.ones(u.shape, dtype=bool)
