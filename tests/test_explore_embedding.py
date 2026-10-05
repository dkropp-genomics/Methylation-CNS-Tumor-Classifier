"""Tests for scripts/explore_embedding.py on a small synthetic store.

Run with `python -m pytest tests/test_explore_embedding.py` or directly.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "explore_embedding.py"
N_SAMPLES, N_PROBES = 90, 400


def make_inputs(root, poison=False, seed=0):
    """3 classes x 30 samples. Class A is all FFPE, B and C are half and half.

    Probes 0-99 carry class signal, 100-149 a material shift, the rest noise.
    Store rows 0-4 are 'dropped' samples and columns 380-399 are filtered-out
    probes; with poison=True those are filled with wild values, which must
    not change the result.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    cls = np.repeat(["A", "B", "C"], 30)
    mat = np.array(["FFPE"] * 30 + (["FFPE"] * 15 + ["Frozen"] * 15) * 2)
    x = rng.normal(0.5, 0.02, (N_SAMPLES, N_PROBES))
    x[cls == "B", :50] += 0.3
    x[cls == "C", 50:100] += 0.3
    x[mat == "Frozen", 100:150] += 0.1
    x[rng.random(x.shape) < 0.01] = np.nan  # scattered missing values
    x[:, 370] = np.nan  # an all-NaN kept probe
    x[::2, 371] = np.nan  # 50% missing: must be skipped
    dropped = np.arange(5)
    if poison:
        x[dropped] = 50.0
        x[:, 380:] = rng.normal(0, 30, (N_SAMPLES, 20))
    x = x.astype("float32")

    g = zarr.open_group(str(root / "store.zarr"), mode="w")
    a = g.create_array("betas", shape=x.shape, dtype="float32", chunks=(16, 128))
    a[:] = x

    keep = np.setdiff1d(np.arange(N_SAMPLES), dropped)
    pd.DataFrame(
        {
            "geo_accession": [f"GSM{i}" for i in keep],
            "zarr_row": keep,
            "mc_class": cls[keep],
            "mc_family": cls[keep],
            "material": mat[keep],
        }
    ).sample(frac=1, random_state=1).to_csv(root / "folds.tsv", sep="\t", index=False)
    pd.DataFrame({"col": range(380), "Probe_ID": [f"cg{i:04d}" for i in range(380)]}).to_csv(
        root / "probes.tsv", sep="\t", index=False
    )
    return root


def run(root, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--store",
            str(root / "store.zarr"),
            "--folds",
            str(root / "folds.tsv"),
            "--probes",
            str(root / "probes.tsv"),
            "--out-dir",
            str(root / "out"),
            "--n-probes",
            "200",
            "--n-pcs",
            "10",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_outputs_and_structure(tmp_path):
    root = make_inputs(tmp_path)
    r = run(root)
    assert r.returncode == 0, r.stderr
    emb = pd.read_csv(root / "out" / "embedding.tsv", sep="\t")
    assert len(emb) == 85 and not emb.isna().any().any()
    assert not emb["geo_accession"].isin([f"GSM{i}" for i in range(5)]).any()
    used = pd.read_csv(root / "out" / "probes_used.tsv", sep="\t")
    assert len(used) == 200 and used["col"].max() < 380
    assert not used["col"].isin([370, 371]).any()
    assoc = pd.read_csv(root / "out" / "pc_associations.tsv", sep="\t")
    # Planted class signal dominates the first two PCs.
    assert assoc.loc[:1, "eta2_class"].min() > 0.9
    # The planted material shift shows up within class on some PC.
    assert assoc["eta2_material_within_class"].max() > 0.5
    for f in ("embedding_by_material", "umap_by_family", "umap_material_within_class"):
        assert (root / "out" / "figures" / f"{f}.png").stat().st_size > 0


def test_dropped_rows_and_filtered_probes_are_never_used(tmp_path):
    """Wild values in dropped samples and filtered probes change nothing."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    clean = make_inputs(tmp_path / "a")
    dirty = make_inputs(tmp_path / "b", poison=True)
    assert run(clean).returncode == 0 and run(dirty).returncode == 0
    a = pd.read_csv(clean / "out" / "embedding.tsv", sep="\t")
    b = pd.read_csv(dirty / "out" / "embedding.tsv", sep="\t")
    pd.testing.assert_frame_equal(a, b)


def test_requires_zarr_row(tmp_path):
    root = make_inputs(tmp_path)
    f = pd.read_csv(root / "folds.tsv", sep="\t").drop(columns="zarr_row")
    f.to_csv(root / "folds.tsv", sep="\t", index=False)
    r = run(root)
    assert r.returncode != 0 and "zarr_row" in r.stderr


def test_too_few_usable_probes_stops(tmp_path):
    root = make_inputs(tmp_path)
    r = run(root, "--n-probes", "379")
    assert r.returncode != 0 and "usable probes" in r.stderr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
