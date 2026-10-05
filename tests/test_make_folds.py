"""Tests for scripts/make_folds.py on synthetic data.

Run with `python -m pytest tests/test_make_folds.py` or directly with python.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_folds.py"


def make_inputs(root, drop=("GSM1000", "GSM1001")):
    """6 classes x 20 samples; QC status drops `drop`; store index is in a
    different (reversed) order so zarr_row must come from a real lookup."""
    root = Path(root)
    ids = [f"GSM{1000 + i}" for i in range(120)]
    pd.DataFrame(
        {
            "geo_accession": ids,
            "methylation.class": [f"CLS,  {i // 20}" for i in range(120)],
            "material": ["FFPE" if i % 3 else "Frozen" for i in range(120)],
        }
    ).to_csv(root / "meta.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "geo_accession": ids,
            "frac_detected": 0.9,
            "keep": [i not in drop for i in ids],
        }
    ).to_csv(root / "status.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "row": range(120),
            "geo_accession": ids[::-1],
        }
    ).to_csv(root / "index.tsv", sep="\t", index=False)
    return root


def run(root, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--meta",
            str(root / "meta.tsv"),
            "--outdir",
            str(root / "out"),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_plain_run_unchanged(tmp_path):
    root = make_inputs(tmp_path)
    r = run(root)
    assert r.returncode == 0, r.stderr
    f = pd.read_csv(root / "out" / "folds_seed42.tsv", sep="\t")
    assert len(f) == 120 and "zarr_row" not in f.columns
    assert f["mc_class"].str.contains("  ").sum() == 0  # whitespace collapsed
    assert f.groupby("outer_fold")["mc_class"].nunique().eq(6).all()


def test_qc_run_drops_and_maps_rows(tmp_path):
    root = make_inputs(tmp_path)
    r = run(root, "--qc-status", str(root / "status.tsv"), "--store-index", str(root / "index.tsv"))
    assert r.returncode == 0, r.stderr
    f = pd.read_csv(root / "out" / "folds_seed42_qc.tsv", sep="\t")
    assert len(f) == 118
    assert not f["geo_accession"].isin(["GSM1000", "GSM1001"]).any()
    assert f.groupby("outer_fold")["mc_class"].nunique().eq(6).all()
    # index is reversed: GSM1002 is row 117, GSM1119 is row 0
    rows = f.set_index("geo_accession")["zarr_row"]
    assert rows["GSM1002"] == 117 and rows["GSM1119"] == 0
    assert not (root / "out" / "folds_seed42.tsv").exists()


def test_qc_run_is_reproducible(tmp_path):
    root = make_inputs(tmp_path)
    args = ("--qc-status", str(root / "status.tsv"))
    run(root, *args)
    a = (root / "out" / "folds_seed42_qc.tsv").read_text()
    run(root, *args)
    assert a == (root / "out" / "folds_seed42_qc.tsv").read_text()


def test_qc_sample_mismatch_stops(tmp_path):
    root = make_inputs(tmp_path)
    st = pd.read_csv(root / "status.tsv", sep="\t").iloc[1:]
    st.to_csv(root / "status.tsv", sep="\t", index=False)
    r = run(root, "--qc-status", str(root / "status.tsv"))
    assert r.returncode != 0 and "GSM1000" in r.stderr


def test_class_too_small_stops(tmp_path):
    """QC leaving a class with fewer samples than outer folds must stop."""
    drop = tuple(f"GSM{1000 + i}" for i in range(16))  # class 0: 20 -> 4
    root = make_inputs(tmp_path, drop=drop)
    r = run(root, "--qc-status", str(root / "status.tsv"))
    assert r.returncode != 0 and "ERROR:" in r.stderr and "CLS, 0" in r.stderr


def test_missing_from_store_index_stops(tmp_path):
    root = make_inputs(tmp_path)
    ix = pd.read_csv(root / "index.tsv", sep="\t")
    ix[ix["geo_accession"] != "GSM1050"].to_csv(root / "index.tsv", sep="\t", index=False)
    r = run(root, "--qc-status", str(root / "status.tsv"), "--store-index", str(root / "index.tsv"))
    assert r.returncode != 0 and "GSM1050" in r.stderr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
