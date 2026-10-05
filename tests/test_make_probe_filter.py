"""Tests for scripts/make_probe_filter.py on synthetic annotation.

Run with `python -m pytest tests/test_make_probe_filter.py` or directly.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_probe_filter.py"

# id, chr, on_epic, recommended  -> expected fate
ROWS = [
    ("cg01", "chr1", True, False),  # kept
    ("cg02", "chrX", True, False),  # removed: X
    ("cg03", "chrY", True, True),  # removed: Y (counted there, not in mask)
    ("cg04", "chr2", True, True),  # removed: mask
    ("cg05", "chr3", False, False),  # removed: not on EPIC
    ("cg06", "chr4", False, True),  # removed: mask (counted before EPIC)
    ("ch.1.1", "chr5", True, False),  # removed: not CpG
    ("rs01", "", True, False),  # removed: not CpG
    ("ctl01", "", False, False),  # removed: not CpG
    ("cg07", "chr22", True, False),  # kept
]


def make_inputs(root, rows=ROWS):
    root = Path(root)
    a = pd.DataFrame(rows, columns=["Probe_ID", "chr", "on_epic", "sesame_recommended"])
    a.insert(0, "col", range(len(a)))
    # R writes logicals as TRUE/FALSE
    for c in ("on_epic", "sesame_recommended"):
        a[c] = a[c].map({True: "TRUE", False: "FALSE"})
    a.to_csv(root / "annot.tsv.gz", sep="\t", index=False)
    a[["col", "Probe_ID"]].to_csv(root / "probes.tsv", sep="\t", index=False)
    return root


def run(root):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--annotation",
            str(root / "annot.tsv.gz"),
            "--probes",
            str(root / "probes.tsv"),
            "--out-dir",
            str(root / "out"),
        ],
        capture_output=True,
        text=True,
    )


def test_filter_counts_and_kept(tmp_path):
    root = make_inputs(tmp_path)
    r = run(root)
    assert r.returncode == 0, r.stderr
    kept = pd.read_csv(root / "out" / "probes_kept.tsv", sep="\t")
    assert kept["Probe_ID"].tolist() == ["cg01", "cg07"]
    assert kept["col"].tolist() == [0, 9]
    s = pd.read_csv(root / "out" / "probe_filter_summary.tsv", sep="\t")
    assert dict(zip(s["step"], s["removed"])) == {
        "start": 0,
        "not_cpg_probe": 3,
        "chrX_chrY": 2,
        "sesame_recommended_mask": 2,
        "not_on_epic": 1,
    }
    assert s["remaining"].iloc[0] == 10 and s["remaining"].iloc[-1] == 2


def test_probe_order_mismatch_stops(tmp_path):
    root = make_inputs(tmp_path)
    p = pd.read_csv(root / "probes.tsv", sep="\t")
    p.loc[[0, 1], "Probe_ID"] = p.loc[[1, 0], "Probe_ID"].to_numpy()
    p.to_csv(root / "probes.tsv", sep="\t", index=False)
    r = run(root)
    assert r.returncode != 0 and "disagree" in r.stderr


def test_length_mismatch_stops(tmp_path):
    root = make_inputs(tmp_path)
    p = pd.read_csv(root / "probes.tsv", sep="\t").iloc[:-1]
    p.to_csv(root / "probes.tsv", sep="\t", index=False)
    r = run(root)
    assert r.returncode != 0 and "ERROR:" in r.stderr


def test_cpg_without_chromosome_stops(tmp_path):
    rows = list(ROWS)
    rows[0] = ("cg01", "", True, False)
    root = make_inputs(tmp_path, rows)
    r = run(root)
    assert r.returncode != 0 and "cg01" in r.stderr


def test_missing_annotation_stops(tmp_path):
    root = make_inputs(tmp_path)
    (root / "annot.tsv.gz").unlink()
    r = run(root)
    assert r.returncode != 0 and "file not found" in r.stderr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
