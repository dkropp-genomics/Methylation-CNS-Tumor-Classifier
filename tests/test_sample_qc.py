"""Tests for scripts/sample_qc.py on synthetic data.

Run with `python -m pytest tests/test_sample_qc.py` or `python tests/test_sample_qc.py`.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_qc.py"

CONFIG = """metric: frac_detected
min_value: {cut}
decided_on: 2026-10-04
decided_from: GSE90496 pooled distribution, no labels
reason: synthetic test
"""


def make_inputs(root, n_train=60, n_valid=20, seed=0):
    """Build QC tables, a fold table, a label table and a config under root.

    Training: 6 classes of 10. Class 'RARE' gets very low detection in 7 of
    its 10 samples, so a 0.8 cut leaves it with 3 (below the minimum of 5).
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "qc").mkdir(parents=True)

    def qc(cohort, n, start):
        return pd.DataFrame(
            {
                "cohort": cohort,
                "geo_accession": [f"GSM{start + i}" for i in range(n)],
                "status": "ok",
                "frac_detected": rng.uniform(0.9, 0.99, n),
                "bisulfite_gct": rng.uniform(1.0, 2.0, n),
                "mean_intensity": rng.uniform(3000, 9000, n),
            }
        )

    tr = qc("GSE90496", n_train, 1000)
    va = qc("GSE109379", n_valid, 5000)
    classes = np.repeat(["A", "B", "C", "D", "E", "RARE"], n_train // 6)
    tr.loc[np.where(classes == "RARE")[0][:7], "frac_detected"] = 0.5
    va.loc[:2, "frac_detected"] = 0.6  # 3 validation samples fail

    tr.to_csv(root / "qc" / "GSE90496_sesame_qc.tsv", sep="\t", index=False)
    va.to_csv(root / "qc" / "GSE109379_sesame_qc.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "geo_accession": tr["geo_accession"],
            "mc_class": classes,
            "material": np.where(np.arange(n_train) % 3 == 0, "frozen", "FFPE"),
        }
    ).to_csv(root / "folds.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "geo_accession": va["geo_accession"],
            "mc_class": "A",
            "material": "FFPE",
        }
    ).to_csv(root / "labels.tsv", sep="\t", index=False)
    (root / "cfg.yaml").write_text(CONFIG.format(cut=0.8))
    return root


def run(root, cmd, *extra):
    args = [
        sys.executable,
        str(SCRIPT),
        cmd,
        "--qc-dir",
        str(root / "qc"),
        "--out-dir",
        str(root / "out"),
    ]
    if cmd == "apply":
        args += [
            "--config",
            str(root / "cfg.yaml"),
            "--folds",
            str(root / "folds.tsv"),
            "--labels",
            str(root / "labels.tsv"),
        ]
    return subprocess.run(args + list(extra), capture_output=True, text=True)


def test_summarize_writes_grid_and_figure(tmp_path):
    root = make_inputs(tmp_path)
    r = run(root, "summarize")
    assert r.returncode == 0, r.stderr
    grid = pd.read_csv(root / "out" / "GSE90496_qc_grid.tsv", sep="\t")
    row = grid[np.isclose(grid["threshold"], 0.8)].iloc[0]
    assert row["n_below"] == 7
    assert (root / "out" / "figures" / "GSE90496_qc_hist.png").stat().st_size > 0


def test_summarize_never_needs_label_tables(tmp_path):
    """summarize must work with the fold and label tables absent."""
    root = make_inputs(tmp_path)
    (root / "folds.tsv").unlink()
    (root / "labels.tsv").unlink()
    assert run(root, "summarize").returncode == 0


def test_summarize_rejects_validation_rows(tmp_path):
    root = make_inputs(tmp_path)
    p = root / "qc" / "GSE90496_sesame_qc.tsv"
    q = pd.read_csv(p, sep="\t")
    q.loc[0, "cohort"] = "GSE109379"
    q.to_csv(p, sep="\t", index=False)
    r = run(root, "summarize")
    assert r.returncode != 0 and "ERROR:" in r.stderr


def test_apply_drops_and_reports(tmp_path):
    root = make_inputs(tmp_path)
    r = run(root, "apply")
    assert r.returncode == 0, r.stderr
    st = pd.read_csv(root / "out" / "GSE90496_sample_qc_status.tsv", sep="\t")
    assert len(st) == 60 and (~st["keep"]).sum() == 7
    sv = pd.read_csv(root / "out" / "GSE109379_sample_qc_status.tsv", sep="\t")
    assert len(sv) == 20 and (~sv["keep"]).sum() == 3
    imp = pd.read_csv(root / "out" / "GSE90496_qc_class_impact.tsv", sep="\t")
    rare = imp[imp["mc_class"] == "RARE"].iloc[0]
    assert rare["n_before"] == 10 and rare["n_after"] == 3
    assert "WARNING" in r.stdout and "RARE" in r.stdout


def test_apply_boundary_is_kept(tmp_path):
    """A sample exactly at the threshold is kept (>=)."""
    root = make_inputs(tmp_path)
    p = root / "qc" / "GSE90496_sesame_qc.tsv"
    q = pd.read_csv(p, sep="\t")
    q.loc[0, "frac_detected"] = 0.8
    q.to_csv(p, sep="\t", index=False)
    assert run(root, "apply").returncode == 0
    st = pd.read_csv(root / "out" / "GSE90496_sample_qc_status.tsv", sep="\t")
    assert bool(st.loc[st["geo_accession"] == "GSM1000", "keep"].iloc[0])


def test_apply_requires_config(tmp_path):
    root = make_inputs(tmp_path)
    (root / "cfg.yaml").unlink()
    r = run(root, "apply")
    assert r.returncode != 0 and "threshold file not found" in r.stderr


def test_apply_rejects_incomplete_config(tmp_path):
    root = make_inputs(tmp_path)
    (root / "cfg.yaml").write_text("metric: frac_detected\nmin_value: 0.8\n")
    r = run(root, "apply")
    assert r.returncode != 0 and "reason" in r.stderr


def test_apply_stops_on_sample_mismatch(tmp_path):
    """A sample in the QC table but not the fold table must stop the run."""
    root = make_inputs(tmp_path)
    f = pd.read_csv(root / "folds.tsv", sep="\t").iloc[1:]
    f.to_csv(root / "folds.tsv", sep="\t", index=False)
    r = run(root, "apply")
    assert r.returncode != 0 and "GSM1000" in r.stderr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
