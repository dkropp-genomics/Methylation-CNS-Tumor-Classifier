"""Tests for scripts/make_validation_labels.py (stdlib only; pytest or run directly)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_validation_labels.py"
HEADER = "geo_accession\ttitle\tsource\tmethylation.class\tmaterial"

TRAIN = [
    ("GSM1", "GBM, MES, sample 1 [reference set]", "GBM, MES", "FFPE"),
    ("GSM2", "GBM, MES, sample 2 [reference set]", "GBM, MES", "Frozen"),
    ("GSM3", "PIN T,  PB A, sample 3 [reference set]", "PIN T,  PB A", "FFPE"),
    ("GSM4", "MNG, sample 4 [reference set]", "MNG", "FFPE"),
    ("GSM5", "O IDH, sample 5 [reference set]", "O IDH", "Frozen"),
]
VAL = [
    ("GSM11", "GBM, MES, sample 1 [validation set]", "GBM, MES", "DNA_FFPE"),
    ("GSM12", "PIN T, PB A, sample 2 [validation set]", "PIN T, PB A", "DNA_KRYO"),
    ("GSM13", "MNG, sample 3 [validation set]", "MNG", "DNA_FFPE"),
    ("GSM14", "GBM, NEW, sample 4 [validation set]", "GBM, NEW", "DNA_FFPE"),
    ("GSM15", "ODD, sample 5 [validation set]", "ODD", "DNA_FFPE"),
]


def write(tmp: Path, name: str, rows) -> None:
    meta = tmp / "data/meta"
    meta.mkdir(parents=True, exist_ok=True)
    lines = [HEADER] + ["\t".join([g, t, "brain tumor", c, m]) for g, t, c, m in rows]
    (meta / f"{name}_samples.tsv").write_text("\n".join(lines) + "\n")


def run(tmp: Path, *args: str, train=TRAIN, val=VAL):
    write(tmp, "GSE90496", train)
    write(tmp, "GSE109379", val)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=tmp, capture_output=True, text=True, timeout=60
    )


def table(tmp: Path, name: str) -> dict[str, dict]:
    lines = (tmp / "results/meta" / name).read_text().splitlines()
    head = lines[0].split("\t")
    return {r.split("\t")[0]: dict(zip(head, r.split("\t"))) for r in lines[1:]}


def ok(res):
    assert res.returncode == 0, res.stdout + res.stderr


def fails(res, text):
    assert res.returncode != 0 and text in res.stdout + res.stderr, res.stdout + res.stderr


def test_matching_material_and_reports():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        res = run(tmp)
        ok(res)
        lab = table(tmp, "GSE109379_labels.tsv")
        assert lab["GSM11"]["mc_class"] == "GBM, MES" and lab["GSM11"]["in_training"] == "1"
        assert lab["GSM11"]["material"] == "FFPE" and lab["GSM12"]["material"] == "Frozen"
        # Double space in training, single in validation: same class after cleaning.
        assert lab["GSM12"]["mc_class"] == "PIN T, PB A" and lab["GSM12"]["mc_family"] == "PIN T"
        # Unknown labels are kept and flagged, not dropped.
        assert lab["GSM14"]["mc_class"] == "NA" and lab["GSM14"]["in_training"] == "0"
        assert lab["GSM14"]["label_family"] == "GBM" and lab["GSM14"]["label_raw"] == "GBM, NEW"
        assert len(lab) == len(VAL)
        assert "Matched to a training class: 3/5" in res.stdout
        assert "GBM, NEW   [family is in training]" in res.stdout
        assert "ODD   [family not in training]" in res.stdout
        assert "Training classes present in validation: 3/4" in res.stdout
        ov = table(tmp, "GSE109379_class_overlap.tsv")
        assert ov["O IDH"]["status"] == "training only" and ov["ODD"]["status"] == "validation only"
        assert ov["GBM, MES"]["n_training"] == "2" and ov["GBM, MES"]["n_validation"] == "1"


def test_alias_maps_a_renamed_class():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "aliases.tsv").write_text(
            "# same class, different spelling\nvalidation_label\ttraining_class\nGBM, NEW\tGBM, MES\n"
        )
        res = run(tmp, "--aliases", "aliases.tsv")
        ok(res)
        lab = table(tmp, "GSE109379_labels.tsv")
        assert lab["GSM14"]["mc_class"] == "GBM, MES" and lab["GSM14"]["match"] == "alias"
        assert "Matched to a training class: 4/5" in res.stdout


def test_bad_aliases_fail():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "a.tsv").write_text("validation_label\ttraining_class\nGBM, NEW\tNOT A CLASS\n")
        fails(run(tmp, "--aliases", "a.tsv"), "is not a training class")
        (tmp / "a.tsv").write_text("validation_label\ttraining_class\nGBM, TYPO\tGBM, MES\n")
        fails(run(tmp, "--aliases", "a.tsv"), "aliases never used")
        (tmp / "a.tsv").write_text("validation_label\ttraining_class\nMNG\tGBM, MES\n")
        fails(run(tmp, "--aliases", "a.tsv"), "already a training class")


def test_unknown_material_fails():
    val = VAL[:-1] + [("GSM15", "ODD, sample 5 [validation set]", "ODD", "DNA_FRESH")]
    with tempfile.TemporaryDirectory() as d:
        fails(run(Path(d), val=val), "unknown material 'DNA_FRESH'")


def test_sample_in_both_cohorts_fails():
    val = VAL[:-1] + [("GSM1", "ODD, sample 5 [validation set]", "ODD", "DNA_FFPE")]
    with tempfile.TemporaryDirectory() as d:
        fails(run(Path(d), val=val), "sample IDs are in both cohorts")


def test_title_mismatch_is_reported():
    val = VAL[:-1] + [("GSM15", "MNG, sample 5 [validation set]", "ODD", "DNA_FFPE")]
    with tempfile.TemporaryDirectory() as d:
        res = run(Path(d), val=val)
        ok(res)
        assert "GSE109379: 4/5 agree" in res.stdout and "GSE90496: 5/5 agree" in res.stdout


def test_missing_metadata_fails():
    with tempfile.TemporaryDirectory() as d:
        res = subprocess.run([sys.executable, str(SCRIPT)], cwd=d, capture_output=True, text=True)
        fails(res, "run scripts/parse_geo_metadata.py first")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
