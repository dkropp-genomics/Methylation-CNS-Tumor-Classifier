"""Checks on results/meta/class_to_family.tsv (Capper et al. 2018, Sup. Table 1).

Run with pytest, or directly:  python tests/test_class_to_family.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results" / "meta" / "class_to_family.tsv"
FOLDS_QC = ROOT / "results" / "splits" / "folds_seed42_qc.tsv"
FOLDS_ALL = ROOT / "results" / "splits" / "folds_seed42.tsv"

EXPECTED = {  # the eight families and their sizes, as in the supplementary table
    "MCF GBM": 6,
    "MCF IDH GLM": 3,
    "MCF ATRT": 3,
    "MCF PA": 3,
    "MCF PLEX T": 3,
    "MCF MB G3G4": 2,
    "MCF MB SHH": 2,
    "MCF ENB": 2,
}


def load():
    return pd.read_csv(TABLE, sep="\t", dtype=str, keep_default_na=False)


def test_every_class_once_and_eight_families():
    t = load()
    assert len(t) == 91 and t["mc_class"].is_unique
    assert not t["mc_class"].str.contains("  ").any()  # whitespace collapsed
    assert (t["mc_class"] == t["mc_class"].str.strip()).all()
    fam = t[t["in_capper_family"] == "1"]
    assert fam["family"].value_counts().to_dict() == EXPECTED
    single = t[t["in_capper_family"] == "0"]
    assert len(single) == 67 and (single["family"] == single["mc_class"]).all()
    assert t["family"].nunique() == 75
    assert t["n_reference"].astype(int).sum() == 2801


def test_classes_match_the_fold_tables():
    t = load()
    for path in (FOLDS_QC, FOLDS_ALL):
        if not path.exists():  # the table can be checked without the repo data
            continue
        f = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        missing = sorted(set(f["mc_class"]) - set(t["mc_class"]))
        extra = sorted(set(t["mc_class"]) - set(f["mc_class"]))
        assert (
            not missing and not extra
        ), f"{path.name}: not in mapping {missing}; not in folds {extra}"
    if FOLDS_ALL.exists():  # class sizes before QC equal the paper's group sizes
        f = pd.read_csv(FOLDS_ALL, sep="\t", dtype=str, keep_default_na=False)
        ours = f["mc_class"].value_counts().to_dict()
        paper = dict(zip(t["mc_class"], t["n_reference"].astype(int)))
        diff = {c: (ours[c], paper[c]) for c in paper if ours[c] != paper[c]}
        assert not diff, f"class sizes differ (ours, paper): {diff}"


if __name__ == "__main__":
    test_every_class_once_and_eight_families()
    test_classes_match_the_fold_tables()
    print("2 tests passed")
