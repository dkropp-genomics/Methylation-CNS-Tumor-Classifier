"""Tests for scripts/make_validation_table.py on small made-up tables."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_validation_table as mvt  # noqa: E402


def tables():
    ids = [f"GSM{i}" for i in range(6)]
    status = pd.DataFrame(
        {"geo_accession": ids, "keep": ["True", "True", "False", "True", "True", "True"]}
    )
    index = pd.DataFrame({"geo_accession": ids, "row": [str(i) for i in range(6)]})
    labels = pd.DataFrame(
        {
            "geo_accession": ids,
            "mc_class": ["A", "A", "B", "PIN T,  PB B", "C, X", "C, Y"],
            "material": ["FFPE"] * 5 + ["Frozen"],
        }
    )
    families = pd.DataFrame(
        {
            "mc_class": ["A", "B", "PIN T, PB B", "C, X", "C, Y"],
            "family": ["A", "B", "PIN T, PB B", "MCF C", "MCF C"],
        }
    )
    subsets = pd.DataFrame(
        {
            "geo_accession": ids,
            "capper_score": ["0.99"] * 5 + ["0.4"],
            "capper_matched": ["True"] * 5 + ["False"],
            "path_concordant": ["True", "False"] * 3,
            "capper_no_match": ["False"] * 5 + ["True"],
        }
    )
    return status, index, labels, families, subsets


def expect_stop(fn, text):
    try:
        fn()
    except SystemExit as e:
        assert str(e).startswith("ERROR:") and text in str(e), str(e)
        return
    raise AssertionError(f"expected a stop mentioning {text!r}")


def test_keeps_qc_passed_samples_with_their_store_rows():
    t = mvt.build(*tables())
    assert list(t["geo_accession"]) == ["GSM0", "GSM1", "GSM3", "GSM4", "GSM5"]
    assert list(t["zarr_row"]) == [0, 1, 3, 4, 5]  # row 2 failed QC
    assert list(t.columns[:5]) == ["geo_accession", "zarr_row", "mc_class", "family", "material"]


def test_collapses_whitespace_and_adds_family():
    t = mvt.build(*tables()).set_index("geo_accession")
    assert t.loc["GSM3", "mc_class"] == "PIN T, PB B"
    assert t.loc["GSM4", "family"] == "MCF C" and t.loc["GSM5", "family"] == "MCF C"


def test_stops_on_a_class_without_family_or_outside_training():
    status, index, labels, families, subsets = tables()
    expect_stop(lambda: mvt.build(status, index, labels, families.iloc[:-1], subsets), "no family")
    expect_stop(
        lambda: mvt.build(status, index, labels, families, subsets, train_classes={"A", "B"}),
        "not training classes",
    )


def test_stops_on_unknown_material_and_mismatched_samples():
    status, index, labels, families, subsets = tables()
    odd = labels.assign(material=["frozen"] + list(labels["material"][1:]))
    expect_stop(lambda: mvt.build(status, index, odd, families, subsets), "material")
    expect_stop(
        lambda: mvt.build(status.iloc[:-1], index, labels, families, subsets), "same samples"
    )


def test_stops_when_subset_flags_overlap():
    status, index, labels, families, subsets = tables()
    bad = subsets.assign(capper_no_match="True")
    expect_stop(lambda: mvt.build(status, index, labels, families, bad), "exactly one")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
