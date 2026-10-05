"""Tests for scripts/make_sample_list.py."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_sample_list as msl  # noqa: E402


def tables():
    ref = pd.DataFrame({"geo_accession": ["GSM3", "GSM1", "GSM2"], "mc_class": list("ABC")})
    ext = pd.DataFrame({"geo_accession": ["GSM9", "GSM8"], "label_raw": list("AB")})
    return ref, ext


def test_keeps_each_table_s_order_reference_first():
    ref, ext = tables()
    out = msl.build([("GSE90496", ref), ("GSE109379", ext)])
    assert list(out.columns) == ["cohort", "geo_accession"]
    assert list(out["geo_accession"]) == ["GSM3", "GSM1", "GSM2", "GSM9", "GSM8"]
    assert list(out["cohort"]) == ["GSE90496"] * 3 + ["GSE109379"] * 2


def test_stops_on_duplicates_empty_tables_and_missing_columns():
    ref, ext = tables()
    for parts, text in (
            ([("GSE90496", ref), ("GSE109379", ref)], "more than once"),
            ([("GSE90496", ref.iloc[:0]), ("GSE109379", ext)], "no samples"),
            ([("GSE90496", ref.rename(columns={"geo_accession": "id"}))], "no geo_accession")):
        try:
            msl.build(parts)
        except SystemExit as e:
            assert text in str(e), str(e)
        else:
            raise AssertionError(f"expected a stop mentioning {text!r}")


def test_command_line_writes_the_file():
    ref, ext = tables()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        ref.to_csv(d / "ref.tsv", sep="\t", index=False)
        ext.to_csv(d / "ext.tsv", sep="\t", index=False)
        msl.main(["--reference", str(d / "ref.tsv"), "--external", str(d / "ext.tsv"),
                  "--out", str(d / "meta" / "all.tsv")])
        assert (d / "meta" / "all.tsv").read_text().splitlines() == [
            "cohort\tgeo_accession", "GSE90496\tGSM3", "GSE90496\tGSM1", "GSE90496\tGSM2",
            "GSE109379\tGSM9", "GSE109379\tGSM8"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
