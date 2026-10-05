"""Tests for R/preprocess_sesame.R on synthetic data (SESAME_MOCK=1, no SeSAMe).

Needs Rscript with the arrow and BiocParallel packages. It is found on PATH, or
through `conda run -n methyl-r`, or set RSCRIPT to the full path.
Run with `python -m pytest tests/` or `python tests/test_preprocess_sesame.py`.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "R" / "preprocess_sesame.R"


def rscript():
    if os.environ.get("RSCRIPT"):
        return [os.environ["RSCRIPT"]]
    if shutil.which("Rscript"):
        return ["Rscript"]
    if shutil.which("conda"):
        return ["conda", "run", "-n", "methyl-r", "Rscript"]
    raise RuntimeError("Rscript not found; set RSCRIPT")


def make_inputs(tmp, ids, skip_red=()):
    """Empty fake IDAT pairs for cohort GSETEST and a sample table."""
    idat = tmp / "idat" / "GSETEST"
    idat.mkdir(parents=True)
    for n, gsm in enumerate(ids):
        slide = "BAD" if gsm == "GSM3" else "5684819014"
        for ch in ("Grn", "Red"):
            if ch == "Red" and gsm in skip_red:
                continue
            (idat / f"{gsm}_{slide}_R0{n + 1}C01_{ch}.idat.gz").touch()
    table = tmp / "samples.tsv"
    table.write_text("cohort\tgeo_accession\n" + "".join(f"GSETEST\t{g}\n" for g in ids))
    return table


def run(tmp, table, *extra):
    cmd = rscript() + [
        str(SCRIPT),
        "--samples",
        str(table),
        "--out",
        str(tmp / "out"),
        "--idat-root",
        str(tmp / "idat"),
        "--batch-size",
        "2",
        "--workers",
        "1",
        *extra,
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, env={**os.environ, "SESAME_MOCK": "1"}
    )


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


IDS = ["GSM1", "GSM2", "GSM3", "GSM4", "GSM5"]  # GSM3 fails in mock mode


def test_batches_qc_and_failed_sample(tmp_path):
    r = run(tmp_path, make_inputs(tmp_path, IDS))
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out" / "GSETEST"
    assert sorted(p.name for p in out.iterdir()) == [
        f"batch_000{b}.{ext}" for b in (1, 2, 3) for ext in ("parquet", "qc.tsv")
    ]
    rows = [row for b in (1, 2, 3) for row in read_tsv(out / f"batch_000{b}.qc.tsv")]
    assert [row["geo_accession"] for row in rows] == IDS
    status = {row["geo_accession"]: row["status"] for row in rows}
    assert status["GSM3"].startswith("error: mock failure")
    assert all(v == "ok" for k, v in status.items() if k != "GSM3")
    assert rows[0]["n_probes"] == "1000" and float(rows[0]["frac_na"]) == 0.1
    assert "FAILED GSM3" in r.stdout
    plan = read_tsv(tmp_path / "out" / "batches.tsv")
    assert [p["batch"] for p in plan] == ["1", "1", "2", "2", "3"]


def test_parquet_is_float32_in_sample_order(tmp_path):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        import pytest

        pytest.skip("pyarrow not installed")
    assert run(tmp_path, make_inputs(tmp_path, IDS)).returncode == 0
    tab = pq.read_table(tmp_path / "out" / "GSETEST" / "batch_0002.parquet")
    assert tab.column_names == ["Probe_ID", "GSM3", "GSM4"]
    assert tab.schema.field("GSM4").type == pa.float32()
    assert tab.num_rows == 1000
    assert tab["GSM3"].null_count == 1000  # failed sample: all NA
    assert tab["GSM4"].null_count == 100  # NA kept as NA
    assert tab["Probe_ID"][0].as_py() == "cg00000001"


def test_rerun_skips_complete_and_redoes_missing(tmp_path):
    table = make_inputs(tmp_path, IDS)
    assert run(tmp_path, table).returncode == 0
    out = tmp_path / "out" / "GSETEST"
    keep = (out / "batch_0001.parquet").stat().st_mtime_ns
    (out / "batch_0002.qc.tsv").unlink()  # batch 2 now looks incomplete
    r = run(tmp_path, table)
    assert r.returncode == 0, r.stderr
    assert "skip  GSETEST batch 1" in r.stdout and "done  GSETEST batch 2" in r.stdout
    assert (out / "batch_0001.parquet").stat().st_mtime_ns == keep
    assert (out / "batch_0002.qc.tsv").exists()


def test_changed_plan_is_refused(tmp_path):
    table = make_inputs(tmp_path, IDS)
    assert run(tmp_path, table).returncode == 0
    table.write_text("cohort\tgeo_accession\nGSETEST\tGSM2\nGSETEST\tGSM1\n")
    r = run(tmp_path, table)
    assert r.returncode != 0 and "ERROR:" in r.stderr and "batches.tsv" in r.stderr


def test_missing_idat_stops_and_names_sample(tmp_path):
    table = make_inputs(tmp_path, IDS, skip_red=("GSM4",))
    r = run(tmp_path, table)
    assert r.returncode != 0 and "ERROR: sample GSM4: Red IDAT missing" in r.stderr
    table.write_text("cohort\tgeo_accession\nGSETEST\tGSM9\n")
    r = run(tmp_path, table)
    assert r.returncode != 0 and "ERROR: sample GSM9: expected 1 Grn IDAT" in r.stderr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print("ok  ", t.__name__)
    print(f"{len(tests)} passed")
    sys.exit(0)
