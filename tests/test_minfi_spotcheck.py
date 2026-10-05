"""Tests for R/minfi_spotcheck.R with MINFI_MOCK=1 (no minfi, no IDATs).
Run with `python -m pytest tests/` or `python tests/test_minfi_spotcheck.py`.
"""

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[1] / "R" / "minfi_spotcheck.R"


def rscript():
    if os.environ.get("RSCRIPT"):
        return [os.environ["RSCRIPT"]]
    if shutil.which("Rscript"):
        return ["Rscript"]
    return ["conda", "run", "-n", "methyl-r", "Rscript"]


def setup(tmp, ids=("GSM1", "GSM2")):
    d = tmp / "sesame" / "C1"
    d.mkdir(parents=True)
    n = 200
    cols = {"Probe_ID": pa.array([f"cg{i:08d}" for i in range(n)])}
    for k, g in enumerate(("GSM1", "GSM2")):
        v = [((i * (k + 3)) % 97) / 97 for i in range(n)]
        v[5] = None
        cols[g] = pa.array(v, type=pa.float32())
    pq.write_table(pa.table(cols), d / "batch_0001.parquet")
    table = tmp / "samples.tsv"
    table.write_text("cohort\tgeo_accession\tmaterial\n" + "".join(f"C1\t{g}\tFFPE\n" for g in ids))
    return table


def run(tmp, table):
    cmd = rscript() + [
        str(SCRIPT),
        "--samples",
        str(table),
        "--sesame",
        str(tmp / "sesame"),
        "--out",
        str(tmp / "out" / "cmp.tsv"),
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, env={**os.environ, "MINFI_MOCK": "1"}
    )


def test_table_written(tmp_path):
    r = run(tmp_path, setup(tmp_path))
    assert r.returncode == 0, r.stderr
    with open(tmp_path / "out" / "cmp.tsv") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert [x["geo_accession"] for x in rows] == ["GSM1", "GSM2"]
    assert all(x["n_probes_compared"] == "199" for x in rows)  # one NA dropped
    assert all(float(x["pearson_r"]) > 0.99 for x in rows)
    assert all(0 < float(x["mean_abs_diff"]) < 0.02 for x in rows)
    assert rows[0]["material"] == "FFPE"


def test_sample_missing_from_sesame_stops(tmp_path):
    r = run(tmp_path, setup(tmp_path, ids=("GSM1", "GSM7")))
    assert r.returncode != 0 and "ERROR: sample GSM7 is not in" in r.stderr


if __name__ == "__main__":
    for t in (test_table_written, test_sample_missing_from_sesame_stops):
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print("ok  ", t.__name__)
    print("2 passed")
