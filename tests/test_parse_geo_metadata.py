"""Tests for scripts/parse_geo_metadata.py (stdlib only; runs under pytest or directly)."""

from __future__ import annotations

import functools
import gzip
import http.server
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "parse_geo_metadata.py"
sys.path.insert(0, str(SCRIPT.parent))
import parse_geo_metadata as pgm  # noqa: E402


def matrix(lines: list[list[str]], table: list[str] = (), crlf: bool = False) -> bytes:
    """Build a series-matrix file: GEO quotes every value."""
    eol = "\r\n" if crlf else "\n"
    out = ['!Series_title\t"A study"']
    for key, *vals in lines:
        out.append("\t".join([key] + [f'"{v}"' for v in vals]))
    out.append("!series_matrix_table_begin")
    out.extend(table)
    out.append("!series_matrix_table_end")
    return gzip.compress((eol.join(out) + eol).encode(), mtime=0)


# Shaped like GSE90496: two characteristics rows, one with GEO's double space.
GSE90496_LIKE = [
    ["!Sample_title", "sample 1", "sample 2", "sample 3"],
    ["!Sample_geo_accession", "GSM1", "GSM2", "GSM3"],
    ["!Sample_status", "Public", "Public", "Public"],
    ["!Sample_source_name_ch1", "brain tumor", "brain tumor", "brain tumor"],
    ["!Sample_characteristics_ch1", "methylation class: GBM, RTK II",
     "methylation class: PIN T,  PB A", "methylation class: CONTR, CEBM"],
    ["!Sample_characteristics_ch1", "material: FFPE", "material: Frozen", "material: FFPE"],
]


def run(tmp: Path, *args: str, gz: bytes | None = None, acc: str = "GSE90496"):
    meta = tmp / "data/meta"
    meta.mkdir(parents=True, exist_ok=True)
    if gz is not None:
        (meta / f"{acc}_series_matrix.txt.gz").write_bytes(gz)
    return subprocess.run([sys.executable, str(SCRIPT), acc, *args], cwd=tmp,
                          capture_output=True, text=True, timeout=60)


def table(tmp: Path, acc: str = "GSE90496") -> list[list[str]]:
    text = (tmp / "data/meta" / f"{acc}_samples.tsv").read_text()
    return [line.split("\t") for line in text.splitlines()]


def ok(res):
    assert res.returncode == 0, res.stdout + res.stderr


def fails(res, text):
    assert res.returncode != 0 and text in res.stdout + res.stderr, res.stdout + res.stderr


# ------------------------------------------------------------------- tests --
def test_matches_r_script_layout_for_gse90496():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ok(run(tmp, gz=matrix(GSE90496_LIKE)))
        rows = table(tmp)
        assert rows[0] == ["geo_accession", "title", "source", "methylation.class", "material"]
        assert rows[2] == ["GSM2", "sample 2", "brain tumor", "PIN T,  PB A", "Frozen"]
        assert len(rows) == 4
        assert (tmp / "results/meta/GSE90496_fields.tsv").exists()


def test_misaligned_characteristics_land_in_the_right_column():
    """GSM2 has no age, so GEO shifts its material into the first row."""
    lines = [
        ["!Sample_title", "a", "b", "c"],
        ["!Sample_geo_accession", "GSM1", "GSM2", "GSM3"],
        ["!Sample_source_name_ch1", "x", "x", "x"],
        ["!Sample_characteristics_ch1", "age: 5", "material: FFPE", "age: 7"],
        ["!Sample_characteristics_ch1", "material: Frozen", "", "material: FFPE"],
    ]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        res = run(tmp, gz=matrix(lines), acc="GSE109379")
        ok(res)
        rows = {r[0]: dict(zip(table(tmp, "GSE109379")[0], r)) for r in table(tmp, "GSE109379")[1:]}
        assert rows["GSM2"]["material"] == "FFPE" and rows["GSM2"]["age"] == "NA"
        assert rows["GSM1"]["material"] == "Frozen"
        assert "'age' is missing for 1 sample" in res.stdout


def test_stops_at_data_table_and_handles_crlf():
    junk = ['"ID_REF"\t"GSM1"', "cg00000029\t0.5\t0.6\t0.7\t0.8"]  # ragged on purpose
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ok(run(tmp, gz=matrix(GSE90496_LIKE, table=junk, crlf=True)))
        assert table(tmp)[1][-1] == "FFPE"  # no stray \r


def test_duplicate_ids_fail():
    lines = [r[:] for r in GSE90496_LIKE]
    lines[1] = ["!Sample_geo_accession", "GSM1", "GSM1", "GSM3"]
    with tempfile.TemporaryDirectory() as d:
        fails(run(Path(d), gz=matrix(lines)), "duplicate sample IDs")


def test_row_length_mismatch_fails():
    lines = [r[:] for r in GSE90496_LIKE]
    lines[5] = ["!Sample_characteristics_ch1", "material: FFPE", "material: FFPE"]
    with tempfile.TemporaryDirectory() as d:
        fails(run(Path(d), gz=matrix(lines)), "has 2 values for 3 samples")


def test_repeated_key_for_one_sample_fails():
    lines = [r[:] for r in GSE90496_LIKE]
    lines.append(["!Sample_characteristics_ch1", "material: FFPE", "", ""])
    with tempfile.TemporaryDirectory() as d:
        fails(run(Path(d), gz=matrix(lines)), "has characteristic 'material' twice")


def test_downloads_when_missing():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        srv_root = tmp / "srv/geo/series/GSE90nnn/GSE90496/matrix"
        srv_root.mkdir(parents=True)
        (srv_root / "GSE90496_series_matrix.txt.gz").write_bytes(matrix(GSE90496_LIKE))
        handler = functools.partial(Quiet, directory=str(tmp / "srv"))
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1"
            url = f"http://127.0.0.1:{httpd.server_port}/geo"
            ok(run(tmp, "--base-url", url))
            assert not list((tmp / "data/meta").glob("*.part"))
            assert len(table(tmp)) == 4
        finally:
            httpd.shutdown()


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_make_names_matches_r():
    cases = {"methylation class": "methylation.class", "material": "material",
             "WHO 2016 diagnosis": "WHO.2016.diagnosis", "2nd opinion": "X2nd.opinion",
             "age (years)": "age..years.", ".5x": "X.5x"}
    for key, want in cases.items():
        assert pgm.r_make_names(key) == want, (key, pgm.r_make_names(key))


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
