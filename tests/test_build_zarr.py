"""Tests for scripts/build_zarr.py on synthetic batch files.
Run with `python -m pytest tests/` or `python tests/test_build_zarr.py`.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_zarr import build, read_tsv  # noqa: E402

PROBES = [f"cg{i:08d}" for i in range(1, 51)]
BATCHES = {1: ["GSM1", "GSM2"], 2: ["GSM3", "GSM4"], 3: ["GSM5"]}  # GSM3 failed


def values(gsm):
    if gsm == "GSM3":
        return [None] * len(PROBES)
    v = [round(int(gsm[3:]) / 10 + j / 1000, 4) for j in range(len(PROBES))]
    v[int(gsm[3:])] = None  # one missing value per sample
    return v


def make_batches(tmp, dtype=pa.float32(), probes_b2=None):
    d = tmp / "batches"
    (d / "C1").mkdir(parents=True)
    lines = ["cohort\tbatch\tgeo_accession"]
    for b, ids in BATCHES.items():
        cols = {"Probe_ID": pa.array(probes_b2 if (b == 2 and probes_b2) else PROBES)}
        cols.update({g: pa.array(values(g), type=dtype) for g in ids})
        pq.write_table(pa.table(cols), d / "C1" / f"batch_{b:04d}.parquet")
        qc = ["cohort\tgeo_accession\tstatus\tfrac_na"]
        qc += [f"C1\t{g}\t{'error: x' if g == 'GSM3' else 'ok'}\t0.02" for g in ids]
        (d / "C1" / f"batch_{b:04d}.qc.tsv").write_text("\n".join(qc) + "\n")
        lines += [f"C1\t{b}\t{g}" for g in ids]
    (d / "batches.tsv").write_text("\n".join(lines) + "\n")
    return d


def run(tmp, d, **kw):
    return build("C1", d, tmp / "zarr", tmp / "qc", **kw)


def expect_error(fn, text):
    try:
        fn()
    except SystemExit as e:
        assert str(e).startswith("ERROR:") and text in str(e), str(e)
    else:
        raise AssertionError("expected ERROR")


def test_store_values_and_index_files(tmp_path):
    store = run(tmp_path, make_batches(tmp_path))
    arr = zarr.open_group(str(store), mode="r")["betas"]
    assert arr.shape == (5, 50) and str(arr.dtype) == "float32"
    x = arr[:]
    assert np.isnan(x[2]).all()                      # failed sample
    assert np.isnan(x[0, 1]) and np.isnan(x).sum() == 50 + 4
    assert x[3, 7] == np.float32(0.407)              # GSM4, probe 8
    samples = read_tsv(tmp_path / "zarr" / "C1.samples.tsv")
    assert [s["geo_accession"] for s in samples] == ["GSM1", "GSM2", "GSM3", "GSM4", "GSM5"]
    assert [s["row"] for s in samples] == list("01234")
    probes = read_tsv(tmp_path / "zarr" / "C1.probes.tsv")
    assert [p["Probe_ID"] for p in probes] == PROBES
    assert len(read_tsv(tmp_path / "qc" / "C1_sesame_qc.tsv")) == 5
    assert not (tmp_path / "zarr" / "C1.zarr.part").exists()


def test_rebuild_replaces_store(tmp_path):
    d = make_batches(tmp_path)
    run(tmp_path, d)
    store = run(tmp_path, d)
    assert zarr.open_group(str(store), mode="r")["betas"].shape == (5, 50)


def test_missing_batch_stops(tmp_path):
    d = make_batches(tmp_path)
    (d / "C1" / "batch_0002.qc.tsv").unlink()
    expect_error(lambda: run(tmp_path, d), "batch 2 is missing or incomplete")
    assert not (tmp_path / "zarr" / "C1.zarr").exists()


def test_probe_order_mismatch_stops(tmp_path):
    d = make_batches(tmp_path, probes_b2=PROBES[::-1])
    expect_error(lambda: run(tmp_path, d), "probe order differs")


def test_float64_input_stops(tmp_path):
    d = make_batches(tmp_path, dtype=pa.float64())
    expect_error(lambda: run(tmp_path, d), "expected float32")


def test_order_from_is_checked(tmp_path):
    d = make_batches(tmp_path)
    ref = tmp_path / "ref.tsv"
    ref.write_text("geo_accession\n" + "\n".join(["GSM2", "GSM1", "GSM3", "GSM4", "GSM5"]) + "\n")
    expect_error(lambda: run(tmp_path, d, order_from=ref), "sample order")
    ref.write_text("geo_accession\n" + "\n".join(["GSM1", "GSM2", "GSM3", "GSM4", "GSM5"]) + "\n")
    run(tmp_path, d, order_from=ref)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        with tempfile.TemporaryDirectory() as tmp:
            t(Path(tmp))
        print("ok  ", t.__name__)
    print(f"{len(tests)} passed")
