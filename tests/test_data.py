"""Tests for methylclf.data on a small synthetic store.

Run with `python -m pytest tests/test_data.py` or `python tests/test_data.py`.

The synthetic store has 40 rows x 60 columns with chunks (8, 16), so every
read crosses chunk borders. Each cell holds row + col/1000, so a value tells
you exactly which store cell it came from.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from methylclf.data import BetaStore, DataError  # noqa: E402

N_ROWS, N_COLS = 40, 60
DROPPED_ROWS = [3, 8, 17, 39]  # "failed QC": in the store, not in the fold table
FILTERED_COLS = [0, 5, 16, 31, 59]  # "filtered probes": in the store, not kept


def make_store(tmp: Path, poison: float = 1e6, fold_edit=None, probe_edit=None):
    import zarr

    data = (np.arange(N_ROWS)[:, None] + np.arange(N_COLS)[None, :] / 1000).astype(np.float32)
    data[10, 7] = np.nan  # a genuine missing value in a kept cell
    data[DROPPED_ROWS, :] = poison
    data[:, FILTERED_COLS] = poison

    zpath = tmp / "GSEtest.zarr"
    arr = zarr.open_group(str(zpath), mode="w").create_array(
        "betas", shape=data.shape, chunks=(8, 16), dtype="float32"
    )
    arr[:] = data

    gsm = [f"GSM{1000 + i}" for i in range(N_ROWS)]
    cg = [f"cg{i:08d}" for i in range(N_COLS)]
    pd.DataFrame({"row": range(N_ROWS), "geo_accession": gsm}).to_csv(
        tmp / "GSEtest.samples.tsv", sep="\t", index=False
    )
    pd.DataFrame({"col": range(N_COLS), "Probe_ID": cg}).to_csv(
        tmp / "GSEtest.probes.tsv", sep="\t", index=False
    )

    keep_r = [i for i in range(N_ROWS) if i not in DROPPED_ROWS]
    folds = pd.DataFrame(
        {"geo_accession": [gsm[i] for i in keep_r], "zarr_row": keep_r, "mc_class": "A"}
    )
    # fold table deliberately NOT in store order
    folds = folds.sample(frac=1, random_state=1).reset_index(drop=True)
    keep_c = [j for j in range(N_COLS) if j not in FILTERED_COLS]
    probes = pd.DataFrame({"col": keep_c, "Probe_ID": [cg[j] for j in keep_c]})
    if fold_edit:
        folds = fold_edit(folds)
    if probe_edit:
        probes = probe_edit(probes)
    folds.to_csv(tmp / "folds.tsv", sep="\t", index=False)
    probes.to_csv(tmp / "probes_kept.tsv", sep="\t", index=False)
    return zpath, tmp / "folds.tsv", tmp / "probes_kept.tsv"


def open_store(**kw):
    tmp = Path(tempfile.mkdtemp())
    return BetaStore.open(*make_store(tmp, **kw))


def expect_error(fn, text):
    try:
        fn()
    except DataError as e:
        assert str(e).startswith("ERROR:"), e
        assert text in str(e), e
        return
    raise AssertionError(f"expected DataError containing '{text}'")


def test_values_come_from_the_right_cells():
    st = open_store()
    ids = st.sample_ids
    X = st.load(ids)
    assert X.dtype == np.float32 and X.shape == (36, 55)
    rows = np.array([int(s[3:]) - 1000 for s in ids])
    cols = np.array([int(p[2:]) for p in st.probe_ids])
    want = (rows[:, None] + cols[None, :] / 1000).astype(np.float32)
    ok = ~np.isnan(X)
    assert np.array_equal(X[ok], want[ok])
    assert np.isnan(X).sum() == 1  # only the one genuine NaN


def test_poison_in_dropped_rows_and_filtered_columns_changes_nothing():
    a = open_store(poison=1e6)
    b = open_store(poison=-777.0)
    Xa, Xb = a.load(a.sample_ids), b.load(b.sample_ids)
    assert np.nanmax(np.abs(Xa)) < 100  # no poison got in
    assert np.array_equal(Xa, Xb, equal_nan=True)
    sa, sb = a.probe_stats(a.sample_ids), b.probe_stats(b.sample_ids)
    pd.testing.assert_frame_equal(sa, sb)


def test_requested_order_is_returned_order():
    st = open_store()
    ids = ["GSM1030", "GSM1002", "GSM1031", "GSM1009"]  # spans chunks, not sorted
    pids = ["cg00000040", "cg00000001", "cg00000017"]
    X = st.load(ids, pids)
    want = np.array([[r + c / 1000 for c in (40, 1, 17)] for r in (30, 2, 31, 9)], np.float32)
    assert np.array_equal(X, want)


def test_probe_stats_match_numpy_on_the_loaded_matrix():
    st = open_store()
    train = list(st.sample_ids[:20])
    X = st.load(train).astype(np.float64)
    s = st.probe_stats(train)
    assert list(s["Probe_ID"]) == list(st.probe_ids)
    assert np.array_equal(s["n_obs"], (~np.isnan(X)).sum(axis=0))
    assert np.allclose(s["frac_missing"], np.isnan(X).mean(axis=0))
    assert np.allclose(s["mean"], np.nanmean(X, axis=0))
    assert np.allclose(s["var"], np.nanvar(X, axis=0, ddof=1), atol=1e-6)


def test_probe_stats_depend_only_on_the_samples_passed():
    st = open_store()
    a = st.probe_stats(list(st.sample_ids[:10]))
    b = st.probe_stats(list(st.sample_ids[10:]))
    assert not np.allclose(a["mean"], b["mean"])


def test_dropped_sample_and_filtered_probe_cannot_be_loaded():
    st = open_store()
    expect_error(lambda: st.load(["GSM1003"]), "not in the fold table")
    expect_error(lambda: st.load(["GSM1001"], ["cg00000005"]), "not in the kept-probe table")
    expect_error(lambda: st.load(["GSM1001", "GSM1001"]), "more than once")
    expect_error(lambda: st.load([]), "no sample IDs")


def test_fold_table_with_wrong_zarr_row_is_refused():
    def swap(f):  # two samples exchange rows: still unique, still in range
        f = f.copy()
        f.loc[[0, 1], "zarr_row"] = f.loc[[1, 0], "zarr_row"].to_numpy()
        return f

    expect_error(lambda: open_store(fold_edit=swap), "store index says row")


def test_fold_table_by_position_instead_of_zarr_row_is_refused():
    # the classic mistake: zarr_row = position in the (QC-filtered) fold table
    expect_error(
        lambda: open_store(fold_edit=lambda f: f.assign(zarr_row=range(len(f)))),
        "store index says row",
    )


def test_probe_table_with_shifted_col_is_refused():
    expect_error(
        lambda: open_store(probe_edit=lambda p: p.assign(col=p["col"] + 1)), "but the store has"
    )


def test_missing_column_is_refused():
    expect_error(
        lambda: open_store(fold_edit=lambda f: f.drop(columns="zarr_row")), "missing column"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
