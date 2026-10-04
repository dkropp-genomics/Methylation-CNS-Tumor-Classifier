"""Tests for scripts/calibrate_inner.py on made-up prediction files (no betas).

Run with pytest, or directly:  python tests/test_calibrate_inner.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from methylclf.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_run_nested_cv import expect_stop, make_data  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "calibrate_inner", ROOT / "scripts" / "calibrate_inner.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
cv = cal.cv

CLASSES = [f"K{i}" for i in range(6)]
FAMILY = {"K0": "F01", "K1": "F01", "K2": "K2", "K3": "K3", "K4": "F45", "K5": "F45"}


def fake_scores(y, rng, sharp=1.0):
    """Mostly-right, underconfident scores, like a random forest's vote shares."""
    z = rng.normal(0, 1, (len(y), len(CLASSES)))
    z[np.arange(len(y)), [CLASSES.index(c) for c in y]] += 3.0
    e = np.exp(sharp * z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def make_run(d, s, sharp=0.5, name="r"):
    rng = np.random.default_rng(0)
    pred, res = Path(d) / "pred" / name, Path(d) / "res" / name
    pred.mkdir(parents=True); res.mkdir(parents=True)
    for k in range(5):
        for j in range(3):
            rows = s[s[f"inner_fold_o{k}"] == str(j)]
            cv.save_pred(cv.pred_path(pred, k, j, "s000"),
                         fake_scores(rows["mc_class"].to_numpy(), rng, sharp),
                         rows["geo_accession"].to_numpy(), CLASSES)
    pd.DataFrame({"outer": range(5), "setting": "s000"}).to_csv(
        res / "selected.tsv", sep="\t", index=False)
    return pred, res


def go(d, s, **kw):
    return cal.run("r", s, FAMILY, Path(d) / "pred", Path(d) / "res", summarize,
                   Cs=(0.1, 1.0), **kw)


def test_calibration_lowers_log_loss_and_writes_tables():
    _, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        _, res = make_run(d, s)
        cands, sel, metrics = go(d, s)
        assert len(cands) == 5 * 2 * 2 and len(sel) == 5
        assert (sel["log_loss_after"] < sel["log_loss_before"]).all()
        for k in range(5):                                   # the chosen one is the best
            c = cands[cands["outer"] == k]
            assert np.isclose(sel.loc[sel["outer"] == k, "log_loss_after"].iloc[0],
                              c["log_loss_heldout"].min())
        assert set(metrics["stage"]) == {"before", "after"}
        assert set(metrics["level"]) == {"class", "family"}
        assert {"all", "FFPE", "Frozen"} <= set(metrics["group"])
        # every training-side sample is scored exactly once per outer fold
        n_all = metrics[(metrics["group"] == "all") & (metrics["level"] == "class")
                        & (metrics["stage"] == "after")].groupby("outer")["n"].first()
        assert list(n_all) == [int((s["outer_fold"] != str(k)).sum()) for k in range(5)]
        o = cal.overview(metrics)
        for f in ("calibration_candidates.tsv", "calibration_selected.tsv",
                  "calibration_inner_metrics.tsv"):
            assert (res / f).exists()


def test_poison_held_out_labels_do_not_change_held_out_probabilities():
    _, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        pred, _ = make_run(d, s)
        parts, classes = cal.load_inner(pred, 0, "s000", s)
        clean, _ = cal.cross_calibrate(parts, "log", 1.0, classes)
        poisoned = [dict(p) for p in parts]
        poisoned[2]["y"] = np.roll(parts[2]["y"], 7)         # wreck fold 2's labels
        dirty, _ = cal.cross_calibrate(poisoned, "log", 1.0, classes)
        assert np.array_equal(clean[2], dirty[2])            # fold 2's output: untouched
        assert not np.allclose(clean[0], dirty[0])           # folds fit on fold 2: changed


def test_probabilities_are_valid_and_order_is_kept():
    _, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        pred, _ = make_run(d, s)
        parts, classes = cal.load_inner(pred, 1, "s000", s)
        for how in cal.TRANSFORMS:
            out, ok = cal.cross_calibrate(parts, how, 1.0, classes)
            for p, c in zip(parts, out):
                assert c.shape == p["P"].shape and np.allclose(c.sum(axis=1), 1)
                assert (c >= 0).all()
        # calibration sharpens underconfident scores: the mean top score goes up
        assert np.vstack(out).max(axis=1).mean() > np.vstack([p["P"] for p in parts]).max(axis=1).mean()


def test_wrong_inputs_stop():
    _, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        expect_stop(lambda: go(d, s), "selected.tsv")
        pred, _ = make_run(d, s)
        fam = dict(FAMILY); del fam["K3"]
        expect_stop(lambda: cal.run("r", s, fam, Path(d) / "pred", Path(d) / "res", summarize),
                    "K3")
        # a file holding an outer-test sample is refused
        path = cv.pred_path(pred, 0, 0, "s000")
        P, ids, cls = cv.load_pred(path)
        ids[0] = s.loc[s["outer_fold"] == "0", "geo_accession"].iloc[0]
        cv.save_pred(path, P, ids, cls)
        expect_stop(lambda: go(d, s), "outer test fold")
        cv.pred_path(pred, 1, 2, "s000").unlink()
        expect_stop(lambda: cal.load_inner(pred, 1, "s000", s), "not found")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
