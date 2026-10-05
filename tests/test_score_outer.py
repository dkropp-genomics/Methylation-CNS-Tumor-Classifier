"""Tests for scripts/score_outer.py on made-up prediction files (no betas).

Run with pytest, or directly:  python tests/test_score_outer.py
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from methylclf.metrics import summarize, to_family

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_calibrate_inner import CLASSES, FAMILY, fake_scores, make_run  # noqa: E402
from test_run_nested_cv import expect_stop, make_data  # noqa: E402

spec = importlib.util.spec_from_file_location("score_outer", ROOT / "scripts" / "score_outer.py")
so = importlib.util.module_from_spec(spec)
spec.loader.exec_module(so)
cv, cal = so.cv, so.cal


def full_run(d, s):
    pred, res = make_run(d, s)
    cal.run("r", s, FAMILY, Path(d) / "pred", Path(d) / "res", summarize, Cs=(0.1, 1.0))
    rng = np.random.default_rng(5)
    for k in range(5):
        rows = s[s["outer_fold"] == str(k)]
        cv.save_pred(
            cv.pred_path(pred, k, None, "s000"),
            fake_scores(rows["mc_class"].to_numpy(), rng, 0.5),
            rows["geo_accession"].to_numpy(),
            CLASSES,
        )
    return pred, res


def go(d, s):
    return so.score_run(
        "r", s, FAMILY, Path(d) / "pred", Path(d) / "res", summarize, to_family, n_boot=20
    )


def test_every_sample_scored_once_and_tables_written():
    _, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        _, res = full_run(d, s)
        metrics, per_class, preds = go(d, s)
        assert sorted(preds["geo_accession"]) == sorted(s["geo_accession"])
        fold = dict(zip(s["geo_accession"], s["outer_fold"].astype(int)))
        assert all(fold[g] == k for g, k in zip(preds["geo_accession"], preds["outer_fold"]))
        assert set(metrics["scores"]) == {"raw", "calibrated"}
        assert set(metrics["level"]) == {"class", "family"}
        assert {"all", "FFPE", "Frozen"} <= set(metrics["group"])
        assert per_class["n"].sum() == len(s) and len(per_class) == 6
        assert per_class["sensitivity"].between(0, 1).all()
        assert (
            per_class["n_family_correct"] >= (per_class["sensitivity"] * per_class["n"]).round()
        ).all()
        assert preds["class_score"].between(0, 1).all()
        # calibration sharpens underconfident scores
        assert preds["class_score"].mean() > preds["raw_class_score"].mean()
        assert not so.show(metrics, "all").empty
        for f in ("outer_metrics.tsv", "outer_per_class.tsv", "outer_predictions.tsv"):
            assert (res / f).exists()


def test_calibrator_inputs_hold_no_test_fold_sample_and_scoring_repeats():
    _, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        full_run(d, s)
        _, _, clean = go(d, s)
        parts_before, classes = cal.load_inner(Path(d) / "pred" / "r", 0, "s000", s)
        # the calibrator for fold 0 is fit on inner files that hold no fold-0 sample
        fold0 = set(s.loc[s["outer_fold"] == "0", "geo_accession"])
        assert not any(set(p["ids"]) & fold0 for p in parts_before)
        _, _, again = go(d, s)
        assert np.array_equal(clean["class_score"].to_numpy(), again["class_score"].to_numpy())


def test_missing_or_wrong_files_stop():
    _, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        pred, res = make_run(d, s)
        expect_stop(lambda: go(d, s), "calibration_selected.tsv")
        cal.run("r", s, FAMILY, Path(d) / "pred", Path(d) / "res", summarize, Cs=(1.0,))
        expect_stop(lambda: go(d, s), "--final-scoring")
    with tempfile.TemporaryDirectory() as d:
        pred, res = full_run(d, s)
        path = cv.pred_path(pred, 3, None, "s000")
        P, ids, cls = cv.load_pred(path)
        cv.save_pred(path, P[1:], ids[1:], cls)  # one test sample missing
        expect_stop(lambda: go(d, s), "outer test fold 3")
    with tempfile.TemporaryDirectory() as d:
        pred, res = full_run(d, s)
        t = pd.read_csv(res / "calibration_selected.tsv", sep="\t")
        t["setting"] = "other"
        t.to_csv(res / "calibration_selected.tsv", sep="\t", index=False)
        expect_stop(lambda: go(d, s), "calibration was chosen for setting")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
