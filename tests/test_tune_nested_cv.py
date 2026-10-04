"""Tests for scripts/tune_nested_cv.py: real Optuna and LightGBM, synthetic data.

Run with pytest, or directly:  python tests/test_tune_nested_cv.py
"""
import copy
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from methylclf.features import FeaturePipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_run_nested_cv import Loader, expect_stop, make_data  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "tune_nested_cv", ROOT / "scripts" / "tune_nested_cv.py")
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)

CFG = {"run_name": "t", "model": "lgbm", "seed": 3, "n_jobs": 1, "select_by": "macro_f1",
       "n_trials": 4,
       "fixed_params": {"class_weight": "balanced", "max_bin": 15, "subsample_freq": 1,
                        "force_col_wise": True, "deterministic": True, "verbose": -1},
       "search_space": {"n_probes": {"choice": [10, 40]},
                        "correct_material": {"choice": [False, True]},
                        "n_estimators": {"int": [5, 15]},
                        "learning_rate": {"float": [0.05, 0.3], "log": True},
                        "num_leaves": {"int": [3, 8]},
                        "min_child_samples": {"int": [2, 5]},
                        "colsample_bytree": {"float": [0.3, 1.0]},
                        "subsample": {"float": [0.6, 1.0]}}}


def go(d, X, s, cfg=CFG, **kw):
    load = Loader(X, s)
    out = tune.run(cfg, s, load, FeaturePipeline, Path(d) / "pred", Path(d) / "res", **kw)
    return out, load


def test_search_never_loads_the_outer_test_fold_and_saves_every_fit():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        (trials, scores, best, n), load = go(d, X, s, outer_folds=[0, 3])
        assert best is None                                # partial run: nothing selected
        assert len(trials) == 8 and len(scores) == 24
        files = list((Path(d) / "pred" / "t").glob("*.npz"))
        assert n == len(files) == 3 * trials.groupby("outer")["setting"].nunique().sum()
        assert len(load.calls) == 2
        for k, ids in zip([0, 3], load.calls):
            assert not set(ids) & set(s.loc[s["outer_fold"] == str(k), "geo_accession"])
        assert trials["macro_f1"].between(0, 1).all()
        assert not (Path(d) / "res" / "t" / "selected.tsv").exists()
        # different outer folds get different proposals (seed + fold)
        assert list(trials.loc[trials.outer == 0, "setting"]) != \
            list(trials.loc[trials.outer == 3, "setting"])


def test_replay_gives_the_same_trials_without_fitting_or_loading():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        (first, _, _, _), _ = go(d, X, s, outer_folds=[1])
        (again, _, _, n), load = go(d, X, s, outer_folds=[1])
        assert n == 0 and load.calls == []
        pd.testing.assert_frame_equal(first, again)


def test_a_short_run_is_a_prefix_of_the_full_run():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        (short, _, _, n2), _ = go(d, X, s, outer_folds=[2], max_trials=2)
        assert len(short) == 2
        (full, _, _, n4), _ = go(d, X, s, outer_folds=[2])
        assert list(full["setting"][:2]) == list(short["setting"])
        assert n2 + n4 == 3 * full["setting"].nunique()    # nothing was fit twice
    with tempfile.TemporaryDirectory() as d:               # and equals an uninterrupted run
        (fresh, _, _, _), _ = go(d, X, s, outer_folds=[2])
        assert list(fresh["setting"]) == list(full["setting"])
        assert np.allclose(fresh["macro_f1"], full["macro_f1"])


def test_a_deleted_fit_is_redone_alone_with_the_same_result():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        (first, _, _, _), _ = go(d, X, s, outer_folds=[4])
        victim = sorted((Path(d) / "pred" / "t").glob("o4_i1_*.npz"))[0]
        before = np.load(victim)["proba"]
        victim.unlink()
        (again, _, _, n), load = go(d, X, s, outer_folds=[4])
        assert n == 1 and len(load.calls) == 1
        assert np.allclose(before, np.load(victim)["proba"], atol=1e-6)
        assert list(again["setting"]) == list(first["setting"])


def test_complete_search_selects_the_best_trial_per_outer_fold():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        (trials, scores, best, _), _ = go(d, X, s)
        assert len(trials) == 20 and len(best) == 5
        for k in range(5):
            t = trials[trials["outer"] == k]
            first_best = t.loc[t["macro_f1"] == t["macro_f1"].max(), "trial"].min()
            assert best.loc[best["outer"] == k, "trial"].iloc[0] == first_best
        res = Path(d) / "res" / "t"
        for f in ("selected.tsv", "settings.tsv", "inner_scores.tsv", "trials_o0.tsv",
                  "manifest.json", "fits.tsv"):
            assert (res / f).exists(), f
        st = pd.read_csv(res / "settings.tsv", sep="\t")
        assert st["setting"].is_unique and set(best["setting"]) <= set(st["setting"])


def test_final_scoring_works_on_a_finished_search():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        (_, _, best, _), _ = go(d, X, s)
        load = Loader(X, s)
        args = ("outer", CFG, s, load, FeaturePipeline, Path(d) / "pred", Path(d) / "res")
        expect_stop(lambda: tune.cv.run(*args), "--final-scoring")
        assert tune.cv.run(*args, final_scoring=True) == 5
        for k in range(5):
            setting = best.loc[best["outer"] == k, "setting"].iloc[0]
            proba, sid, _ = tune.cv.load_pred(tune.cv.pred_path(Path(d) / "pred" / "t", k, None, setting))
            assert set(sid) == set(s.loc[s["outer_fold"] == str(k), "geo_accession"])
            assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)


def test_bad_or_changed_config_stops():
    X, s = make_data()
    bad = copy.deepcopy(CFG); del bad["search_space"]["n_probes"]
    with tempfile.TemporaryDirectory() as d:
        expect_stop(lambda: go(d, X, s, cfg=bad), "n_probes")
    bad = copy.deepcopy(CFG); bad["fixed_params"]["num_leaves"] = 5
    with tempfile.TemporaryDirectory() as d:
        expect_stop(lambda: go(d, X, s, cfg=bad), "both")
    with tempfile.TemporaryDirectory() as d:
        go(d, X, s, outer_folds=[0], max_trials=1)
        other = copy.deepcopy(CFG); other["search_space"]["num_leaves"]["int"] = [3, 9]
        expect_stop(lambda: go(d, X, s, cfg=other, outer_folds=[0], max_trials=1),
                    "different settings")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
