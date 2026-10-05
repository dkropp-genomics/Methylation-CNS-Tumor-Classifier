"""Tests for scripts/run_nested_cv.py on synthetic data with a fake loader.

Run with pytest, or directly:  python tests/test_run_nested_cv.py
"""

import copy
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from methylclf.features import FeaturePipeline

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_nested_cv", ROOT / "scripts" / "run_nested_cv.py"
)
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)

CFG = {
    "run_name": "t",
    "model": "rf",
    "seed": 1,
    "n_jobs": 1,
    "select_by": "macro_f1",
    "features": {"n_probes": [10, 40], "correct_material": [False, True]},
    "model_params": {"n_estimators": [15], "max_features": ["sqrt"]},
}
N_SETTINGS = 4


def make_data(seed=0, n_classes=6, per=30, p=120):
    rng = np.random.default_rng(seed)
    y = np.repeat([f"K{i}" for i in range(n_classes)], per).astype(object)
    n = y.size
    centers = rng.uniform(0.2, 0.8, (n_classes, p))
    X = centers[np.repeat(np.arange(n_classes), per)] + rng.normal(0, 0.08, (n, p))
    mat = np.where(np.arange(n) % 2 == 0, "FFPE", "Frozen").astype(object)
    X[mat == "FFPE", :20] += 0.05
    X = np.clip(X, 0, 1).astype(np.float32)
    X[rng.random((n, p)) < 0.01] = np.nan
    X[:, -1] = np.nan  # a probe missing everywhere
    s = pd.DataFrame(
        {
            "geo_accession": [f"GSM{i:04d}" for i in range(n)],
            "mc_class": y,
            "material": mat,
            "outer_fold": "-1",
        }
    )
    outer = np.empty(n, dtype=int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1).split(X, y)):
        outer[te] = k
    s["outer_fold"] = outer.astype(str)  # fold columns are strings, as in BetaStore
    for k in range(5):
        col = np.full(n, -1)
        tr = np.flatnonzero(outer != k)
        for j, (_, te) in enumerate(
            StratifiedKFold(3, shuffle=True, random_state=k).split(tr, y[tr])
        ):
            col[tr[te]] = j
        s[f"inner_fold_o{k}"] = col.astype(str)
    return X, s


class Loader:
    """Fake BetaStore.load that records every sample ID it was asked for."""

    def __init__(self, X, samples):
        self.X, self.row = X, {g: i for i, g in enumerate(samples["geo_accession"])}
        self.calls = []

    def __call__(self, ids):
        self.calls.append(list(ids))
        return self.X[[self.row[i] for i in ids]].copy()


def go(stage, d, X, s, cfg=CFG, **kw):
    load = Loader(X, s)
    out = cv.run(stage, cfg, s, load, FeaturePipeline, Path(d) / "pred", Path(d) / "res", **kw)
    return out, load


def expect_stop(fn, text):
    try:
        fn()
    except SystemExit as e:
        assert "ERROR" in str(e) and text in str(e), str(e)
    else:
        raise AssertionError(f"expected a stop mentioning '{text}'")


def test_settings_grid():
    t = cv.expand_settings(CFG)
    assert list(t["setting"]) == ["s000", "s001", "s002", "s003"]
    assert sorted(set(zip(t["n_probes"], t["correct_material"]))) == [
        (10, False),
        (10, True),
        (40, False),
        (40, True),
    ]
    bad = copy.deepcopy(CFG)
    bad["select_by"] = "auc"
    expect_stop(lambda: cv.expand_settings(bad), "select_by")


def test_inner_stage_never_loads_the_outer_test_fold():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        n, load = go("inner", d, X, s)
        assert n == 5 * 3 * N_SETTINGS
        assert len(load.calls) == 5  # one load per outer fold
        for k, ids in enumerate(load.calls):
            test_ids = set(s.loc[s["outer_fold"] == str(k), "geo_accession"])
            assert not (set(ids) & test_ids)
        assert not list((Path(d) / "pred" / "t").glob("*test*"))
        assert not list((Path(d) / "pred" / "t").glob("*.tmp.npz"))
        # each file holds exactly its inner fold, with valid probabilities
        proba, sid, classes = cv.load_pred(Path(d) / "pred" / "t" / "o2_i1_s003.npz")
        want = set(s.loc[s["inner_fold_o2"] == "1", "geo_accession"])
        assert set(sid) == want and len(sid) == len(want)
        assert list(classes) == sorted(set(s["mc_class"]))
        assert proba.shape == (len(want), 6) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
        fits = pd.read_csv(Path(d) / "res" / "t" / "fits.tsv", sep="\t")
        assert len(fits) == 60 and set(fits["n_features"]) == {10, 40}


def test_resume_skips_finished_fits_and_redoes_a_missing_one():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        go("inner", d, X, s, outer_folds=[0, 1])
        n, load = go("inner", d, X, s, outer_folds=[0, 1])
        assert n == 0 and load.calls == []  # nothing loaded, nothing fit
        target = Path(d) / "pred" / "t" / "o1_i2_s001.npz"
        before = cv.load_pred(target)[0]
        target.unlink()
        n, load = go("inner", d, X, s, outer_folds=[0, 1])
        assert n == 1 and len(load.calls) == 1
        assert np.array_equal(before, cv.load_pred(target)[0])  # same seed, same answer
        n, _ = go("inner", d, X, s)  # the remaining three outer folds
        assert n == 3 * 3 * N_SETTINGS


def test_changed_config_or_folds_is_refused():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        go("inner", d, X, s, outer_folds=[0])
        other = copy.deepcopy(CFG)
        other["model_params"]["n_estimators"] = [16]
        expect_stop(lambda: go("inner", d, X, s, cfg=other, outer_folds=[0]), "different settings")
        s2 = s.copy()
        s2.loc[0, "material"] = "Frozen"
        expect_stop(lambda: go("inner", d, X, s2, outer_folds=[0]), "different settings")


def test_select_needs_all_fits_then_picks_the_best_per_outer_fold():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        go("inner", d, X, s, outer_folds=[0])
        expect_stop(lambda: go("select", d, X, s), "missing")
        go("inner", d, X, s)
        (scores, mean, best), load = go("select", d, X, s)
        assert load.calls == []  # select reads no betas
        assert len(scores) == 60 and len(mean) == 20 and len(best) == 5
        assert scores["accuracy"].between(0, 1).all() and scores["accuracy"].mean() > 0.8
        for k in range(5):
            m = mean[mean["outer"] == k]
            top = m["macro_f1"].max()
            pick = best.loc[best["outer"] == k, "setting"].iloc[0]
            assert pick == sorted(m.loc[m["macro_f1"] == top, "setting"])[0]
        for f in ("inner_scores.tsv", "selected.tsv", "settings.tsv", "manifest.json"):
            assert (Path(d) / "res" / "t" / f).exists()


def test_outer_stage_is_gated_and_scores_each_sample_once():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        go("inner", d, X, s)
        expect_stop(lambda: go("outer", d, X, s, final_scoring=True), "select")
        go("select", d, X, s)
        expect_stop(lambda: go("outer", d, X, s), "--final-scoring")
        n, _ = go("outer", d, X, s, final_scoring=True)
        assert n == 5
        seen = []
        for path in sorted((Path(d) / "pred" / "t").glob("o*_test_*.npz")):
            k = int(path.name[1])
            proba, sid, _ = cv.load_pred(path)
            assert set(sid) == set(s.loc[s["outer_fold"] == str(k), "geo_accession"])
            seen += list(sid)
        assert sorted(seen) == sorted(s["geo_accession"])
        n, _ = go("outer", d, X, s, final_scoring=True)
        assert n == 0


def test_a_training_set_without_every_class_stops():
    X, s = make_data()
    s = s.copy()
    lone = s.index[(s["mc_class"] == "K5")]
    s.loc[lone, "inner_fold_o0"] = np.where(s.loc[lone, "outer_fold"] == "0", "-1", "0")
    with (
        tempfile.TemporaryDirectory() as d
    ):  # all K5 in inner fold 0 -> absent from its training set
        expect_stop(lambda: go("inner", d, X, s, outer_folds=[0]), "classes")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
