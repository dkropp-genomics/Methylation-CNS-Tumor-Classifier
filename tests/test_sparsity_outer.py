"""Tests for scripts/sparsity_outer.py on synthetic data.

Run with pytest, or directly:  python tests/test_sparsity_outer.py
The network models need PyTorch; without it the tests cover the tree models only.
"""
import copy
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from methylclf.features import FeaturePipeline
from methylclf.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_run_nested_cv as t_rf  # noqa: E402
import test_train_nn as t_nn  # noqa: E402
import test_tune_nested_cv as t_lgbm  # noqa: E402
from test_run_nested_cv import Loader, expect_stop, make_data  # noqa: E402

spec = importlib.util.spec_from_file_location("sparsity_outer", ROOT / "scripts" / "sparsity_outer.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)
cv = sp.cv

PROBES = t_nn.PROBES
FAMILY = {"K0": "F01", "K1": "F01", "K2": "K2", "K3": "K3", "K4": "F45", "K5": "F45"}
RF_CFG = dict(t_rf.CFG, run_name="rf")
LG_CFG = dict(t_lgbm.CFG, run_name="lg")
NN_CFG = dict(copy.deepcopy(t_nn.CFG), run_name="nn")
NN_CFG["train"].update({"max_epochs": {"plain": 20, "masked": 30}, "schedule": "cosine"})


def setup(d, X, s, with_nn):
    """Run each model's own search and Phase 3 outer stage on synthetic data."""
    pred, res = Path(d) / "pred", Path(d) / "res"
    args = (s, Loader(X, s), FeaturePipeline, pred, res)
    cv.run("inner", RF_CFG, *args); cv.run("select", RF_CFG, *args)
    cv.run("outer", RF_CFG, *args, final_scoring=True)
    t_lgbm.tune.run(LG_CFG, *args)
    cv.run("outer", LG_CFG, *args, final_scoring=True)
    models = {"rf": {"kind": "tree", "run": "rf", "config": "-"},
              "lgbm": {"kind": "tree", "run": "lg", "config": "-"}}
    cfgs = {"rf": RF_CFG, "lgbm": LG_CFG}
    if with_nn:
        t_nn.nn.run("inner", NN_CFG, s, Loader(X, s), PROBES, FeaturePipeline, pred, res)
        t_nn.nn.run("select", NN_CFG, s, Loader(X, s), PROBES, FeaturePipeline, pred, res)
        for net in ("plain", "masked"):
            models[f"nn_{net}"] = {"kind": "nn", "run": "nn", "config": "-", "network": net}
            cfgs[f"nn_{net}"] = NN_CFG
    cfg = {"run_name": "sp", "levels": [1.0, 0.1, 0.01], "mask_seed": 9, "models": models}
    return cfg, cfgs, pred, res


def fit(cfg, cfgs, X, s, pred, res, **kw):
    load = Loader(X, s)
    n = sp.run("fit", cfg, cfgs, s, load, PROBES, FeaturePipeline, pred, res, **kw)
    return n, load


def test_fit_is_gated_reproduces_phase3_and_report_pools_every_sample():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        cfg, cfgs, pred, res = setup(d, X, s, t_nn.HAVE_TORCH)
        n_models = len(cfg["models"])
        expect_stop(lambda: fit(cfg, cfgs, X, s, pred, res), "--final-scoring")
        n, load = fit(cfg, cfgs, X, s, pred, res, final_scoring=True, outer_folds=[0, 1])
        assert n == 2 * n_models and len(load.calls) == 4          # train + test per fold
        n, _ = fit(cfg, cfgs, X, s, pred, res, final_scoring=True)
        assert n == 3 * n_models                                   # the other three folds
        n, load = fit(cfg, cfgs, X, s, pred, res, final_scoring=True)
        assert n == 0 and load.calls == []                         # resume: nothing to do
        # at 100% coverage the forest refit equals the saved Phase 3 predictions
        check = pd.read_csv(res / "sp" / "repro_check.tsv", sep="\t")
        rf = check[check["model"] == "rf"]
        assert len(rf) == 5 and rf["same_sample_order"].all()
        assert (rf["max_abs_diff"] < 1e-6).all() and (rf["n_top_class_changed"] == 0).all()
        with np.load(sp.out_path(pred / "sp", "rf", 3)) as z:
            assert z["proba"].shape[0] == 3 and z["proba"].shape[2] == 6
            assert np.allclose(z["proba"].sum(axis=2), 1, atol=1e-4)
            assert set(z["sample_ids"]) == set(s.loc[s["outer_fold"] == "3", "geo_accession"])

        metrics, diffs = sp.run("report", cfg, cfgs, s, pred_root=pred, res_root=res,
                                family_of=FAMILY, summarize=summarize, n_boot=20)
        assert set(metrics["model"]) == set(cfg["models"])
        assert set(metrics["coverage"]) == {1.0, 0.1, 0.01}
        assert set(metrics["label_level"]) == {"class", "family"}
        assert (metrics["n"] == len(s)).all()                      # every sample, once
        acc = metrics[(metrics["metric"] == "accuracy") & (metrics["label_level"] == "class")]
        acc = acc.pivot_table(index="model", columns="coverage", values="value")
        assert (acc[1.0] > 0.8).loc[["rf"]].all()
        assert (acc[1.0] >= acc[0.01]).all()                       # fewer CpGs cannot help
        assert sp.table(metrics[(metrics["metric"] == "accuracy")
                                & (metrics["group"] == "all")], "class").shape[1] == 3
        if t_nn.HAVE_TORCH:
            assert {"nn_masked", "nn_plain"} <= set(diffs["model_a"])
            assert set(diffs["metric"]) == {"accuracy", "macro_f1"}
            assert (diffs["ci_low"] <= diffs["difference"] + 1e-9).all()
        for f in ("metrics.tsv", "differences.tsv", "manifest.json", "fits.tsv"):
            assert (res / "sp" / f).exists()


def test_fast_metrics_and_paired_difference():
    t = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    from sklearn.metrics import f1_score
    assert np.isclose(sp.macro_f1_fast(t, p, 3), f1_score(t, p, average="macro"))
    # a class absent from the true labels is left out of the average
    assert np.isclose(sp.macro_f1_fast(np.array([0, 0, 1]), np.array([0, 2, 1]), 3),
                      f1_score([0, 0, 1], [0, 2, 1], labels=[0, 1], average="macro"))
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 4, 400)
    good = np.where(rng.random(400) < 0.9, truth, (truth + 1) % 4)
    bad = np.where(rng.random(400) < 0.5, truth, (truth + 1) % 4)
    out = {r["metric"]: r for r in sp.paired_difference(truth, good, bad, 4, 200, seed=0)}
    assert out["accuracy"]["ci_low"] > 0 and out["macro_f1"]["ci_low"] > 0
    assert out["accuracy"]["share_of_draws_above_0"] == 1.0
    same = {r["metric"]: r for r in sp.paired_difference(truth, good, good, 4, 50, seed=0)}
    assert same["accuracy"]["difference"] == 0 and same["accuracy"]["ci_high"] == 0


def test_config_checks():
    X, s = make_data()
    base = {"run_name": "sp", "levels": [1.0, 0.1], "mask_seed": 9,
            "models": {"n": {"kind": "nn", "run": "nn", "config": "-", "network": "masked"}}}
    expect_stop(lambda: sp.check_config(base, {"n": NN_CFG}), "levels and mask_seed")
    bad = copy.deepcopy(base); bad["models"]["n"]["kind"] = "svm"
    expect_stop(lambda: sp.check_config(bad, {"n": NN_CFG}), "kind")
    bad = copy.deepcopy(base); bad["levels"] = [1.0, 0.1, 0.01]; bad["models"]["n"]["network"] = "x"
    expect_stop(lambda: sp.check_config(bad, {"n": NN_CFG}), "network")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests run")
