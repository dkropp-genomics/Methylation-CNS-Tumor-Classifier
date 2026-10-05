"""Tests for scripts/train_nn.py and methylclf.nn on synthetic data.

Run with pytest, or directly:  python tests/test_train_nn.py
The tests that train a network need PyTorch and are skipped without it.
"""

import copy
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

from methylclf.features import FeaturePipeline

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_run_nested_cv import Loader, expect_stop, make_data  # noqa: E402

spec = importlib.util.spec_from_file_location("train_nn", ROOT / "scripts" / "train_nn.py")
nn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nn)
cv = nn.cv

PROBES = np.array([f"cg{i:04d}" for i in range(120)], dtype=object)
CFG = {
    "run_name": "t",
    "model": "mlp",
    "seed": 1,
    "n_threads": 1,
    "select_by": "macro_f1",
    "levels": [1.0, 0.1, 0.01],
    "mask_seed": 9,
    "features": {"n_probes": 60, "correct_material": False},
    "train": {
        "hidden": [32, 16],
        "batch_size": 32,
        "weight_decay": 0.01,
        "max_epochs": 30,
        "eval_every": 10,
        "min_fraction": 0.01,
    },
    "grid": {"masked": [False, True], "lr": [0.01], "dropout": [0.0]},
}


def go(stage, d, X, s, cfg=CFG, **kw):
    load = Loader(X, s)
    out = nn.run(
        stage, cfg, s, load, PROBES, FeaturePipeline, Path(d) / "pred", Path(d) / "res", **kw
    )
    return out, load


def fake_fits(d, s, cfg=CFG):
    """Write made-up learning curves for every fit, without training anything."""
    settings = nn.expand_settings(cfg)
    pred = Path(d) / "pred" / cfg["run_name"]
    pred.mkdir(parents=True)
    classes = sorted(set(s["mc_class"]))
    for k in range(5):
        for j in range(3):
            ids = s.loc[s[f"inner_fold_o{k}"] == str(j), "geo_accession"].to_numpy()
            for a, r in settings.iterrows():
                # epochs 10, 20, 30; levels 1, 0.1, 0.01; metrics in nn.METRICS order
                v = np.zeros((3, 3, len(nn.METRICS)))
                peak = 1 if r["masked"] else 2  # masked peaks at epoch 20
                for e in range(3):
                    for b, drop in enumerate((0.0, 0.2, 0.5 if not r["masked"] else 0.3)):
                        v[e, b, :] = 0.9 - 0.1 * abs(e - peak) - drop + 0.001 * k
                nn.save_curve(
                    cv.pred_path(pred, k, j, r["setting"]),
                    [10, 20, 30],
                    v,
                    np.zeros((3, len(ids), len(classes))),
                    ids,
                    classes,
                    cfg["levels"],
                )


def test_settings_and_config_checks():
    t = nn.expand_settings(CFG)
    assert list(t["setting"]) == ["s000", "s001"] and list(t["masked"]) == [False, True]
    bad = copy.deepcopy(CFG)
    del bad["grid"]["masked"]
    expect_stop(lambda: nn.check_config(bad), "masked")
    bad = copy.deepcopy(CFG)
    bad["levels"] = [1.0, 0.0]
    expect_stop(lambda: nn.check_config(bad), "levels")


def test_select_picks_setting_and_epoch_per_network_and_outer_fold():
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        expect_stop(lambda: go("select", d, X, s), "missing")
    with tempfile.TemporaryDirectory() as d:
        fake_fits(d, s)
        (curves, best), load = go("select", d, X, s)
        assert load.calls == []  # select reads no betas
        assert len(curves) == 5 * 3 * 2 * 3 * 3
        assert len(best) == 10 and set(best["network"]) == {"plain", "masked"}
        assert (best.loc[best["network"] == "plain", "epoch"] == 30).all()
        assert (best.loc[best["network"] == "masked", "epoch"] == 20).all()
        assert (best.loc[best["network"] == "masked", "setting"] == "s001").all()
        m = best[best["network"] == "masked"].iloc[0]
        assert np.isclose(
            m["macro_f1_mean_over_levels"],
            np.mean([m["macro_f1_at_1"], m["macro_f1_at_0.1"], m["macro_f1_at_0.01"]]),
        )
        for f in ("inner_curves.tsv", "selected.tsv", "settings.tsv", "manifest.json"):
            assert (Path(d) / "res" / "t" / f).exists()
        pred = Path(d) / "pred" / "t"
        cv.pred_path(pred, 2, 1, "s000").unlink()
        expect_stop(lambda: go("select", d, X, s), "missing")


def test_training_learns_saves_curves_and_never_loads_the_test_fold():
    if not HAVE_TORCH:
        print("   (skipped: no PyTorch)")
        return
    X, s = make_data()
    with tempfile.TemporaryDirectory() as d:
        n, load = go("inner", d, X, s, outer_folds=[0], inner_folds=[0, 1])
        assert n == 4 and len(load.calls) == 1
        assert not set(load.calls[0]) & set(s.loc[s["outer_fold"] == "0", "geo_accession"])
        curves, _ = go("curves", d, X, s, outer_folds=[0], inner_folds=[0, 1])
        assert set(curves["epoch"]) == {10, 20, 30} and set(curves["level"]) == {1.0, 0.1, 0.01}
        last = curves[curves["epoch"] == 30]
        full = last[last["level"] == 1.0]
        assert (full.loc[~full["masked"], "accuracy"] > 0.8).all()  # an easy problem is learned
        assert (full.loc[full["masked"], "accuracy"] > 0.4).all()  # chance is 0.17
        # fewer observed probes cannot help
        by_level = last.groupby(["masked", "level"])["accuracy"].mean().unstack("level")
        assert (by_level[1.0] >= by_level[0.01]).all()
        with np.load(cv.pred_path(Path(d) / "pred" / "t", 0, 1, "s001")) as z:
            want = set(s.loc[s["inner_fold_o0"] == "1", "geo_accession"])
            assert set(z["sample_ids"]) == want
            assert z["proba_full"].shape == (3, len(want), 6)
            assert np.allclose(z["proba_full"].astype(np.float64).sum(axis=2), 1, atol=0.01)
        # resume: nothing refit; one deleted file: refit alone, same curve
        n, load = go("inner", d, X, s, outer_folds=[0], inner_folds=[0, 1])
        assert n == 0 and load.calls == []
        path = cv.pred_path(Path(d) / "pred" / "t", 0, 0, "s001")
        before = nn.load_curve(path)
        path.unlink()
        n, _ = go("inner", d, X, s, outer_folds=[0], inner_folds=[0, 1])
        assert n == 1
        assert np.allclose(before[nn.METRICS], nn.load_curve(path)[nn.METRICS], atol=1e-3)


def test_cosine_schedule_and_separate_epoch_limits():
    if not HAVE_TORCH:
        print("   (skipped: no PyTorch)")
        return
    X, s = make_data()
    cfg = copy.deepcopy(CFG)
    cfg["train"].update({"max_epochs": {"plain": 20, "masked": 30}, "schedule": "cosine"})
    assert nn.max_epochs_for(cfg, False) == 20 and nn.max_epochs_for(cfg, True) == 30
    with tempfile.TemporaryDirectory() as d:
        go("inner", d, X, s, cfg=cfg, outer_folds=[0], inner_folds=[0])
        curves, _ = go("curves", d, X, s, cfg=cfg, outer_folds=[0], inner_folds=[0])
        last = curves.groupby("masked")["epoch"].max()
        assert last[False] == 20 and last[True] == 30
        end = curves[(curves["epoch"] == 20) & ~curves["masked"] & (curves["level"] == 1.0)]
        assert (end["accuracy"] > 0.8).all()
    from methylclf.nn import Standardizer, train_mlp

    rng = np.random.default_rng(0)
    Zs = Standardizer().fit(Z := rng.normal(0.5, 0.1, (40, 6)).astype(np.float32)).transform(Z)
    y = (Zs[:, 0] > 0).astype(np.int64)
    seen = []
    kw = dict(
        masked=False,
        hidden=(4,),
        dropout=0.0,
        lr=0.01,
        batch_size=8,
        eval_every=4,
        seed=0,
        n_threads=1,
        schedule="cosine",
    )
    full = train_mlp(Zs, y, 2, max_epochs=10, **kw)
    stopped = train_mlp(
        Zs, y, 2, max_epochs=10, stop_epoch=6, on_checkpoint=lambda e, n_: seen.append(e), **kw
    )
    assert seen == [4, 6]
    assert stopped is not full
    expect_stop(lambda: train_mlp(Zs, y, 2, max_epochs=10, stop_epoch=11, **kw), "stop_epoch")
    bad = dict(kw)
    bad["schedule"] = "linear"
    expect_stop(lambda: train_mlp(Zs, y, 2, max_epochs=10, **bad), "schedule")


def test_network_input_and_masked_training_pieces():
    if not HAVE_TORCH:
        print("   (skipped: no PyTorch)")
        return
    import torch
    from methylclf.nn import (
        Standardizer,
        build_mlp,
        class_weights,
        make_input,
        predict_proba,
        train_mlp,
    )

    rng = np.random.default_rng(0)
    Z = rng.normal(0.5, 0.1, (50, 8)).astype(np.float32)
    sc = Standardizer().fit(Z)
    Zs = sc.transform(Z)
    assert np.allclose(Zs.mean(axis=0), 0, atol=1e-4) and np.allclose(Zs.std(axis=0), 1, atol=1e-3)
    obs = torch.tensor([[True, False] * 4])
    x = make_input(torch.from_numpy(Zs[:1]), obs)
    assert x.shape == (1, 16)
    assert (x[0, :8][~obs[0]] == 0).all()  # hidden values are zeroed
    assert x[0, 8:].tolist() == [1, 0] * 4  # and flagged
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    assert np.allclose(w, [4 / 6, 4 / 2])
    y = (Zs[:, 0] > 0).astype(np.int64)
    seen = []
    net = train_mlp(
        Zs,
        y,
        2,
        masked=True,
        hidden=(8,),
        dropout=0.0,
        lr=0.01,
        batch_size=16,
        max_epochs=4,
        eval_every=3,
        min_fraction=0.1,
        seed=0,
        n_threads=1,
        on_checkpoint=lambda e, n_: seen.append(e),
    )
    assert seen == [3, 4]  # every 3 epochs and the last
    p = predict_proba(net, Zs, np.ones(Zs.shape, dtype=bool))
    assert p.shape == (50, 2) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    # a hidden value has no effect: changing it leaves the output unchanged
    hide = np.ones(Zs.shape, dtype=bool)
    hide[:, 3] = False
    Z2 = Zs.copy()
    Z2[:, 3] = 99.0
    assert np.allclose(predict_proba(net, Zs, hide), predict_proba(net, Z2, hide))
    assert isinstance(build_mlp(8, 2, (4,), 0.1), torch.nn.Sequential)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests run")
