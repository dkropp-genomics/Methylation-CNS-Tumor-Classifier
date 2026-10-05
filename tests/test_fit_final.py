"""Tests for scripts/fit_final.py on synthetic data (no real betas, no validation cohort)."""

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fit_final as ff  # noqa: E402
import run_nested_cv as cv  # noqa: E402
import sparsity_outer as so  # noqa: E402
from methylclf.features import FeaturePipeline  # noqa: E402
from methylclf.masking import observed_uniform  # noqa: E402

LEVELS = [1.0, 0.1, 0.01]
RF_CFG = {"run_name": "rf_t", "model": "rf", "seed": 7, "n_jobs": 2}
RF_PARAMS = {"n_estimators": 30, "max_features": "sqrt"}
NN_CFG = {
    "run_name": "nn_t",
    "model": "mlp",
    "seed": 7,
    "n_threads": 1,
    "features": {"n_probes": 60, "correct_material": False},
    "train": {
        "hidden": [16],
        "batch_size": 16,
        "weight_decay": 0.01,
        "max_epochs": {"plain": 4, "masked": 6},
        "eval_every": 2,
        "min_fraction": 0.01,
        "schedule": "cosine",
    },
}


def has_torch():
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def make_data(n_classes=6, per_class=20, n_probes=300, seed=0):
    """Samples with class signal, some missing values, two materials, 5 outer folds."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, size=(n_classes, n_probes))
    y = np.repeat([f"C{i}" for i in range(n_classes)], per_class).astype(object)
    X = np.clip(
        centers[np.repeat(np.arange(n_classes), per_class)]
        + rng.normal(0, 0.08, size=(y.size, n_probes)),
        0,
        1,
    ).astype(np.float32)
    X[rng.random(X.shape) < 0.01] = np.nan
    ids = np.array([f"S{i:03d}" for i in range(y.size)], dtype=object)
    outer = np.tile(np.arange(5), y.size // 5 + 1)[: y.size]  # every class in every fold
    samples = pd.DataFrame(
        {
            "geo_accession": ids,
            "mc_class": y,
            "material": np.where(np.arange(y.size) % 3 == 0, "Frozen", "FFPE"),
            "outer_fold": outer.astype(str),
        }
    )
    for k in range(5):
        samples[f"inner_fold_o{k}"] = np.where(outer == k, -1, np.arange(y.size) % 3).astype(str)
    probe_ids = np.array([f"cg{i:05d}" for i in range(n_probes)], dtype=object)
    row_of = {s: i for i, s in enumerate(ids)}
    load = lambda want: X[[row_of[s] for s in want]]
    return samples, load, probe_ids, X


def make_runs(cv_root: Path, with_nn: bool):
    (cv_root / "rf_t").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "setting": "s000",
                "correct_material": True,
                "n_probes": 80,
                "model_params": json.dumps(RF_PARAMS),
            }
        ]
    ).to_csv(cv_root / "rf_t" / "settings.tsv", sep="\t", index=False)
    if with_nn:
        (cv_root / "nn_t").mkdir(parents=True)
        pd.DataFrame(
            {
                "setting": ["s000", "s001"],
                "dropout": [0.2, 0.2],
                "lr": [0.01, 0.01],
                "masked": [False, True],
            }
        ).to_csv(cv_root / "nn_t" / "settings.tsv", sep="\t", index=False)


def make_cfg(with_nn: bool):
    cfg = {
        "run_name": "final_t",
        "models": {
            "rf": {"kind": "tree", "run": "rf_t", "setting": "s000"},
            "centroid": {
                "kind": "centroid",
                "features": {"n_probes": 60, "correct_material": False},
            },
        },
        "calibration": {"rf": {"transform": "raw", "C": 1.0}},
        "analyses": {"sparsity": {"levels": LEVELS, "mask_seed": 11}},
    }
    model_cfgs = {"rf": RF_CFG}
    if with_nn:
        cfg["models"]["nn_masked"] = {
            "kind": "nn",
            "run": "nn_t",
            "network": "masked",
            "setting": "s001",
            "epoch": 5,
        }
        model_cfgs["nn_masked"] = NN_CFG
    return cfg, model_cfgs


def do_run(d: Path, samples, load, probe_ids, cfg, model_cfgs, stage="all"):
    return ff.run(
        stage,
        cfg,
        model_cfgs,
        samples,
        load,
        probe_ids,
        FeaturePipeline,
        pred_root=d / "pred",
        res_root=d / "res",
        model_root=d / "models",
        cv_res_root=d / "cv",
    )


def split(X, samples, k=0):
    ids, y, mat, outer = cv.columns(samples)
    tr = outer != k
    pipe = FeaturePipeline(n_probes=60).fit(X[tr], y[tr], mat[tr])
    return pipe.transform(X[tr], mat[tr]), y[tr], pipe.transform(X[~tr], mat[~tr]), ids[~tr]


# ---- fit + predict here must equal the one-step functions used in Phase 4
def test_tree_matches_sparsity_outer():
    samples, _, probe_ids, X = make_data()
    Z_tr, y_tr, Z_te, ids_te = split(X, samples)
    classes = sorted(set(y_tr))
    u = observed_uniform(ids_te, 300, np.arange(60), 11)
    old = so.predict_tree(
        {"run": "rf_t"},
        RF_CFG,
        {"model_params": json.dumps(RF_PARAMS)},
        Z_tr,
        y_tr,
        Z_te,
        u,
        LEVELS,
        classes,
    )
    art = {
        "kind": "tree",
        "classes": classes,
        "fit": ff.fit_tree(RF_CFG, RF_PARAMS, Z_tr, y_tr, classes),
    }
    assert np.array_equal(old, ff.predict_levels(art, Z_te, u, LEVELS))


def test_centroid_matches_sparsity_outer():
    samples, _, probe_ids, X = make_data()
    Z_tr, y_tr, Z_te, ids_te = split(X, samples)
    classes = sorted(set(y_tr))
    u = observed_uniform(ids_te, 300, np.arange(60), 11)
    old = so.predict_centroid(Z_tr, y_tr, Z_te, u, LEVELS, classes)
    art = {"kind": "centroid", "classes": classes, "fit": ff.fit_centroid(Z_tr, y_tr, classes)}
    assert np.allclose(old, ff.predict_levels(art, Z_te, u, LEVELS), atol=1e-12)


def test_nn_matches_sparsity_outer():
    if not has_torch():
        print("  (skipped: PyTorch not installed)")
        return
    samples, _, probe_ids, X = make_data()
    Z_tr, y_tr, Z_te, ids_te = split(X, samples)
    classes = sorted(set(y_tr))
    u = observed_uniform(ids_te, 300, np.arange(60), 11)
    row = {"masked": True, "dropout": 0.2, "lr": 0.01, "epoch": 5}
    old = so.predict_nn(NN_CFG, row, Z_tr, y_tr, Z_te, u, LEVELS, classes)
    s = {"masked": True, "dropout": 0.2, "lr": 0.01, "epoch": 5, "max_epochs": 6}
    art = {"kind": "nn", "classes": classes, "fit": ff.fit_nn(NN_CFG, s, Z_tr, y_tr, classes)}
    assert np.allclose(old, ff.predict_levels(art, Z_te, u, LEVELS), atol=1e-6)


# ---- the whole run
def test_run_writes_models_that_reload_and_is_resumable():
    samples, load, probe_ids, X = make_data()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cfg, model_cfgs = make_cfg(has_torch())
        make_runs(d / "cv", has_torch())
        n = do_run(d, samples, load, probe_ids, cfg, model_cfgs)
        assert n == 5 + len(cfg["models"])  # 5 oof fits, then the models
        ids, y, mat, outer = cv.columns(samples)
        P, y_oof, classes = ff.pooled_oof(d / "pred" / "final_t", "rf", samples)
        assert P.shape == (len(ids), 6) and list(y_oof) == list(y)

        summary = pd.read_csv(d / "res" / "final_t" / "fit_summary.tsv", sep="\t")
        assert set(summary["model"]) == set(cfg["models"])
        assert (summary["reload_max_abs_diff"] <= 1e-5).all()
        trained = summary[summary["kind"] != "nn"]  # the toy network trains 5 epochs
        assert (trained["train_accuracy"] > 0.9).all()

        art = ff.load_model(ff.model_path(d / "models" / "final_t", "rf"))
        assert art["n_train"] == len(ids) and art["probes_sha256"] == ff.probes_digest(probe_ids)
        Z = art["pipe"].transform(X[:10], mat[:10])
        raw = ff.predict_levels(art, Z, np.zeros(Z.shape, dtype=np.float32), [1.0])[0]
        calibrated = ff.calibrated(art, raw)
        assert calibrated.shape == raw.shape and np.allclose(calibrated.sum(axis=1), 1.0)
        centroid = ff.load_model(ff.model_path(d / "models" / "final_t", "centroid"))
        assert centroid["calibrator"] is None

        assert do_run(d, samples, load, probe_ids, cfg, model_cfgs) == 0  # nothing refits


def test_a_changed_config_is_refused():
    samples, load, probe_ids, _ = make_data()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cfg, model_cfgs = make_cfg(False)
        make_runs(d / "cv", False)
        do_run(d, samples, load, probe_ids, cfg, model_cfgs, stage="oof")
        changed = copy.deepcopy(cfg)
        changed["calibration"]["rf"]["C"] = 10.0
        try:
            do_run(d, samples, load, probe_ids, changed, model_cfgs, stage="fit")
        except SystemExit as e:
            assert "different settings" in str(e)
        else:
            raise AssertionError("expected a stop")


def test_out_of_fold_scores_ignore_the_held_out_labels():
    """Poison test: shuffling the labels of fold 0 must not change fold 0's scores."""
    samples, load, probe_ids, _ = make_data()
    poisoned = samples.copy()
    in0 = (poisoned["outer_fold"] == "0").to_numpy()
    poisoned.loc[in0, "mc_class"] = poisoned.loc[in0, "mc_class"].to_numpy()[::-1]
    assert (poisoned["mc_class"] != samples["mc_class"]).any()
    out = []
    for table in (samples, poisoned):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cfg, model_cfgs = make_cfg(False)
            make_runs(d / "cv", False)
            do_run(d, table, load, probe_ids, cfg, model_cfgs, stage="oof")
            out.append(cv.load_pred(ff.oof_path(d / "pred" / "final_t", "rf", 0))[0])
    assert np.array_equal(out[0], out[1])


def test_oof_is_compared_with_a_saved_phase3_file():
    samples, load, probe_ids, X = make_data()
    ids, y, mat, outer = cv.columns(samples)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cfg, model_cfgs = make_cfg(False)
        make_runs(d / "cv", False)
        tr = outer != 2  # a "Phase 3" file for fold 2
        pipe = FeaturePipeline(n_probes=80, correct_material=True).fit(X[tr], y[tr], mat[tr])
        model = cv.make_model("rf", RF_PARAMS, 7, 2).fit(pipe.transform(X[tr], mat[tr]), y[tr])
        (d / "pred" / "rf_t").mkdir(parents=True)
        cv.save_pred(
            cv.pred_path(d / "pred" / "rf_t", 2, None, "s000"),
            model.predict_proba(pipe.transform(X[~tr], mat[~tr])),
            ids[~tr],
            sorted(set(y)),
        )
        do_run(d, samples, load, probe_ids, cfg, model_cfgs, stage="oof")
        chk = pd.read_csv(d / "res" / "final_t" / "repro_check.tsv", sep="\t")
        assert list(chk["outer"]) == [2] and chk["max_abs_diff"].iloc[0] < 1e-6  # float32 on disk
        assert chk["n_top_class_changed"].iloc[0] == 0


def test_bad_settings_stop():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        make_runs(d / "cv", True)
        for m, mcfg, text in (
            ({"kind": "tree", "run": "rf_t", "setting": "s009"}, RF_CFG, "not found"),
            (
                {"kind": "nn", "run": "nn_t", "network": "plain", "setting": "s001", "epoch": 3},
                NN_CFG,
                "network says",
            ),
            (
                {"kind": "nn", "run": "nn_t", "network": "masked", "setting": "s001", "epoch": 7},
                NN_CFG,
                "within 1..6",
            ),
        ):
            try:
                ff.final_setting("x", m, mcfg, d / "cv")
            except SystemExit as e:
                assert text in str(e), str(e)
            else:
                raise AssertionError(f"expected a stop mentioning {text!r}")


def test_the_command_line_refuses_another_cohort():
    try:
        ff.main(
            [str(ROOT / "configs" / "final_v1.yaml"), "--store", "data/betas/zarr/GSE109379.zarr"]
        )
    except SystemExit as e:
        assert "training cohort" in str(e)
    else:
        raise AssertionError("expected a stop")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
