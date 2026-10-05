#!/usr/bin/env python
"""Fit the final models on all of GSE90496 and save them to disk.

Cross-validation estimated how well each recipe works. The models that score
the validation cohort are those same recipes fit once more, on every training
sample. Nothing is chosen here: every setting comes from configs/final_v1.yaml,
which scripts/check_final_config.py ties to inner-fold results.

Two stages:

  oof   Tree models only. One 5-fold pass over the existing outer folds with
        the final setting. Each training sample gets one score vector from a
        model that did not train on it ("out-of-fold"). The calibrator needs
        these: on its own training samples a forest is overconfident.
  fit   Every model: fit the feature pipeline and the model on all samples,
        fit the calibrator on the out-of-fold scores, save one file per model.
        The saved file is reloaded and must reproduce the in-memory predictions.

This script reads GSE90496 only. It has no argument for the validation cohort.

  python scripts/fit_final.py configs/final_v1.yaml

Outputs:
  data/models/final_v1/<model>.joblib            pipeline + model (+ calibrator)
  data/predictions/final_v1/oof_<model>_o<k>.npz out-of-fold scores (tree models)
  results/final_v1/manifest.json, fits.tsv, fit_summary.tsv, repro_check.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_inner as cal  # noqa: E402
import run_nested_cv as cv  # noqa: E402
import sparsity_outer as so  # noqa: E402
import train_nn as tnn  # noqa: E402

KINDS = ("tree", "nn", "centroid")


# --------------------------------------------------------------------------
# Settings: what the config names, looked up in each run's own tables
# --------------------------------------------------------------------------
def load_model_cfgs(cfg) -> dict:
    import yaml

    out = {}
    for name, m in cfg["models"].items():
        if m.get("kind") not in KINDS:
            cv.fail(f"config: models.{name}.kind must be one of {KINDS}")
        if m["kind"] == "centroid":
            continue
        if not Path(m.get("config", "")).is_file():
            cv.fail(f"models.{name}.config: {m.get('config')} not found")
        out[name] = yaml.safe_load(Path(m["config"]).read_text())
    return out


def final_setting(name, m, mcfg, res_root) -> dict:
    """Features and model settings of one final model."""
    if m["kind"] == "centroid":
        return {
            "n_probes": int(m["features"]["n_probes"]),
            "correct_material": bool(m["features"]["correct_material"]),
        }
    path = Path(res_root) / m["run"] / "settings.tsv"
    if not path.is_file():
        cv.fail(f"{path} not found")
    t = pd.read_csv(path, sep="\t", dtype={"setting": str})
    hit = t[t["setting"] == str(m["setting"])]
    if len(hit) != 1:
        cv.fail(f"{path}: setting {m['setting']} of model {name} not found")
    row = hit.iloc[0]
    if m["kind"] == "tree":
        return {
            "n_probes": int(row["n_probes"]),
            "correct_material": bool(row["correct_material"]),
            "params": json.loads(row["model_params"]),
        }
    masked = bool(row["masked"])
    if masked != (m["network"] == "masked"):
        cv.fail(
            f"models.{name}: setting {m['setting']} is "
            f"{'masked' if masked else 'plain'}, but network says {m['network']}"
        )
    limit = tnn.max_epochs_for(mcfg, masked)
    if not 1 <= int(m["epoch"]) <= limit:
        cv.fail(f"models.{name}.epoch must be within 1..{limit}")
    return {
        "n_probes": int(mcfg["features"]["n_probes"]),
        "correct_material": bool(mcfg["features"]["correct_material"]),
        "masked": masked,
        "lr": float(row["lr"]),
        "dropout": float(row["dropout"]),
        "epoch": int(m["epoch"]),
        "max_epochs": limit,
    }


def probes_digest(probe_ids) -> str:
    """Fingerprint of the array's probe list and order."""
    return hashlib.sha256("\n".join(map(str, probe_ids)).encode()).hexdigest()


# --------------------------------------------------------------------------
# Fit and predict, one pair per kind. These mirror scripts/sparsity_outer.py
# (predict_tree, predict_centroid, predict_nn) split into "fit" and "predict",
# so a model can be saved in between. tests/test_fit_final.py checks that the
# two give identical numbers.
# --------------------------------------------------------------------------
def fit_tree(mcfg, params, Z, y, classes) -> dict:
    model = cv.make_model(mcfg["model"], params, mcfg["seed"], mcfg["n_jobs"])
    model.fit(Z, y)
    if list(model.classes_) != list(classes):
        cv.fail("the training samples do not hold every class")
    return {"model": model, "median": np.median(Z, axis=0).astype(np.float32)}


def fit_centroid(Z, y, classes) -> dict:
    y = np.asarray(y, dtype=object)
    C = np.vstack([Z[y == c].mean(axis=0, dtype=np.float64) for c in classes])
    resid = np.zeros(Z.shape[1], dtype=np.float64)
    for i, c in enumerate(classes):
        d = Z[y == c].astype(np.float64) - C[i]
        resid += (d * d).sum(axis=0)
    var = resid / max(len(y) - len(classes), 1) + so.VAR_FLOOR
    return {"A": (C / var).T, "B": (C * C / var).T}


def fit_nn(mcfg, s, Z, y, classes) -> dict:
    from methylclf.nn import Standardizer, train_mlp

    scaler = Standardizer().fit(Z)
    index = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([index[c] for c in y], dtype=np.int64)
    t = mcfg["train"]
    net = train_mlp(
        scaler.transform(Z),
        y_idx,
        len(classes),
        masked=s["masked"],
        hidden=tuple(t["hidden"]),
        dropout=s["dropout"],
        lr=s["lr"],
        weight_decay=float(t["weight_decay"]),
        batch_size=int(t["batch_size"]),
        max_epochs=s["max_epochs"],
        stop_epoch=s["epoch"],
        eval_every=10**9,
        min_fraction=float(t["min_fraction"]),
        seed=int(mcfg["seed"]),
        n_threads=int(mcfg["n_threads"]),
        schedule=str(t.get("schedule", "constant")),
    )
    # weights as plain arrays, so the saved file does not depend on how torch pickles
    state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    return {
        "mean": scaler.mean_,
        "sd": scaler.sd_,
        "state": state,
        "hidden": tuple(t["hidden"]),
        "dropout": s["dropout"],
        "n_threads": int(mcfg["n_threads"]),
    }


def predict_levels(art: dict, Z, u, levels) -> np.ndarray:
    """Raw scores at each coverage level, shaped (level, sample, class)."""
    from methylclf.masking import observed

    kind, fit = art["kind"], art["fit"]
    if kind == "tree":
        out = [
            fit["model"].predict_proba(
                np.where(observed(u, lv), Z, fit["median"][None, :]).astype(np.float32)
            )
            for lv in levels
        ]
    elif kind == "centroid":
        X, out = Z.astype(np.float64), []
        for lv in levels:
            O = observed(u, lv).astype(np.float64)
            dist = -2.0 * (X * O) @ fit["A"] + O @ fit["B"]
            n_obs = np.maximum(O.sum(axis=1, keepdims=True), 1.0)
            e = np.exp(-0.5 * (dist - dist.min(axis=1, keepdims=True)) / n_obs)
            out.append(e / e.sum(axis=1, keepdims=True))
    elif kind == "nn":
        import torch
        from methylclf.nn import build_mlp, predict_proba

        torch.set_num_threads(fit["n_threads"])
        net = build_mlp(Z.shape[1], len(art["classes"]), fit["hidden"], fit["dropout"])
        net.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in fit["state"].items()})
        Zs = ((Z - fit["mean"]) / fit["sd"]).astype(np.float32)
        out = [predict_proba(net, Zs, observed(u, lv)) for lv in levels]
    else:
        cv.fail(f"unknown model kind '{kind}'")
    return np.stack(out)


def calibrated(art: dict, raw) -> np.ndarray:
    """Calibrated scores from full-coverage raw scores (tree models only)."""
    c = art.get("calibrator")
    if c is None:
        cv.fail(f"model {art['name']} has no calibrator")
    return cal.apply_calibrator(c["model"], raw, c["transform"])


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
def model_path(model_dir: Path, name: str) -> Path:
    return Path(model_dir) / f"{name}.joblib"


def save_model(path: Path, art: dict):
    import joblib

    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(art, tmp, compress=3)
    os.replace(tmp, path)  # a file that exists is whole


def load_model(path: Path) -> dict:
    import joblib

    if not Path(path).is_file():
        cv.fail(f"{path} not found; run scripts/fit_final.py first")
    return joblib.load(path)


def oof_path(pred_dir: Path, name: str, k: int) -> Path:
    return Path(pred_dir) / f"oof_{name}_o{k}.npz"


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_oof(
    cfg, model_cfgs, settings, samples, X, pipeline_factory, pred_dir, res_dir, pred_root
):
    """Out-of-fold raw scores of each tree model's final setting."""
    ids, y, mat, outer = cv.columns(samples)
    classes = sorted(set(y))
    trees = [n for n, m in cfg["models"].items() if m["kind"] == "tree"]
    checks, n_done = [], 0
    for k in range(cv.N_OUTER):
        todo = [n for n in trees if not oof_path(pred_dir, n, k).exists()]
        if not todo:
            continue
        train = outer != k
        X_tr, X_te = X[train], X[~train]
        for name in todo:
            s, m, t0 = settings[name], cfg["models"][name], time.time()
            pipe = pipeline_factory(n_probes=s["n_probes"], correct_material=s["correct_material"])
            pipe.fit(X_tr, y[train], mat[train])
            Z_tr, Z_te = pipe.transform(X_tr, mat[train]), pipe.transform(X_te, mat[~train])
            model = fit_tree(model_cfgs[name], s["params"], Z_tr, y[train], classes)["model"]
            proba = model.predict_proba(Z_te)
            cv.save_pred(oof_path(pred_dir, name, k), proba, ids[~train], classes)
            # same setting, same fold, same seed as a saved Phase 3 file: must agree
            old = cv.pred_path(Path(pred_root) / m["run"], k, None, str(m["setting"]))
            if old.exists():
                P, old_ids, _ = cv.load_pred(old)
                same = list(old_ids) == list(ids[~train])
                checks.append(
                    {
                        "model": name,
                        "outer": k,
                        "phase3_file": old.name,
                        "max_abs_diff": float(np.abs(P - proba).max()) if same else np.nan,
                        "n_top_class_changed": (
                            int((P.argmax(1) != proba.argmax(1)).sum()) if same else -1
                        ),
                    }
                )
            n_done += 1
            print(f"  oof outer {k} {name}: {time.time() - t0:.0f} s", flush=True)
            del pipe, Z_tr, Z_te
        del X_tr, X_te
    if checks:
        path = res_dir / "repro_check.tsv"
        t = pd.DataFrame(checks)
        if path.exists():
            t = pd.concat([pd.read_csv(path, sep="\t"), t]).drop_duplicates(
                ["model", "outer"], keep="last"
            )
        t.sort_values(["model", "outer"]).to_csv(path, sep="\t", index=False)
    print(f"oof stage: {n_done} fits done now")
    return n_done


def pooled_oof(pred_dir, name, samples):
    """One out-of-fold score vector per training sample, in fold-table order."""
    ids, y, _, outer = cv.columns(samples)
    by_id, classes = {}, None
    for k in range(cv.N_OUTER):
        path = oof_path(pred_dir, name, k)
        if not path.exists():
            cv.fail(f"{path} not found; run --stage oof first")
        P, pid, cls = cv.load_pred(path)
        if set(pid) != set(ids[outer == k]) or len(pid) != int((outer == k).sum()):
            cv.fail(f"{path.name}: samples are not exactly outer fold {k}")
        if classes is not None and list(cls) != classes:
            cv.fail(f"{path.name}: class order differs between folds")
        classes = list(cls)
        by_id.update(zip(pid, P.astype(np.float64)))
    return np.vstack([by_id[i] for i in ids]), y, classes


def stage_fit(
    cfg,
    model_cfgs,
    settings,
    samples,
    X,
    probe_ids,
    pipeline_factory,
    pred_dir,
    model_dir,
    res_dir,
    only=None,
):
    from methylclf.masking import observed_uniform, probe_positions

    ids, y, mat, _ = cv.columns(samples)
    classes = sorted(set(y))
    digest = probes_digest(probe_ids)
    levels = [float(v) for v in cfg["analyses"]["sparsity"]["levels"]]
    seed = int(cfg["analyses"]["sparsity"]["mask_seed"])
    cache, rows, n_done = {}, [], 0
    for name, m in cfg["models"].items():
        if only and name not in only:
            continue
        path = model_path(model_dir, name)
        if path.exists():
            print(f"{name}: already fitted", flush=True)
            continue
        s, t0 = settings[name], time.time()
        key = (s["n_probes"], s["correct_material"])
        if key not in cache:
            pipe = pipeline_factory(n_probes=key[0], correct_material=key[1])
            pipe.fit(X, y, mat)
            cache[key] = (pipe, pipe.transform(X, mat), pipe.get_feature_names_out(probe_ids))
        pipe, Z, names = cache[key]
        if np.isnan(Z).any():
            cv.fail(f"{name}: NaN after the feature pipeline")
        t_feat, t1 = time.time() - t0, time.time()
        if m["kind"] == "tree":
            fit = fit_tree(model_cfgs[name], s["params"], Z, y, classes)
        elif m["kind"] == "centroid":
            fit = fit_centroid(Z, y, classes)
        else:
            fit = fit_nn(model_cfgs[name], s, Z, y, classes)
        art = {
            "name": name,
            "kind": m["kind"],
            "run_name": cfg["run_name"],
            "settings": s,
            "classes": classes,
            "pipe": pipe,
            "feature_names": np.asarray(names, dtype=object),
            "n_array_probes": len(probe_ids),
            "probes_sha256": digest,
            "n_train": len(ids),
            "fit": fit,
            "calibrator": None,
        }
        row = {
            "model": name,
            "kind": m["kind"],
            "n_train": len(ids),
            "n_features": Z.shape[1],
            "correct_material": key[1],
        }
        if m["kind"] == "tree":
            P, y_oof, oof_classes = pooled_oof(pred_dir, name, samples)
            if oof_classes != classes:
                cv.fail(f"{name}: out-of-fold class order differs from the final model")
            c = cfg["calibration"][name]
            cmodel = cal.fit_calibrator(P, y_oof, c["transform"], float(c["C"]), classes)
            art["calibrator"] = {
                "model": cmodel,
                "transform": c["transform"],
                "C": float(c["C"]),
                "n_fit": len(y_oof),
            }
            pred = np.asarray(classes, dtype=object)[P.argmax(1)]
            row.update(
                oof_accuracy_raw=float((pred == y_oof).mean()),
                calibrator=f"{c['transform']} C={float(c['C']):g}",
                calibrator_converged=bool(cmodel.converged_),
            )
        save_model(path, art)

        # the saved file must give the same numbers as the model in memory
        n_chk = min(64, len(ids))
        u = observed_uniform(ids[:n_chk], len(probe_ids), probe_positions(names, probe_ids), seed)
        before = predict_levels(art, Z[:n_chk], u, levels)
        back = load_model(path)
        Z_back = back["pipe"].transform(X[:n_chk], mat[:n_chk])
        after = predict_levels(back, Z_back, u, levels)
        diff = float(np.abs(before - after).max())
        if diff > 1e-5:
            cv.fail(f"{name}: the reloaded model differs from the fitted one (max {diff:.2e})")
        # agreement with the labels it was trained on: a sanity number, not a result
        full = predict_levels(art, Z, np.zeros(Z.shape, dtype=np.float32), [1.0])[0]
        row.update(
            reload_max_abs_diff=diff,
            train_accuracy=float((np.asarray(classes, dtype=object)[full.argmax(1)] == y).mean()),
            seconds_features=round(t_feat, 1),
            seconds_model=round(time.time() - t1, 1),
            file_mb=round(path.stat().st_size / 1e6, 1),
        )
        rows.append(row)
        cv.log_fit(
            res_dir,
            {
                "stage": "final",
                "outer": "all",
                "inner": "all",
                "setting": name,
                "n_train": len(ids),
                "n_test": 0,
                "n_features": Z.shape[1],
                "seconds_features": t_feat,
                "seconds_model": time.time() - t1,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        n_done += 1
        print(
            f"{name}: fitted on {len(ids)} samples x {Z.shape[1]} probes in "
            f"{time.time() - t0:.0f} s; reload check {diff:.1e}",
            flush=True,
        )
    if rows:
        path = res_dir / "fit_summary.tsv"
        t = pd.DataFrame(rows)
        if path.exists():
            t = pd.concat([pd.read_csv(path, sep="\t"), t]).drop_duplicates("model", keep="last")
        t.to_csv(path, sep="\t", index=False, float_format="%.6g")
    print(f"fit stage: {n_done} models fitted now")
    return n_done


def run(
    stage,
    cfg,
    model_cfgs,
    samples,
    load,
    probe_ids,
    pipeline_factory,
    pred_root="data/predictions",
    res_root="results",
    model_root="data/models",
    cv_res_root="results/cv",
    only=None,
):
    """Everything the command line does, on any loader (the tests pass a fake one)."""
    for key in ("run_name", "models", "calibration", "analyses"):
        if key not in cfg:
            cv.fail(f"config: missing '{key}'")
    if stage not in ("oof", "fit", "all"):
        cv.fail(f"unknown stage '{stage}'")
    if only:
        unknown = sorted(set(only) - set(cfg["models"]))
        if unknown:
            cv.fail(f"unknown models {unknown}; known: {list(cfg['models'])}")
    pred_dir = Path(pred_root) / cfg["run_name"]
    res_dir = Path(res_root) / cfg["run_name"]
    model_dir = Path(model_root) / cfg["run_name"]
    settings = {
        n: final_setting(n, m, model_cfgs.get(n), cv_res_root) for n, m in cfg["models"].items()
    }
    cv.check_manifest({"final": cfg, "models": model_cfgs}, samples, res_dir, pd.DataFrame())
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    ids = cv.columns(samples)[0]
    t0 = time.time()
    X = load(ids)  # every training sample, once
    print(
        f"loaded {X.shape[0]} x {X.shape[1]} training samples in {time.time() - t0:.0f} s",
        flush=True,
    )
    n = 0
    if stage in ("oof", "all"):
        n += stage_oof(
            cfg, model_cfgs, settings, samples, X, pipeline_factory, pred_dir, res_dir, pred_root
        )
    if stage in ("fit", "all"):
        n += stage_fit(
            cfg,
            model_cfgs,
            settings,
            samples,
            X,
            probe_ids,
            pipeline_factory,
            pred_dir,
            model_dir,
            res_dir,
            only,
        )
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--stage", default="all", choices=["oof", "fit", "all"])
    ap.add_argument("--models", nargs="+", default=None, help="limit the fit stage to these")
    ap.add_argument("--store", default="data/betas/zarr/GSE90496.zarr")
    ap.add_argument("--folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--probes", default="results/probes/probes_kept.tsv")
    ap.add_argument("--pred-root", default="data/predictions")
    ap.add_argument("--res-root", default="results")
    ap.add_argument("--model-root", default="data/models")
    ap.add_argument("--cv-res-root", default="results/cv")
    args = ap.parse_args(argv)

    import yaml
    from methylclf.data import BetaStore
    from methylclf.features import FeaturePipeline

    if not Path(args.config).is_file():
        cv.fail(f"{args.config}: config file not found")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if "GSE90496" not in Path(args.store).name:
        cv.fail("fit_final.py reads the training cohort (GSE90496) only")
    st = BetaStore.open(args.store, args.folds, args.probes)
    run(
        args.stage,
        cfg,
        load_model_cfgs(cfg),
        st.samples,
        st.load,
        np.asarray(st.probe_ids, dtype=object),
        FeaturePipeline,
        args.pred_root,
        args.res_root,
        args.model_root,
        args.cv_res_root,
        args.models,
    )
    res_dir = Path(args.res_root) / cfg["run_name"]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for name, title in (
        ("repro_check.tsv", "out-of-fold scores against the Phase 3 files"),
        ("fit_summary.tsv", "final models"),
    ):
        if (res_dir / name).exists():
            print(f"\n== {title} ==")
            print(pd.read_csv(res_dir / name, sep="\t").to_string(index=False))


if __name__ == "__main__":
    main()
