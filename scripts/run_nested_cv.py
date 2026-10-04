#!/usr/bin/env python
"""Resumable nested cross-validation: fit, predict, save probabilities.

Three stages, run separately:

  inner   For every outer fold k, inner fold j and setting: fit the feature
          pipeline and the model on the inner training samples, save predicted
          probabilities for the inner held-out samples. Outer test fold k is
          never loaded in this stage.
  select  Score the saved inner predictions and pick, per outer fold, the
          setting with the best mean inner score. Writes inner_scores.tsv and
          selected.tsv. Fits nothing.
  outer   FINAL SCORING. For every outer fold k: refit the selected setting on
          the whole training side and save probabilities for outer test fold k.
          Refused unless --final-scoring is given.

Only probabilities are saved by the fitting stages; metrics, calibration and
family scores are computed afterwards from the saved files. Each fit writes one
file, renamed into place when complete, so a crash costs one fit: rerun the same
command and finished fits are skipped.

  python scripts/run_nested_cv.py configs/rf_v1.yaml --stage inner
  python scripts/run_nested_cv.py configs/rf_v1.yaml --stage select

Outputs:
  data/predictions/<run_name>/*.npz     probabilities (git-ignored)
  results/cv/<run_name>/manifest.json   the settings this run was started with
  results/cv/<run_name>/settings.tsv    one row per setting
  results/cv/<run_name>/fits.tsv        one row per finished fit, with timings
  results/cv/<run_name>/inner_scores.tsv, selected.tsv   (stage select)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

N_OUTER, N_INNER = 5, 3
SCORE_COLUMNS = ["accuracy", "macro_f1", "balanced_accuracy", "log_loss"]


def fail(msg: str):
    sys.exit(f"ERROR: {msg}")


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def expand_settings(cfg: dict) -> pd.DataFrame:
    """Every combination of the feature and model values in the config."""
    for key in ("run_name", "model", "seed", "n_jobs", "select_by", "features", "model_params"):
        if key not in cfg:
            fail(f"config: missing '{key}'")
    if cfg["select_by"] not in SCORE_COLUMNS:
        fail(f"config: select_by must be one of {SCORE_COLUMNS}")
    feats, params = cfg["features"], cfg["model_params"]
    if set(feats) != {"n_probes", "correct_material"}:
        fail("config: features must list exactly n_probes and correct_material")
    rows = []
    f_keys, p_keys = sorted(feats), sorted(params)
    for fv in itertools.product(*(feats[k] for k in f_keys)):
        for pv in itertools.product(*(params[k] for k in p_keys)):
            rows.append({"setting": f"s{len(rows):03d}", **dict(zip(f_keys, fv)),
                         "model_params": json.dumps(dict(zip(p_keys, pv)), sort_keys=True)})
    return pd.DataFrame(rows)


def make_model(name: str, params: dict, seed: int, n_jobs: int):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=n_jobs, **params)
    fail(f"unknown model '{name}' (known: rf, lgbm)")


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
def pred_path(pred_dir: Path, outer: int, inner, setting: str) -> Path:
    part = "test" if inner is None else f"i{inner}"
    return Path(pred_dir) / f"o{outer}_{part}_{setting}.npz"


def save_pred(path: Path, proba, sample_ids, classes):
    """Write under a temporary name, then rename: a file that exists is whole."""
    tmp = path.with_name(path.name[:-4] + ".tmp.npz")
    np.savez(tmp, proba=np.asarray(proba, dtype=np.float32),
             sample_ids=np.asarray(sample_ids, dtype=str),
             classes=np.asarray(classes, dtype=str))
    os.replace(tmp, path)


def load_pred(path: Path):
    with np.load(path) as z:
        return z["proba"], z["sample_ids"].astype(object), z["classes"].astype(object)


def check_manifest(cfg: dict, samples: pd.DataFrame, res_dir: Path, settings: pd.DataFrame):
    """Refuse to resume a run whose settings or fold table have changed."""
    fold_cols = ["geo_accession", "mc_class", "material", "outer_fold"] + \
                [f"inner_fold_o{k}" for k in range(N_OUTER)]
    digest = hashlib.sha256(
        samples[fold_cols].astype(str).to_csv(index=False).encode()).hexdigest()
    now = {"config": cfg, "folds_sha256": digest, "n_samples": int(len(samples))}
    path = res_dir / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != json.loads(json.dumps(now)):
            fail(f"{path}: this run was started with different settings or a different "
                 f"fold table. Use a new run_name, or delete the run's two folders "
                 f"to start it again.")
    else:
        res_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(now, indent=2) + "\n")
        if len(settings):
            settings.to_csv(res_dir / "settings.tsv", sep="\t", index=False)


def log_fit(res_dir: Path, row: dict):
    path = res_dir / "fits.tsv"
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a",
                               header=not path.exists(), float_format="%.1f")


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_group(X_tr, y_tr, m_tr, X_te, m_te, ids_te, todo, cfg, all_classes,
              pipeline_factory, res_dir, stage, outer, inner):
    """Fit every setting in `todo` (setting row, output path) for one split.

    Settings that share the same features share one pipeline fit.
    """
    done = 0
    key = lambda item: (int(item[0]["n_probes"]), bool(item[0]["correct_material"]))
    for (n_probes, correct), items in itertools.groupby(sorted(todo, key=key), key=key):
        t0 = time.time()
        pipe = pipeline_factory(n_probes=n_probes, correct_material=correct)
        pipe.fit(X_tr, y_tr, m_tr)
        Z_tr, Z_te = pipe.transform(X_tr, m_tr), pipe.transform(X_te, m_te)
        t_feat = time.time() - t0
        if np.isnan(Z_tr).any() or np.isnan(Z_te).any():
            fail(f"outer {outer} inner {inner}: NaN after the feature pipeline")
        for row, path in items:
            t1 = time.time()
            model = make_model(cfg["model"], json.loads(row["model_params"]),
                               cfg["seed"], cfg["n_jobs"])
            model.fit(Z_tr, y_tr)
            if list(model.classes_) != list(all_classes):
                fail(f"outer {outer} inner {inner}: the training samples hold "
                     f"{len(model.classes_)} classes, expected {len(all_classes)}")
            proba = model.predict_proba(Z_te)
            save_pred(path, proba, ids_te, all_classes)
            log_fit(res_dir, {"stage": stage, "outer": outer,
                              "inner": "test" if inner is None else inner,
                              "setting": row["setting"], "n_train": len(y_tr),
                              "n_test": len(ids_te), "n_features": Z_tr.shape[1],
                              "seconds_features": t_feat, "seconds_model": time.time() - t1,
                              "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
            done += 1
            print(f"  outer {outer} {'test' if inner is None else 'inner ' + str(inner)} "
                  f"{row['setting']} probes={n_probes} correct={int(correct)}: "
                  f"features {t_feat:.0f} s, model {time.time() - t1:.0f} s", flush=True)
        del pipe, Z_tr, Z_te
    return done


def columns(samples: pd.DataFrame):
    ids, y, mat = (samples[c].to_numpy(dtype=object)
                   for c in ("geo_accession", "mc_class", "material"))
    outer = samples["outer_fold"].astype(int).to_numpy()
    return ids, y, mat, outer


def stage_inner(load, samples, cfg, settings, pred_dir, res_dir, pipeline_factory,
                outer_folds):
    ids, y, mat, outer = columns(samples)
    all_classes = sorted(set(y))
    n_done = n_skipped = 0
    for k in outer_folds:
        inner = samples[f"inner_fold_o{k}"].astype(int).to_numpy()
        train = outer != k
        if not ((inner == -1) == ~train).all():
            fail(f"fold table: inner_fold_o{k} is not -1 exactly on outer test fold {k}")
        todo = {j: [(r, pred_path(pred_dir, k, j, r["setting"]))
                    for _, r in settings.iterrows()] for j in range(N_INNER)}
        total = sum(len(v) for v in todo.values())
        todo = {j: [(r, p) for r, p in v if not p.exists()] for j, v in todo.items()}
        left = sum(len(v) for v in todo.values())
        n_skipped += total - left
        if left == 0:
            print(f"outer {k}: all {total} inner fits already done", flush=True)
            continue
        t0 = time.time()
        X = load(ids[train])              # the training side only; test fold k is not read
        y_k, m_k, ids_k, inner_k = y[train], mat[train], ids[train], inner[train]
        print(f"outer {k}: loaded {X.shape[0]} x {X.shape[1]} training side in "
              f"{time.time() - t0:.0f} s; {left} of {total} fits to do", flush=True)
        for j in range(N_INNER):
            if not todo[j]:
                continue
            tr, te = np.flatnonzero(inner_k != j), np.flatnonzero(inner_k == j)
            X_tr, X_te = X[tr], X[te]
            n_done += fit_group(X_tr, y_k[tr], m_k[tr], X_te, m_k[te], ids_k[te], todo[j],
                                cfg, all_classes, pipeline_factory, res_dir, "inner", k, j)
            del X_tr, X_te
        del X
    print(f"inner stage: {n_done} fits done now, {n_skipped} already on disk")
    return n_done


def score_pred(y_true, proba, classes):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 log_loss)
    classes = np.asarray(classes, dtype=object)
    proba = proba.astype(np.float64)
    proba = proba / proba.sum(axis=1, keepdims=True)   # float32 rows sum to 1 +- 1e-7
    pred = classes[np.argmax(proba, axis=1)]
    present = sorted(set(y_true))        # macro average over classes with true samples
    return {"accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, labels=present, average="macro",
                                 zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "log_loss": log_loss(y_true, proba, labels=list(classes))}


def stage_select(samples, cfg, settings, pred_dir, res_dir):
    ids, y, mat, outer = columns(samples)
    truth = dict(zip(ids, y))
    rows, missing = [], []
    for k in range(N_OUTER):
        inner = dict(zip(ids, samples[f"inner_fold_o{k}"].astype(int)))
        for j in range(N_INNER):
            for _, r in settings.iterrows():
                path = pred_path(pred_dir, k, j, r["setting"])
                if not path.exists():
                    missing.append(path.name)
                    continue
                proba, sid, classes = load_pred(path)
                expected = {i for i in ids if inner[i] == j}
                if set(sid) != expected or len(sid) != len(expected):
                    fail(f"{path.name}: samples are not inner fold {j} of outer fold {k}")
                s = score_pred(np.array([truth[i] for i in sid], dtype=object), proba, classes)
                rows.append({"outer": k, "inner": j, "setting": r["setting"],
                             "n": len(sid), **s})
    if missing:
        fail(f"select: {len(missing)} inner fits are missing (first: {missing[0]}); "
             f"finish --stage inner first")
    scores = pd.DataFrame(rows)
    scores.to_csv(res_dir / "inner_scores.tsv", sep="\t", index=False, float_format="%.5f")
    by = cfg["select_by"]
    mean = (scores.groupby(["outer", "setting"], as_index=False)[SCORE_COLUMNS].mean()
            .merge(settings, on="setting"))
    better = -1 if by == "log_loss" else 1          # log loss: lower is better
    mean["_key"] = better * mean[by]
    # best score; ties go to the earlier setting (stable sort on the setting id)
    best = (mean.sort_values(["outer", "_key", "setting"], ascending=[True, False, True])
            .groupby("outer", as_index=False).head(1).drop(columns="_key"))
    best.insert(1, "selected_by", by)
    best.to_csv(res_dir / "selected.tsv", sep="\t", index=False, float_format="%.5f")
    return scores, mean.drop(columns="_key"), best


def stage_outer(load, samples, cfg, settings, pred_dir, res_dir, pipeline_factory,
                outer_folds):
    path = res_dir / "selected.tsv"
    if not path.exists():
        fail(f"outer: {path} not found; run --stage select first")
    chosen = pd.read_csv(path, sep="\t", dtype={"setting": str}).set_index("outer")["setting"]
    by_id = settings.set_index("setting", drop=False)
    ids, y, mat, outer = columns(samples)
    all_classes = sorted(set(y))
    n_done = 0
    for k in outer_folds:
        if k not in chosen.index or chosen[k] not in by_id.index:
            fail(f"outer: no valid selected setting for outer fold {k} in {path}")
        row = by_id.loc[chosen[k]]
        out = pred_path(pred_dir, k, None, row["setting"])
        if out.exists():
            print(f"outer {k}: already done", flush=True)
            continue
        train = outer != k
        X_tr, X_te = load(ids[train]), load(ids[~train])
        n_done += fit_group(X_tr, y[train], mat[train], X_te, mat[~train], ids[~train],
                            [(row, out)], cfg, all_classes, pipeline_factory, res_dir,
                            "outer", k, None)
        del X_tr, X_te
    print(f"outer stage: {n_done} fits done now")
    return n_done


# --------------------------------------------------------------------------
def run(stage, cfg, samples, load, pipeline_factory, pred_root, res_root,
        outer_folds=None, final_scoring=False):
    """Everything the command line does, on any loader (the tests pass a fake one)."""
    pred_dir = Path(pred_root) / cfg["run_name"]
    res_dir = Path(res_root) / cfg["run_name"]
    if stage == "outer":
        # the settings table written by the search (grid or Optuna), not the config
        for name in ("manifest.json", "settings.tsv", "selected.tsv"):
            if not (res_dir / name).exists():
                fail(f"outer: {res_dir / name} not found; finish the inner search and "
                     f"select first")
        settings = pd.read_csv(res_dir / "settings.tsv", sep="\t", dtype={"setting": str})
    else:
        settings = expand_settings(cfg)
    outer_folds = list(range(N_OUTER)) if outer_folds is None else list(outer_folds)
    if any(k not in range(N_OUTER) for k in outer_folds):
        fail(f"outer folds must be within 0..{N_OUTER - 1}")
    check_manifest(cfg, samples, res_dir, settings)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if stage == "inner":
        return stage_inner(load, samples, cfg, settings, pred_dir, res_dir,
                           pipeline_factory, outer_folds)
    if stage == "select":
        return stage_select(samples, cfg, settings, pred_dir, res_dir)
    if stage == "outer":
        if not final_scoring:
            fail("outer: this stage scores the outer test folds, which is done once, "
                 "at the end of Phase 3. Add --final-scoring to confirm.")
        return stage_outer(load, samples, cfg, settings, pred_dir, res_dir,
                           pipeline_factory, outer_folds)
    fail(f"unknown stage '{stage}'")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--stage", required=True, choices=["inner", "select", "outer"])
    ap.add_argument("--outer-folds", type=int, nargs="+", default=None,
                    help="limit the inner or outer stage to these outer folds")
    ap.add_argument("--final-scoring", action="store_true")
    ap.add_argument("--store", default="data/betas/zarr/GSE90496.zarr")
    ap.add_argument("--folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--probes", default="results/probes/probes_kept.tsv")
    ap.add_argument("--pred-root", default="data/predictions")
    ap.add_argument("--res-root", default="results/cv")
    args = ap.parse_args(argv)

    import yaml
    from methylclf.data import BetaStore
    from methylclf.features import FeaturePipeline

    if not Path(args.config).exists():
        fail(f"{args.config}: config file not found")
    cfg = yaml.safe_load(Path(args.config).read_text())
    st = BetaStore.open(args.store, args.folds, args.probes)
    out = run(args.stage, cfg, st.samples, st.load, FeaturePipeline, args.pred_root,
              args.res_root, args.outer_folds, args.final_scoring)
    if args.stage == "select":
        scores, mean, best = out
        pd.set_option("display.width", 250, "display.max_columns", 30,
                      "display.max_colwidth", 90, "display.float_format", "{:.4f}".format)
        print("== mean inner score per setting, averaged over the 5 outer folds ==")
        print(mean.groupby("setting")[SCORE_COLUMNS].mean()
              .join(expand_settings(cfg).set_index("setting")).to_string())
        print("\n== selected per outer fold ==")
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()
