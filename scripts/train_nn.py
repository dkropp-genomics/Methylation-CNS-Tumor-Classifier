#!/usr/bin/env python
"""Neural network inside nested CV: inner fits with learning curves, then selection.

  inner   For every outer fold k, inner fold j and setting: fit the feature
          pipeline and the network on the inner training samples. Every few
          epochs, score the inner held-out samples at each coverage level
          (100%, 10%, 1%, 0.1% of CpGs observed). One file per fit holds the
          whole learning curve. Outer test fold k is never loaded.
  select  Per outer fold and per network type (plain, masked): pick the setting
          AND the number of epochs with the best inner score, averaged over the
          coverage levels and the 3 inner folds. Fits nothing.

The number of epochs is chosen like any other setting, from inner held-out
scores. The outer refit (a later step) trains for exactly that many epochs, so
no test fold ever decides when training stops.

  python scripts/train_nn.py configs/nn_v1.yaml --stage inner --outer-folds 0 --inner-folds 0 --settings s000 s001
  python scripts/train_nn.py configs/nn_v1.yaml --stage inner
  python scripts/train_nn.py configs/nn_v1.yaml --stage select

Outputs:
  data/predictions/<run>/o<k>_i<j>_<setting>.npz   learning curve + 100%-coverage probabilities
  results/cv/<run>/settings.tsv, fits.tsv, manifest.json
  results/cv/<run>/inner_curves.tsv, selected.tsv   (stage select)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nested_cv as cv  # noqa: E402

METRICS = cv.SCORE_COLUMNS


def check_config(cfg):
    for key in ("run_name", "model", "seed", "n_threads", "select_by", "levels", "mask_seed",
                "features", "train", "grid"):
        if key not in cfg:
            cv.fail(f"config: missing '{key}'")
    if cfg["select_by"] not in METRICS:
        cv.fail(f"config: select_by must be one of {METRICS}")
    if "masked" not in cfg["grid"]:
        cv.fail("config: grid must include 'masked'")
    if not all(0 < float(v) <= 1 for v in cfg["levels"]):
        cv.fail("config: levels must be in (0, 1]")
    for key in ("hidden", "batch_size", "weight_decay", "max_epochs", "eval_every",
                "min_fraction"):
        if key not in cfg["train"]:
            cv.fail(f"config: train is missing '{key}'")


def expand_settings(cfg) -> pd.DataFrame:
    keys = sorted(cfg["grid"])
    rows = []
    for values in itertools.product(*(cfg["grid"][k] for k in keys)):
        rows.append({"setting": f"s{len(rows):03d}", **dict(zip(keys, values))})
    return pd.DataFrame(rows)


def save_curve(path: Path, epochs, values, proba_full, sample_ids, classes, levels):
    tmp = path.with_name(path.name[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, epochs=np.asarray(epochs, dtype=np.int64),
                        values=np.asarray(values, dtype=np.float64),      # (epoch, level, metric)
                        proba_full=np.asarray(proba_full, dtype=np.float16),  # (epoch, n, classes)
                        sample_ids=np.asarray(sample_ids, dtype=str),
                        classes=np.asarray(classes, dtype=str),
                        levels=np.asarray(levels, dtype=np.float64),
                        metrics=np.asarray(METRICS, dtype=str))
    os.replace(tmp, path)


def load_curve(path: Path) -> pd.DataFrame:
    """Long table: epoch, level, one column per metric."""
    with np.load(path) as z:
        epochs, values, levels, metrics = z["epochs"], z["values"], z["levels"], z["metrics"]
    rows = [{"epoch": int(e), "level": float(lv), **dict(zip(metrics, values[a, b]))}
            for a, e in enumerate(epochs) for b, lv in enumerate(levels)]
    return pd.DataFrame(rows)


def max_epochs_for(cfg, masked) -> int:
    """max_epochs is one number, or {plain: ..., masked: ...}."""
    m = cfg["train"]["max_epochs"]
    if isinstance(m, dict):
        if set(m) != {"plain", "masked"}:
            cv.fail("config: train.max_epochs must be a number or {plain: , masked: }")
        return int(m["masked" if masked else "plain"])
    return int(m)


def fit_one(Zs_tr, y_idx, Zs_te, y_te, u_te, classes, row, cfg):
    """Train one network; return its learning curve on the held-out samples."""
    from methylclf.masking import observed
    from methylclf.nn import predict_proba, train_mlp
    levels = [float(v) for v in cfg["levels"]]
    epochs, values, proba_full = [], [], []

    def checkpoint(epoch, net):
        per_level = []
        for lv in levels:
            proba = predict_proba(net, Zs_te, observed(u_te, lv))
            s = cv.score_pred(y_te, proba, classes)
            per_level.append([s[m] for m in METRICS])
            if lv == 1.0:
                proba_full.append(proba)
        epochs.append(epoch)
        values.append(per_level)

    t = cfg["train"]
    train_mlp(Zs_tr, y_idx, len(classes), masked=bool(row["masked"]), hidden=tuple(t["hidden"]),
              dropout=float(row.get("dropout", 0.2)), lr=float(row.get("lr", 1e-3)),
              weight_decay=float(t["weight_decay"]), batch_size=int(t["batch_size"]),
              max_epochs=max_epochs_for(cfg, row["masked"]), eval_every=int(t["eval_every"]),
              schedule=str(t.get("schedule", "constant")),
              min_fraction=float(t["min_fraction"]), seed=int(cfg["seed"]),
              n_threads=int(cfg["n_threads"]), on_checkpoint=checkpoint)
    if not proba_full:                      # 1.0 not among the levels
        proba_full = np.zeros((len(epochs), 0, len(classes)))
    return epochs, values, proba_full


def stage_inner(load, array_probe_ids, samples, cfg, settings, pred_dir, res_dir,
                pipeline_factory, outer_folds, inner_folds):
    from methylclf.masking import observed_uniform, probe_positions
    from methylclf.nn import Standardizer
    ids, y, mat, outer = cv.columns(samples)
    classes = sorted(set(y))
    index = {c: i for i, c in enumerate(classes)}
    n_done = 0
    for k in outer_folds:
        inner = samples[f"inner_fold_o{k}"].astype(int).to_numpy()
        train = outer != k
        if not ((inner == -1) == ~train).all():
            cv.fail(f"fold table: inner_fold_o{k} is not -1 exactly on outer test fold {k}")
        todo = {j: [r for _, r in settings.iterrows()
                    if not cv.pred_path(pred_dir, k, j, r["setting"]).exists()]
                for j in inner_folds}
        if not any(todo.values()):
            print(f"outer {k}: all requested inner fits already done", flush=True)
            continue
        t0 = time.time()
        X = load(ids[train])               # training side only; test fold k is not read
        y_k, m_k, ids_k, inner_k = y[train], mat[train], ids[train], inner[train]
        print(f"outer {k}: loaded {X.shape[0]} x {X.shape[1]} training side in "
              f"{time.time() - t0:.0f} s", flush=True)
        for j in inner_folds:
            if not todo[j]:
                continue
            tr, te = np.flatnonzero(inner_k != j), np.flatnonzero(inner_k == j)
            if sorted(set(y_k[tr])) != classes:
                cv.fail(f"outer {k} inner {j}: the training samples do not hold every class")
            t0 = time.time()
            pipe = pipeline_factory(n_probes=int(cfg["features"]["n_probes"]),
                                    correct_material=bool(cfg["features"]["correct_material"]))
            pipe.fit(X[tr], y_k[tr], m_k[tr])
            Z_tr, Z_te = pipe.transform(X[tr], m_k[tr]), pipe.transform(X[te], m_k[te])
            scaler = Standardizer().fit(Z_tr)              # training fold only
            Zs_tr, Zs_te = scaler.transform(Z_tr), scaler.transform(Z_te)
            del Z_tr, Z_te
            names = pipe.get_feature_names_out(array_probe_ids)
            u_te = observed_uniform(ids_k[te], len(array_probe_ids),
                                    probe_positions(names, array_probe_ids),
                                    int(cfg["mask_seed"]))
            t_feat = time.time() - t0
            y_idx = np.array([index[c] for c in y_k[tr]], dtype=np.int64)
            for row in todo[j]:
                t1 = time.time()
                epochs, values, proba_full = fit_one(Zs_tr, y_idx, Zs_te, y_k[te], u_te,
                                                     classes, row, cfg)
                save_curve(cv.pred_path(pred_dir, k, j, row["setting"]), epochs, values,
                           proba_full, ids_k[te], classes, cfg["levels"])
                cv.log_fit(res_dir, {"stage": "inner", "outer": k, "inner": j,
                                     "setting": row["setting"], "n_train": len(tr),
                                     "n_test": len(te), "n_features": Zs_tr.shape[1],
                                     "seconds_features": t_feat,
                                     "seconds_model": time.time() - t1,
                                     "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
                n_done += 1
                last = dict(zip(METRICS, np.asarray(values)[-1].T))
                by = cfg["select_by"]
                print(f"  outer {k} inner {j} {row['setting']} "
                      f"{'masked' if row['masked'] else 'plain '}: {time.time() - t1:.0f} s; "
                      f"{by} at epoch {epochs[-1]} by level: "
                      + "  ".join(f"{lv:g}={v:.3f}" for lv, v in zip(cfg['levels'], last[by])),
                      flush=True)
            del Zs_tr, Zs_te, u_te
        del X
    print(f"inner stage: {n_done} fits done now")
    return n_done


def read_curves(settings, pred_dir, outer_folds, inner_folds, require_all=True):
    tables, missing = [], []
    for k in outer_folds:
        for j in inner_folds:
            for _, r in settings.iterrows():
                path = cv.pred_path(pred_dir, k, j, r["setting"])
                if not path.exists():
                    missing.append(path.name)
                    continue
                t = load_curve(path)
                t.insert(0, "setting", r["setting"])
                t.insert(0, "inner", j)
                t.insert(0, "outer", k)
                tables.append(t)
    if missing and require_all:
        cv.fail(f"select: {len(missing)} inner fits are missing (first: {missing[0]}); "
                f"finish --stage inner first")
    if not tables:
        cv.fail("no finished fits found for the requested folds and settings")
    return pd.concat(tables, ignore_index=True)


def stage_select(cfg, settings, pred_dir, res_dir):
    by = cfg["select_by"]
    curves = read_curves(settings, pred_dir, range(cv.N_OUTER), range(cv.N_INNER))
    curves = curves.merge(settings, on="setting")
    curves.to_csv(res_dir / "inner_curves.tsv", sep="\t", index=False, float_format="%.5f")
    # mean over the 3 inner folds, per coverage level
    lvl = curves.groupby(["outer", "masked", "setting", "epoch", "level"],
                         as_index=False)[METRICS].mean()
    # then the mean over coverage levels: the number that picks setting and epochs
    mean = lvl.groupby(["outer", "masked", "setting", "epoch"], as_index=False)[by].mean()
    mean["_key"] = mean[by] * (-1 if by == "log_loss" else 1)
    best = (mean.sort_values(["outer", "masked", "_key", "epoch", "setting"],
                             ascending=[True, True, False, True, True])
            .groupby(["outer", "masked"], as_index=False).head(1).drop(columns="_key"))
    best = best.rename(columns={by: f"{by}_mean_over_levels"})
    wide = lvl.pivot_table(index=["outer", "setting", "epoch"], columns="level", values=by)
    wide.columns = [f"{by}_at_{c:g}" for c in wide.columns]
    best = best.merge(wide.reset_index(), on=["outer", "setting", "epoch"]).merge(
        settings.drop(columns="masked"), on="setting")
    best.insert(0, "network", np.where(best["masked"], "masked", "plain"))
    best = best.sort_values(["network", "outer"]).reset_index(drop=True)
    best.to_csv(res_dir / "selected.tsv", sep="\t", index=False, float_format="%.5f")
    return curves, best


def run(stage, cfg, samples, load, array_probe_ids, pipeline_factory, pred_root, res_root,
        outer_folds=None, inner_folds=None, only_settings=None):
    check_config(cfg)
    settings = expand_settings(cfg)
    pred_dir = Path(pred_root) / cfg["run_name"]
    res_dir = Path(res_root) / cfg["run_name"]
    outer_folds = list(range(cv.N_OUTER)) if outer_folds is None else list(outer_folds)
    inner_folds = list(range(cv.N_INNER)) if inner_folds is None else list(inner_folds)
    if any(k not in range(cv.N_OUTER) for k in outer_folds) or \
            any(j not in range(cv.N_INNER) for j in inner_folds):
        cv.fail("outer folds must be within 0..4 and inner folds within 0..2")
    cv.check_manifest(cfg, samples, res_dir, settings)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if stage == "inner":
        chosen = settings
        if only_settings:
            unknown = sorted(set(only_settings) - set(settings["setting"]))
            if unknown:
                cv.fail(f"unknown settings {unknown}; known: {list(settings['setting'])}")
            chosen = settings[settings["setting"].isin(only_settings)]
        return stage_inner(load, array_probe_ids, samples, cfg, chosen, pred_dir, res_dir,
                           pipeline_factory, outer_folds, inner_folds)
    if stage == "select":
        return stage_select(cfg, settings, pred_dir, res_dir)
    if stage == "curves":      # look at whatever is finished, without selecting anything
        chosen = settings if not only_settings else settings[settings["setting"].isin(only_settings)]
        return read_curves(chosen, pred_dir, outer_folds, inner_folds,
                           require_all=False).merge(settings, on="setting")
    cv.fail(f"unknown stage '{stage}'")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--stage", required=True, choices=["inner", "select", "curves"])
    ap.add_argument("--outer-folds", type=int, nargs="+", default=None)
    ap.add_argument("--inner-folds", type=int, nargs="+", default=None)
    ap.add_argument("--settings", nargs="+", default=None)
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
        cv.fail(f"{args.config}: config file not found")
    cfg = yaml.safe_load(Path(args.config).read_text())
    st = BetaStore.open(args.store, args.folds, args.probes)
    out = run(args.stage, cfg, st.samples, st.load, np.asarray(st.probe_ids, dtype=object),
              FeaturePipeline, args.pred_root, args.res_root, args.outer_folds,
              args.inner_folds, args.settings)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400,
                  "display.float_format", "{:.3f}".format)
    by = cfg["select_by"]
    if args.stage == "curves":
        print(expand_settings(cfg).to_string(index=False))
        for metric in dict.fromkeys([by, "accuracy"]):
            t = out.groupby(["setting", "epoch", "level"])[metric].mean().unstack("level")
            print(f"\n== {metric} on inner held-out samples, by epoch and share observed ==")
            print(t[sorted(t.columns, reverse=True)].to_string())
    if args.stage == "select":
        print("== selected per network and outer fold ==")
        print(out[1].to_string(index=False))


if __name__ == "__main__":
    main()
