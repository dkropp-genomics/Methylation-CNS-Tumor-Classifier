#!/usr/bin/env python
"""Hyperparameter search with Optuna inside nested cross-validation.

For each outer fold k, one Optuna study proposes `n_trials` settings in turn.
Each setting is fit on the 3 inner folds of outer fold k and scored on the
inner held-out samples; the mean inner score goes back to Optuna, which uses it
to propose the next setting. Outer test fold k is never loaded.

Resuming: every fit saves its probabilities under a name derived from its
settings. On every start the study is replayed from trial 0 with the same seed;
trials whose files exist are scored from disk in milliseconds, so Optuna
proposes exactly the same sequence and the run continues where it stopped.
An interrupted run and an uninterrupted one give the same table.

  python scripts/tune_nested_cv.py configs/lgbm_v1.yaml --outer-folds 0 --max-trials 2   # timing pilot
  python scripts/tune_nested_cv.py configs/lgbm_v1.yaml                                  # everything

Outputs:
  data/predictions/<run_name>/o<k>_i<j>_<setting>.npz   probabilities (git-ignored)
  results/cv/<run_name>/trials_o<k>.tsv   one row per trial of outer fold k
  results/cv/<run_name>/fits.tsv          one row per finished fit, with timings
  when every outer fold has all its trials:
  results/cv/<run_name>/settings.tsv, inner_scores.tsv, selected.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nested_cv as cv  # noqa: E402  (shared: files, manifest, fitting, scoring)

FEATURE_KEYS = ("n_probes", "correct_material")


def check_config(cfg: dict):
    for key in (
        "run_name",
        "model",
        "seed",
        "n_jobs",
        "select_by",
        "n_trials",
        "fixed_params",
        "search_space",
    ):
        if key not in cfg:
            cv.fail(f"config: missing '{key}'")
    if cfg["select_by"] not in cv.SCORE_COLUMNS:
        cv.fail(f"config: select_by must be one of {cv.SCORE_COLUMNS}")
    space = cfg["search_space"]
    for key in FEATURE_KEYS:
        if key not in space:
            cv.fail(f"config: search_space must include {key}")
    for name, spec in space.items():
        kinds = [k for k in ("choice", "int", "float") if k in spec]
        if len(kinds) != 1:
            cv.fail(f"config: search_space.{name} needs exactly one of choice, int, float")
        if kinds[0] != "choice" and len(spec[kinds[0]]) != 2:
            cv.fail(f"config: search_space.{name} needs [low, high]")
    overlap = set(space) & set(cfg["fixed_params"])
    if overlap:
        cv.fail(f"config: {sorted(overlap)} appear in both fixed_params and search_space")


def suggest(trial, space: dict) -> dict:
    """Ask Optuna for one value per entry of the search space, in file order."""
    out = {}
    for name, spec in space.items():
        if "choice" in spec:
            out[name] = trial.suggest_categorical(name, list(spec["choice"]))
        elif "int" in spec:
            out[name] = int(
                trial.suggest_int(
                    name, spec["int"][0], spec["int"][1], log=bool(spec.get("log", False))
                )
            )
        else:
            out[name] = float(
                trial.suggest_float(
                    name, spec["float"][0], spec["float"][1], log=bool(spec.get("log", False))
                )
            )
    return out


def setting_row(values: dict, fixed: dict) -> dict:
    """Split suggested values into feature and model parts; name the setting by content."""
    model_params = {**fixed, **{k: v for k, v in values.items() if k not in FEATURE_KEYS}}
    text = json.dumps(
        {
            "n_probes": values["n_probes"],
            "correct_material": values["correct_material"],
            "model_params": model_params,
        },
        sort_keys=True,
    )
    return {
        "setting": "t" + hashlib.sha1(text.encode()).hexdigest()[:10],
        "n_probes": int(values["n_probes"]),
        "correct_material": bool(values["correct_material"]),
        "model_params": json.dumps(model_params, sort_keys=True),
    }


class FoldData:
    """Training side of one outer fold, loaded only when a fit is actually needed."""

    def __init__(self, load, samples, k):
        ids, y, mat, outer = cv.columns(samples)
        inner = samples[f"inner_fold_o{k}"].astype(int).to_numpy()
        train = outer != k
        if not ((inner == -1) == ~train).all():
            cv.fail(f"fold table: inner_fold_o{k} is not -1 exactly on outer test fold {k}")
        self.k, self._load = k, load
        self.ids, self.y, self.mat, self.inner = ids[train], y[train], mat[train], inner[train]
        self.truth = dict(zip(self.ids, self.y))
        self.X = None

    def split(self, j):
        if self.X is None:
            t0 = time.time()
            self.X = self._load(self.ids)  # training side only; test fold k is not read
            print(
                f"outer {self.k}: loaded {self.X.shape[0]} x {self.X.shape[1]} training "
                f"side in {time.time() - t0:.0f} s",
                flush=True,
            )
        tr, te = np.flatnonzero(self.inner != j), np.flatnonzero(self.inner == j)
        return (self.X[tr], self.y[tr], self.mat[tr], self.X[te], self.mat[te], self.ids[te])

    def expected_ids(self, j):
        return set(self.ids[self.inner == j])


def tune_fold(k, load, samples, cfg, pred_dir, res_dir, pipeline_factory, max_trials):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    by = cfg["select_by"]
    study = optuna.create_study(
        direction="minimize" if by == "log_loss" else "maximize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg["seed"]) + k),
    )
    all_classes = sorted(set(samples["mc_class"]))
    data = FoldData(load, samples, k)
    rows, score_rows, n_fit = [], [], 0
    for t in range(max_trials):
        t0 = time.time()
        trial = study.ask()
        row = setting_row(suggest(trial, cfg["search_space"]), cfg["fixed_params"])
        fit_now = 0
        scores = []
        for j in range(cv.N_INNER):
            path = cv.pred_path(pred_dir, k, j, row["setting"])
            if not path.exists():
                X_tr, y_tr, m_tr, X_te, m_te, ids_te = data.split(j)
                fit_now += cv.fit_group(
                    X_tr,
                    y_tr,
                    m_tr,
                    X_te,
                    m_te,
                    ids_te,
                    [(row, path)],
                    cfg,
                    all_classes,
                    pipeline_factory,
                    res_dir,
                    "inner",
                    k,
                    j,
                )
                del X_tr, X_te
            proba, sid, classes = cv.load_pred(path)
            if set(sid) != data.expected_ids(j) or len(sid) != len(set(sid)):
                cv.fail(f"{path.name}: samples are not inner fold {j} of outer fold {k}")
            s = cv.score_pred(np.array([data.truth[i] for i in sid], dtype=object), proba, classes)
            scores.append(s)
            score_rows.append(
                {"outer": k, "inner": j, "setting": row["setting"], "n": len(sid), **s}
            )
        mean = {m: float(np.mean([s[m] for s in scores])) for m in cv.SCORE_COLUMNS}
        study.tell(trial, mean[by])
        n_fit += fit_now
        rows.append({"outer": k, "trial": t, **row, **mean})
        print(
            f"outer {k} trial {t:2d} {row['setting']}: {by} {mean[by]:.4f}"
            f"{'' if fit_now else '  (from disk)'}"
            f"{f'  [{time.time() - t0:.0f} s]' if fit_now else ''}",
            flush=True,
        )
    trials = pd.DataFrame(rows)
    trials.to_csv(res_dir / f"trials_o{k}.tsv", sep="\t", index=False, float_format="%.6g")
    return trials, pd.DataFrame(score_rows), n_fit


def pick_best(trials: pd.DataFrame, by: str) -> pd.DataFrame:
    """Best trial per outer fold; ties go to the earlier trial."""
    t = trials.copy()
    t["_key"] = t[by] * (-1 if by == "log_loss" else 1)
    best = (
        t.sort_values(["outer", "_key", "trial"], ascending=[True, False, True])
        .groupby("outer", as_index=False)
        .head(1)
        .drop(columns="_key")
    )
    best.insert(1, "selected_by", by)
    return best


def run(
    cfg, samples, load, pipeline_factory, pred_root, res_root, outer_folds=None, max_trials=None
):
    check_config(cfg)
    pred_dir = Path(pred_root) / cfg["run_name"]
    res_dir = Path(res_root) / cfg["run_name"]
    n_trials = int(cfg["n_trials"])
    max_trials = n_trials if max_trials is None else min(int(max_trials), n_trials)
    outer_folds = list(range(cv.N_OUTER)) if outer_folds is None else list(outer_folds)
    if any(k not in range(cv.N_OUTER) for k in outer_folds):
        cv.fail(f"outer folds must be within 0..{cv.N_OUTER - 1}")
    cv.check_manifest(cfg, samples, res_dir, pd.DataFrame())
    pred_dir.mkdir(parents=True, exist_ok=True)

    trials, scores, n_fit = [], [], 0
    for k in outer_folds:
        t, s, n = tune_fold(k, load, samples, cfg, pred_dir, res_dir, pipeline_factory, max_trials)
        trials.append(t)
        scores.append(s)
        n_fit += n
    trials, scores = pd.concat(trials, ignore_index=True), pd.concat(scores, ignore_index=True)
    complete = outer_folds == list(range(cv.N_OUTER)) and max_trials == n_trials
    best = None
    if complete:
        best = pick_best(trials, cfg["select_by"])
        (
            trials.drop_duplicates("setting")[
                ["setting", "n_probes", "correct_material", "model_params"]
            ].to_csv(res_dir / "settings.tsv", sep="\t", index=False)
        )
        scores.to_csv(res_dir / "inner_scores.tsv", sep="\t", index=False, float_format="%.5f")
        best.to_csv(res_dir / "selected.tsv", sep="\t", index=False, float_format="%.6g")
    print(
        f"{n_fit} fits done now; "
        + (
            "search complete, selected.tsv written"
            if complete
            else "partial run: selected.tsv is written only when all 5 outer folds "
            "have all their trials"
        )
    )
    return trials, scores, best, n_fit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--outer-folds", type=int, nargs="+", default=None)
    ap.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="stop each study after this many trials (a later run continues)",
    )
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
    trials, scores, best, _ = run(
        cfg,
        st.samples,
        st.load,
        FeaturePipeline,
        args.pred_root,
        args.res_root,
        args.outer_folds,
        args.max_trials,
    )
    pd.set_option(
        "display.width",
        250,
        "display.max_columns",
        30,
        "display.max_colwidth",
        60,
        "display.float_format",
        "{:.4f}".format,
    )
    if best is not None:
        print("\n== selected per outer fold ==")
        print(best.drop(columns="model_params").to_string(index=False))
        for _, r in best.iterrows():
            print(f"outer {r['outer']}: {r['model_params']}")


if __name__ == "__main__":
    main()
