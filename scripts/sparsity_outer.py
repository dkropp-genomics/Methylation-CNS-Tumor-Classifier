#!/usr/bin/env python
"""Sparse-coverage stress test on the outer test folds (final scoring).

  fit     FINAL SCORING, refused without --final-scoring. For every outer fold
          k and every model: refit the model's selected setting on the whole
          training side, then predict test fold k at each coverage level
          (share of array CpGs observed). The observed CpGs of a sample are the
          same for every model (methylclf.masking).
            tree models: unobserved CpGs are filled with the training-fold
                         median of that feature;
            networks:    unobserved CpGs are zeroed and flagged as unobserved.
          At 100% coverage a tree model's refit is compared with the
          predictions saved in Phase 3 (repro_check.tsv).
  report  Pool the five test folds and compute every metric per model and
          level with bootstrap intervals, plus paired differences between
          models. Reads saved predictions only.

No setting is chosen here: settings and epochs come from each run's
selected.tsv, which was written from inner folds.

  python scripts/sparsity_outer.py configs/sparsity_v1.yaml --stage fit --final-scoring
  python scripts/sparsity_outer.py configs/sparsity_v1.yaml --stage report

Outputs:
  data/predictions/<run>/<model>_o<k>.npz        probabilities at every level
  results/cv/<run>/repro_check.tsv, fits.tsv, manifest.json
  results/cv/<run>/metrics.tsv, differences.tsv  (stage report)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nested_cv as cv  # noqa: E402
import train_nn as tnn  # noqa: E402

PAIRS = (("nn_masked", "rf"), ("nn_masked", "nn_plain"), ("nn_masked", "lgbm"),
         ("nn_plain", "rf"))


def check_config(cfg, model_cfgs):
    for key in ("run_name", "levels", "mask_seed", "models"):
        if key not in cfg:
            cv.fail(f"config: missing '{key}'")
    if not all(0 < float(v) <= 1 for v in cfg["levels"]):
        cv.fail("config: levels must be in (0, 1]")
    for name, m in cfg["models"].items():
        if m.get("kind") not in ("tree", "nn"):
            cv.fail(f"config: models.{name}.kind must be 'tree' or 'nn'")
        if m["kind"] == "nn":
            if m.get("network") not in ("plain", "masked"):
                cv.fail(f"config: models.{name}.network must be 'plain' or 'masked'")
            c = model_cfgs[name]
            if [float(v) for v in c["levels"]] != [float(v) for v in cfg["levels"]] or \
                    int(c["mask_seed"]) != int(cfg["mask_seed"]):
                cv.fail(f"config: levels and mask_seed must equal those of {m['config']}, "
                        f"which the network was selected on")


def out_path(pred_dir: Path, name: str, k: int) -> Path:
    return Path(pred_dir) / f"{name}_o{k}.npz"


def save_levels(path, proba, sample_ids, classes, levels):
    tmp = path.with_name(path.name[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, proba=np.asarray(proba, dtype=np.float32),   # (level, n, class)
                        sample_ids=np.asarray(sample_ids, dtype=str),
                        classes=np.asarray(classes, dtype=str),
                        levels=np.asarray(levels, dtype=np.float64))
    os.replace(tmp, path)


def selected_tree(res_root, m, k):
    res = Path(res_root) / m["run"]
    for f in ("selected.tsv", "settings.tsv"):
        if not (res / f).exists():
            cv.fail(f"{res / f} not found")
    chosen = pd.read_csv(res / "selected.tsv", sep="\t", dtype={"setting": str})
    settings = pd.read_csv(res / "settings.tsv", sep="\t", dtype={"setting": str})
    row = chosen[chosen["outer"] == k]
    if len(row) != 1:
        cv.fail(f"{res / 'selected.tsv'}: expected one row for outer fold {k}")
    hit = settings[settings["setting"] == row["setting"].iloc[0]]
    if len(hit) != 1:
        cv.fail(f"{res / 'settings.tsv'}: setting {row['setting'].iloc[0]} not found")
    return hit.iloc[0]


def selected_nn(res_root, m, k):
    path = Path(res_root) / m["run"] / "selected.tsv"
    if not path.exists():
        cv.fail(f"{path} not found; run train_nn.py --stage select first")
    t = pd.read_csv(path, sep="\t", dtype={"setting": str})
    row = t[(t["network"] == m["network"]) & (t["outer"] == k)]
    if len(row) != 1:
        cv.fail(f"{path}: expected one row for network {m['network']}, outer fold {k}")
    return row.iloc[0]


def features(pipeline_factory, n_probes, correct, X_tr, y_tr, m_tr, X_te, m_te, probe_ids):
    pipe = pipeline_factory(n_probes=int(n_probes), correct_material=bool(correct))
    pipe.fit(X_tr, y_tr, m_tr)
    return (pipe.transform(X_tr, m_tr), pipe.transform(X_te, m_te),
            pipe.get_feature_names_out(probe_ids))


def predict_tree(m, mcfg, row, Z_tr, y_tr, Z_te, u, levels, classes):
    """Fit once; at each level fill unobserved features with the training median."""
    from methylclf.masking import observed
    model = cv.make_model(mcfg["model"], json.loads(row["model_params"]), mcfg["seed"],
                          mcfg["n_jobs"])
    model.fit(Z_tr, y_tr)
    if list(model.classes_) != list(classes):
        cv.fail(f"{m['run']}: the training samples do not hold every class")
    median = np.median(Z_tr, axis=0).astype(np.float32)      # training fold only
    out = []
    for lv in levels:
        obs = observed(u, lv)
        out.append(model.predict_proba(np.where(obs, Z_te, median[None, :]).astype(np.float32)))
    return np.stack(out)


def predict_nn(mcfg, row, Z_tr, y_tr, Z_te, u, levels, classes):
    from methylclf.masking import observed
    from methylclf.nn import Standardizer, predict_proba, train_mlp
    scaler = Standardizer().fit(Z_tr)                         # training fold only
    Zs_tr, Zs_te = scaler.transform(Z_tr), scaler.transform(Z_te)
    index = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([index[c] for c in y_tr], dtype=np.int64)
    t = mcfg["train"]
    masked = bool(row["masked"])
    net = train_mlp(Zs_tr, y_idx, len(classes), masked=masked, hidden=tuple(t["hidden"]),
                    dropout=float(row["dropout"]), lr=float(row["lr"]),
                    weight_decay=float(t["weight_decay"]), batch_size=int(t["batch_size"]),
                    max_epochs=tnn.max_epochs_for(mcfg, masked), stop_epoch=int(row["epoch"]),
                    eval_every=10 ** 9, min_fraction=float(t["min_fraction"]),
                    seed=int(mcfg["seed"]), n_threads=int(mcfg["n_threads"]),
                    schedule=str(t.get("schedule", "constant")))
    return np.stack([predict_proba(net, Zs_te, observed(u, lv)) for lv in levels])


def stage_fit(cfg, model_cfgs, samples, load, probe_ids, pipeline_factory, pred_root,
              res_root, outer_folds):
    from methylclf.masking import observed_uniform, probe_positions
    pred_dir, res_dir = Path(pred_root) / cfg["run_name"], Path(res_root) / cfg["run_name"]
    ids, y, mat, outer = cv.columns(samples)
    classes = sorted(set(y))
    levels = [float(v) for v in cfg["levels"]]
    n_done, checks = 0, []
    for k in outer_folds:
        todo = [n for n in cfg["models"] if not out_path(pred_dir, n, k).exists()]
        if not todo:
            print(f"outer {k}: all models already done", flush=True)
            continue
        train = outer != k
        t0 = time.time()
        X_tr, X_te = load(ids[train]), load(ids[~train])
        print(f"outer {k}: loaded {X_tr.shape[0]} training and {X_te.shape[0]} test samples "
              f"in {time.time() - t0:.0f} s", flush=True)
        cache = {}                                   # features shared by models
        for name in todo:
            m, mcfg = cfg["models"][name], model_cfgs[name]
            t1 = time.time()
            if m["kind"] == "tree":
                row = selected_tree(res_root, m, k)
                key = (int(row["n_probes"]), bool(row["correct_material"]))
            else:
                row = selected_nn(res_root, m, k)
                key = (int(mcfg["features"]["n_probes"]),
                       bool(mcfg["features"]["correct_material"]))
            if key not in cache:
                cache[key] = features(pipeline_factory, key[0], key[1], X_tr, y[train],
                                      mat[train], X_te, mat[~train], probe_ids)
            Z_tr, Z_te, names = cache[key]
            u = observed_uniform(ids[~train], len(probe_ids),
                                 probe_positions(names, probe_ids), int(cfg["mask_seed"]))
            if m["kind"] == "tree":
                proba = predict_tree(m, mcfg, row, Z_tr, y[train], Z_te, u, levels, classes)
            else:
                proba = predict_nn(mcfg, row, Z_tr, y[train], Z_te, u, levels, classes)
            save_levels(out_path(pred_dir, name, k), proba, ids[~train], classes, levels)
            if m["kind"] == "tree" and 1.0 in levels:
                old = cv.pred_path(Path(pred_root) / m["run"], k, None, row["setting"])
                if old.exists():
                    P, old_ids, _ = cv.load_pred(old)
                    same = list(old_ids) == list(ids[~train])
                    new = proba[levels.index(1.0)]
                    checks.append({
                        "model": name, "outer": k, "same_sample_order": same,
                        "max_abs_diff": float(np.abs(P - new).max()) if same else np.nan,
                        "n_top_class_changed": int((P.argmax(1) != new.argmax(1)).sum())
                        if same else -1})
            acc = [float((np.asarray(classes, dtype=object)[p.argmax(1)] == y[~train]).mean())
                   for p in proba]
            cv.log_fit(res_dir, {"stage": "sparsity", "outer": k, "inner": "test",
                                 "setting": f"{name}:{row['setting']}",
                                 "n_train": int(train.sum()), "n_test": int((~train).sum()),
                                 "n_features": Z_tr.shape[1], "seconds_features": 0.0,
                                 "seconds_model": time.time() - t1,
                                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
            n_done += 1
            print(f"  outer {k} {name:9s} {row['setting']}: {time.time() - t1:.0f} s; "
                  "accuracy by level: "
                  + "  ".join(f"{lv:g}={a:.3f}" for lv, a in zip(levels, acc)), flush=True)
        del X_tr, X_te, cache
    if checks:
        path = res_dir / "repro_check.tsv"
        t = pd.DataFrame(checks)
        if path.exists():
            t = pd.concat([pd.read_csv(path, sep="\t"), t]).drop_duplicates(
                ["model", "outer"], keep="last")
        t.sort_values(["model", "outer"]).to_csv(path, sep="\t", index=False)
    print(f"fit stage: {n_done} model fits done now")
    return n_done


# --------------------------------------------------------------------------
def pooled(cfg, pred_dir, name, folds):
    """One prediction per sample and level, in fold-table order."""
    outer_of = dict(zip(folds["geo_accession"], folds["outer_fold"].astype(int)))
    by_id, classes, levels = {}, None, None
    for k in range(cv.N_OUTER):
        path = out_path(pred_dir, name, k)
        if not path.exists():
            cv.fail(f"{path} not found; finish --stage fit first")
        with np.load(path) as z:
            proba, ids = z["proba"], z["sample_ids"].astype(object)
            cls, lv = list(z["classes"].astype(object)), [float(v) for v in z["levels"]]
        want = {i for i, o in outer_of.items() if o == k}
        if set(ids) != want or len(ids) != len(want):
            cv.fail(f"{path.name}: samples are not exactly outer test fold {k}")
        if (classes is not None and cls != classes) or (levels is not None and lv != levels):
            cv.fail(f"{path.name}: classes or levels differ between folds")
        classes, levels = cls, lv
        for i, sid in enumerate(ids):
            by_id[sid] = proba[:, i, :]
    order = folds["geo_accession"].to_numpy()
    return np.stack([by_id[s] for s in order], axis=1).astype(np.float64), classes, levels


def macro_f1_fast(t, p, n_classes):
    """Macro-F1 over the classes present in the true labels (integer codes)."""
    tp = np.bincount(t[t == p], minlength=n_classes).astype(np.float64)
    n_true = np.bincount(t, minlength=n_classes).astype(np.float64)
    n_pred = np.bincount(p, minlength=n_classes).astype(np.float64)
    present = n_true > 0
    return float((2 * tp[present] / (n_true[present] + n_pred[present])).mean())


def paired_difference(t, pa, pb, n_classes, n_boot, seed):
    """Model A minus model B on the same samples, with a paired bootstrap interval."""
    rng = np.random.default_rng(seed)
    point = {"accuracy": float((pa == t).mean() - (pb == t).mean()),
             "macro_f1": macro_f1_fast(t, pa, n_classes) - macro_f1_fast(t, pb, n_classes)}
    boots = {m: np.empty(n_boot) for m in point}
    for b in range(n_boot):
        i = rng.integers(0, t.size, t.size)          # the same resample for both models
        boots["accuracy"][b] = (pa[i] == t[i]).mean() - (pb[i] == t[i]).mean()
        boots["macro_f1"][b] = (macro_f1_fast(t[i], pa[i], n_classes)
                                - macro_f1_fast(t[i], pb[i], n_classes))
    out = []
    for m in point:
        lo, hi = (np.percentile(boots[m], [2.5, 97.5]) if n_boot else (np.nan, np.nan))
        out.append({"metric": m, "difference": point[m], "ci_low": lo, "ci_high": hi,
                    "share_of_draws_above_0": float((boots[m] > 0).mean()) if n_boot else np.nan})
    return out


def stage_report(cfg, folds, family_of, summarize, pred_root, res_root, n_boot=1000):
    pred_dir, res_dir = Path(pred_root) / cfg["run_name"], Path(res_root) / cfg["run_name"]
    y = folds["mc_class"].to_numpy(dtype=object)
    tables, top = [], {}
    classes = levels = None
    for name in cfg["models"]:
        P, classes, levels = pooled(cfg, pred_dir, name, folds)
        if [float(v) for v in cfg["levels"]] != levels:
            cv.fail(f"{name}: saved levels {levels} differ from the config")
        for a, lv in enumerate(levels):
            t = summarize(y, P[a], classes, family_of=family_of, n_boot=n_boot)
            t = t.rename(columns={"level": "label_level"})     # class or family
            t.insert(0, "coverage", lv)                         # share of CpGs observed
            t.insert(0, "model", name)
            tables.append(t)
            top[(name, lv)] = P[a].argmax(axis=1)
        print(f"{name}: metrics done", flush=True)
    metrics = pd.concat(tables, ignore_index=True)
    index = {c: i for i, c in enumerate(classes)}
    t_idx = np.array([index[c] for c in y])
    rows = []
    for a, b in PAIRS:
        if a not in cfg["models"] or b not in cfg["models"]:
            continue
        for lv in levels:
            for r in paired_difference(t_idx, top[(a, lv)], top[(b, lv)], len(classes),
                                       n_boot, seed=0):
                rows.append({"model_a": a, "model_b": b, "coverage": lv, **r})
    diffs = pd.DataFrame(rows)
    res_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(res_dir / "metrics.tsv", sep="\t", index=False, float_format="%.5f")
    diffs.to_csv(res_dir / "differences.tsv", sep="\t", index=False, float_format="%.5f")
    return metrics, diffs


def run(stage, cfg, model_cfgs, samples, load=None, probe_ids=None, pipeline_factory=None,
        pred_root="data/predictions", res_root="results/cv", outer_folds=None,
        final_scoring=False, family_of=None, summarize=None, n_boot=1000):
    check_config(cfg, model_cfgs)
    res_dir = Path(res_root) / cfg["run_name"]
    if stage == "fit":
        if not final_scoring:
            cv.fail("fit: this stage scores the outer test folds. Add --final-scoring to confirm.")
        outer_folds = list(range(cv.N_OUTER)) if outer_folds is None else list(outer_folds)
        cv.check_manifest({"sparsity": cfg, "models": model_cfgs}, samples, res_dir,
                          pd.DataFrame())
        (Path(pred_root) / cfg["run_name"]).mkdir(parents=True, exist_ok=True)
        return stage_fit(cfg, model_cfgs, samples, load, probe_ids, pipeline_factory,
                         pred_root, res_root, outer_folds)
    if stage == "report":
        return stage_report(cfg, samples, family_of, summarize, pred_root, res_root, n_boot)
    cv.fail(f"unknown stage '{stage}'")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--stage", required=True, choices=["fit", "report"])
    ap.add_argument("--outer-folds", type=int, nargs="+", default=None)
    ap.add_argument("--final-scoring", action="store_true")
    ap.add_argument("--store", default="data/betas/zarr/GSE90496.zarr")
    ap.add_argument("--folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--probes", default="results/probes/probes_kept.tsv")
    ap.add_argument("--families", default="results/meta/class_to_family.tsv")
    ap.add_argument("--pred-root", default="data/predictions")
    ap.add_argument("--res-root", default="results/cv")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    import yaml
    for path in (args.config, args.folds, args.families):
        if not Path(path).exists():
            cv.fail(f"{path}: not found")
    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfgs = {}
    for name, m in cfg.get("models", {}).items():
        if not Path(m.get("config", "")).exists():
            cv.fail(f"models.{name}.config: {m.get('config')} not found")
        model_cfgs[name] = yaml.safe_load(Path(m["config"]).read_text())
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    if args.stage == "fit":
        from methylclf.data import BetaStore
        from methylclf.features import FeaturePipeline
        st = BetaStore.open(args.store, args.folds, args.probes)
        run("fit", cfg, model_cfgs, st.samples, st.load,
            np.asarray(st.probe_ids, dtype=object), FeaturePipeline, args.pred_root,
            args.res_root, args.outer_folds, args.final_scoring)
        check = Path(args.res_root) / cfg["run_name"] / "repro_check.tsv"
        if check.exists():
            print("\n== tree models at 100% coverage against the Phase 3 predictions ==")
            print(pd.read_csv(check, sep="\t").to_string(index=False))
        return
    from methylclf.metrics import summarize
    folds = pd.read_csv(args.folds, sep="\t", dtype=str, keep_default_na=False)
    fam = pd.read_csv(args.families, sep="\t", dtype=str, keep_default_na=False)
    metrics, diffs = run("report", cfg, model_cfgs, folds, pred_root=args.pred_root,
                         res_root=args.res_root, family_of=dict(zip(fam["mc_class"], fam["family"])),
                         summarize=summarize, n_boot=args.n_boot)
    for lvl_name in ("class", "family"):
        for metric in ("accuracy", "macro_f1"):
            sub = metrics[(metrics["metric"] == metric) & (metrics["group"] == "all")]
            print(f"\n== {metric}, {lvl_name} level; columns = share of CpGs observed; "
                  f"value (95% interval) ==")
            print(table(sub, lvl_name).to_string())
    print("\n== paired differences, model_a minus model_b (class level) ==")
    print(diffs.to_string(index=False, float_format="{:.4f}".format))


def table(sub, lvl_name):
    s = sub[sub["label_level"] == lvl_name].copy()
    s["text"] = [f"{v:.3f} ({lo:.3f}-{hi:.3f})" for v, lo, hi in
                 zip(s["value"], s["ci_low"], s["ci_high"])]
    t = s.pivot_table(index="model", columns="coverage", values="text", aggfunc="first",
                      sort=False)
    return t[sorted(t.columns, reverse=True)]


if __name__ == "__main__":
    main()
