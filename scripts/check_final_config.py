#!/usr/bin/env python
"""Confirm that configs/final_v1.yaml follows its own rules.

The config names one final setting per model. Each choice has a stated rule
that uses inner-fold results only. This script recomputes every rule from the
committed run tables and stops if the config says something else. It reads no
beta values and no outer-test or validation score.

  python scripts/check_final_config.py configs/final_v1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def fail(msg: str):
    sys.exit(f"ERROR: {msg}")


def table(path, **kw):
    if not Path(path).is_file():
        fail(f"{path}: not found")
    return pd.read_csv(path, sep="\t", dtype={"setting": str}, **kw)


def most_common(values, what):
    """The value chosen in most outer folds; a tie for first place is an error."""
    counts = pd.Series(list(values)).value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        fail(f"{what}: no single most frequent choice ({counts.to_dict()})")
    return counts.index[0], f"{int(counts.iloc[0])} of {int(counts.sum())} outer folds"


def expected(cfg, res_root):
    """One row per choice: what the rule gives, from the run tables."""
    rows = []
    res = lambda name: Path(res_root) / cfg["models"][name]["run"]

    # forest: best mean inner macro-F1 over all inner fits; ties to the earlier setting
    s = table(res("rf") / "inner_scores.tsv")
    mean = s.groupby("setting")["macro_f1"].mean().sort_index()
    best = mean.idxmax()  # idxmax returns the first maximum
    rows.append(
        (
            "rf",
            "setting",
            best,
            f"mean inner macro-F1 {mean[best]:.4f} "
            f"over {int(s.groupby('setting').size()[best])} fits",
        )
    )

    # LightGBM: the per-fold selected trial with the highest inner macro-F1
    s = table(res("lgbm") / "selected.tsv").sort_values(
        ["macro_f1", "outer"], ascending=[False, True]
    )
    rows.append(
        (
            "lgbm",
            "setting",
            s["setting"].iloc[0],
            f"outer fold {int(s['outer'].iloc[0])}, inner macro-F1 " f"{s['macro_f1'].iloc[0]:.4f}",
        )
    )

    # networks: setting and epochs selected in most outer folds
    for name in ("nn_plain", "nn_masked"):
        s = table(res(name) / "selected.tsv")
        s = s[s["network"] == cfg["models"][name]["network"]]
        if len(s) == 0:
            fail(f"{name}: no rows for network {cfg['models'][name]['network']}")
        value, why = most_common(s["setting"], f"{name} setting")
        rows.append((name, "setting", value, why))
        value, why = most_common(s["epoch"].astype(int), f"{name} epoch")
        rows.append((name, "epoch", int(value), why))

    # calibrators: the form chosen in most outer folds
    for name in ("rf", "lgbm"):
        s = table(res(name) / "calibration_selected.tsv")
        value, why = most_common(zip(s["transform"], s["C"].astype(float)), f"{name} calibrator")
        rows.append((name, "transform", value[0], why))
        rows.append((name, "C", float(value[1]), why))
    return rows


def check(cfg, res_root="results/cv") -> pd.DataFrame:
    for key in ("models", "calibration"):
        if key not in cfg:
            fail(f"config: missing '{key}'")
    rows = []
    for name, field, want, why in expected(cfg, res_root):
        block = cfg["calibration"] if field in ("transform", "C") else cfg["models"]
        have = block.get(name, {}).get(field)
        same = (
            float(have) == float(want)
            if field in ("C", "epoch") and have is not None
            else have == want
        )
        rows.append(
            {
                "model": name,
                "field": field,
                "config": have,
                "rule_gives": want,
                "ok": bool(same),
                "basis": why,
            }
        )
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--res-root", default="results/cv")
    args = ap.parse_args(argv)
    import yaml

    if not Path(args.config).is_file():
        fail(f"{args.config}: config file not found")
    t = check(yaml.safe_load(Path(args.config).read_text()), args.res_root)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    if not t["ok"].all():
        fail(f"{int((~t['ok']).sum())} choice(s) in {args.config} do not follow their rule")
    print(f"\nall {len(t)} choices follow their rules")


if __name__ == "__main__":
    main()
