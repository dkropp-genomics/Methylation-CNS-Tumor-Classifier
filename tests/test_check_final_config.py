"""Tests for scripts/check_final_config.py, and for the committed config itself."""
import copy
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_final_config as cfc  # noqa: E402

CFG = {
    "models": {
        "rf": {"run": "rf", "setting": "s001"},
        "lgbm": {"run": "lgbm", "setting": "tb"},
        "nn_plain": {"run": "nn", "network": "plain", "setting": "s000", "epoch": 60},
        "nn_masked": {"run": "nn", "network": "masked", "setting": "s001", "epoch": 150},
        "centroid": {"kind": "centroid"},
    },
    "calibration": {"rf": {"transform": "raw", "C": 1.0},
                    "lgbm": {"transform": "log", "C": 10.0}},
}


def make_runs(root: Path):
    w = lambda run, name, df: ((root / run).mkdir(parents=True, exist_ok=True),
                               df.to_csv(root / run / name, sep="\t", index=False))
    rows = [{"outer": k, "inner": j, "setting": s, "macro_f1": v}
            for k in range(5) for j in range(3)
            for s, v in (("s000", 0.90), ("s001", 0.93), ("s002", 0.93))]
    w("rf", "inner_scores.tsv", pd.DataFrame(rows))        # s001 and s002 tie: earlier wins
    w("rf", "calibration_selected.tsv", pd.DataFrame(
        {"outer": range(5), "transform": ["raw"] * 5, "C": [1.0, 1.0, 10.0, 1.0, 1.0]}))
    w("lgbm", "selected.tsv", pd.DataFrame(
        {"outer": range(5), "setting": ["ta", "tb", "tc", "td", "te"],
         "macro_f1": [0.91, 0.92, 0.90, 0.89, 0.88]}))
    w("lgbm", "calibration_selected.tsv", pd.DataFrame(
        {"outer": range(5), "transform": ["log"] * 5, "C": [10.0] * 5}))
    w("nn", "selected.tsv", pd.DataFrame(
        {"network": ["masked"] * 5 + ["plain"] * 5, "outer": list(range(5)) * 2,
         "setting": ["s001"] * 5 + ["s000"] * 5,
         "epoch": [150] * 5 + [60, 50, 60, 60, 60]}))


def test_a_config_that_follows_the_rules_passes():
    with tempfile.TemporaryDirectory() as d:
        make_runs(Path(d))
        t = cfc.check(CFG, d)
    assert len(t) == 10 and t["ok"].all(), t.to_string()


def test_each_wrong_choice_is_caught():
    edits = [(("models", "rf", "setting"), "s002"), (("models", "lgbm", "setting"), "ta"),
             (("models", "nn_plain", "epoch"), 50), (("models", "nn_masked", "setting"), "s003"),
             (("calibration", "rf", "C"), 10.0), (("calibration", "lgbm", "transform"), "raw")]
    with tempfile.TemporaryDirectory() as d:
        make_runs(Path(d))
        for (a, b, c), value in edits:
            cfg = copy.deepcopy(CFG)
            cfg[a][b][c] = value
            t = cfc.check(cfg, d)
            assert int((~t["ok"]).sum()) == 1, (a, b, c)


def test_a_tie_between_folds_stops():
    with tempfile.TemporaryDirectory() as d:
        make_runs(Path(d))
        pd.DataFrame({"outer": range(4), "transform": ["raw", "raw", "log", "log"],
                      "C": [1.0] * 4}).to_csv(Path(d) / "rf" / "calibration_selected.tsv",
                                              sep="\t", index=False)
        try:
            cfc.check(CFG, d)
        except SystemExit as e:
            assert "no single most frequent" in str(e)
        else:
            raise AssertionError("expected a stop")


def test_the_committed_config_has_every_decision():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "final_v1.yaml").read_text())
    assert set(cfg["models"]) == {"rf", "lgbm", "nn_plain", "nn_masked", "centroid"}
    assert cfg["target"] == {**cfg["target"], "model": "rf", "level": "family",
                             "metric": "macro_f1", "max_drop": 0.05}
    assert cfg["analyses"]["sparsity"]["mask_seed"] == 20261004
    assert set(cfg["analyses"]["full_coverage"]) == set(cfg["models"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
