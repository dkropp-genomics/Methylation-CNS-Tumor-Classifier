"""Tests for scripts/cv_reference.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cv_reference as ref  # noqa: E402


def preds():
    # classes A1, A2 (family A), B, C; C does not occur in validation
    true = ["A1"] * 4 + ["A2"] * 4 + ["B"] * 4 + ["C"] * 4
    pred = [
        "A1",
        "A1",
        "A1",
        "A2",
        "A2",
        "A2",
        "A2",
        "C",
        "B",
        "B",
        "B",
        "B",
        "A1",
        "A1",
        "A1",
        "A1",
    ]
    fam = {"A1": "A", "A2": "A", "B": "B", "C": "C"}
    return pd.DataFrame(
        {
            "mc_class": true,
            "predicted_class": pred,
            "family": [fam[c] for c in true],
            "predicted_family": [fam[c] for c in pred],
        }
    )


def get(t, level, metric):
    return t[(t["level"] == level) & (t["metric"] == metric)].iloc[0]


def test_only_shared_classes_are_scored():
    t = ref.reference(preds(), {"A1", "A2", "B"}, n_boot=0)
    row = get(t, "class", "accuracy")
    assert row["n"] == 12 and row["n_groups"] == 3
    assert abs(row["value"] - 10 / 12) < 1e-12  # class C's 4 errors are left out


def test_macro_f1_by_hand():
    t = ref.reference(preds(), {"A1", "A2", "B"}, n_boot=0)
    # among the 12 scored samples: A1 tp 3, 4 true, 3 predicted; A2 tp 3, 4 true,
    # 4 predicted; B perfect
    want = np.mean([2 * 3 / 7, 2 * 3 / 8, 1.0])
    assert abs(get(t, "class", "macro_f1")["value"] - want) < 1e-12


def test_family_level_counts_a_wrong_class_in_the_right_family_as_correct():
    t = ref.reference(preds(), {"A1", "A2", "B"}, n_boot=0)
    row = get(t, "family", "accuracy")
    assert row["n_groups"] == 2 and abs(row["value"] - 11 / 12) < 1e-12


def test_intervals_bracket_the_value_and_repeat():
    a = ref.reference(preds(), {"A1", "A2", "B"}, n_boot=200)
    b = ref.reference(preds(), {"A1", "A2", "B"}, n_boot=200)
    assert a.equals(b)
    assert (a["ci_low"] <= a["value"]).all() and (a["value"] <= a["ci_high"]).all()


def test_stops_on_a_validation_class_missing_from_cv():
    try:
        ref.reference(preds(), {"A1", "Z"}, n_boot=0)
    except SystemExit as e:
        assert "no CV samples" in str(e)
    else:
        raise AssertionError("expected a stop")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
