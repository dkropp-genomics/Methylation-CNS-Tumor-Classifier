"""Tests for methylclf.metrics on small hand-checkable inputs.

Run with `python -m pytest tests/test_metrics.py` or `python tests/test_metrics.py`.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from methylclf.metrics import (  # noqa: E402
    confusion_table, reliability_table, score, summarize, to_family)

CLASSES = ["A", "B", "C"]


def expect_error(fn, text):
    try:
        fn()
    except ValueError as e:
        assert str(e).startswith("ERROR:") and text in str(e), e
        return
    raise AssertionError(f"expected an error containing '{text}'")


def onehot(pred, k=3, top=1.0):
    p = np.full((len(pred), k), (1 - top) / (k - 1))
    p[np.arange(len(pred)), pred] = top
    return p


def random_case(n=400, k=6, seed=0, skill=2.0):
    rng = np.random.default_rng(seed)
    classes = [f"c{i}" for i in range(k)]
    t = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), t] += skill
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return [classes[i] for i in t], p, classes


def test_by_hand():
    # true: A A A A B B C C ; predicted: A A A B B B C A
    y = list("AAAABBCC")
    p = onehot([0, 0, 0, 1, 1, 1, 2, 0])
    s = score(y, p, CLASSES)
    # F1: A = 2*3/(4+4) = .75, B = 2*2/(2+3) = .8, C = 2*1/(2+1) = .6667
    assert np.isclose(s["macro_f1"], (0.75 + 0.8 + 2 / 3) / 3)
    assert np.isclose(s["balanced_accuracy"], (0.75 + 1.0 + 0.5) / 3)
    assert np.isclose(s["accuracy"], 6 / 8)
    assert np.isclose(s["brier"], 2 * 2 / 8)        # each wrong one-hot row costs 2
    assert np.isclose(s["ece"], 2 / 8)              # confidence 1.0, accuracy 0.75
    assert s["confident_share"] == 1.0 and np.isclose(s["confident_accuracy"], 0.75)
    cm = confusion_table(y, p, CLASSES)
    assert cm.loc["A"].tolist() == [3, 1, 0] and cm.loc["C"].tolist() == [1, 0, 1]


def test_matches_sklearn_on_random_data():
    y, p, classes = random_case()
    pred = [classes[i] for i in p.argmax(axis=1)]
    s = score(y, p, classes)
    assert np.isclose(s["macro_f1"], f1_score(y, pred, average="macro", labels=classes))
    assert np.isclose(s["balanced_accuracy"], balanced_accuracy_score(y, pred))


def test_classes_absent_from_y_true_are_left_out_of_macro_averages():
    # C never occurs as a true label, but is predicted once
    y = list("AAABBB")
    p = onehot([0, 0, 2, 1, 1, 1])
    s = score(y, p, CLASSES)
    assert np.isclose(s["macro_f1"], (0.8 + 1.0) / 2)       # over A and B only
    assert np.isclose(s["balanced_accuracy"], (2 / 3 + 1.0) / 2)


def test_brier_and_ece_by_hand():
    y = ["A", "A", "B", "B"]
    p = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.2, 0.8]])
    s = score(y, p, ["A", "B"])
    # Brier: three right at .8 -> .08 each; one wrong at .8 -> 1.28
    assert np.isclose(s["brier"], (3 * 0.08 + 1.28) / 4)
    assert np.isclose(s["ece"], abs(0.75 - 0.8))            # all four in one bin
    assert s["confident_share"] == 0.0 and np.isnan(s["confident_accuracy"])
    r = reliability_table(y, p, ["A", "B"])
    assert r["n"].sum() == 4 and r.loc[r["n"] > 0, "accuracy"].iloc[0] == 0.75


def test_well_calibrated_scores_have_low_ece_and_overconfident_ones_do_not():
    rng = np.random.default_rng(1)
    n = 20000
    conf = rng.uniform(0.5, 1.0, n)
    right = rng.random(n) < conf                             # correct with probability = conf
    y = ["A"] * n
    p = np.column_stack([np.where(right, conf, 1 - conf), np.where(right, 1 - conf, conf)])
    assert score(y, p, ["A", "B"])["ece"] < 0.02
    sharp = np.where(p > 0.5, 0.99, 0.01)                    # same decisions, inflated scores
    assert score(y, sharp, ["A", "B"])["ece"] > 0.15


def test_threshold_report():
    y = list("AABB")
    p = np.array([[0.95, 0.05, 0], [0.6, 0.4, 0], [0.92, 0.08, 0], [0.3, 0.7, 0]])
    s = score(y, p, CLASSES)
    assert s["confident_share"] == 0.5 and s["confident_accuracy"] == 0.5


def test_family_scores_are_sums_and_can_rescue_within_family_errors():
    family_of = {"A": "F1", "B": "F1", "C": "F2"}
    p = np.array([[0.4, 0.35, 0.25], [0.3, 0.3, 0.4]])
    fam, names = to_family(p, CLASSES, family_of)
    assert names == ["F1", "F2"] and np.allclose(fam, [[0.75, 0.25], [0.6, 0.4]])
    # row 2: class call is C (wrong for a true B), family call is F1 (right)
    out = summarize(["A", "B"], p, CLASSES, family_of=family_of, n_boot=20)
    get = lambda lvl, m: out[(out.level == lvl) & (out.metric == m)]["value"].iloc[0]
    assert get("class", "accuracy") == 0.5 and get("family", "accuracy") == 1.0
    expect_error(lambda: to_family(p, CLASSES, {"A": "F1"}), "have no family")


def test_bootstrap_interval_contains_the_value_is_repeatable_and_narrows_with_n():
    y, p, classes = random_case(n=300)
    a = summarize(y, p, classes, n_boot=300, seed=7)
    b = summarize(y, p, classes, n_boot=300, seed=7)
    assert a.equals(b)
    assert ((a.ci_low <= a.value) & (a.value <= a.ci_high)).all()
    assert list(a.metric) == ["macro_f1", "balanced_accuracy", "accuracy", "brier", "ece",
                              "confident_share", "confident_accuracy"]
    width = lambda t: (t.ci_high - t.ci_low)[t.metric == "accuracy"].iloc[0]
    y2, p2, _ = random_case(n=3000)
    assert width(summarize(y2, p2, classes, n_boot=300)) < width(a) / 2


def test_groups_give_separate_rows_that_match_scoring_the_subset():
    y, p, classes = random_case(n=300)
    material = np.where(np.arange(300) % 3 == 0, "Frozen", "FFPE")
    out = summarize(y, p, classes, groups=material, n_boot=50)
    assert sorted(out.group.unique()) == ["FFPE", "Frozen", "all"]
    m = material == "Frozen"
    want = score([v for v, k in zip(y, m) if k], p[m], classes)["macro_f1"]
    got = out[(out.group == "Frozen") & (out.metric == "macro_f1")]["value"].iloc[0]
    assert np.isclose(got, want) and out[out.group == "Frozen"]["n"].iloc[0] == 100


def test_errors():
    p = onehot([0, 1])
    expect_error(lambda: score(["A", "Z"], p, CLASSES), "not among the model's classes")
    expect_error(lambda: score(["A"], p, CLASSES), "true labels but")
    expect_error(lambda: score(["A", "B"], p[:, :2], CLASSES), "expected (n_samples, 3)")
    expect_error(lambda: score(["A", "B"], p * 0.5, CLASSES), "do not sum to 1")
    expect_error(lambda: summarize(["A", "B"], p, CLASSES, groups=["x"], n_boot=5), "group values")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
