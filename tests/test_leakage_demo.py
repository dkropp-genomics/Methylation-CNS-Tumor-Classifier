"""The leakage test on toy data: pure noise, so the honest answer is chance.

Run with `python -m pytest tests/test_leakage_demo.py` or `python tests/test_leakage_demo.py`.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_selection import f_classif

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from leakage_demo import f_statistic, run_demo, shuffle_within_folds, small_noise_case  # noqa: E402


def noise(n=120, p=3000, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, p)).astype(np.float32)
    X[rng.random((n, p)) < 0.01] = np.nan
    y = np.array([f"class{i % n_classes}" for i in range(n)], dtype=object)
    fold = (np.arange(n) // n_classes) % 3          # every fold has every class
    return X, y, fold


def test_f_statistic_matches_sklearn():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(60, 50)).astype(np.float32)
    codes = np.arange(60) % 5
    assert np.allclose(f_statistic(Z, codes, 5), f_classif(Z, codes)[0], rtol=1e-4)


def test_shuffle_keeps_class_counts_in_every_fold():
    _, y, fold = noise()
    s = shuffle_within_folds(y, fold, seed=3)
    assert (s != y).any()
    for f in range(3):
        assert sorted(s[fold == f]) == sorted(y[fold == f])


def test_small_noise_case_is_balanced_in_every_fold():
    rows, y, fold = small_noise_case(2000, n=150, n_classes=3, seed=1)
    assert rows.size == 150 and np.unique(rows).size == 150 and rows.max() < 2000
    for f in range(3):
        assert np.unique(y[fold == f], return_counts=True)[1].tolist() == [17, 17, 16] or \
            sorted(np.unique(y[fold == f], return_counts=True)[1].tolist()) in ([16, 17, 17], [17, 17, 17], [16, 16, 17], [16, 16, 16])


def test_on_pure_noise_the_leaky_arm_scores_above_chance_and_the_inside_arm_does_not():
    X, y, fold = noise()
    per_fold, summary = run_demo(X, y, fold, k=20, n_trees=100, n_jobs=1, seed=0,
                                 log=lambda *_: None)
    assert len(per_fold) == 3 * 2 * 3               # experiments x arms x folds
    row = summary.set_index("experiment").loc["F-statistic, real labels"]
    # 4 balanced classes of noise: chance is 0.25
    assert row["accuracy_leaky"] > 0.45, row
    assert row["accuracy_inside"] < 0.38, row
    assert row["accuracy_gap"] > 0.15
    # variance selection uses no labels, so on noise neither arm can beat chance
    var = summary.set_index("experiment").loc["variance, real labels"]
    assert var["accuracy_leaky"] < 0.40 and var["accuracy_inside"] < 0.40


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
