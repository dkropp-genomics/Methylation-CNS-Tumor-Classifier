"""Tests for methylclf.features on small synthetic matrices.

Run with `python -m pytest tests/test_features.py` or `python tests/test_features.py`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from methylclf.features import (  # noqa: E402
    MedianImputer, MissingnessFilter, TopVarianceSelector, make_feature_pipeline)


def toy(n=100, p=40, seed=0, frac_nan=0.02):
    rng = np.random.default_rng(seed)
    scale = np.linspace(0.01, 0.2, p)  # later columns vary more
    X = (0.5 + rng.normal(size=(n, p)) * scale).astype(np.float32)
    X[rng.random((n, p)) < frac_nan] = np.nan
    return X


def expect_error(fn, text):
    try:
        fn()
    except ValueError as e:
        assert str(e).startswith("ERROR:") and text in str(e), e
        return
    raise AssertionError(f"expected an error containing '{text}'")


def test_missingness_threshold_is_exact_and_fit_only():
    X = np.full((100, 4), 0.5, dtype=np.float32)
    X[:5, 1] = np.nan   # exactly 5% -> kept
    X[:6, 2] = np.nan   # 6%        -> dropped
    X[:, 3] = np.nan    # all       -> dropped
    f = MissingnessFilter(0.05).fit(X)
    assert list(f.keep_) == [0, 1]
    # test rows that are entirely missing in column 0 do not change the decision
    T = np.full((20, 4), np.nan, dtype=np.float32)
    assert f.transform(T).shape == (20, 2)
    assert list(f.get_feature_names_out(["a", "b", "c", "d"])) == ["a", "b"]


def test_imputer_uses_fit_medians_not_the_new_samples():
    X = toy(101, 30)
    imp = MedianImputer().fit(X)
    assert np.allclose(imp.medians_, np.nanmedian(X, axis=0), atol=1e-7)
    T = np.full((3, 30), 0.9, dtype=np.float32)   # a very different "test fold"
    T[0, :] = np.nan
    out = imp.transform(T)
    assert out.dtype == np.float32 and not np.isnan(out).any()
    assert np.array_equal(out[0], imp.medians_)       # filled with TRAINING medians
    assert np.array_equal(out[1:], T[1:])             # observed values untouched
    assert np.isnan(T[0]).all()                       # copy=True left the input alone


def test_imputer_even_count_and_in_place():
    X = np.array([[0.1], [0.4], [np.nan], [0.2], [0.3]], dtype=np.float32)  # 4 observed
    imp = MedianImputer(copy=False).fit(X)
    assert np.isclose(imp.medians_[0], 0.25)
    out = imp.transform(X)
    assert out is X and np.isclose(X[2, 0], 0.25)


def test_imputer_refuses_all_missing_column_and_wrong_width():
    X = toy(20, 5, frac_nan=0)
    X[:, 2] = np.nan
    expect_error(lambda: MedianImputer().fit(X), "missing in every fit sample")
    imp = MedianImputer().fit(toy(20, 5, frac_nan=0))
    expect_error(lambda: imp.transform(toy(4, 6)), "fitted on 5 probes")


def test_selector_picks_highest_variance_in_column_order():
    X = toy(200, 40, frac_nan=0)
    s = TopVarianceSelector(k=5).fit(X)
    want = np.sort(np.argsort(-X.var(axis=0, ddof=1, dtype=np.float64))[:5])
    assert np.array_equal(s.keep_, want)
    assert np.array_equal(s.transform(X), X[:, want])
    expect_error(lambda: TopVarianceSelector(k=41).fit(X), "probes are available")
    Xn = X.copy(); Xn[0, 0] = np.nan
    expect_error(lambda: TopVarianceSelector(k=5).fit(Xn), "run MedianImputer first")


def test_selector_ties_are_deterministic():
    X = np.tile(np.array([[0.0], [1.0]], dtype=np.float32), (10, 6))  # 6 identical columns
    assert list(TopVarianceSelector(k=3).fit(X).keep_) == [0, 1, 2]


def test_pipeline_poison_test_rows_cannot_change_what_was_fit():
    """The leakage test. Fit on training rows; then replace the test rows with
    wild values and fit again. Nothing learned may differ."""
    X = toy(120, 60, seed=3)
    train, test = np.arange(90), np.arange(90, 120)
    names = np.array([f"cg{i:04d}" for i in range(60)])

    def run(X):
        pipe = make_feature_pipeline(n_probes=10).fit(X[train])
        return (pipe.get_feature_names_out(names), pipe.transform(X[train]),
                pipe.named_steps["impute"].medians_)

    Xp = X.copy()
    Xp[test] = np.random.default_rng(9).uniform(-50, 50, size=(30, 60)).astype(np.float32)
    Xp[test, :20] = np.nan
    a, b = run(X), run(Xp)
    assert list(a[0]) == list(b[0]) and len(a[0]) == 10
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_fitting_on_all_rows_does_change_the_result():
    """The contrast: when the 'test' rows are included in the fit, they DO
    change the selected probes. This is what leakage looks like."""
    X = toy(120, 60, seed=3, frac_nan=0)
    X[90:, :10] += np.random.default_rng(1).normal(size=(30, 10)).astype(np.float32)
    names = np.array([f"cg{i:04d}" for i in range(60)])
    inside = make_feature_pipeline(n_probes=10).fit(X[:90]).get_feature_names_out(names)
    leaky = make_feature_pipeline(n_probes=10).fit(X).get_feature_names_out(names)
    assert set(inside) != set(leaky)


def test_pipeline_output_is_float32_complete_and_right_shape():
    X = toy(80, 50, seed=5, frac_nan=0.03)
    X[:, 7] = np.nan
    pipe = make_feature_pipeline(n_probes=12).fit(X[:60])
    out = pipe.transform(X[60:])
    assert out.shape == (20, 12) and out.dtype == np.float32
    assert not np.isnan(out).any()
    assert pipe.named_steps["missing"].n_features_in_ == 50


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
