"""Tests for methylclf.features on small synthetic matrices.

Run with `python -m pytest tests/test_features.py` or `python tests/test_features.py`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from methylclf.features import (  # noqa: E402
    FeaturePipeline,
    MaterialCorrector,
    MedianImputer,
    MissingnessFilter,
    TopVarianceSelector,
)


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
    X[:5, 1] = np.nan  # exactly 5% -> kept
    X[:6, 2] = np.nan  # 6%        -> dropped
    X[:, 3] = np.nan  # all       -> dropped
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
    T = np.full((3, 30), 0.9, dtype=np.float32)  # a very different "test fold"
    T[0, :] = np.nan
    out = imp.transform(T)
    assert out.dtype == np.float32 and not np.isnan(out).any()
    assert np.array_equal(out[0], imp.medians_)  # filled with TRAINING medians
    assert np.array_equal(out[1:], T[1:])  # observed values untouched
    assert np.isnan(T[0]).all()  # copy=True left the input alone


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
    Xn = X.copy()
    Xn[0, 0] = np.nan
    expect_error(lambda: TopVarianceSelector(k=5).fit(Xn), "run MedianImputer first")


def test_selector_ties_are_deterministic():
    X = np.tile(np.array([[0.0], [1.0]], dtype=np.float32), (10, 6))  # 6 identical columns
    assert list(TopVarianceSelector(k=3).fit(X).keep_) == [0, 1, 2]


def toy_material(seed=0, p=30, shift=0.08):
    """6 classes x 40 samples. Classes A-C have both materials, D-E are all
    FFPE, F is all frozen. Class means differ a lot; FFPE adds `shift` to
    the first 10 probes. Returns X, y, material, true shift."""
    rng = np.random.default_rng(seed)
    y = np.repeat(list("ABCDEF"), 40).astype(object)
    material = np.empty(240, dtype=object)
    for c, frac in zip("ABCDEF", [0.5, 0.25, 0.75, 1.0, 1.0, 0.0]):
        idx = np.flatnonzero(y == c)
        material[idx] = np.where(np.arange(40) < 40 * frac, "FFPE", "frozen")
    class_mean = {c: rng.uniform(0.2, 0.7, p) for c in "ABCDEF"}
    class_mean["D"] = class_mean["D"] * 0 + 0.75  # the all-FFPE classes sit high:
    class_mean["E"] = class_mean["E"] * 0 + 0.75  # the confound
    d = np.zeros(p)
    d[:10] = shift
    X = np.stack([class_mean[c] for c in y]) + rng.normal(0, 0.02, (240, p))
    X += (material == "FFPE")[:, None] * d
    return X.astype(np.float32), y, material, d


def test_corrector_weights_by_hand():
    # class A: 2 FFPE, 2 frozen, difference 0.2 (weight 1.0)
    # class B: 2 FFPE, 6 frozen, difference 0.4 (weight 1.5)  -> (0.2 + 0.6) / 2.5 = 0.32
    X = np.array([[0.7], [0.7], [0.5], [0.5]] + [[0.8], [0.8]] + [[0.4]] * 6, dtype=np.float32)
    y = np.array(list("AAAA") + list("BBBBBBBB"), dtype=object)
    m = np.array(
        ["FFPE", "FFPE", "frozen", "frozen", "FFPE", "FFPE"] + ["frozen"] * 6, dtype=object
    )
    mc = MaterialCorrector().fit(X, y, m)
    assert np.isclose(mc.shift_[0], 0.32) and mc.classes_used_ == ["A", "B"]


def test_corrector_recovers_the_shift_where_the_plain_difference_does_not():
    X, y, m, d = toy_material()
    mc = MaterialCorrector().fit(X, y, m)
    assert mc.classes_used_ == ["A", "B", "C"]  # D, E, F have one material
    assert np.abs(mc.shift_ - d).max() < 0.02  # within-class: close to the truth
    plain = X[m == "FFPE"].mean(axis=0) - X[m == "frozen"].mean(axis=0)
    assert np.abs(plain - d).max() > 0.1  # plain: polluted by class
    out = mc.transform(X, m)
    for c in "ABC":  # after: materials agree within class
        gap = out[(y == c) & (m == "FFPE")].mean(0) - out[(y == c) & (m == "frozen")].mean(0)
        assert np.abs(gap).max() < 0.03


def test_corrector_leaves_frozen_alone_clips_and_needs_no_labels_to_transform():
    X, y, m, _ = toy_material()
    mc = MaterialCorrector().fit(X, y, m)
    out = mc.transform(X, m)  # note: no y passed
    assert out.dtype == np.float32
    assert np.array_equal(out[m == "frozen"], X[m == "frozen"])
    low = np.zeros((1, X.shape[1]), dtype=np.float32)  # 0 minus a positive shift
    assert mc.transform(low, np.array(["FFPE"], dtype=object)).min() == 0.0
    assert out is not X and mc.set_params(copy=False).transform(X, m) is X


def test_corrector_errors():
    X, y, m, _ = toy_material()
    expect_error(lambda: MaterialCorrector().fit(X, y), "material is required")
    expect_error(lambda: MaterialCorrector().fit(X, None, m), "class labels y are required")
    expect_error(lambda: MaterialCorrector().fit(X, y, m[:-1]), "material values")
    expect_error(
        lambda: MaterialCorrector(ffpe_label="ffpe").fit(X, y, m), "found ['FFPE', 'frozen']"
    )
    only = np.isin(y, list("DEF"))  # no class with both materials
    expect_error(lambda: MaterialCorrector().fit(X[only], y[only], m[only]), "cannot be estimated")
    mc = MaterialCorrector().fit(X, y, m)
    expect_error(lambda: mc.transform(X[:1], np.array(["fresh"], dtype=object)), "unknown material")


NAMES = np.array([f"cg{i:04d}" for i in range(60)])


def _fitted(X, y, m, rows, on):
    pipe = FeaturePipeline(n_probes=10, correct_material=on).fit(X[rows], y[rows], m[rows])
    extra = pipe.correct_.shift_ if on else pipe.impute_.medians_
    return pipe.get_feature_names_out(NAMES), pipe.transform(X[rows], m[rows]), extra


def test_pipeline_poison_test_rows_cannot_change_what_was_fit():
    """The leakage test, with correction off and on. Fit on training rows;
    then replace the test rows (values, labels, material) with wild ones and
    fit again. Nothing learned may differ."""
    X, y, m, _ = toy_material(seed=3, p=60)
    X[np.random.default_rng(4).random(X.shape) < 0.02] = np.nan
    order = np.random.default_rng(5).permutation(240)
    train, test = order[:180], order[180:]
    Xp, yp, mp = X.copy(), y.copy(), m.copy()
    Xp[test] = np.random.default_rng(9).uniform(-50, 50, (60, 60)).astype(np.float32)
    Xp[test, :20] = np.nan
    yp[test], mp[test] = "A", "frozen"
    for on in (False, True):
        a, b = _fitted(X, y, m, train, on), _fitted(Xp, yp, mp, train, on)
        assert list(a[0]) == list(b[0]) and len(a[0]) == 10
        assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_fitting_on_all_rows_does_change_the_result():
    """The contrast: when the 'test' rows are included in the fit, they DO
    change the selected probes. This is what leakage looks like."""
    X = toy(120, 60, seed=3, frac_nan=0)
    X[90:, :10] += np.random.default_rng(1).normal(size=(30, 10)).astype(np.float32)
    inside = FeaturePipeline(n_probes=10).fit(X[:90]).get_feature_names_out(NAMES)
    leaky = FeaturePipeline(n_probes=10).fit(X).get_feature_names_out(NAMES)
    assert set(inside) != set(leaky)


def test_pipeline_off_equals_the_three_steps_by_hand():
    X = toy(80, 50, seed=5, frac_nan=0.03)
    X[:, 7] = np.nan
    pipe = FeaturePipeline(n_probes=12).fit(X[:60])
    out = pipe.transform(X[60:])
    f = MissingnessFilter().fit(X[:60])
    Z = f.transform(X[:60])
    i = MedianImputer().fit(Z)
    s = TopVarianceSelector(12).fit(i.transform(Z))
    want = s.transform(i.transform(f.transform(X[60:])))
    assert np.array_equal(out, want)
    assert out.shape == (20, 12) and out.dtype == np.float32 and not np.isnan(out).any()
    assert pipe.correct_ is None and pipe.n_features_in_ == 50
    assert np.isnan(X[:60]).any()  # the caller's array was not modified


def test_pipeline_on_requires_material_and_changes_ffpe_rows_only():
    X, y, m, _ = toy_material(p=60)
    expect_error(
        lambda: FeaturePipeline(n_probes=10, correct_material=True).fit(X, y),
        "material is required",
    )
    on = FeaturePipeline(n_probes=60, correct_material=True).fit(X, y, m)
    off = FeaturePipeline(n_probes=60).fit(X, y, m)
    a, b = on.transform(X, m), off.transform(X, m)
    assert np.array_equal(a[m == "frozen"], b[m == "frozen"])
    assert not np.array_equal(a[m == "FFPE"], b[m == "FFPE"])
    expect_error(lambda: on.transform(X), "material is required")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
