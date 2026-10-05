"""Tests for methylclf.masking. Run with pytest, or: python tests/test_masking.py"""

import numpy as np

from methylclf.masking import observed, observed_uniform, probe_positions

IDS = np.array([f"GSM{i}" for i in range(40)], dtype=object)
ARRAY = np.array([f"cg{i:06d}" for i in range(5000)], dtype=object)


def test_same_sample_same_subset_whatever_else_is_asked():
    cols = np.arange(0, 5000, 7)
    a = observed_uniform(IDS, 5000, cols, seed=1)
    b = observed_uniform(IDS[::-1], 5000, cols, seed=1)[::-1]  # other order
    c = observed_uniform(IDS[:5], 5000, cols, seed=1)  # other companions
    d = observed_uniform(IDS, 5000, np.arange(5000), seed=1)[:, cols]  # other features
    assert np.array_equal(a, b) and np.array_equal(a[:5], c) and np.array_equal(a, d)
    assert not np.array_equal(a, observed_uniform(IDS, 5000, cols, seed=2))
    assert not np.array_equal(a[0], a[1])  # samples differ


def test_levels_are_nested_and_have_the_right_share():
    u = observed_uniform(IDS, 5000, np.arange(5000), seed=3)
    masks = {lv: observed(u, lv) for lv in (1.0, 0.1, 0.01, 0.001)}
    assert masks[1.0].all()
    for small, big in ((0.001, 0.01), (0.01, 0.1), (0.1, 1.0)):
        assert not (masks[small] & ~masks[big]).any()
    for lv in (0.1, 0.01, 0.001):
        assert abs(masks[lv].mean() - lv) < 0.2 * lv
    try:
        observed(u, 0)
    except SystemExit as e:
        assert "ERROR" in str(e)
    else:
        raise AssertionError("level 0 should stop")


def test_probe_positions():
    names = ["cg000003", "cg004999", "cg000000"]
    assert list(probe_positions(names, ARRAY)) == [3, 4999, 0]
    try:
        probe_positions(["nope"], ARRAY)
    except SystemExit as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("an unknown probe should stop")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
