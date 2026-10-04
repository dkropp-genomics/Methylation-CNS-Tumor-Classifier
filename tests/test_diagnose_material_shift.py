"""Tests for scripts/diagnose_material_shift.py on synthetic data.

Run with pytest, or directly:  python tests/test_diagnose_material_shift.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "diagnose_material_shift", ROOT / "scripts" / "diagnose_material_shift.py")
dms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dms)

P = 400          # probes
COMMON = 0.08    # shift shared by every class on probes 0..49
ODD = 0.60       # shift present in ONE class only, on probe 399


def make_data(seed=0, n_classes=8, per=(12, 10)):
    """8 classes with both materials, plus one all-FFPE and one 1-frozen class."""
    rng = np.random.default_rng(seed)
    X, y, m = [], [], []
    def add(name, n_f, n_z, odd=False):
        base = rng.uniform(0.2, 0.7, P)
        for mat, n in (("FFPE", n_f), ("Frozen", n_z)):
            b = base + rng.normal(0, 0.02, (n, P))
            if mat == "FFPE":
                b[:, :50] += COMMON
                if odd:
                    b[:, 399] += ODD
            X.append(b); y.extend([name] * n); m.extend([mat] * n)
    for i in range(n_classes):
        add(f"C{i}", per[0], per[1], odd=(i == 3))
    add("ALLFFPE", 15, 0)
    add("ONEFROZEN", 9, 1)
    X = np.clip(np.vstack(X), 0, 1).astype(np.float32)
    return X, np.array(y, dtype=object), np.array(m, dtype=object)


PROBES = np.array([f"cg{i:08d}" for i in range(P)], dtype=object)


def test_only_classes_with_both_materials_are_used():
    X, y, m = make_data()
    classes, D = dms.per_class_differences(X, y, m)
    assert list(classes["mc_class"]) == [f"C{i}" for i in range(8)]
    assert D.shape == (8, P)
    assert (classes["n_ffpe"] == 12).all() and (classes["n_frozen"] == 10).all()
    # raising the minimum above the smaller group removes every class -> stop
    try:
        dms.per_class_differences(X, y, m, min_per_material=11)
    except SystemExit as e:
        assert "ERROR" in str(e)
    else:
        raise AssertionError("expected a stop when no class qualifies")


def test_shift_matches_material_corrector():
    from methylclf.features import MaterialCorrector
    X, y, m = make_data()
    classes, D = dms.per_class_differences(X, y, m)
    shift = dms.pooled_shift(D, classes["weight"])
    fitted = MaterialCorrector("FFPE").fit(X, y, m)
    assert np.allclose(shift, fitted.shift_, atol=1e-5)
    assert abs(shift[:50].mean() - COMMON) < 0.01


def test_one_class_probe_is_flagged_and_shared_probes_are_not():
    X, y, m = make_data()
    with tempfile.TemporaryDirectory() as d:
        res = dms.run(X, y, m, PROBES, d, top=60, skip_model=True)
        for f in ("class_table.tsv", "top_probes.tsv", "top_probes_by_class.tsv",
                  "summary.tsv"):
            assert (Path(d) / f).exists(), f
    p = res["probes"].set_index("Probe_ID")
    odd = p.loc["cg00000399"]
    assert odd["top_class"] == "C3"
    assert odd["top_class_share"] > 0.9
    assert abs(odd["shift"] - ODD / 8) < 0.02          # diluted over 8 classes
    assert abs(odd["shift_without_top_class"]) < 0.02  # gone without C3
    assert odd["n_same_sign_gt_0.05"] == 1
    shared = p.loc[[f"cg{i:08d}" for i in range(50)]]
    assert (shared["n_same_sign"] == 8).all()
    assert (shared["top_class_share"] < 0.3).all()
    assert (shared["shift_without_top_class"] > 0.06).all()
    c = res["classes"]
    c = c.set_index("mc_class")["corr_with_other_classes"]
    assert (c.drop("C3") > 0.8).all() and c.idxmin() == "C3"
    assert abs(res["classes"]["weight_share"].sum() - 1) < 1e-9
    assert res["wide"].shape == (60, 9)


def test_material_model_drops_to_chance_when_shift_is_shared():
    X, y, m = make_data(seed=1, n_classes=10, per=(20, 20))
    X[:, 399] = X[:, 398]  # remove the one-class probe: the shift is now fully shared
    t = dms.material_model(X, y, m, [f"C{i}" for i in range(10)], n_probes=P)
    assert len(t) == 10 and set(t["mc_class"]) == {f"C{i}" for i in range(10)}
    w = t["weight"]
    assert np.average(t["auc_off"], weights=w) > 0.95
    assert 0.3 < np.average(t["auc_on"], weights=w) < 0.7


def test_material_model_shows_over_and_under_correction():
    """Same direction of shift in every class but a different size: one
    average shift cannot fit them all."""
    X, y, m = make_data(seed=2, n_classes=10, per=(20, 20))
    for i in range(10):  # a different sign pattern per class on probes 100..199
        rows = np.flatnonzero((y == f"C{i}") & (m == "FFPE"))
        scale = 0.05 + 0.02 * i                 # same direction, different size
        X[np.ix_(rows, np.arange(100, 200))] += np.float32(scale)
    X = np.clip(X, 0, 1)
    t = dms.material_model(X, y, m, [f"C{i}" for i in range(10)], n_probes=P)
    # after removing the average, small-shift classes are over-corrected and
    # large-shift classes under-corrected: AUC leaves 0.5 in both directions
    a = t.set_index("mc_class")["auc_on"]
    assert a["C0"] < 0.2 and a["C1"] < 0.2
    assert a["C8"] > 0.8 and a["C9"] > 0.8
    assert (t["auc_off"] > 0.95).all()


def test_full_run_writes_model_table_and_summary():
    X, y, m = make_data(seed=3)
    with tempfile.TemporaryDirectory() as d:
        res = dms.run(X, y, m, PROBES, d, top=60, model_probes=P)
        assert (Path(d) / "material_model.tsv").exists()
    s = dict(zip(res["summary"]["item"], res["summary"]["value"]))
    assert s["n_classes_used"] == 8
    assert s["max_abs_diff_vs_MaterialCorrector"] < 1e-4
    assert s["top_n_one_class_gives_over_half"] >= 1
    assert "model_auc_on_weighted" in s
    dms.report(res, show=5)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
