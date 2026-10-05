"""Tests for scripts/score_validation.py: synthetic training and "validation" cohorts."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import fit_final as ff  # noqa: E402
import score_validation as sv  # noqa: E402
import test_fit_final as tff  # noqa: E402
from methylclf.features import FeaturePipeline  # noqa: E402
from methylclf.metrics import summarize, to_family  # noqa: E402

FAMILY = {"C0": "F01", "C1": "F01", "C2": "C2", "C3": "C3", "C4": "C4", "C5": "C5"}


def make_validation(n=60, n_probes=300, seed=5):
    """New samples from the same class centers as tff.make_data(seed=0)."""
    centers = np.random.default_rng(0).uniform(0.2, 0.8, size=(6, n_probes))
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, 5, n)                              # class C5 never occurs
    X = np.clip(centers[cls] + rng.normal(0, 0.08, size=(n, n_probes)), 0, 1).astype(np.float32)
    X[rng.random(X.shape) < 0.01] = np.nan
    ids = np.array([f"V{i:03d}" for i in range(n)], dtype=object)
    score = rng.uniform(0.5, 1.0, n)
    kept = pd.DataFrame({
        "geo_accession": ids, "zarr_row": np.arange(n),
        "mc_class": [f"C{i}" for i in cls], "family": [FAMILY[f"C{i}"] for i in cls],
        "material": np.where(np.arange(n) % 6 == 0, "Frozen", "FFPE"),
        "capper_score": [f"{v:.3f}" for v in score],
        "capper_matched": [str(v >= 0.9) for v in score],
        "path_concordant": [str(v >= 0.95) for v in score],
        "capper_no_match": [str(v < 0.9) for v in score]})
    row_of = {s: i for i, s in enumerate(ids)}
    return kept, (lambda want: X[[row_of[s] for s in want]])


def setup(d: Path):
    """Fit the final models on the synthetic training cohort."""
    samples, load, probe_ids, _ = tff.make_data()
    cfg, model_cfgs = tff.make_cfg(tff.has_torch())
    tff.make_runs(d / "cv", tff.has_torch())
    cfg.update({
        "validation_cohort": "VAL",
        "labels": {"subsets": ["capper_matched", "path_concordant", "capper_no_match"]},
        "target": {"model": "rf", "scores": "calibrated", "level": "family",
                   "metric": "macro_f1", "max_drop": 0.05, "reference": "unused"}})
    cfg["analyses"].update({"full_coverage": list(cfg["models"]), "headline": "rf",
                            "confident_share": {"threshold": 0.9, "cv_value": 0.9},
                            "n_boot": 20})
    tff.do_run(d, samples, load, probe_ids, cfg, model_cfgs)
    return cfg, probe_ids


def score(d, cfg, kept, load, probe_ids, final_scoring=True):
    return sv.run("score", cfg, kept, load, probe_ids, d / "pred", d / "res", d / "models",
                  final_scoring=final_scoring)


def report(d, cfg, kept, reference=None):
    return sv.run("report", cfg, kept, pred_root=d / "pred", res_root=d / "res",
                  family_of=FAMILY, summarize=summarize, to_family=to_family,
                  reference=reference)


REFERENCE = pd.DataFrame([{"run": "rf_t", "level": "family", "metric": "macro_f1",
                           "value": 0.99, "n_groups": 4}])


def test_scoring_is_refused_without_the_flag():
    kept, load = make_validation()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cfg, probe_ids = setup(d)
        try:
            score(d, cfg, kept, load, probe_ids, final_scoring=False)
        except SystemExit as e:
            assert "--final-scoring" in str(e)
        else:
            raise AssertionError("expected a stop")
        assert not list((d / "pred" / "final_t").glob("val_*.npz"))


def test_score_then_report():
    kept, load = make_validation()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cfg, probe_ids = setup(d)
        assert score(d, cfg, kept, load, probe_ids) == len(cfg["models"])
        metrics, top, diffs, target, per_class, per_sample = report(d, cfg, kept, REFERENCE)
        res = d / "res" / "final_t"
        for name in ("validation_metrics.tsv", "validation_top_class.tsv",
                     "validation_target.tsv", "validation_per_class.tsv",
                     "validation_predictions.tsv", "validation_sparsity_curve.png",
                     "validation_sparsity_curve_family.png", "validation_scored.json"):
            assert (res / name).exists(), name

        head = metrics[(metrics["model"] == "rf") & (metrics["scores"] == "calibrated")
                       & (metrics["subset"] == "all") & (metrics["group"] == "all")]
        acc = head[(head["level"] == "class") & (head["metric"] == "accuracy")].iloc[0]
        assert acc["n"] == len(kept) and acc["value"] > 0.9       # an easy synthetic task
        assert acc["n_classes"] == kept["mc_class"].nunique()     # 5 of the 6 classes occur
        # the saved per-sample table gives the same number
        same = (per_sample["mc_class"] == per_sample["predicted_class"]).mean()
        assert abs(same - acc["value"]) < 1e-12

        # family scores only for calibrated scores; no calibration metrics for the centroid
        fam = metrics[metrics["level"] == "family"]
        assert set(fam["scores"]) == {"calibrated"} and set(fam["model"]) == {"rf"}
        assert set(metrics.loc[metrics["model"] == "centroid", "metric"]) == set(sv.ORDER_METRICS)

        # subsets hold the right samples; material groups only for "all"
        n_by = metrics[
            (metrics["model"] == "rf") & (metrics["scores"] == "raw")
            & (metrics["metric"] == "accuracy") & (metrics["group"] == "all")]
        n_by = n_by.set_index("subset")["n"]
        assert n_by["capper_matched"] + n_by["capper_no_match"] == len(kept)
        assert n_by["path_concordant"] == (kept["path_concordant"] == "True").sum()
        assert set(metrics.loc[metrics["subset"] != "all", "group"]) == {"all"}
        assert {"FFPE", "Frozen"} <= set(metrics.loc[metrics["subset"] == "all", "group"])

        r = target.iloc[0]
        assert abs(r["threshold"] - 0.94) < 1e-12 and r["met"] == (r["validation"] >= 0.94)
        assert set(top["coverage"]) == set(tff.LEVELS)
        assert (top["ci_low"] <= top["agreement"]).all()
        assert (top["agreement"] <= top["ci_high"]).all()
        assert set(per_class["mc_class"]) == set(kept["mc_class"])


def test_a_saved_score_is_never_overwritten():
    kept, load = make_validation()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cfg, probe_ids = setup(d)
        score(d, cfg, kept, load, probe_ids)
        path = sv.val_path(d / "pred" / "final_t", "rf")
        before = path.read_bytes()
        broken = lambda want: load(want) * 0                 # different data on a second try
        assert score(d, cfg, kept, broken, probe_ids) == 0
        assert path.read_bytes() == before


def test_a_sample_does_not_depend_on_the_other_samples():
    """Poison test: scoring half the cohort gives those samples the same scores."""
    kept, load = make_validation()
    half = kept.iloc[::2].reset_index(drop=True)
    out = []
    for table in (kept, half):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cfg, probe_ids = setup(d)
            score(d, cfg, table, load, probe_ids)
            with np.load(sv.val_path(d / "pred" / "final_t", "rf")) as z:
                out.append((z["proba"], z["calibrated"]))
    assert np.allclose(out[0][0][:, ::2], out[1][0], atol=1e-7)
    assert np.allclose(out[0][1][::2], out[1][1], atol=1e-6)


def test_a_different_probe_list_stops_before_scoring():
    kept, load = make_validation()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cfg, probe_ids = setup(d)
        try:
            score(d, cfg, kept, load, probe_ids[::-1].copy())
        except SystemExit as e:
            assert "probe list or order differs" in str(e)
        else:
            raise AssertionError("expected a stop")
        assert not list((d / "pred" / "final_t").glob("val_*.npz"))


def test_the_command_line_refuses_before_opening_the_store():
    for extra, text in (([], "--final-scoring"),
                        (["--final-scoring", "--store", "data/betas/zarr/GSE90496.zarr"],
                         "must be the GSE109379 store")):
        with tempfile.TemporaryDirectory() as tmp:
            kept = Path(tmp) / "kept.tsv"
            kept.write_text("geo_accession\tzarr_row\n")
            try:
                sv.main([str(ROOT / "configs" / "final_v1.yaml"), "--stage", "score",
                         "--kept", str(kept), "--families", str(kept)] + extra)
            except SystemExit as e:
                assert text in str(e), str(e)
            else:
                raise AssertionError("expected a stop")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
