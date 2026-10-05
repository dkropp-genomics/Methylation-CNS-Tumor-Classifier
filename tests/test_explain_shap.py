"""Tests for scripts/explain_shap.py. A made-up explainer stands in for TreeSHAP,
so the measures are checked against attributions whose answer is known."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import explain_shap as es  # noqa: E402

F, CLASSES = 60, ["A", "B", "C"]
SIZES = {"A": 30, "B": 20, "C": 10}
PROBES = np.array([f"cg{i:03d}" for i in range(F)], dtype=object)
# class A leans on probes 0-4, class B on 5-9, class C on 10-14
WEIGHT = np.zeros((F, 3))
for k in range(3):
    WEIGHT[5 * k: 5 * k + 5, k] = np.arange(5, 0, -1)


class Pipe:
    def transform(self, X, material=None):
        return X


class Model:
    def predict_proba(self, Z):
        return fake_explainer(self, Z)[0].sum(axis=1) + 1 / 3


def fake_explainer(model, Z):
    """Attribution = weight x (value - 0.5): additive by construction."""
    sv = (Z[:, :, None] - 0.5) * WEIGHT[None, :, :]
    return sv, np.full(3, 1 / 3)


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(CLASSES, [SIZES[c] for c in CLASSES]).astype(object)
    X = rng.uniform(0.3, 0.7, size=(y.size, F)).astype(np.float32)
    X[y == "A", 0:5] += 0.25                         # class A is hypermethylated there
    X[y == "B", 5:10] -= 0.25                        # class B is hypomethylated there
    ids = np.array([f"S{i:02d}" for i in range(y.size)], dtype=object)
    samples = pd.DataFrame({"geo_accession": ids, "mc_class": y, "material": "FFPE",
                            "outer_fold": "0"})
    row_of = {s: i for i, s in enumerate(ids)}
    load = lambda want: X[[row_of[s] for s in want]]
    genes = [""] * F
    genes[0], genes[1], genes[2] = "GENE1;GENE1", "GENE1", "GENE2"
    genes[5], genes[6], genes[40] = "OLDNAME", "GENE3", "FARAWAY"
    annotation = pd.DataFrame({
        "Probe_ID": PROBES[::-1], "chr": "chr1", "pos": [str(i) for i in range(F)][::-1],
        "genes": genes[::-1], "gene_groups": "Body",
        "relation_to_island": (["Island"] * 5 + ["N_Shore"] * 5 + ["OpenSea"] * 50)[::-1]})
    art = {"kind": "tree", "classes": CLASSES, "feature_names": PROBES, "pipe": Pipe(),
           "fit": {"model": Model()}}
    cfg = {"run_name": "shap_t", "top_n": 5, "largest_classes": ["A", "B"],
           "stability": {"n_splits": 10, "seed": 0}, "gene_min_cpgs": 2,
           "pairs": [{"class": "A", "gene": "GENE2", "aliases": ["GENE2"], "direction": "hyper"},
                     {"class": "B", "gene": "NEWNAME", "aliases": ["NEWNAME", "OLDNAME"],
                      "direction": "hyper"},
                     {"class": "C", "gene": "FARAWAY", "aliases": ["FARAWAY"],
                      "direction": "hypo"},
                     {"class": "C", "gene": "ABSENT", "aliases": ["ABSENT"],
                      "direction": "hypo"}]}
    return cfg, art, samples, load, annotation


def do_run(d, cfg, art, samples, load, annotation, explainer=fake_explainer):
    return es.run(cfg, art, samples, load, annotation, d / "shap", d / "res", explainer)


def test_top_cpgs_context_and_genes():
    cfg, art, samples, load, annotation = make_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        out = do_run(Path(tmp), cfg, art, samples, load, annotation)
        for name in out:
            assert (Path(tmp) / "res" / "shap_t" / name).exists()
    top = out["top_cpgs.tsv"]
    assert set(top.loc[top["mc_class"] == "A", "Probe_ID"]) == set(PROBES[0:5])
    assert set(top.loc[top["mc_class"] == "B", "Probe_ID"]) == set(PROBES[5:10])
    assert (top.loc[top["mc_class"] == "A", "delta_beta"] > 0).all()
    assert (top.loc[top["mc_class"] == "B", "delta_beta"] < 0).all()
    assert set(top["mc_class"]) == {"A", "B", "C"}           # C only because of its pairs
    ctx = out["context.tsv"].set_index(["mc_class", "context"]).sort_index()
    assert ctx.loc[("A", "island"), "share_top"] == 1.0      # probes 0-4 are islands
    assert ctx.loc[("B", "shore"), "share_top"] == 1.0
    assert abs(ctx.loc[("A", "island"), "share_model"] - 5 / 60) < 1e-12
    g = out["genes.tsv"]
    a = g[g["mc_class"] == "A"]
    assert list(a["gene"]) == ["GENE1"] and a["n_top_cpgs"].iloc[0] == 2   # symbol repeats count once
    assert a["direction"].iloc[0] == "hyper"
    s = out["summary.tsv"].set_index("mc_class")
    assert s.loc["A", "in_largest"] and not s.loc["C", "in_largest"]
    assert s.loc["A", "additivity_error"] < 1e-6
    assert abs(s.loc["A", "share_of_total_shap_in_top"] - 1.0) < 1e-6


def test_stability_is_high_for_real_signal_and_near_chance_for_noise():
    cfg, art, samples, load, annotation = make_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        out = do_run(Path(tmp), cfg, art, samples, load, annotation)
    st = out["stability.tsv"].set_index("mc_class")
    assert st.loc["A", "shared_mean"] == 5 and st.loc["A", "shared_min"] == 5
    assert abs(st.loc["A", "shared_by_chance"] - 25 / 60) < 1e-12
    noise = np.abs(np.random.default_rng(1).normal(size=(40, 500)))
    s = es.stability(noise, 20, 30, 0)
    assert s["shared_mean"] < 4                              # chance is 0.8


def test_published_pairs():
    cfg, art, samples, load, annotation = make_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        p = do_run(Path(tmp), cfg, art, samples, load, annotation)["pairs.tsv"]
    p = p.set_index("gene")
    assert p.loc["GENE2", "recovered"] and p.loc["GENE2", "best_probe"] == "cg002"
    assert p.loc["GENE2", "best_rank"] == 3 and p.loc["GENE2", "direction_matches"]
    # found through its old symbol; the class is hypomethylated there, the claim said hyper
    assert p.loc["NEWNAME", "recovered"] and not p.loc["NEWNAME", "direction_matches"]
    assert p.loc["FARAWAY", "testable"] and not p.loc["FARAWAY", "recovered"]
    assert p.loc["FARAWAY", "best_rank"] > 5
    assert not p.loc["ABSENT", "testable"] and not p.loc["ABSENT", "recovered"]
    assert p.loc["ABSENT", "n_model_probes"] == 0


def test_saved_values_are_reused_and_non_additive_values_stop():
    cfg, art, samples, load, annotation = make_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        first = do_run(d, cfg, art, samples, load, annotation)

        def never(model, Z):
            raise AssertionError("SHAP values were recomputed")
        again = do_run(d, cfg, art, samples, load, annotation, explainer=never)
        assert first["top_cpgs.tsv"].equals(again["top_cpgs.tsv"])
    with tempfile.TemporaryDirectory() as tmp:
        broken = lambda model, Z: (fake_explainer(model, Z)[0] * 1.5, np.full(3, 1 / 3))
        try:
            do_run(Path(tmp), cfg, art, samples, load, annotation, explainer=broken)
        except SystemExit as e:
            assert "do not add up" in str(e)
        else:
            raise AssertionError("expected a stop")


def test_config_checks():
    cfg, art, samples, load, annotation = make_inputs()
    for change, text in (({"largest_classes": ["A", "C"]}, "not the 2 largest"),
                         ({"largest_classes": ["A", "Z"]}, "not in the fold table")):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                do_run(Path(tmp), {**cfg, **change}, art, samples, load, annotation)
            except SystemExit as e:
                assert text in str(e), str(e)
            else:
                raise AssertionError(f"expected a stop mentioning {text!r}")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            do_run(Path(tmp), cfg, art, samples, load, annotation.iloc[1:])
        except SystemExit as e:
            assert "not in the annotation" in str(e)
        else:
            raise AssertionError("expected a stop")


def test_real_treeshap_on_a_small_forest():
    try:
        import shap  # noqa: F401
    except ImportError:
        print("  (skipped: shap not installed)")
        return
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    y = np.repeat(["A", "B", "C"], 30)
    X = rng.uniform(0.3, 0.7, size=(90, 12)).astype(np.float32)
    X[y == "A", 0] += 0.3
    X[y == "B", 1] += 0.3
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    sv, err = es.class_shap(model, X[y == "A"], 0, es.tree_explainer)
    assert sv.shape == (30, 12) and err < 1e-6
    assert es.rank_order(np.abs(sv).mean(axis=0))[0] == 0     # probe 0 marks class A


def test_the_committed_config():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "shap_v1.yaml").read_text())
    assert cfg["top_n"] == 100 and len(cfg["largest_classes"]) == 5
    assert [p["gene"] for p in cfg["pairs"]] == ["SHPRH", "PWWP3A", "TBX19", "RET"]
    assert all(p["direction"] in ("hyper", "hypo") and p["gene"] in p["aliases"]
               for p in cfg["pairs"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
