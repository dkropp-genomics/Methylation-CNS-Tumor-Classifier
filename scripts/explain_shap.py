#!/usr/bin/env python
"""Which CpGs the final random forest relies on, class by class (TreeSHAP).

For one sample and one class, SHAP splits the forest's score among the CpGs:
each CpG gets a number saying how far its value moved the score up or down,
and the numbers add up to the score. Averaging their absolute size over the
samples of a class ranks the CpGs the forest leans on for that class.

Everything that is reported was fixed in configs/shap_v1.yaml before any
attribution was computed:

  top CpGs      the top_n CpGs of each class by mean absolute SHAP
  stability     how many of the top_n are shared by two random halves of the
                class's samples (correlated CpGs share credit arbitrarily,
                so single-CpG ranks are not stable; this measures how much)
  context       island / shore / shelf / open sea: top_n against all model probes
  genes         genes with several top_n CpGs, and the direction of methylation
  pairs         four published class-gene pairs: is the gene among the top_n?

The samples explained are the training samples of each class. This describes
what the model uses. It is not a performance estimate and no model changes.

  python scripts/explain_shap.py configs/shap_v1.yaml

Outputs:
  data/shap/shap_v1/<class>.npz           SHAP values of that class's samples
  results/shap_v1/top_cpgs.tsv, stability.tsv, context.tsv, genes.tsv, pairs.tsv,
  results/shap_v1/summary.tsv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_final as ff  # noqa: E402

cv = ff.cv
CONTEXT = {
    "Island": "island",
    "N_Shore": "shore",
    "S_Shore": "shore",
    "N_Shelf": "shelf",
    "S_Shelf": "shelf",
    "OpenSea": "open sea",
}
CONTEXT_ORDER = ["island", "shore", "shelf", "open sea"]


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


# --------------------------------------------------------------------------
# SHAP values
# --------------------------------------------------------------------------
def tree_explainer(model, Z, batch=32):
    """SHAP values shaped (sample, probe, class), and the base value per class."""
    import shap

    ex = shap.TreeExplainer(model)
    out = []
    for i in range(0, len(Z), batch):
        sv = np.asarray(ex.shap_values(Z[i : i + batch], check_additivity=False))
        if sv.ndim != 3:
            cv.fail(f"shap returned an array of shape {sv.shape}; expected 3 dimensions")
        if sv.shape[1] != Z.shape[1]:  # older layout: class first
            sv = np.moveaxis(sv, 0, 2)
        out.append(sv)
    return np.concatenate(out), np.asarray(ex.expected_value, dtype=np.float64).ravel()


def class_shap(model, Z, k, explainer):
    """SHAP values of class k for every row of Z, with the additivity error.

    Additivity: base value + the sum of a sample's SHAP values must equal the
    forest's score for that sample. It is checked on every class at once.
    """
    sv, base = explainer(model, Z)
    if sv.shape[:2] != Z.shape:
        cv.fail(f"SHAP values have shape {sv.shape} for input {Z.shape}")
    err = float(np.abs(sv.sum(axis=1) + base[None, :] - model.predict_proba(Z)).max())
    return sv[:, :, k].astype(np.float32), err


# --------------------------------------------------------------------------
# The measures (plain arrays in, tables out)
# --------------------------------------------------------------------------
def rank_order(score) -> np.ndarray:
    """Probe positions from the highest score down; ties by position."""
    return np.lexsort((np.arange(score.size), -np.asarray(score, dtype=np.float64)))


def stability(abs_sv, top_n, n_splits, seed) -> dict:
    """Overlap of the top_n between two random halves of the samples."""
    n, F = abs_sv.shape
    if n < 4:
        cv.fail(f"stability needs at least 4 samples, got {n}")
    rng = np.random.default_rng(seed)
    shared = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2 :]
        top_a = set(rank_order(abs_sv[a].mean(axis=0))[:top_n])
        top_b = set(rank_order(abs_sv[b].mean(axis=0))[:top_n])
        shared.append(len(top_a & top_b))
    return {
        "n_samples": n,
        "top_n": top_n,
        "n_splits": n_splits,
        "shared_mean": float(np.mean(shared)),
        "shared_min": int(min(shared)),
        "shared_max": int(max(shared)),
        "shared_by_chance": top_n * top_n / F,
    }  # two random sets of top_n


def context_table(top, relation) -> pd.DataFrame:
    """Share of the top CpGs in each island context, against all model probes."""
    relation = np.asarray(relation, dtype=object)
    rows = []
    for c in CONTEXT_ORDER:
        rows.append(
            {
                "context": c,
                "n_top": int((relation[top] == c).sum()),
                "share_top": float((relation[top] == c).mean()),
                "n_model": int((relation == c).sum()),
                "share_model": float((relation == c).mean()),
            }
        )
    return pd.DataFrame(rows)


def split_genes(text) -> list:
    """'TP53;TP53;WRAP53' -> ['TP53', 'WRAP53'] (the annotation repeats symbols)."""
    return sorted({g for g in str(text).split(";") if g and g != "nan"})


def gene_table(top, genes, mean_abs, delta, min_cpgs) -> pd.DataFrame:
    """Genes holding at least min_cpgs of the top CpGs."""
    hits = {}
    for i in top:
        for g in split_genes(genes[i]):
            hits.setdefault(g, []).append(i)
    rows = []
    for g, idx in hits.items():
        if len(idx) < min_cpgs:
            continue
        d = delta[idx]
        rows.append(
            {
                "gene": g,
                "n_top_cpgs": len(idx),
                "sum_mean_abs_shap": float(mean_abs[idx].sum()),
                "mean_delta_beta": float(d.mean()),
                "direction": "hyper" if (d > 0).all() else "hypo" if (d < 0).all() else "mixed",
            }
        )
    cols = ["gene", "n_top_cpgs", "sum_mean_abs_shap", "mean_delta_beta", "direction"]
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["n_top_cpgs", "sum_mean_abs_shap"], ascending=False)
        .reset_index(drop=True)
    )


def pair_row(pair, order, genes, groups, probe_ids, delta, top_n) -> dict:
    """One published class-gene pair: where does the gene's best CpG rank?"""
    aliases = set(pair["aliases"])
    idx = np.array(
        [i for i in range(len(genes)) if aliases & set(split_genes(genes[i]))], dtype=np.int64
    )
    row = {
        "mc_class": pair["class"],
        "gene": pair["gene"],
        "expected_direction": pair["direction"],
        "n_model_probes": int(idx.size),
    }
    if idx.size == 0:
        row.update(testable=False, recovered=False)
        return row
    rank_of = np.empty(order.size, dtype=np.int64)
    rank_of[order] = np.arange(1, order.size + 1)
    best = idx[np.argmin(rank_of[idx])]
    seen = "hyper" if delta[best] > 0 else "hypo"
    row.update(
        testable=True,
        best_rank=int(rank_of[best]),
        best_probe=probe_ids[best],
        best_probe_gene_groups=groups[best],
        best_delta_beta=float(delta[best]),
        observed_direction=seen,
        direction_matches=bool(seen == pair["direction"]),
        n_in_top=int((rank_of[idx] <= top_n).sum()),
        recovered=bool(rank_of[best] <= top_n),
    )
    return row


# --------------------------------------------------------------------------
def run(
    cfg,
    art,
    samples,
    load,
    annotation,
    shap_root="data/shap",
    res_root="results",
    explainer=tree_explainer,
):
    for key in ("run_name", "top_n", "largest_classes", "stability", "gene_min_cpgs", "pairs"):
        if key not in cfg:
            cv.fail(f"config: missing '{key}'")
    if art["kind"] != "tree":
        cv.fail("explain_shap.py explains the tree model only")
    shap_dir, res_dir = Path(shap_root) / cfg["run_name"], Path(res_root) / cfg["run_name"]
    shap_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    ids, y, mat, _ = cv.columns(samples)
    classes, model, top_n = list(art["classes"]), art["fit"]["model"], int(cfg["top_n"])

    # the "largest classes" in the config must be the largest classes in the data
    sizes = pd.Series(y).value_counts()
    largest = list(cfg["largest_classes"])
    absent = [c for c in largest if c not in sizes.index]
    if absent:
        cv.fail(f"config: largest_classes not in the fold table: {absent}")
    cutoff = sizes[largest].min()
    if set(sizes.index[sizes > cutoff]) - set(largest) or (sizes == cutoff).sum() > 1:
        cv.fail(
            f"config: largest_classes are not the {len(largest)} largest classes of the "
            f"fold table (largest: {list(sizes.index[:len(largest)])})"
        )
    todo = list(dict.fromkeys(largest + [p["class"] for p in cfg["pairs"]]))  # each class once
    unknown = [c for c in todo if c not in classes]
    if unknown:
        cv.fail(f"class(es) not in the model: {unknown}")

    # annotation, in the order of the model's probes
    probe_ids = np.asarray(art["feature_names"], dtype=object)
    ann = annotation.set_index("Probe_ID")
    missing = [p for p in probe_ids if p not in ann.index]
    if missing:
        cv.fail(f"{len(missing)} model probes are not in the annotation, first: {missing[0]}")
    ann = ann.loc[probe_ids]
    bad = sorted(set(ann["relation_to_island"]) - set(CONTEXT))
    if bad:
        cv.fail(f"annotation: unknown relation_to_island values {bad}")
    relation = ann["relation_to_island"].map(CONTEXT).to_numpy(dtype=object)
    genes = ann["genes"].fillna("").astype(str).to_numpy(dtype=object)
    groups = ann["gene_groups"].fillna("").astype(str).to_numpy(dtype=object)

    t0 = time.time()
    Z = art["pipe"].transform(load(ids), mat)  # every training sample, model features
    print(
        f"features for {Z.shape[0]} samples x {Z.shape[1]} probes in {time.time() - t0:.0f} s",
        flush=True,
    )

    tops, stab, ctx, gene_rows, summary, order_of, delta_of = [], [], [], [], [], {}, {}
    for c in todo:
        k, rows = classes.index(c), np.flatnonzero(y == c)
        path = shap_dir / f"{slug(c)}.npz"
        if path.exists():
            with np.load(path) as z:
                if list(z["sample_ids"].astype(object)) != list(ids[rows]):
                    cv.fail(f"{path}: saved for different samples; delete it to recompute")
                sv, err = z["shap"], float(z["additivity_error"])
        else:
            t1 = time.time()
            sv, err = class_shap(model, Z[rows], k, explainer)
            tmp = path.with_name(path.name[:-4] + ".tmp.npz")
            np.savez_compressed(
                tmp, shap=sv, sample_ids=np.asarray(ids[rows], dtype=str), additivity_error=err
            )
            os.replace(tmp, path)
            print(f"{c}: {len(rows)} samples explained in {time.time() - t1:.0f} s", flush=True)
        if err > 1e-6:
            cv.fail(f"{c}: SHAP values do not add up to the forest's score (error {err:.2e})")
        abs_sv = np.abs(sv)
        mean_abs, mean_sv = abs_sv.mean(axis=0), sv.mean(axis=0)
        # direction: mean methylation in the class minus the mean in all other samples
        delta = Z[rows].mean(axis=0, dtype=np.float64) - Z[y != c].mean(axis=0, dtype=np.float64)
        order = rank_order(mean_abs)
        order_of[c], delta_of[c] = order, delta
        top = order[:top_n]
        tops.append(
            pd.DataFrame(
                {
                    "mc_class": c,
                    "rank": np.arange(1, top_n + 1),
                    "Probe_ID": probe_ids[top],
                    "mean_abs_shap": mean_abs[top],
                    "mean_shap": mean_sv[top],
                    "delta_beta": delta[top],
                    "chr": ann["chr"].to_numpy()[top],
                    "pos": ann["pos"].to_numpy()[top],
                    "genes": genes[top],
                    "gene_groups": groups[top],
                    "context": relation[top],
                }
            )
        )
        s = stability(
            abs_sv, top_n, int(cfg["stability"]["n_splits"]), int(cfg["stability"]["seed"])
        )
        stab.append({"mc_class": c, **s})
        t = context_table(top, relation)
        t.insert(0, "mc_class", c)
        ctx.append(t)
        g = gene_table(top, genes, mean_abs, delta, int(cfg["gene_min_cpgs"]))
        g.insert(0, "mc_class", c)
        gene_rows.append(g)
        summary.append(
            {
                "mc_class": c,
                "in_largest": c in largest,
                "n_samples": len(rows),
                "additivity_error": err,
                "share_of_total_shap_in_top": float(mean_abs[top].sum() / mean_abs.sum()),
                "top_hyper": int((delta[top] > 0).sum()),
                "top_hypo": int((delta[top] < 0).sum()),
                "top_with_gene": int(sum(bool(split_genes(x)) for x in genes[top])),
                "stability_shared_mean": s["shared_mean"],
            }
        )
    pairs = pd.DataFrame(
        [
            pair_row(p, order_of[p["class"]], genes, groups, probe_ids, delta_of[p["class"]], top_n)
            for p in cfg["pairs"]
        ]
    )
    out = {
        "top_cpgs.tsv": pd.concat(tops, ignore_index=True),
        "stability.tsv": pd.DataFrame(stab),
        "context.tsv": pd.concat(ctx, ignore_index=True),
        "genes.tsv": pd.concat(gene_rows, ignore_index=True),
        "pairs.tsv": pairs,
        "summary.tsv": pd.DataFrame(summary),
    }
    for name, t in out.items():
        t.to_csv(res_dir / name, sep="\t", index=False, float_format="%.6g")
    return out


def show(out, cfg):
    """Print the tables a reader checks first."""
    pd.set_option(
        "display.width",
        250,
        "display.max_columns",
        30,
        "display.max_rows",
        300,
        "display.max_colwidth",
        60,
    )
    fmt = "{:.3g}".format
    print("\n== per class ==")
    print(out["summary.tsv"].to_string(index=False, float_format=fmt))
    print(f"\n== stability: top-{cfg['top_n']} CpGs shared by two random halves ==")
    print(out["stability.tsv"].to_string(index=False, float_format=fmt))
    c = out["context.tsv"]
    c = c[c["mc_class"].isin(cfg["largest_classes"])]
    print("\n== genomic context: share of the top CpGs (all model probes in the last row) ==")
    wide = c.pivot_table(index="mc_class", columns="context", values="share_top", sort=False)
    wide.loc["all model probes"] = c.drop_duplicates("context").set_index("context")["share_model"]
    print(wide[CONTEXT_ORDER].to_string(float_format="{:.2f}".format))
    g = out["genes.tsv"]
    for name in cfg["largest_classes"]:
        print(f"\n== {name}: genes with >= {cfg['gene_min_cpgs']} top CpGs (first 12) ==")
        print(
            g[g["mc_class"] == name]
            .drop(columns="mc_class")
            .head(12)
            .to_string(index=False, float_format=fmt)
        )
    print("\n== published class-gene pairs (Benfatto et al. 2025) ==")
    print(out["pairs.tsv"].to_string(index=False, float_format=fmt))
    p = out["pairs.tsv"]
    print(
        f"\nrecovered {int(p['recovered'].sum())} of {int(p['testable'].sum())} testable "
        f"pairs ({int((~p['testable']).sum())} not testable)"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--store", default="data/betas/zarr/GSE90496.zarr")
    ap.add_argument("--folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--probes", default="results/probes/probes_kept.tsv")
    ap.add_argument("--annotation", default="data/meta/hm450_gene_annotation.tsv.gz")
    ap.add_argument("--shap-root", default="data/shap")
    ap.add_argument("--res-root", default="results")
    args = ap.parse_args(argv)

    import yaml
    from methylclf.data import BetaStore

    for path in (args.config, args.annotation):
        if not Path(path).is_file():
            cv.fail(
                f"{path}: not found"
                + (
                    " (run R/export_gene_annotation.R in methyl-r)"
                    if path == args.annotation
                    else ""
                )
            )
    cfg = yaml.safe_load(Path(args.config).read_text())
    if "GSE90496" not in Path(args.store).name:
        cv.fail("explain_shap.py reads the training cohort (GSE90496) only")
    annotation = pd.read_csv(args.annotation, sep="\t", dtype=str, keep_default_na=False)
    st = BetaStore.open(args.store, args.folds, args.probes)
    out = run(
        cfg,
        ff.load_model(cfg["model"]),
        st.samples,
        st.load,
        annotation,
        args.shap_root,
        args.res_root,
    )
    show(out, cfg)


if __name__ == "__main__":
    main()
