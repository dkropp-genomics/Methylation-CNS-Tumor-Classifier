#!/usr/bin/env python3
"""PCA and UMAP of the training cohort. FOR LOOKING ONLY.

Nothing chosen from these plots may feed a model. The probe selection and
the median fill below use all training samples at once, which is fine for a
picture and would be leakage if reused for modeling (that happens inside
folds in Phase 2).

What it does:
  1. Reads the QC-passed training samples (rows from zarr_row in the fold
     table) and the fixed probe filter (cols from probes_kept.tsv).
  2. Pass 1 over the store: per-probe missing fraction and variance.
     Keeps probes missing in <= 5% of samples, then the top-variance N.
  3. Pass 2: loads only those N probes, fills NaN with the probe median,
     runs PCA, then UMAP on the leading PCs.
  4. Measures, per PC, how much of it is explained by class and by material
     (eta squared = share of variance between groups), and by material
     WITHIN class, using only classes that contain both materials.

The validation cohort is never read.

Usage:
  python scripts/explore_embedding.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MATERIAL_COLORS = {"FFPE": "#EE7733", "Frozen": "#0077BB"}
CHUNK = 256


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def open_betas(store):
    import zarr

    if not Path(store).exists():
        die(f"store not found: {store}")
    return zarr.open_group(str(store), mode="r")["betas"]


def read_tables(folds_path, probes_path, n_rows, n_cols):
    for p in (folds_path, probes_path):
        if not Path(p).exists():
            die(f"file not found: {p}")
    folds = pd.read_csv(folds_path, sep="\t")
    need = {"geo_accession", "zarr_row", "mc_class", "mc_family", "material"}
    if need - set(folds.columns):
        die(
            f"{folds_path}: missing columns {sorted(need - set(folds.columns))} "
            "(use the QC fold table made with --store-index)"
        )
    if folds["zarr_row"].duplicated().any():
        die(f"{folds_path}: duplicated zarr_row")
    if folds["zarr_row"].min() < 0 or folds["zarr_row"].max() >= n_rows:
        die(f"{folds_path}: zarr_row outside the store's {n_rows} rows")
    unknown = set(folds["material"]) - set(MATERIAL_COLORS)
    if unknown:
        die(f"{folds_path}: unexpected material values {sorted(unknown)}")
    folds = folds.sort_values("zarr_row").reset_index(drop=True)

    probes = pd.read_csv(probes_path, sep="\t")
    if {"col", "Probe_ID"} - set(probes.columns):
        die(f"{probes_path}: need columns col, Probe_ID")
    if probes["col"].duplicated().any() or probes["col"].max() >= n_cols:
        die(f"{probes_path}: bad col values for a store with {n_cols} columns")
    return folds, probes.sort_values("col").reset_index(drop=True)


def iter_blocks(z, rows):
    """Yield (positions in `rows`, block of those store rows), in order."""
    rows = np.asarray(rows)
    for start in range(0, z.shape[0], CHUNK):
        pos = np.where((rows >= start) & (rows < start + CHUNK))[0]
        if len(pos):
            yield pos, np.asarray(z[start : start + CHUNK])[rows[pos] - start]


def probe_stats(z, rows, cols):
    """Pass 1: missing fraction and variance of each kept probe."""
    n = np.zeros(len(cols))
    s = np.zeros(len(cols))
    ss = np.zeros(len(cols))
    for _, block in iter_blocks(z, rows):
        b = block[:, cols].astype(np.float64)
        ok = ~np.isnan(b)
        b[~ok] = 0.0
        n += ok.sum(axis=0)
        s += b.sum(axis=0)
        ss += (b * b).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (ss - s * s / n) / (n - 1)
    return 1 - n / len(rows), var


def load_matrix(z, rows, cols):
    """Pass 2: samples x selected probes, NaN filled with the probe median."""
    x = np.empty((len(rows), len(cols)), dtype=np.float32)
    for pos, block in iter_blocks(z, rows):
        x[pos] = block[:, cols]
    med = np.nanmedian(x, axis=0)
    filled = int(np.isnan(x).sum())
    idx = np.where(np.isnan(x))
    x[idx] = med[idx[1]]
    return x, filled


def eta_squared(values, groups):
    """Share of the variance of `values` that lies between `groups`."""
    values = np.asarray(values, dtype=float)
    total = ((values - values.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    means = pd.Series(values).groupby(np.asarray(groups)).transform("mean")
    return float(((means - values.mean()) ** 2).sum() / total)


def pc_associations(pcs, explained, folds, mixed_classes):
    """Per PC: variance explained, and eta squared for class and material."""
    in_mixed = folds["mc_class"].isin(mixed_classes).to_numpy()
    rows = []
    for i in range(pcs.shape[1]):
        v = pcs[:, i]
        row = {
            "pc": i + 1,
            "var_explained": explained[i],
            "eta2_class": eta_squared(v, folds["mc_class"]),
            "eta2_material": eta_squared(v, folds["material"]),
            "eta2_material_within_class": np.nan,
        }
        if in_mixed.any():
            sub = folds[in_mixed]
            # Remove each class's own mean, then ask what material explains.
            resid = (
                v[in_mixed]
                - pd.Series(v[in_mixed])
                .groupby(sub["mc_class"].to_numpy())
                .transform("mean")
                .to_numpy()
            )
            row["eta2_material_within_class"] = eta_squared(resid, sub["material"])
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def make_figures(emb, mixed, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    def by_material(ax, xcol, ycol):
        for m, color in MATERIAL_COLORS.items():
            d = emb[emb["material"] == m]
            ax.scatter(
                d[xcol],
                d[ycol],
                s=5,
                alpha=0.6,
                color=color,
                label=f"{m} (n={len(d)})",
                linewidths=0,
            )
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

    # 1. Material: PCA and UMAP side by side.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    by_material(axes[0], "PC1", "PC2")
    by_material(axes[1], "UMAP1", "UMAP2")
    axes[0].legend(markerscale=3, frameon=False)
    fig.suptitle("Training cohort by material (look-only)")
    fig.tight_layout()
    fig.savefig(fig_dir / "embedding_by_material.png", dpi=140)
    plt.close(fig)

    # 2. Family: UMAP, one color per family, name written at its centre.
    fams = sorted(emb["mc_family"].unique())
    cmap = [plt.get_cmap(name)(i) for name in ("tab20", "tab20b", "tab20c") for i in range(20)]
    fig, ax = plt.subplots(figsize=(13, 11))
    for i, f in enumerate(fams):
        d = emb[emb["mc_family"] == f]
        ax.scatter(d["UMAP1"], d["UMAP2"], s=6, alpha=0.7, color=cmap[i % len(cmap)], linewidths=0)
        ax.text(
            d["UMAP1"].median(),
            d["UMAP2"].median(),
            f,
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"Training cohort by class family ({len(fams)} families, look-only)")
    fig.tight_layout()
    fig.savefig(fig_dir / "umap_by_family.png", dpi=140)
    plt.close(fig)

    # 3. Within-class: the largest classes that contain both materials.
    show = mixed[:6]
    if show:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
        for ax, cls in zip(axes.ravel(), show):
            ax.scatter(emb["UMAP1"], emb["UMAP2"], s=2, color="#dddddd", linewidths=0)
            for m, color in MATERIAL_COLORS.items():
                d = emb[(emb["mc_class"] == cls) & (emb["material"] == m)]
                ax.scatter(
                    d["UMAP1"],
                    d["UMAP2"],
                    s=14,
                    color=color,
                    label=f"{m} (n={len(d)})",
                    linewidths=0,
                )
            ax.set_title(cls, fontsize=10)
            ax.legend(fontsize=8, frameon=False)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.ravel()[len(show) :]:
            ax.axis("off")
        fig.suptitle("Do FFPE and frozen samples of the same class sit together?")
        fig.tight_layout()
        fig.savefig(fig_dir / "umap_material_within_class.png", dpi=140)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default="data/betas/zarr/GSE90496.zarr")
    ap.add_argument("--folds", default="results/splits/folds_seed42_qc.tsv")
    ap.add_argument("--probes", default="results/probes/probes_kept.tsv")
    ap.add_argument("--out-dir", default="results/explore")
    ap.add_argument("--n-probes", type=int, default=20000)
    ap.add_argument("--max-missing", type=float, default=0.05)
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument(
        "--min-per-material",
        type=int,
        default=5,
        help="a class counts as mixed with at least this many of each",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from sklearn.decomposition import PCA
    import umap

    z = open_betas(args.store)
    folds, probes = read_tables(args.folds, args.probes, z.shape[0], z.shape[1])
    rows = folds["zarr_row"].to_numpy()
    cols = probes["col"].to_numpy()
    print(f"{len(rows)} samples, {len(cols)} filtered probes")

    miss, var = probe_stats(z, rows, cols)
    usable = (miss <= args.max_missing) & np.isfinite(var)
    print(f"Probes missing in <= {args.max_missing:.0%} of samples: {int(usable.sum())}")
    if usable.sum() < args.n_probes:
        die(f"only {int(usable.sum())} usable probes, fewer than --n-probes {args.n_probes}")
    order = np.argsort(-np.where(usable, var, -np.inf), kind="stable")[: args.n_probes]
    sel = np.sort(order)

    x, filled = load_matrix(z, rows, cols[sel])
    print(
        f"Loaded {x.shape[0]} x {x.shape[1]}; filled {filled} missing values "
        f"({filled / x.size:.2%}) with probe medians"
    )

    n_pcs = min(args.n_pcs, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=args.seed)
    pcs = pca.fit_transform(x - x.mean(axis=0))
    um = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=args.seed).fit_transform(pcs)

    counts = folds.groupby(["mc_class", "material"]).size().unstack(fill_value=0)
    for m in MATERIAL_COLORS:
        if m not in counts:
            counts[m] = 0
    is_mixed = (counts[list(MATERIAL_COLORS)] >= args.min_per_material).all(axis=1)
    mixed = counts[is_mixed].sum(axis=1).sort_values(ascending=False).index.tolist()
    print(
        f"Classes with >= {args.min_per_material} samples of each material: "
        f"{len(mixed)} of {len(counts)}"
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = folds[["geo_accession", "mc_class", "mc_family", "material"]].copy()
    for i in range(min(10, n_pcs)):
        emb[f"PC{i + 1}"] = pcs[:, i].round(4)
    emb["UMAP1"], emb["UMAP2"] = um[:, 0].round(4), um[:, 1].round(4)
    emb.to_csv(out / "embedding.tsv", sep="\t", index=False)

    n_show = min(10, n_pcs)
    assoc = pc_associations(pcs[:, :n_show], pca.explained_variance_ratio_, folds, mixed)
    assoc.to_csv(out / "pc_associations.tsv", sep="\t", index=False)
    probes.iloc[sel].to_csv(out / "probes_used.tsv", sep="\t", index=False)
    print("\nPer PC: variance explained, and share explained by class and material")
    print(assoc.to_string(index=False))

    make_figures(emb, mixed, out / "figures")
    print(f"\nWrote {out}/embedding.tsv, pc_associations.tsv, probes_used.tsv, figures/")


if __name__ == "__main__":
    main()
