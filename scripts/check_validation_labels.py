"""Compare GEO's validation labels with Capper Supplementary Table 4 (labels only; no betas)."""

import re
import pandas as pd

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()
cols = {1: "sentrix", 11: "top_abbr", 12: "score", 14: "sub_abbr", 17: "interp"}
t = pd.read_excel("data/meta/capper_sup_table_4.xlsx", header=1, usecols=list(cols))
t.columns = [cols[i] for i in sorted(cols)]
for c in ("sentrix", "top_abbr", "sub_abbr", "interp"):
    t[c] = t[c].map(norm)

is_family = t.top_abbr.str.startswith("MCF_")
has_sub = ~t.sub_abbr.str.contains("not applicable|not performed|failed")
t["table_class"] = t.top_abbr.where(~is_family).mask(is_family & has_sub, t.sub_abbr)
t["table_class"] = t.table_class.replace({"PITUI, SCO, GCT": "PITUI"})
t["group"] = "matched, class named"
t.loc[t.score < 0.9, "group"] = "no match, class named"
t.loc[t.table_class.isna() & (t.score >= 0.9), "group"] = "matched, subclass failed"
t.loc[t.table_class.isna() & (t.score < 0.9), "group"] = "no match, family only"

qc = pd.read_csv("results/qc/GSE109379_sesame_qc.tsv", sep="\t")
qc["sentrix"] = qc.idat_prefix.str.extract(r"(\d+_R\d\dC\d\d)")
lab = pd.read_csv("results/meta/GSE109379_labels.tsv", sep="\t")
m = lab.merge(qc[["geo_accession", "sentrix"]], on="geo_accession").merge(
    t, on="sentrix", how="left"
)
print("GEO samples:", len(lab), "| joined to table:", m.interp.notna().sum())

m["mc_class"] = m.mc_class.map(norm)
m["same"] = m.mc_class == m.table_class
out = m.groupby("group").agg(n=("same", "size"), agree=("same", "sum"))
print(out.to_string())
named = m[m.table_class.notna()]
print("\nMismatches where the table names a class:", (~named.same).sum())
print(
    named.loc[~named.same, ["geo_accession", "mc_class", "table_class", "score"]]
    .head(15)
    .to_string(index=False)
)
print("\nGEO class for the family-only cases, by table family:")
print(
    m[m.table_class.isna()]
    .groupby("top_abbr")
    .mc_class.agg(lambda s: dict(s.value_counts()))
    .to_string()
)

# Subset flags for external validation, fixed before any model scores the cohort.
m["kept_qc"] = m.geo_accession.map(qc.set_index("geo_accession").frac_detected) >= 0.70
m["capper_matched"] = m.score >= 0.9
m["path_concordant"] = m.interp.str.startswith("confirmation")
m["capper_no_match"] = m.score < 0.9
keep = [
    "geo_accession",
    "sentrix",
    "mc_class",
    "score",
    "interp",
    "group",
    "kept_qc",
    "capper_matched",
    "path_concordant",
    "capper_no_match",
]
m[keep].rename(
    columns={"score": "capper_score", "interp": "capper_interpretation", "group": "label_source"}
).to_csv("results/meta/GSE109379_subsets.tsv", sep="\t", index=False)
k = m[m.kept_qc]
print("\nAfter sample QC:", len(k))
for c in ("capper_matched", "path_concordant", "capper_no_match"):
    print(f"  {c:16s} n={k[c].sum():5d}  classes={k.loc[k[c], 'mc_class'].nunique()}")
print("  all              classes=", k.mc_class.nunique())
