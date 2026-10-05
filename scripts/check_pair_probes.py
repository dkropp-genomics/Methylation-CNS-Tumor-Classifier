#!/usr/bin/env python
"""Post hoc check (made after the SHAP results): why the four published
class-gene pairs were not testable. For each gene, where its CpGs rank by
variance in the final forest's feature pipeline. Reads the saved model and the
array annotation only.

  python scripts/check_pair_probes.py configs/shap_v1.yaml
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_final as ff  # noqa: E402

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
pipe = ff.load_model(cfg["model"])["pipe"]
kept = pd.read_csv("results/probes/probes_kept.tsv", sep="\t")["Probe_ID"].to_numpy(dtype=object)
names = pipe.missing_.get_feature_names_out(kept)  # probes past the missingness filter
var = pipe.select_.variances_
rank = np.empty(var.size, dtype=int)
rank[np.lexsort((np.arange(var.size), -var))] = np.arange(1, var.size + 1)
rank_of = dict(zip(names, rank))
ann = pd.read_csv(
    "data/meta/hm450_gene_annotation.tsv.gz", sep="\t", dtype=str, keep_default_na=False
)
rows = []
for p in cfg["pairs"]:
    aliases = set(p["aliases"])
    hit = [q for q, g in zip(ann["Probe_ID"], ann["genes"]) if aliases & set(g.split(";"))]
    ranks = sorted(int(rank_of[q]) for q in hit if q in rank_of)
    rows.append(
        {
            "mc_class": p["class"],
            "gene": p["gene"],
            "n_kept_probes": len(hit),
            "n_past_missingness": len(ranks),
            "best_variance_rank": ranks[0] if ranks else np.nan,
            "n_ranked_probes": var.size,
            "n_model_probes": int(pipe.n_probes),
            "n_in_model": sum(r <= pipe.n_probes for r in ranks),
        }
    )
out = pd.DataFrame(rows)
out.to_csv(f"results/{cfg['run_name']}/pairs_variance_rank.tsv", sep="\t", index=False)
print(out.to_string(index=False))
