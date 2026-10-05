#!/usr/bin/env python
"""Does refitting a final network give the same numbers?

Refits one saved network on the same data with the same seed and settings and
compares it with the saved model: the weights, and the predictions for every
training tumor at each coverage level. Reads the training cohort only.

  python scripts/check_nn_reproducibility.py configs/final_v1.yaml nn_plain

Output: results/final_v1/nn_repro_check.tsv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_final as ff  # noqa: E402
from methylclf.data import BetaStore  # noqa: E402
from methylclf.masking import observed_uniform, probe_positions  # noqa: E402

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
name = sys.argv[2] if len(sys.argv) > 2 else "nn_plain"
if cfg["models"][name]["kind"] != "nn":
    ff.cv.fail(f"{name} is not a network")
art = ff.load_model(ff.model_path(Path("data/models") / cfg["run_name"], name))
mcfg = yaml.safe_load(Path(cfg["models"][name]["config"]).read_text())
st = BetaStore.open("data/betas/zarr/GSE90496.zarr", "results/splits/folds_seed42_qc.tsv",
                    "results/probes/probes_kept.tsv")
ids, y, mat, _ = ff.cv.columns(st.samples)
Z = art["pipe"].transform(st.load(ids), mat)
t0 = time.time()
again = dict(art, fit=ff.fit_nn(mcfg, art["settings"], Z, y, art["classes"]))
print(f"{name}: refitted in {time.time() - t0:.0f} s")

weights = max(float(np.abs(art["fit"]["state"][k] - again["fit"]["state"][k]).max())
              for k in art["fit"]["state"])
levels = [float(v) for v in cfg["analyses"]["sparsity"]["levels"]]
probe_ids = np.asarray(st.probe_ids, dtype=object)
u = observed_uniform(ids, len(probe_ids), probe_positions(art["feature_names"], probe_ids),
                     int(cfg["analyses"]["sparsity"]["mask_seed"]))
a, b = ff.predict_levels(art, Z, u, levels), ff.predict_levels(again, Z, u, levels)
rows = [{"model": name, "coverage": lv, "n_samples": len(ids), "max_abs_weight_diff": weights,
         "max_abs_score_diff": float(np.abs(a[i] - b[i]).max()),
         "n_top_class_changed": int((a[i].argmax(1) != b[i].argmax(1)).sum())}
        for i, lv in enumerate(levels)]
out = pd.DataFrame(rows)
path = Path("results") / cfg["run_name"] / "nn_repro_check.tsv"
if path.exists():
    old = pd.read_csv(path, sep="\t")
    out = pd.concat([old[old["model"] != name], out], ignore_index=True)
out.to_csv(path, sep="\t", index=False, float_format="%.3g")
print(out.to_string(index=False))
