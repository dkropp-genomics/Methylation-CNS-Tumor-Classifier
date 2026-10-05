#!/usr/bin/env python3
"""Build the clean label table for the external validation cohort.

Why this script exists
----------------------
A classifier can only ever predict a class it was trained on. So before the
validation cohort (GSE109379) is scored, every one of its labels has to be
matched to one of the 91 training classes (GSE90496), and anything that
doesn't match has to be listed, never silently dropped. This script does
that matching once, writes the result to a small committed table, and every
later step reads that table.

It uses only labels and never looks at methylation values, so it cannot leak
anything about the validation cohort into the model.

What it does
------------
1. Reads data/meta/<GSE>_samples.tsv for both cohorts (written by
   scripts/parse_geo_metadata.py).
2. Cleans labels exactly as scripts/make_folds.py does (collapse repeated
   whitespace; family = text before the first comma), so a label means the
   same thing in both cohorts.
3. Puts material on one vocabulary: DNA_FFPE -> FFPE, DNA_KRYO -> Frozen
   ("Kryo" is German for frozen tissue). An unknown value stops the run.
4. Matches each validation label to the training classes by exact name, or
   through an optional alias table (configs/label_aliases.tsv) for labels
   that are the same class under a different spelling.
5. Cross-checks each sample's label against its GEO title, which starts
   with the class name ("GBM, MES, sample 1 [validation set]").
6. Writes
     results/meta/<VAL>_labels.tsv        one row per validation sample
     results/meta/<VAL>_class_overlap.tsv one row per class, both cohorts
   and prints a summary.

Usage (from the repo root):
    python scripts/make_validation_labels.py
    python scripts/make_validation_labels.py --aliases configs/label_aliases.tsv

Exit status: 0 on success, including when some labels are unmatched (they
are reported and flagged in_training=0). Non-zero for broken input: missing
columns, duplicate or empty values, unknown material, a bad alias table.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

MATERIAL_MAP = {
    "FFPE": "FFPE",
    "Frozen": "Frozen",
    "DNA_FFPE": "FFPE",
    "DNA_KRYO": "Frozen",  # "Kryo" = cryopreserved / fresh-frozen
}
TITLE_SUFFIX = re.compile(r",\s*sample\s+\d+\s*\[[^\]]*\]\s*$")


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def clean_label(label: str) -> str:
    """Collapse repeated whitespace (GEO has 'PIN T,  PB A' with two spaces)."""
    return re.sub(r"\s+", " ", label).strip()


def family_of(mc_class: str) -> str:
    """Rough family = text before the first comma (same rule as make_folds.py)."""
    return mc_class.split(",")[0].strip()


def read_samples(path: Path) -> list[dict]:
    """Load a <GSE>_samples.tsv and return rows with cleaned label columns."""
    if not path.exists():
        die(f"{path} not found; run scripts/parse_geo_metadata.py first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    if not rows:
        die(f"{path} has no samples")
    for col in ("geo_accession", "title", "methylation.class", "material"):
        if col not in rows[0]:
            die(f"{path} has no '{col}' column (found: {', '.join(rows[0])})")

    out = []
    for r in rows:
        gsm, raw = r["geo_accession"], r["methylation.class"]
        if raw in ("", "NA"):
            die(f"{path}: {gsm} has no methylation class")
        if r["material"] not in MATERIAL_MAP:
            die(
                f"{path}: {gsm} has unknown material '{r['material']}' "
                f"(known: {', '.join(MATERIAL_MAP)})"
            )
        out.append(
            {
                "geo_accession": gsm,
                "title": r["title"],
                "label": clean_label(raw),
                "material": MATERIAL_MAP[r["material"]],
                "material_raw": r["material"],
            }
        )
    dup = [k for k, c in Counter(r["geo_accession"] for r in out).items() if c > 1]
    if dup:
        die(f"{path}: duplicate sample IDs, e.g. {dup[:3]}")
    return out


def read_aliases(path: Path | None, train_classes: set[str]) -> dict[str, str]:
    """Optional table: validation_label <TAB> training_class."""
    if path is None:
        return {}
    if not path.exists():
        die(f"alias file {path} not found")
    aliases = {}
    with open(path, newline="") as f:
        for r in csv.DictReader((ln for ln in f if not ln.startswith("#")), delimiter="\t"):
            try:
                src, dst = clean_label(r["validation_label"]), clean_label(r["training_class"])
            except KeyError:
                die(f"{path} needs columns validation_label and training_class")
            if dst not in train_classes:
                die(f"{path}: '{dst}' is not a training class")
            if src in train_classes:
                die(f"{path}: '{src}' is already a training class; an alias would hide it")
            if src in aliases:
                die(f"{path}: '{src}' listed twice")
            aliases[src] = dst
    return aliases


def title_mismatches(samples: list[dict]) -> list[dict]:
    """Samples whose GEO title doesn't start with their class label."""
    bad = []
    for s in samples:
        stem = clean_label(TITLE_SUFFIX.sub("", s["title"]))
        if stem != s["label"]:
            bad.append(s)
    return bad


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(
            f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, quotechar=None
        )
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train", default="GSE90496", help="training series")
    p.add_argument("--val", default="GSE109379", help="validation series")
    p.add_argument("--meta-dir", type=Path, default=Path("data/meta"))
    p.add_argument("--outdir", type=Path, default=Path("results/meta"))
    p.add_argument(
        "--aliases",
        type=Path,
        default=None,
        help="TSV with columns validation_label, training_class",
    )
    a = p.parse_args()

    train = read_samples(a.meta_dir / f"{a.train}_samples.tsv")
    val = read_samples(a.meta_dir / f"{a.val}_samples.tsv")
    overlap = {s["geo_accession"] for s in train} & {s["geo_accession"] for s in val}
    if overlap:
        die(f"{len(overlap)} sample IDs are in both cohorts, e.g. {sorted(overlap)[:3]}")

    train_counts = Counter(s["label"] for s in train)
    train_classes = set(train_counts)
    aliases = read_aliases(a.aliases, train_classes)

    # Match every validation label to a training class (or to nothing).
    rows = []
    for s in val:
        label = s["label"]
        if label in train_classes:
            mc_class, how = label, "exact"
        elif label in aliases:
            mc_class, how = aliases[label], "alias"
        else:
            mc_class, how = "", "none"
        s["mc_class"], s["match"] = mc_class, how
        rows.append(
            [
                s["geo_accession"],
                label,
                mc_class or "NA",
                family_of(mc_class) if mc_class else "NA",
                family_of(label),
                s["material"],
                int(bool(mc_class)),
                how,
            ]
        )
    unused = sorted(set(aliases) - {s["label"] for s in val})
    if unused:
        die(f"aliases never used (typo?): {unused}")

    labels_out = a.outdir / f"{a.val}_labels.tsv"
    write_tsv(
        labels_out,
        [
            "geo_accession",
            "label_raw",
            "mc_class",
            "mc_family",
            "label_family",
            "material",
            "in_training",
            "match",
        ],
        rows,
    )

    # One row per class across both cohorts.
    val_matched = Counter(s["mc_class"] for s in val if s["mc_class"])
    val_unmatched = Counter(s["label"] for s in val if not s["mc_class"])
    train_families = {family_of(c) for c in train_classes}
    class_rows = [
        [
            c,
            family_of(c),
            train_counts[c],
            val_matched.get(c, 0),
            "both" if val_matched.get(c, 0) else "training only",
        ]
        for c in sorted(train_classes)
    ]
    class_rows += [
        [c, family_of(c), 0, n, "validation only"] for c, n in sorted(val_unmatched.items())
    ]
    overlap_out = a.outdir / f"{a.val}_class_overlap.tsv"
    write_tsv(overlap_out, ["class", "family", "n_training", "n_validation", "status"], class_rows)

    # ------------------------------------------------------------- summary --
    n = len(val)
    n_ok = sum(1 for s in val if s["mc_class"])
    n_alias = sum(1 for s in val if s["match"] == "alias")
    print(f"Training   {a.train}: {len(train)} samples, {len(train_classes)} classes")
    print(f"Validation {a.val}: {n} samples, {len({s['label'] for s in val})} distinct labels")
    print(
        f"\nMatched to a training class: {n_ok}/{n} samples ({n_ok / n:.1%})"
        + (f", {n_alias} of them through aliases" if n_alias else "")
    )
    print(f"Training classes present in validation: {len(val_matched)}/{len(train_classes)}")

    if val_unmatched:
        print(
            f"\nValidation labels with NO training class "
            f"({len(val_unmatched)} labels, {n - n_ok} samples):"
        )
        for c, k in val_unmatched.most_common():
            note = (
                "family is in training"
                if family_of(c) in train_families
                else "family not in training"
            )
            print(f"  {k:5d}  {c}   [{note}]")
    else:
        print("\nEvery validation label is a training class.")

    absent = sorted(c for c in train_classes if c not in val_matched)
    print(f"\nTraining classes with no validation samples ({len(absent)}):")
    print("  " + "; ".join(absent) if absent else "  none")

    small = sorted((k, c) for c, k in val_matched.items() if k < 5)
    print(
        f"\nMatched classes with fewer than 5 validation samples: {len(small)} "
        f"(per-class scores for these will be very noisy)"
    )

    mat = Counter(s["material"] for s in val)
    tmat = Counter(s["material"] for s in train)
    print(
        "\nMaterial, validation: "
        + ", ".join(f"{k} {v} ({v / n:.1%})" for k, v in sorted(mat.items()))
    )
    print(
        "Material, training:   "
        + ", ".join(f"{k} {v} ({v / len(train):.1%})" for k, v in sorted(tmat.items()))
    )

    for name, cohort in ((a.train, train), (a.val, val)):
        bad = title_mismatches(cohort)
        print(f"\nTitle vs label check, {name}: {len(cohort) - len(bad)}/{len(cohort)} agree")
        for s in bad[:5]:
            print(f"  {s['geo_accession']}: title '{s['title']}' vs label '{s['label']}'")

    print(f"\nWrote {labels_out}\nWrote {overlap_out}")


if __name__ == "__main__":
    main()
