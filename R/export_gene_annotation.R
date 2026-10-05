#!/usr/bin/env Rscript
# Gene and CpG-island annotation for the kept probes, from Illumina's 450K
# annotation (ilmn12.hg19). These are facts about the array, not about our
# samples, so nothing here can leak.
#
#   Rscript R/export_gene_annotation.R            (in the methyl-r environment)
#
# Output: data/meta/hm450_gene_annotation.tsv.gz
#   Probe_ID, chr, pos, genes, gene_groups, relation_to_island
suppressMessages(library(IlluminaHumanMethylation450kanno.ilmn12.hg19))

args <- commandArgs(trailingOnly = TRUE)
probes_path <- if (length(args) >= 1) args[1] else "results/probes/probes_kept.tsv"
out_path <- if (length(args) >= 2) args[2] else "data/meta/hm450_gene_annotation.tsv.gz"

if (!file.exists(probes_path)) stop("ERROR: ", probes_path, " not found")
probes <- read.delim(probes_path, stringsAsFactors = FALSE)
if (!"Probe_ID" %in% names(probes)) stop("ERROR: ", probes_path, " has no Probe_ID column")

a <- as.data.frame(getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19))
missing <- setdiff(probes$Probe_ID, rownames(a))
if (length(missing) > 0)
  stop("ERROR: ", length(missing), " probes are not in the annotation, first: ", missing[1])
a <- a[probes$Probe_ID, ]

out <- data.frame(Probe_ID = probes$Probe_ID, chr = a$chr, pos = a$pos,
                  genes = a$UCSC_RefGene_Name, gene_groups = a$UCSC_RefGene_Group,
                  relation_to_island = a$Relation_to_Island, stringsAsFactors = FALSE)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
con <- gzfile(out_path, "w")
write.table(out, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)
cat("wrote", out_path, ":", nrow(out), "probes;",
    sum(out$genes != ""), "with a gene;\n")
print(table(out$relation_to_island))
