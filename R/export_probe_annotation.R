#!/usr/bin/env Rscript
# Export fixed, public probe annotation for the 450K array, one row per probe
# in the beta store, in store column order. Nothing here depends on our samples.
#
#   chr                 chromosome (hg19), from the minfi 450k annotation package
#   on_epic             probe ID is also on the EPIC array (sesameData manifest)
#   sesame_recommended  probe is in SeSAMe's "recommended" quality mask for HM450
#                       (the mask already applied by prep code Q)
#
# Usage (in methyl-r):
#   Rscript R/export_probe_annotation.R \
#     --probes data/betas/zarr/GSE90496.probes.tsv \
#     --out results/probes/hm450_annotation.tsv.gz

fail <- function(...) { message("ERROR: ", ...); quit(status = 1) }

a <- commandArgs(trailingOnly = TRUE)
opt <- function(flag, default) {
  i <- match(flag, a)
  if (is.na(i)) default else a[i + 1]
}
probes_path <- opt("--probes", "data/betas/zarr/GSE90496.probes.tsv")
out_path <- opt("--out", "results/probes/hm450_annotation.tsv.gz")

suppressPackageStartupMessages({library(sesame); library(sesameData)})

if (!file.exists(probes_path)) fail("probe index not found: ", probes_path)
p <- read.delim(probes_path, stringsAsFactors = FALSE)
if (!all(c("col", "Probe_ID") %in% names(p))) fail(probes_path, ": need columns col, Probe_ID")
if (anyDuplicated(p$Probe_ID)) fail(probes_path, ": duplicated probe IDs")

loc <- as.data.frame(IlluminaHumanMethylation450kanno.ilmn12.hg19::Locations)
chr <- as.character(loc$chr)[match(p$Probe_ID, rownames(loc))]

rec <- getMask("HM450", mask_names = "recommended")
if (length(rec) == 0) fail("SeSAMe returned an empty 'recommended' mask for HM450")
if (!all(rec %in% p$Probe_ID)) fail("recommended mask has probes that are not in the store")

epic <- names(sesameData_getManifestGRanges("EPIC"))
if (length(epic) < 800000) fail("EPIC manifest looks too small: ", length(epic))

out <- data.frame(
  col = p$col, Probe_ID = p$Probe_ID,
  chr = ifelse(is.na(chr), "", chr),
  on_epic = p$Probe_ID %in% epic,
  sesame_recommended = p$Probe_ID %in% rec
)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
con <- gzfile(out_path, "w")
write.table(out, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)
cat(sprintf("sesame %s | %d probes | with chr %d | on EPIC %d | recommended mask %d\nWrote %s\n",
            as.character(packageVersion("sesame")), nrow(out), sum(out$chr != ""),
            sum(out$on_epic), sum(out$sesame_recommended), out_path))
