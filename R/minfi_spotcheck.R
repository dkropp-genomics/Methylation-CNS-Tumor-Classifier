#!/usr/bin/env Rscript
# Phase 1c robustness spot-check: do SeSAMe betas agree with minfi preprocessNoob?
#
# For the samples in --samples (all must be in batch 1 of a SeSAMe output
# folder, as in the 20-sample pilot), reprocess the IDATs with minfi
# preprocessNoob (per-sample: noob background + single-sample dye correction)
# and compare with the SeSAMe betas, per sample, over probes that are present
# in both and not NA in either.
#
# Output: a small table (committed) with Pearson r and mean absolute difference.
#
# Usage:
#   Rscript R/minfi_spotcheck.R --samples data/meta/pilot_samples.tsv \
#       --sesame data/betas/pilot3 [--idat-root data/idat] \
#       [--out results/qc/minfi_vs_sesame.tsv]
#
# MINFI_MOCK=1 replaces minfi with SeSAMe betas plus noise (tests only).

fail <- function(...) {
  cat("ERROR: ", ..., "\n", sep = "", file = stderr())
  quit(status = 1)
}

opt <- list(samples = NULL, sesame = NULL, `idat-root` = "data/idat",
            out = "results/qc/minfi_vs_sesame.tsv")
argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) %% 2 != 0) fail("arguments must come as --name value pairs")
for (i in seq(1, length(argv), by = 2)[seq_len(length(argv) / 2)]) {
  key <- sub("^--", "", argv[i])
  if (!(key %in% names(opt))) fail("unknown argument '", argv[i], "'")
  opt[[key]] <- argv[i + 1]
}
if (is.null(opt$samples) || is.null(opt$sesame)) fail("--samples and --sesame are required")
mock <- identical(Sys.getenv("MINFI_MOCK"), "1")

if (!file.exists(opt$samples)) fail("sample table not found: ", opt$samples)
s <- read.delim(opt$samples, stringsAsFactors = FALSE)
if (!all(c("cohort", "geo_accession") %in% names(s))) fail(opt$samples, " needs columns cohort and geo_accession")

minfi_betas <- function(idat_dir, gsm) {
  grn <- list.files(idat_dir, pattern = "_Grn\\.idat(\\.gz)?$")
  pre <- sub("_Grn\\.idat(\\.gz)?$", "", grn)
  ids <- sub("_.*$", "", pre)
  miss <- setdiff(gsm, ids)
  if (length(miss)) fail("sample ", miss[1], ": no Grn IDAT in ", idat_dir)
  rg <- minfi::read.metharray(file.path(idat_dir, pre[match(gsm, ids)]), verbose = FALSE)
  b <- minfi::getBeta(minfi::preprocessNoob(rg))
  colnames(b) <- gsm
  b
}

rows <- list()
for (co in unique(s$cohort)) {
  gsm <- s$geo_accession[s$cohort == co]
  pq <- file.path(opt$sesame, co, "batch_0001.parquet")
  if (!file.exists(pq)) fail("SeSAMe batch file not found: ", pq)
  tab <- as.data.frame(arrow::read_parquet(pq))
  miss <- setdiff(gsm, names(tab))
  if (length(miss)) fail("sample ", miss[1], " is not in ", pq)
  ses <- as.matrix(tab[, gsm, drop = FALSE])
  rownames(ses) <- tab$Probe_ID
  if (mock) {
    set.seed(1)
    mf <- ses + rnorm(length(ses), sd = 0.01)
  } else {
    mf <- minfi_betas(file.path(opt$`idat-root`, co), gsm)
  }
  common <- intersect(rownames(ses), rownames(mf))
  if (length(common) == 0) fail("no probes in common for ", co)
  for (g in gsm) {
    x <- ses[common, g]; y <- mf[common, g]
    ok <- !is.na(x) & !is.na(y)
    rows[[length(rows) + 1]] <- data.frame(
      cohort = co, geo_accession = g, n_probes_compared = sum(ok),
      pearson_r = round(cor(x[ok], y[ok]), 5),
      mean_abs_diff = round(mean(abs(x[ok] - y[ok])), 5),
      stringsAsFactors = FALSE)
  }
}
res <- do.call(rbind, rows)
if ("material" %in% names(s)) res$material <- s$material[match(res$geo_accession, s$geo_accession)]
dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)
write.table(res, opt$out, sep = "\t", quote = FALSE, row.names = FALSE)
print(res, row.names = FALSE)
cat(sprintf("samples: %d | Pearson r min %.4f, median %.4f | mean abs diff median %.4f\n",
            nrow(res), min(res$pearson_r), median(res$pearson_r), median(res$mean_abs_diff)))
cat("wrote ", opt$out, "\n", sep = "")
