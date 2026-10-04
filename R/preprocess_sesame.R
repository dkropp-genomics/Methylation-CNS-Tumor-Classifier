#!/usr/bin/env Rscript
# Phase 1c: per-sample SeSAMe preprocessing of 450K IDATs, written in batches.
#
# For every sample: openSesame prep "QCDPB" (quality mask, channel inference,
# nonlinear dye-bias correction, pOOBAH detection masking, noob background).
# Nothing is fit across samples, so both cohorts go through this one script.
#
# Output, per cohort and batch, under <out>/<cohort>/:
#   batch_0001.parquet  Probe_ID + one float32 beta column per sample (NA kept)
#   batch_0001.qc.tsv   one QC row per sample (written last = "batch complete")
# plus <out>/batches.tsv, the sample-to-batch plan. Rerunning skips complete
# batches. A sample that fails is recorded in the QC table with an all-NA
# column; it does not stop the batch.
#
# Usage:
#   Rscript R/preprocess_sesame.R --samples data/meta/pilot_samples.tsv \
#       --out data/betas/pilot [--idat-root data/idat] [--batch-size 100] \
#       [--workers 4] [--verify-first]
#
# --samples: TSV with columns cohort, geo_accession (row order = output order).
# --verify-first: check on the first sample that the two-step call used here
#   gives the same betas as a plain openSesame(prep = "QCDPB").
#
# Environment note: preprocessCore must be built from source with
#   configure.args = "--disable-threading"  (pthread_create error 22 otherwise).
#
# SESAME_MOCK=1 replaces SeSAMe with a fake sample generator (tests only).

fail <- function(...) {
  cat("ERROR: ", ..., "\n", sep = "", file = stderr())
  quit(status = 1)
}

parse_args <- function(argv) {
  opt <- list(`idat-root` = "data/idat", `batch-size` = "100", workers = "4",
              `verify-first` = FALSE, samples = NULL, out = NULL)
  i <- 1
  while (i <= length(argv)) {
    key <- sub("^--", "", argv[i])
    if (!startsWith(argv[i], "--") || !(key %in% names(opt)))
      fail("unknown argument '", argv[i], "'")
    if (key == "verify-first") { opt[[key]] <- TRUE; i <- i + 1; next }
    if (i == length(argv)) fail("argument --", key, " needs a value")
    opt[[key]] <- argv[i + 1]; i <- i + 2
  }
  if (is.null(opt$samples) || is.null(opt$out)) fail("--samples and --out are required")
  opt$`batch-size` <- as.integer(opt$`batch-size`)
  opt$workers <- as.integer(opt$workers)
  if (is.na(opt$`batch-size`) || opt$`batch-size` < 1) fail("--batch-size must be a positive integer")
  if (is.na(opt$workers) || opt$workers < 1) fail("--workers must be a positive integer")
  opt
}

# Map each GSM to its IDAT prefix; stop unless exactly one Grn/Red pair exists.
find_prefixes <- function(idat_dir, gsm) {
  if (!dir.exists(idat_dir)) fail("IDAT directory not found: ", idat_dir)
  grn <- list.files(idat_dir, pattern = "_Grn\\.idat(\\.gz)?$")
  pre <- sub("_Grn\\.idat(\\.gz)?$", "", grn)
  ids <- sub("_.*$", "", pre)
  out <- character(length(gsm))
  for (k in seq_along(gsm)) {
    hit <- pre[ids == gsm[k]]
    if (length(hit) != 1)
      fail("sample ", gsm[k], ": expected 1 Grn IDAT in ", idat_dir, ", found ", length(hit))
    red <- file.path(idat_dir, paste0(hit, c("_Red.idat.gz", "_Red.idat")))
    if (!any(file.exists(red))) fail("sample ", gsm[k], ": Red IDAT missing in ", idat_dir)
    out[k] <- file.path(idat_dir, hit)
  }
  out
}

mock_sample <- function(prefix) {
  if (grepl("BAD", basename(prefix))) stop("mock failure")
  set.seed(sum(utf8ToInt(basename(prefix))))
  b <- round(runif(1000), 4)
  b[sample(1000, 100)] <- NA
  names(b) <- sprintf("cg%08d", 1:1000)
  list(betas = b, frac_detected = 0.95, frac_pval_na = 0, mean_intensity = 5000, bisulfite_gct = 1.1)
}

safe_num <- function(expr) tryCatch(as.numeric(expr)[1], error = function(e) NA_real_)

sesame_sample <- function(prefix) {
  # Stop after "QCD" to read the pOOBAH p-values, then finish with "PB".
  sdf <- sesame::openSesame(prefix, prep = "QCD", func = NULL)
  pval <- sesame::pOOBAH(sdf, return.pval = TRUE)
  mi <- safe_num(sesame::meanIntensity(sdf))
  gct <- safe_num(sesame::bisConversionControl(sdf))
  sdf <- sesame::prepSesame(sdf, "PB")
  # pOOBAH returns NA p-values for some probes; report their share separately.
  list(betas = sesame::getBetas(sdf), frac_detected = mean(pval < 0.05, na.rm = TRUE),
       frac_pval_na = mean(is.na(pval)), mean_intensity = mi, bisulfite_gct = gct)
}

# Never throws: a failure becomes status = "error: ...".
process_one <- function(prefix, mock) {
  t0 <- proc.time()[["elapsed"]]
  r <- tryCatch(if (mock) mock_sample(prefix) else sesame_sample(prefix),
                error = function(e) list(error = conditionMessage(e)))
  r$seconds <- round(proc.time()[["elapsed"]] - t0, 2)
  r
}

write_batch <- function(res, cohort, gsm, prefixes, pq, qc_path) {
  ok <- vapply(res, function(r) is.null(r$error), logical(1))
  if (!any(ok)) fail("every sample failed in ", basename(pq), " (", cohort,
                     "); first error: ", res[[1]]$error)
  probes <- names(res[[which(ok)[1]]]$betas)
  cols <- list(Probe_ID = probes)
  for (k in seq_along(res)) {
    if (ok[k]) {
      if (!identical(names(res[[k]]$betas), probes))
        fail("sample ", gsm[k], ": probe order differs from ", gsm[which(ok)[1]])
      cols[[gsm[k]]] <- unname(res[[k]]$betas)
    } else {
      cols[[gsm[k]]] <- rep(NA_real_, length(probes))
    }
  }
  df <- as.data.frame(cols, check.names = FALSE, stringsAsFactors = FALSE)
  fields <- c(list(Probe_ID = arrow::utf8()),
              stats::setNames(rep(list(arrow::float32()), length(gsm)), gsm))
  tab <- arrow::Table$create(df)$cast(do.call(arrow::schema, fields))
  arrow::write_parquet(tab, paste0(pq, ".part"))

  num <- function(f) vapply(res, function(r) if (is.null(r[[f]])) NA_real_ else as.numeric(r[[f]]), numeric(1))
  qc <- data.frame(
    cohort = cohort, geo_accession = gsm, idat_prefix = basename(prefixes),
    status = vapply(res, function(r) if (is.null(r$error)) "ok" else
      paste("error:", gsub("[\t\n]", " ", r$error)), character(1)),
    n_probes = ifelse(ok, length(probes), NA),
    frac_na = vapply(seq_along(res), function(k)
      if (ok[k]) mean(is.na(res[[k]]$betas)) else NA_real_, numeric(1)),
    frac_detected = num("frac_detected"), frac_pval_na = num("frac_pval_na"),
    mean_intensity = num("mean_intensity"),
    bisulfite_gct = num("bisulfite_gct"), seconds = num("seconds"),
    stringsAsFactors = FALSE)
  write.table(qc, paste0(qc_path, ".part"), sep = "\t", quote = FALSE, row.names = FALSE)
  if (!file.rename(paste0(pq, ".part"), pq)) fail("could not write ", pq)
  if (!file.rename(paste0(qc_path, ".part"), qc_path)) fail("could not write ", qc_path)
  qc
}

main <- function() {
  opt <- parse_args(commandArgs(trailingOnly = TRUE))
  mock <- identical(Sys.getenv("SESAME_MOCK"), "1")
  if (!file.exists(opt$samples)) fail("sample table not found: ", opt$samples)
  s <- read.delim(opt$samples, stringsAsFactors = FALSE, check.names = FALSE)
  if (!all(c("cohort", "geo_accession") %in% names(s)))
    fail(opt$samples, " needs columns cohort and geo_accession")
  if (nrow(s) == 0) fail(opt$samples, " has no samples")
  if (anyDuplicated(s$geo_accession))
    fail("duplicated sample in ", opt$samples, ": ", s$geo_accession[anyDuplicated(s$geo_accession)])

  # Batch plan: consecutive chunks within each cohort, in table order.
  s$batch <- ave(seq_len(nrow(s)), s$cohort,
                 FUN = function(i) (seq_along(i) - 1) %/% opt$`batch-size` + 1)
  plan <- s[, c("cohort", "batch", "geo_accession")]

  # Check every input before writing anything, so a failed start leaves no plan behind.
  s$prefix <- NA_character_
  for (co in unique(s$cohort)) {
    i <- s$cohort == co
    s$prefix[i] <- find_prefixes(file.path(opt$`idat-root`, co), s$geo_accession[i])
  }

  dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)
  plan_path <- file.path(opt$out, "batches.tsv")
  if (file.exists(plan_path)) {
    old <- read.delim(plan_path, stringsAsFactors = FALSE)
    if (!isTRUE(all.equal(old, plan, check.attributes = FALSE)))
      fail(plan_path, " does not match this sample table and batch size; ",
           "use a new --out directory or the original settings")
  } else {
    write.table(plan, plan_path, sep = "\t", quote = FALSE, row.names = FALSE)
  }

  if (!mock) suppressMessages(library(sesame))
  if (!mock) {
    # Warm-up: process one sample here so SeSAMe's annotation data is loaded
    # before the workers fork. Workers that each open the ExperimentHub cache
    # at the same moment can fail with "error reading from connection".
    a <- sesame_sample(s$prefix[1])$betas
    if (opt$`verify-first`) {
      b <- sesame::openSesame(s$prefix[1], prep = "QCDPB")
      if (!identical(a, b)) fail("two-step QCD + PB betas differ from prep = 'QCDPB' for ", s$geo_accession[1])
      cat("verify-first: two-step betas identical to prep = 'QCDPB'\n")
    }
  }
  bp <- if (opt$workers > 1) BiocParallel::MulticoreParam(opt$workers) else BiocParallel::SerialParam()

  n_fail <- 0
  for (co in unique(s$cohort)) {
    dir.create(file.path(opt$out, co), showWarnings = FALSE)
    for (b in unique(s$batch[s$cohort == co])) {
      i <- which(s$cohort == co & s$batch == b)
      stem <- file.path(opt$out, co, sprintf("batch_%04d", b))
      pq <- paste0(stem, ".parquet"); qc_path <- paste0(stem, ".qc.tsv")
      if (file.exists(pq) && file.exists(qc_path)) {
        cat(sprintf("skip  %s batch %d (complete)\n", co, b)); next
      }
      t0 <- proc.time()[["elapsed"]]
      res <- BiocParallel::bplapply(s$prefix[i], process_one, mock = mock, BPPARAM = bp)
      # Retry failures once, one at a time, so a passing glitch is not recorded as a failed sample.
      for (k in which(vapply(res, function(r) !is.null(r$error), logical(1)))) {
        cat("  retry ", s$geo_accession[i][k], " after: ", res[[k]]$error, "\n", sep = "")
        res[[k]] <- process_one(s$prefix[i][k], mock)
      }
      qc <- write_batch(res, co, s$geo_accession[i], s$prefix[i], pq, qc_path)
      bad <- qc$geo_accession[qc$status != "ok"]
      n_fail <- n_fail + length(bad)
      cat(sprintf("done  %s batch %d: %d samples, %d failed, %.0f s wall, median %.1f s/sample\n",
                  co, b, length(i), length(bad), proc.time()[["elapsed"]] - t0,
                  median(qc$seconds)))
      for (g in bad) cat("  FAILED ", g, ": ", qc$status[qc$geo_accession == g], "\n", sep = "")
    }
  }
  cat(sprintf("finished: %d samples in plan, %d failed in batches run now\n", nrow(s), n_fail))
}

main()
