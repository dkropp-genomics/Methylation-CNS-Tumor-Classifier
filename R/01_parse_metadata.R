# Parse GEO series-matrix sample header lines into a tidy per-sample table.
lines  <- readLines("data/meta/GSE90496_sample_header.txt")
fields <- strsplit(lines, "\t", fixed = TRUE)
keys   <- vapply(fields, `[`, character(1), 1)
vals   <- lapply(fields, function(x) gsub('^"|"$', "", x[-1]))

meta <- data.frame(
  geo_accession = vals[[which(keys == "!Sample_geo_accession")]],
  title         = vals[[which(keys == "!Sample_title")]],
  source        = vals[[which(keys == "!Sample_source_name_ch1")]]
)

# Characteristics rows look like "key: value"; each row is one field.
for (i in which(keys == "!Sample_characteristics_ch1")) {
  v   <- vals[[i]]
  key <- names(which.max(table(sub(":.*$", "", v))))   # most common key in the row
  meta[[make.names(trimws(key))]] <- trimws(sub("^[^:]*:", "", v))
}

str(meta)
write.table(meta, "data/meta/GSE90496_samples.tsv",
            sep = "\t", quote = FALSE, row.names = FALSE)