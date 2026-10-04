#!/usr/bin/env bash
# =============================================================================
# download_idats.sh -- download, extract and verify raw 450K IDATs from GEO
# =============================================================================
#
# WHAT IT DOES (per GEO series, e.g. GSE90496)
#   1. Asks GEO for the size of <GSE>_RAW.tar and checks there is enough disk.
#   2. Downloads it (resumable: re-running continues a broken download) to
#      data/raw/<GSE>_RAW.tar, via a ".part" file so a half-finished download
#      is never mistaken for a finished one.
#   3. Lists the tar and checks it BEFORE extracting: every IDAT entry must be
#      a GSM..._Grn.idat.gz / _Red.idat.gz file, and every array must have both
#      colour channels. Illumina platform files GEO bundles (GPL..._*) are
#      logged and skipped; anything else stops the run. A truncated tar fails
#      here too.
#   4. Extracts only the IDATs into data/idat/<GSE>/ (skipped if the IDATs
#      already there match the tar listing, name for name and byte for byte).
#   5. Tests every .idat.gz with `gzip -t` and records its MD5 in
#      results/download/<GSE>_idat_md5.tsv. That small table is COMMITTED to
#      Git: on any later rerun (or on someone else's computer) new MD5s are
#      compared with it, so a changed or corrupted file is caught.
#   6. If data/meta/<GSE>_samples.tsv exists, checks that the GSM IDs in the
#      IDATs match the metadata exactly (no missing or extra samples).
#   7. Appends a row to results/download/download_summary.tsv (sizes, counts,
#      tar SHA-256, disk use) and writes a stamp file so the next run skips
#      the dataset in seconds.
#
# Nothing is ever deleted unless you pass --remove-tar, and even then only
# after every check above has passed. SeSAMe reads .idat.gz directly, so the
# IDATs stay compressed.
#
# USAGE (from the repo root, any conda env; needs only standard Linux tools)
#   bash scripts/download_idats.sh                    # both cohorts
#   bash scripts/download_idats.sh GSE90496           # one cohort
#   bash scripts/download_idats.sh --jobs 6 --remove-tar
#   bash scripts/download_idats.sh --reverify         # re-check existing files
#
# OPTIONS
#   --data-dir DIR    root data folder              (default: data)
#   --results-dir DIR where manifests/summary go    (default: results/download)
#   --jobs N          parallel gzip/MD5 checks      (default: 4)
#   --remove-tar      delete the tar after everything verifies
#   --reverify        ignore stamp files and re-check everything
#   --offline         don't contact GEO; use the local tar as-is
#   -h, --help        show this help
#
# ENVIRONMENT (mainly for tests)
#   GEO_BASE_URL   default https://ftp.ncbi.nlm.nih.gov/geo
#   MIN_FREE_GB    free space to keep in reserve (default: 5)
#
# EXIT STATUS: 0 when every requested dataset is verified, non-zero otherwise.
# =============================================================================

set -Eeuo pipefail
shopt -s nullglob

# ---------------------------------------------------------------- defaults --
DATA_DIR="data"
RESULTS_DIR="results/download"
LOG_DIR="logs"
JOBS=4
REMOVE_TAR=0
REVERIFY=0
OFFLINE=0
DEFAULT_DATASETS=(GSE90496 GSE109379)
GEO_BASE_URL="${GEO_BASE_URL:-https://ftp.ncbi.nlm.nih.gov/geo}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"

# Every IDAT GEO serves for a 450K series looks like
#   GSM2402855_5775041068_R04C01_Grn.idat.gz
IDAT_REGEX='^GSM[0-9]+_[A-Za-z0-9_]+_(Grn|Red)\.idat\.gz$'
# Illumina platform files GEO sometimes bundles, e.g.
#   GPL13534_HumanMethylation450_15017482_v.1.2.bpm.gz
PLATFORM_REGEX='^GPL[0-9]+_[^/]+$'

# ----------------------------------------------------------------- helpers --
log()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die()  { log "ERROR: $*"; exit 1; }
usage() { sed -n '5,/^# EXIT STATUS/{s/^# \{0,1\}//;p}' "$0"; }
trap 'log "ERROR: command failed (line $LINENO): $BASH_COMMAND"' ERR

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "${1} bytes"; }
file_bytes() { stat -c %s "$1"; }
free_bytes() { df --output=avail -B1 "$1" | tail -n 1 | tr -d ' '; }

# GEO groups series in folders like GSE90nnn / GSE109nnn.
geo_tar_url() {
    local acc=$1
    printf '%s/series/%snnn/%s/suppl/%s_RAW.tar\n' \
        "$GEO_BASE_URL" "${acc:0:${#acc}-3}" "$acc" "$acc"
}

# Size of the remote file in bytes, following redirects (last header wins).
remote_bytes() {
    curl -sfIL --retry 3 --retry-delay 5 "$1" | tr -d '\r' |
        awk 'tolower($1) == "content-length:" { n = $2 } END { if (n != "") print n; else exit 1 }'
}

require_space() {  # require_space <bytes needed> <dir> <what for>
    local need=$(( $1 + MIN_FREE_GB * 1024 * 1024 * 1024 )) have
    have=$(free_bytes "$2")
    (( have >= need )) ||
        die "not enough disk for $3: need $(human "$need") (incl. ${MIN_FREE_GB} GB reserve), have $(human "$have")"
}

# ------------------------------------------------------------- arguments ----
DATASETS=()
while (( $# )); do
    case $1 in
        --data-dir)    DATA_DIR=${2:?}; shift 2 ;;
        --results-dir) RESULTS_DIR=${2:?}; shift 2 ;;
        --jobs)        JOBS=${2:?}; shift 2 ;;
        --remove-tar)  REMOVE_TAR=1; shift ;;
        --reverify)    REVERIFY=1; shift ;;
        --offline)     OFFLINE=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        GSE*)          DATASETS+=("$1"); shift ;;
        *)             die "unknown argument: $1 (see --help)" ;;
    esac
done
(( ${#DATASETS[@]} )) || DATASETS=("${DEFAULT_DATASETS[@]}")
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
for acc in "${DATASETS[@]}"; do
    [[ $acc =~ ^GSE[0-9]{4,}$ ]] || die "not a GEO series accession: $acc"
done

for cmd in wget curl tar gzip md5sum sha256sum df du stat xargs awk sort comm numfmt; do
    command -v "$cmd" >/dev/null || die "required command not found: $cmd"
done

RAW_DIR="$DATA_DIR/raw"
IDAT_ROOT="$DATA_DIR/idat"
mkdir -p "$RAW_DIR" "$IDAT_ROOT" "$RESULTS_DIR" "$LOG_DIR"
SUMMARY="$RESULTS_DIR/download_summary.tsv"
LOG_FILE="$LOG_DIR/download_idats_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

log "download_idats.sh  datasets: ${DATASETS[*]}  jobs: $JOBS  log: $LOG_FILE"

# -------------------------------------------------------------- steps -------

# Step 2: download with resume; only a size-verified file gets the final name.
download_tar() {  # download_tar <acc> <url> <tar path> <remote bytes or "">
    local acc=$1 url=$2 tar=$3 remote=$4 part="$3.part" have=0
    if [[ -f $tar ]]; then
        if [[ -z $remote ]]; then
            log "$acc: using local $tar (offline; integrity is checked by listing it)"
            return
        fi
        have=$(file_bytes "$tar")
        if (( have == remote )); then
            log "$acc: $tar already complete ($(human "$have")); not downloading"
            return
        fi
        # A final-named file of the wrong size came from outside this script.
        log "$acc: $tar is $(human "$have") but GEO says $(human "$remote"); resuming it"
        mv "$tar" "$part"
    fi
    [[ -n $remote ]] || die "$acc: no local tar and --offline was given"
    [[ -f $part ]] && have=$(file_bytes "$part") || have=0
    # Space for the rest of the tar plus the extracted copy of it.
    require_space $(( remote - have + remote )) "$RAW_DIR" "$acc download + extraction"

    log "$acc: downloading $(human "$remote") from $url"
    wget --continue --tries=20 --waitretry=30 --retry-connrefused \
         --progress=dot:giga -O "$part" "$url"
    have=$(file_bytes "$part")
    (( have == remote )) || die "$acc: downloaded $have bytes, expected $remote; rerun to resume"
    mv "$part" "$tar"
    log "$acc: download complete"
}

# Step 3: list the tar as "name<TAB>bytes" and check its contents.
# The listing written to <out> holds IDATs only. GEO also bundles Illumina
# platform files (e.g. GPL13534_HumanMethylation450_..._v.1.2.bpm.gz); those
# are logged and not extracted, because SeSAMe ships its own 450K manifest.
# Anything else is unexpected and stops the run.
list_and_check_tar() {  # list_and_check_tar <acc> <tar> <out listing>
    local acc=$1 tar=$2 out=$3 all="$WORK/$1.all" platform bad
    # `tar -tv` reads every header, so a truncated archive fails here.
    tar -tvf "$tar" | awk '{ print $6 "\t" $3 }' | LC_ALL=C sort > "$all" ||
        die "$acc: cannot list $tar (truncated or corrupt?); delete it and rerun"
    [[ -s $all ]] || die "$acc: $tar is empty"
    # Pass the regex via ENVIRON, not `awk -v`: -v turns `\.` into `.`
    # (any character) and gawk warns about it.
    RE="$IDAT_REGEX" awk -F'\t' '$1 ~ ENVIRON["RE"]' "$all" > "$out"
    platform=$(cut -f1 "$all" | grep -E "$PLATFORM_REGEX" || true)
    bad=$(cut -f1 "$all" | grep -Ev "$IDAT_REGEX" | grep -Ev "$PLATFORM_REGEX" || true)
    [[ -z $bad ]] || die "$acc: unexpected entries in tar:"$'\n'"$(head <<< "$bad")"
    [[ -s $out ]] || die "$acc: $tar contains no IDAT files"
    if [[ -n $platform ]]; then
        log "$acc: tar also holds $(wc -l <<< "$platform") Illumina platform file(s); not extracting them:"
        sed 's/^/    /' <<< "$platform"
    fi
    check_pairs "$acc" "$out"
}

# Every array (GSM + slide + position) needs exactly one Grn and one Red file.
check_pairs() {  # check_pairs <acc> <file whose column 1 is IDAT names>
    local unpaired
    unpaired=$(cut -f1 "$2" | sed -E 's/_(Grn|Red)\.idat\.gz$//' | sort | uniq -c |
               awk '$1 != 2 { print $2 }')
    [[ -z $unpaired ]] || die "$1: arrays without both Grn and Red IDATs:"$'\n'"$(head <<< "$unpaired")"
    # One array per GSM (a GSM with two arrays would be ambiguous downstream).
    local dup
    dup=$(cut -f1 "$2" | cut -d_ -f1 | sort | uniq -c | awk '$1 != 2 { print $2 }')
    [[ -z $dup ]] || die "$1: GSM IDs with more than one array:"$'\n'"$(head <<< "$dup")"
}

# Files currently on disk, same format as the tar listing.
list_dir() {  # list_dir <dir> <out>
    find "$1" -maxdepth 1 -type f -name '*.idat*' -printf '%f\t%s\n' | LC_ALL=C sort > "$2"
}

# Step 4: extract unless the directory already matches the tar exactly.
extract_tar() {  # extract_tar <acc> <tar> <idat dir> <tar listing>
    local acc=$1 tar=$2 dir=$3 listing=$4 present="$WORK/$1.present" extra
    mkdir -p "$dir"
    list_dir "$dir" "$present"
    if cmp -s "$listing" "$present"; then
        log "$acc: $dir already matches the tar ($(wc -l < "$listing") files); not extracting"
        return
    fi
    extra=$(LC_ALL=C comm -13 <(cut -f1 "$listing") <(cut -f1 "$present"))
    [[ -z $extra ]] || die "$acc: $dir holds files that are not in the tar (move them away first):"$'\n'"$(head <<< "$extra")"
    require_space "$(awk -F'\t' '{ s += $2 } END { print s + 0 }' "$listing")" "$dir" "$acc extraction"
    log "$acc: extracting into $dir"
    tar -xf "$tar" -C "$dir" --wildcards '*.idat.gz'
    list_dir "$dir" "$present"
    cmp -s "$listing" "$present" || die "$acc: extracted files don't match the tar listing"
    log "$acc: extracted $(wc -l < "$present") files"
}

# Step 5: gzip-test every IDAT and record its MD5 (parallel).
checksum_idats() {  # checksum_idats <acc> <idat dir> <out: name<TAB>md5>
    local acc=$1 dir=$2 out=$3 n
    n=$(find "$dir" -maxdepth 1 -name '*.idat.gz' | wc -l)
    log "$acc: gzip-testing and checksumming $n files with $JOBS workers"
    ( cd "$dir" && find . -maxdepth 1 -name '*.idat.gz' -printf '%f\0' |
        xargs -0 -n 64 -P "$JOBS" bash -c '
            for f; do
                gzip -t "$f" 2>/dev/null || { echo "CORRUPT (gzip -t failed): $f" >&2; exit 255; }
                md5sum "$f"
            done' _ ) |
        awk '{ print $2 "\t" $1 }' | LC_ALL=C sort > "$out" ||
        die "$acc: integrity check failed (see CORRUPT lines above)"
    (( $(wc -l < "$out") == n )) || die "$acc: checksummed $(wc -l < "$out") of $n files"
}

# Compare with a manifest from an earlier run, or write the first one.
reconcile_manifest() {  # reconcile_manifest <acc> <new md5 table> <manifest>
    local acc=$1 new=$2 manifest=$3 old="$WORK/$1.old"
    if [[ -f $manifest ]]; then
        tail -n +2 "$manifest" | LC_ALL=C sort > "$old"
        if ! cmp -s "$old" "$new"; then
            log "$acc: files differ from $manifest (first differences):"
            diff "$old" "$new" | head -n 20 || true
            die "$acc: checksum mismatch; investigate before continuing"
        fi
        log "$acc: all MD5s match the committed manifest"
    else
        { printf 'file\tmd5\n'; cat "$new"; } > "$manifest"
        log "$acc: wrote $manifest (commit this file)"
    fi
}

# Step 6: GSM IDs in the IDATs vs the parsed series-matrix metadata.
check_against_metadata() {  # check_against_metadata <acc> <file with IDAT names in col 1>
    local acc=$1 meta="$DATA_DIR/meta/$1_samples.tsv" col missing extra
    if [[ ! -f $meta ]]; then
        log "$acc: no $meta yet; skipping sample-ID cross-check"
        return
    fi
    col=$(head -n 1 "$meta" | tr '\t' '\n' | grep -nx 'geo_accession' | cut -d: -f1 || true)
    [[ -n $col ]] || die "$acc: $meta has no geo_accession column"
    tail -n +2 "$meta" | cut -f "$col" | tr -d '\r' | LC_ALL=C sort -u > "$WORK/$acc.meta_ids"
    cut -f1 "$2" | cut -d_ -f1 | LC_ALL=C sort -u > "$WORK/$acc.idat_ids"
    missing=$(LC_ALL=C comm -23 "$WORK/$acc.meta_ids" "$WORK/$acc.idat_ids")
    extra=$(LC_ALL=C comm -13 "$WORK/$acc.meta_ids" "$WORK/$acc.idat_ids")
    [[ -z $missing ]] || die "$acc: $(wc -l <<< "$missing") samples in metadata have no IDATs, e.g. $(head -n 3 <<< "$missing" | paste -sd' ')"
    [[ -z $extra ]]   || die "$acc: $(wc -l <<< "$extra") IDAT samples are not in metadata, e.g. $(head -n 3 <<< "$extra" | paste -sd' ')"
    log "$acc: all $(wc -l < "$WORK/$acc.meta_ids") metadata samples have IDATs, and no extras"
}

# Step 7: one summary row per dataset (replaces any earlier row for it).
write_summary() {  # write_summary <acc> <url> <tar bytes> <tar sha256> <n files> <idat dir>
    local acc=$1 n_samples dir_bytes row
    n_samples=$(( $5 / 2 ))
    dir_bytes=$(du -sb "$6" | cut -f1)
    row=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
        "$acc" "$2" "$3" "$4" "$5" "$n_samples" "$dir_bytes" "$(date -u +%FT%TZ)")
    {
        printf 'accession\turl\ttar_bytes\ttar_sha256\tn_idat_files\tn_samples\tidat_dir_bytes\tverified_utc\n'
        [[ -f $SUMMARY ]] && tail -n +2 "$SUMMARY" | awk -F'\t' -v a="$acc" '$1 != a'
        printf '%s\n' "$row"
    } > "$WORK/summary" && mv "$WORK/summary" "$SUMMARY"
    log "$acc: $n_samples samples, $5 IDATs, $(human "$dir_bytes") on disk"
}

# ------------------------------------------------------------ one dataset ---
process_dataset() {
    local acc=$1 url tar idat_dir manifest stamp remote="" listing md5s tar_sha n
    url=$(geo_tar_url "$acc")
    tar="$RAW_DIR/${acc}_RAW.tar"
    idat_dir="$IDAT_ROOT/$acc"
    manifest="$RESULTS_DIR/${acc}_idat_md5.tsv"
    stamp="$idat_dir/.verified"
    listing="$WORK/$acc.listing"
    md5s="$WORK/$acc.md5"

    log "=== $acc ==="
    if [[ -f $stamp && $REVERIFY -eq 0 ]]; then
        log "$acc: already verified on $(cat "$stamp"); skipping (use --reverify to re-check)"
        # --remove-tar on a dataset verified earlier: safe, because the IDATs
        # passed every check and the MD5 manifest can re-verify them later.
        if (( REMOVE_TAR )) && [[ -f $tar && -f $manifest ]]; then
            rm -f "$tar"
            log "$acc: removed $tar (verified earlier; manifest kept)"
        fi
        return
    fi

    if [[ ! -f $tar && -f $manifest && -d $idat_dir ]]; then
        # Tar was removed after an earlier verified run: check the IDATs
        # against the committed manifest instead of downloading again.
        log "$acc: no tar, but $manifest exists; verifying $idat_dir against it"
        tail -n +2 "$manifest" | cut -f1 > "$listing"
        list_dir "$idat_dir" "$WORK/$acc.present"
        cmp -s "$listing" <(cut -f1 "$WORK/$acc.present") ||
            die "$acc: file names in $idat_dir differ from $manifest; rerun without the manifest to re-download"
        tar_sha="removed"
    else
        if (( OFFLINE == 0 )); then
            remote=$(remote_bytes "$url") || die "$acc: could not get the file size from GEO ($url)"
            log "$acc: GEO reports $(human "$remote")"
        fi
        download_tar "$acc" "$url" "$tar" "$remote"
        list_and_check_tar "$acc" "$tar" "$listing"
        extract_tar "$acc" "$tar" "$idat_dir" "$listing"
        log "$acc: computing SHA-256 of the tar"
        tar_sha=$(sha256sum "$tar" | cut -d' ' -f1)
    fi

    check_pairs "$acc" "$listing"
    check_against_metadata "$acc" "$listing"
    checksum_idats "$acc" "$idat_dir" "$md5s"
    reconcile_manifest "$acc" "$md5s" "$manifest"
    n=$(wc -l < "$md5s")
    write_summary "$acc" "$url" "$([[ -f $tar ]] && file_bytes "$tar" || echo NA)" "$tar_sha" "$n" "$idat_dir"

    date '+%F %T' > "$stamp"
    if (( REMOVE_TAR )) && [[ -f $tar ]]; then
        rm -f "$tar"
        log "$acc: removed $tar (all checks passed)"
    fi
    log "$acc: verified"
}

for acc in "${DATASETS[@]}"; do
    process_dataset "$acc"
done

log "Disk use:"
du -sh "$RAW_DIR" "$IDAT_ROOT"/* 2>/dev/null || true
df -h "$DATA_DIR" | tail -n 1
log "Done. Summary: $SUMMARY"
