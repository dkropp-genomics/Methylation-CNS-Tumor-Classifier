#!/usr/bin/env nextflow
// Rebuilds the project from public data on one CPU desktop.
//
//   nextflow run pipeline/main.nf -profile quick    external cohort, final models, SHAP
//   nextflow run pipeline/main.nf -profile full     everything, including the searches
//
// Each process is a stage: a short list of the repository's scripts, run in
// order from the repository root. Stages pass a "done" signal to the next one;
// the data itself stays in data/ and results/, where every script expects it.
// A stage that stops can be continued by running the same command again.

process TESTS {
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} -m pytest tests/ -q --tb=short
    """
    stub:
    """
    echo "stub: TESTS"
    """
}

process METADATA {
    // Sample tables from GEO, validation labels, the pre-QC fold table (it
    // fixes the row order of the beta store) and the list of arrays to process.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/parse_geo_metadata.py GSE90496
    ${params.python} scripts/parse_geo_metadata.py GSE109379
    ${params.python} scripts/make_validation_labels.py
    ${params.python} scripts/make_folds.py
    ${params.python} scripts/make_sample_list.py
    """
    stub:
    """
    echo "stub: METADATA"
    """
}

process DOWNLOAD {
    // 32 GB of raw arrays; every file is checked against the committed MD5 lists.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    bash scripts/download_idats.sh --jobs ${params.workers} --remove-tar
    """
    stub:
    """
    echo "stub: DOWNLOAD"
    """
}

process PREPROCESS {
    // SeSAMe, one array at a time, written in resumable batches of 100.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.rscript} R/preprocess_sesame.R --samples data/meta/all_samples.tsv \\
        --out data/betas/sesame_qcdpb --workers ${params.workers}
    """
    stub:
    """
    echo "stub: PREPROCESS"
    """
}

process STORES {
    // Beta stores, sample QC, the modeling fold table, the fixed probe filter,
    // the kept-sample table for the external cohort and the gene annotation.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/build_zarr.py GSE90496 --order-from results/splits/folds_seed42.tsv
    ${params.python} scripts/build_zarr.py GSE109379 --order-from results/meta/GSE109379_labels.tsv
    ${params.python} scripts/sample_qc.py summarize
    ${params.python} scripts/sample_qc.py apply
    ${params.python} scripts/make_folds.py --qc-status results/qc/GSE90496_sample_qc_status.tsv \\
        --store-index data/betas/zarr/GSE90496.samples.tsv
    ${params.rscript} R/export_probe_annotation.R
    ${params.python} scripts/make_probe_filter.py
    ${params.python} scripts/make_validation_table.py
    ${params.rscript} R/export_gene_annotation.R
    """
    stub:
    """
    echo "stub: STORES"
    """
}

process EXPLORE {
    // Look-only checks and diagnostics: nothing here feeds a model.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/check_data_access.py
    ${params.python} scripts/check_features.py
    ${params.python} scripts/explore_embedding.py
    ${params.python} scripts/leakage_demo.py
    ${params.python} scripts/leakage_demo.py --small-repeats 20
    ${params.python} scripts/diagnose_material_shift.py
    """
    stub:
    """
    echo "stub: EXPLORE"
    """
}

process TREES {
    // Random forest grid and LightGBM search inside nested CV, calibration,
    // and the single scoring of the outer test folds.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/run_nested_cv.py configs/rf_v1.yaml --stage inner
    ${params.python} scripts/run_nested_cv.py configs/rf_v1.yaml --stage select
    ${params.python} scripts/run_nested_cv.py configs/rf_v1.yaml --stage outer --final-scoring
    ${params.python} scripts/tune_nested_cv.py configs/lgbm_v1.yaml
    ${params.python} scripts/run_nested_cv.py configs/lgbm_v1.yaml --stage outer --final-scoring
    ${params.python} scripts/calibrate_inner.py rf_v1 lgbm_v1
    ${params.python} scripts/score_outer.py rf_v1 lgbm_v1
    """
    stub:
    """
    echo "stub: TREES"
    """
}

process NETWORKS {
    // Plain and masked networks: inner-fold learning curves, then selection.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/train_nn.py configs/nn_v2.yaml --stage inner
    ${params.python} scripts/train_nn.py configs/nn_v2.yaml --stage select
    """
    stub:
    """
    echo "stub: NETWORKS"
    """
}

process SPARSITY {
    // Sparse-coverage scoring of all five models on the outer test folds.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/sparsity_outer.py configs/sparsity_v1.yaml --stage fit --final-scoring
    ${params.python} scripts/sparsity_outer.py configs/sparsity_centroid_v1.yaml --stage fit --final-scoring
    ${params.python} scripts/sparsity_outer.py configs/sparsity_report_v1.yaml --stage report
    """
    stub:
    """
    echo "stub: SPARSITY"
    """
}

process FINAL_MODELS {
    // The CV reference for the external target, the check that the final
    // config follows its rules, and the five models fit on all reference tumors.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/cv_reference.py rf_v1 lgbm_v1
    ${params.python} scripts/check_final_config.py configs/final_v1.yaml
    ${params.python} scripts/fit_final.py configs/final_v1.yaml
    """
    stub:
    """
    echo "stub: FINAL_MODELS"
    """
}

process EXTERNAL {
    // The external cohort: scored (a saved score is never overwritten), then reported.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/score_validation.py configs/final_v1.yaml --stage score --final-scoring
    ${params.python} scripts/score_validation.py configs/final_v1.yaml --stage report
    """
    stub:
    """
    echo "stub: EXTERNAL"
    """
}

process INTERPRET {
    // TreeSHAP on the final forest, and the check on the published pairs.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/explain_shap.py configs/shap_v1.yaml
    ${params.python} scripts/check_pair_probes.py configs/shap_v1.yaml
    """
    stub:
    """
    echo "stub: INTERPRET"
    """
}

process REPRODUCIBILITY {
    // Refit the plain network and compare it with the saved one.
    input: val ready
    output: val true
    script:
    """
    set -euo pipefail
    cd "${params.repo}"
    ${params.python} scripts/check_nn_reproducibility.py configs/final_v1.yaml nn_plain
    """
    stub:
    """
    echo "stub: REPRODUCIBILITY"
    """
}

workflow {
    // -profile full reruns everything; anything else is the quick rebuild.
    def profiles = workflow.profile.tokenize(',')
    def unknown = profiles - ['quick', 'full', 'standard']
    if (unknown) {
        error "ERROR: unknown profile ${unknown}; use -profile quick or -profile full"
    }
    def full = profiles.contains('full')

    tested = TESTS(channel.value(true))
    meta = METADATA(tested)
    arrays = DOWNLOAD(meta)
    betas = PREPROCESS(arrays)
    stores = STORES(betas)

    // quick: the selected settings come from the committed results/cv/ tables.
    // full: the searches are rerun first, and the final models wait for them.
    selected = stores
    if (full) {
        explored = EXPLORE(stores)
        trees = TREES(explored)
        networks = NETWORKS(trees)
        selected = SPARSITY(networks)
    }

    models = FINAL_MODELS(selected)
    external = EXTERNAL(models)
    interpreted = INTERPRET(external)
    if (full) {
        REPRODUCIBILITY(interpreted)
    }
}
