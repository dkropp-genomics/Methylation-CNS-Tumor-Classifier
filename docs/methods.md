# Methods in detail

This file holds the detail behind the one-line statements in the README. Each
section ends with the files that hold its numbers.

## 1. Data

| Accession | Role | Arrays | Tissue |
|---|---|---|---|
| GSE90496 | Reference cohort | 2,801 tumors and controls in 91 classes | 1,878 FFPE (67.0%), 923 frozen |
| GSE109379 | External cohort | 1,104 tumors in 69 classes | 1,094 FFPE (99.1%), 10 frozen |

Both are Illumina HumanMethylation450 arrays from Capper et al. (2018). Class
sizes in the reference cohort run from 8 to 143 (median 21); 41 classes have
fewer than 20 samples. Tissue type is confounded with class: 34 classes are
entirely FFPE and one is entirely frozen.

Every downloaded file was tested with `gzip -t`, and its MD5 is committed, so a
later download can be checked against the one used here.

Files: `results/download/`, `results/meta/`.

## 2. Preprocessing

Arrays were processed one at a time with SeSAMe 1.24.0, preparation code
`QCDPB`: quality mask, channel inference, nonlinear dye-bias correction, pOOBAH
detection masking, and noob background correction. No step uses more than one
sample, so nothing can leak between samples or cohorts and a new sample can be
processed alone. This is the reason SeSAMe was chosen over minfi's
across-sample normalizations.

All 3,905 arrays were processed with no failures. As a spot-check, 20 samples
were also processed with minfi's `preprocessNoob`: Pearson correlation with the
SeSAMe values had a minimum of 0.9935 and a median of 0.9983, with a median
absolute difference of 0.025 in beta value. Twenty samples is a spot-check, not
a validation.

Files: `results/qc/GSE90496_sesame_qc.tsv`, `results/qc/GSE109379_sesame_qc.tsv`,
`results/qc/minfi_vs_sesame.tsv`.

## 3. Sample quality control

Arrays were kept when at least 70% of probes were detected (pOOBAH p < 0.05).
The threshold was chosen from the pooled reference-cohort distribution only,
without looking at class, tissue type or the external cohort, and committed in
`configs/sample_qc.yaml` before it was applied.

There was no natural gap in the distribution. The reasoning was technical:
arrays below 60% detected have a median intensity of 903 against 6,861 overall
(failed arrays), while above 70% intensity approaches normal (degraded FFPE DNA
on a working array, which this project is meant to handle). The conventional
90 to 95% cuts would have removed 16 to 39% of the cohort.

| Cohort | Removed | Kept |
|---|---|---|
| Reference | 34 (1.2%), all FFPE | 2,767 |
| External | 8 (0.7%) | 1,096 |

Seventeen reference classes lost at least one sample. One class, `PLEX, PED B`,
lost 10 of 46; every other class lost 1 to 3. No class fell below 8 samples.
Folds were redrawn after this step, not filtered, so that every class stays in
every test fold.

Files: `results/qc/*_sample_qc_status.tsv`, `results/qc/*_qc_class_impact.tsv`.

## 4. Probe filter

Filters that depend only on the array design were applied to all samples at once:

| Step | Removed | Remaining |
|---|---|---|
| Start | | 486,427 |
| Non-CpG, control and SNP probes | 4,006 | 482,421 |
| Sex chromosomes | 11,551 | 470,870 |
| SeSAMe recommended quality mask | 61,904 | 408,966 |
| Not on the EPIC array | 27,611 | 381,355 |

Capper et al. kept 428,799 probes. The difference is the quality mask: theirs
removed about 12,000 probes for SNPs and non-unique mapping, SeSAMe's
recommended mask removes about 62,000.

Files: `results/probes/`.

## 5. Cross-validation design

Nested cross-validation with 5 outer and 3 inner folds, stratified by class,
seed 42. The fold assignment is a committed file, so every model sees the same
splits. All 91 classes occur in every outer test fold and every inner training
set; the smallest class count in an inner training set is 4.

Settings are chosen on inner folds. Each outer test fold was scored once per
model. "Test" always means an outer fold; the external cohort is called
"validation" in the code and was never used for tuning.

Files: `results/splits/folds_seed42_qc.tsv`.

## 6. Feature steps inside each fold

Four steps are learned from the training fold only and applied unchanged to
held-out samples:

1. Drop CpGs missing in more than 5% of training-fold samples.
2. Fill remaining gaps with the training-fold median of that CpG.
3. Optionally remove the FFPE-against-frozen shift (section 7).
4. Keep the most variable CpGs of the training fold.

The rule for what belongs inside a fold: if a sample's result would change when
other samples are added or removed, the step is fit on training samples only.
The tests include "poison" tests that corrupt held-out rows or labels and check
that nothing fitted changes.

## 7. FFPE and frozen tissue

A plain FFPE-minus-frozen correction would remove class signal, because many
classes have only one tissue type. The correction used here is estimated within
classes: for each CpG, the FFPE-minus-frozen difference is taken inside every
class that has at least 2 samples of each type, then averaged with weights
n_ffpe x n_frozen / (n_ffpe + n_frozen). This equals the tissue coefficient of
the linear model `beta ~ class + tissue`. FFPE samples are shifted onto the
frozen scale. Whether to apply it was decided on inner folds, per model.

What a diagnostic on one training fold showed (2,213 samples, 40 classes with
both tissue types):

- The largest shifts are shared across classes. For the top CpG (shift 0.605)
  all 40 classes have the same sign.
- Tissue type is a strong, shared signature: a model trained to tell FFPE from
  frozen on some classes scores an AUC of 0.999 on classes it never saw.
- The correction removes this on average (the same detector scores 0.44 on
  corrected samples) but not exactly per class: 7 of 40 held-out classes are
  over-corrected and 3 under-corrected.
- The cause of a 0.6 beta difference between tissue types is not known.

The assumption that cannot be tested: that the shift is the same in the 34
classes with only FFPE samples.

Files: `results/material_shift/`.

## 8. Models and how settings were chosen

**Random forest.** 500 trees, balanced class weights. Six settings: 5,000,
10,000 or 20,000 CpGs, with and without the tissue correction. Inner macro-F1
ranged from 0.933 to 0.938, less than the fold-to-fold spread, and the five
outer folds chose five different settings.

**LightGBM.** Optuna, 20 trials per outer fold (300 fits), searching the number
of CpGs, the correction, boosting rounds, learning rate, tree size and
regularization. There is no early stopping; the number of rounds is a searched
setting, because stopping on a fold and then reporting that fold's score
flatters the model. The search had not settled (one fold's best trial was its
last), and 8 of 100 trials collapsed to macro-F1 between 0.27 and 0.39 for
reasons that were not investigated.

**Networks.** Two hidden layers (512 and 256 units), dropout 0.2, AdamW, weight
decay 0.01, batch size 128, class-weighted loss, cosine learning-rate schedule,
50,000 CpGs, no tissue correction. Each CpG enters as two numbers: its
standardized value (0 when unobserved) and a flag saying whether it was
observed. The masked network hides a random share of each training sample's
CpGs, drawn between 0.1% and 100% on a log scale. The number of epochs was
chosen from inner held-out scores recorded every 10 epochs.

The schedule, the epoch limits (60 plain, 150 masked) and the learning-rate grid
were set from a pilot on one inner fold of one outer fold. That fold's training
data contains samples that are test samples in other outer folds.

**Nearest centroid.** Each class's mean profile and one within-class variance
per CpG, from the training fold. A sample is assigned to the class with the
smallest variance-scaled squared distance over the CpGs that sample observed.
Nothing is tuned. It was added after the first sparse-coverage results, when
the forest was seen to collapse.

Files: `configs/`, `results/cv/<run>/selected.tsv`, `results/cv/<run>/fits.tsv`.

## 9. Calibration

A forest's vote shares are not probabilities. As in Capper et al., an
L2-penalized multinomial logistic regression maps the 91 raw scores to 91
probabilities. It is fit on out-of-fold scores, never on the scores of samples
the model trained on. Ten candidate forms were compared on inner folds (raw or
log scores; penalty C of 0.01 to 100). The forest chose raw scores with C = 1 in
four of five folds; LightGBM chose log scores with C = 10 in all five.

The networks and the centroid are not calibrated.

Files: `results/cv/<run>/calibration_*.tsv`.

## 10. Metric conventions

- Macro averages run over the classes present in the true labels.
- A family score is the sum of the class scores in that family, as in Capper et
  al. This is meaningful for calibrated scores at full coverage only. Elsewhere,
  "family" means the family of the predicted class.
- Families are the eight methylation class families of Capper et al.; every
  other class is its own family, giving 75 groups.
- Expected calibration error uses the top score in 15 equal-width bins.
- Intervals are 95% percentile bootstrap intervals from 1,000 resamples of the
  scored samples. They show how a number would move with a different draw of
  test samples, not with a retrained model. Resampling inflates the calibration
  error slightly, so its point value can sit at or below the interval's lower end.
- Differences between models use a paired bootstrap: the same resampled samples
  for both models.

## 11. Sparse coverage

Each sample gets one random number per array CpG, drawn from a fixed seed and
the sample's ID. A CpG counts as observed at level p when its number is below
p. So the observed sets are nested across levels, identical for every model,
and independent of every other sample.

Unobserved CpGs are handled as each model allows: tree models receive the
training-fold median, the networks receive zero plus the "unobserved" flag, and
the centroid leaves them out of the distance.

Files: `results/cv/sparsity_report_v1/`, `results/final_v1/validation_top_class.tsv`.

## 12. Leakage demonstration

On one training fold with its three inner folds, feature selection was run two
ways: on all samples before cross-validating ("leaky") and inside each training
fold ("correct"). On the real task the leak adds 0.5 points of accuracy. With
150 samples given three made-up classes at random, repeated 20 times, the leaky
version scores 0.557 (chance is 0.333) and the correct version 0.336; the leaky
version was ahead in all 20 repeats.

Files: `results/leakage_demo/`.

## 13. Final models and the external cohort

Five decisions were committed in `configs/final_v1.yaml` before any model read
the external cohort, and `scripts/check_final_config.py` recomputes each from
inner-fold results:

1. **Labels.** GEO's class label equals the published classifier's top-scoring
   class in all 1,035 cases where its Supplementary Table 4 names a class. All
   external figures are therefore agreement with that classifier.
2. **Target.** Family-level macro-F1 of the calibrated forest, compared with
   cross-validation recomputed on the 69 classes present in the external cohort
   (0.993), allowing a drop of 0.05.
3. **One setting per model.** Forest: best mean inner macro-F1 (10,000 CpGs,
   correction on). LightGBM: the per-fold selected trial with the highest inner
   macro-F1. Networks: the setting and epochs selected in most outer folds.
4. **Calibrator.** Fit on out-of-fold scores from one 5-fold pass with the final
   setting, using the form chosen in most folds.
5. **Analyses.** All five models at full coverage, the sparse-coverage curve
   with the same seed, the share scoring 0.9 or higher, results by tissue type,
   and three subsets fixed in advance.

The scoring stage receives sample IDs and tissue types but never labels, and it
does not overwrite a saved prediction.

Checks made when the final models were fit: the out-of-fold scores match saved
cross-validation predictions to 3e-8 where the setting coincides; every saved
model reproduces its in-memory predictions exactly after reloading; a refit of
the plain network gives identical weights.

Files: `results/final_v1/`.

## 14. Interpretation

TreeSHAP (Lundberg et al. 2020) was run on the final forest for the training
samples of nine classes: the five largest, and four classes named in published
class and gene pairs. The base value plus a sample's SHAP values equals the
forest's score to within 2e-11.

The plan in `configs/shap_v1.yaml` was committed before any attribution was
computed. It reworded the original target, for the reason given in the README.

| Measure | Definition |
|---|---|
| Top CpGs | The 100 CpGs with the highest mean absolute SHAP value over a class's samples |
| Stability | CpGs shared by the top 100 of two random halves of the samples, over 20 splits |
| Genomic context | Share of the top 100 in islands, shores, shelves and open sea, against the forest's 10,000 CpGs |
| Gene level | Genes with at least two top CpGs; direction is the class mean minus the mean of all other samples |
| Published pairs | For each pair from Benfatto et al. (2025), whether the gene has a CpG in the class's top 100 |

Gene and island annotation comes from Illumina's 450K annotation, whose gene
symbols date from 2011 (*PWWP3A* appears there as *MUM1*).

Files: `results/shap_v1/`.

## References

- Capper D, et al. DNA methylation-based classification of central nervous
  system tumours. *Nature* 2018;555:469-474.
- Zhou W, Triche TJ Jr, Laird PW, Shen H. SeSAMe: reducing artifactual detection
  of DNA methylation by Infinium BeadChips in genomic deletions. *Nucleic Acids
  Research* 2018.
- Lundberg SM, et al. From local explanations to global understanding with
  explainable AI for trees. *Nature Machine Intelligence* 2020.
- Benfatto S, et al. Explainable artificial intelligence of DNA
  methylation-based brain tumor diagnostics. *Nature Communications* 2025.
