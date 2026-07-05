# Autism is classifiable but not spatially mappable from functional connectivity

Code and figure-level results for a two-cohort benchmark and a brain–transcriptome null.

**Finding.** Under a leakage-free protocol in two independent cohorts (ABIDE-I, *n* = 1,064;
ABIDE-II, *n* = 500), resting-state functional connectivity classifies autism spectrum
disorder only at a modest, motion-robust ceiling (ROC-AUC ≈ 0.72) that a BrainNetCNN does not
exceed. The cortical **topography** of autism connectivity disruption shows **no** concordance
with the cortical expression of autism-risk genes beyond spatial autocorrelation — across four
operationalizations, spin-test spatial nulls, two cohorts and two preprocessing pipelines —
while a positive control confirms the same analysis detects genuine brain–gene coupling when
it is present. A power analysis shows why: classification saturates within a few hundred
participants, whereas a reliable disruption map would need ~1,740. The molecular signal is
real and developmental, not adult-spatial.

Authors: **Swarnabha Chowdhury** and **Dinesh Gupta** (corresponding, `dinesh@icgeb.res.in`),
Translational Bioinformatics Group, International Centre for Genetic Engineering and
Biotechnology (ICGEB), New Delhi, India.

---

## Repository layout

| Script | Role |
|---|---|
| `imaging_transcriptomics.py` | Core module: signed Fisher-*z* FC, case–control disruption GLM, AHBA/SFARI expression maps, spin-test null |
| `classify_eval.py` | Leave-one-site-out (LOSO) logistic / RBF-SVM classification with bootstrap CIs |
| `deep_baselines_loso.py` | BrainNetCNN deep-learning baseline under the identical LOSO protocol |
| `combat.py` / `harmonized_classify.py` | ComBat harmonization and harmonized classification |
| `full_transcriptome.py` | Transcriptome-wide enrichment test (abagen, left-hemisphere) |
| `spin_pls_sfari.py` | Per-gene FDR and spin-refit PLS operationalizations |
| `aggregate_lh_concordance.py` | Aggregate SFARI + cell-type + biotype concordance on the unified abagen LH-50 pipeline (both cohorts) |
| `coupling.py` | Positive control: functional connectivity ↔ correlated gene expression |
| `biotypes.py` | Normative model, connectome biotypes, stability, transcriptomic test |
| `power_analysis.py` | Reliability (split-half) vs classification-AUC power curves |
| `developmental.py` | BrainSpan developmental co-expression convergence |
| `abide2_preprocess.py` / `replicate_abide2.py` | ABIDE-II local FSL preprocessing and independent replication |
| `boot_abide2_ci.py` | ABIDE-II bootstrap confidence intervals |
| `make_submission_figures.py` | Main figures 2, 4–8 (native vector; real arrays only) |
| `make_schematics.py` | Figure 1 (workflow) and Figure 3 (BrainNetCNN architecture) |
| `make_surface_figure.py` | Supplementary Fig. 1 (cortical-surface renderings) |

`results/` holds the saved result arrays and JSON summaries that the figure scripts read;
`figures_submission/` holds the rendered figures.

## Reproducing

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt

# Regenerate the figures from the saved result arrays:
python make_submission_figures.py
python make_schematics.py
python make_surface_figure.py
```

Re-running the full analysis from raw data requires the public datasets below. The figure
scripts fail loudly (raise an error) rather than plotting simulated data.

## Data availability

This repository contains **analysis code and the derived, group/region-level result arrays
that underlie the figures**. It does **not** redistribute per-subject imaging or the large
reference atlases; obtain those from the original open repositories and regenerate the derived
inputs with the provided code:

- **ABIDE-I / ABIDE-II** — Preprocessed Connectomes Project / NITRC (`abide2_preprocess.py`
  reproduces the ABIDE-II connectivity matrices from the raw NIfTI).
- **Allen Human Brain Atlas** — Allen Institute; parcellated with the `abagen` toolbox
  (`full_transcriptome.py` regenerates the region × gene matrix on first run).
- **BrainSpan** — Allen Institute (developmental transcriptomics).
- **SFARI Gene** — the SFARI Gene database (categories 1–2).
- **Velmeshev et al. 2019** — single-cell atlas, from the repository cited therein.

## Statistical standards

Leakage-free leave-one-site-out cross-validation; spin-test spatial nulls (1,000 rotations) on
every map–map comparison; Benjamini–Hochberg FDR for gene-wise tests; 2,000-sample bootstrap
AUC confidence intervals; a positive control to exclude insensitivity; a power analysis; and
two-cohort, two-pipeline replication.

## Citation

Chowdhury, S. & Gupta, D. *Autism is classifiable but not spatially mappable from functional
connectivity: a two-cohort benchmark and a brain–transcriptome null.* (2026). *(Citation to be
updated on publication.)*

## License

MIT — see [LICENSE](LICENSE).
