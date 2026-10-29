# liver-federated-simulation

Experiment configurations and validation pipeline of the study

> *End-to-end federated learning for prognostic survival models in an international biliary tract cancer registry*
> (manuscript under submission; S-RACE, University Vita-Salute San Raffaele, Milan).

The study evaluates **SraceFed**, a federated-learning framework built on NVIDIA FLARE, on an international
real-world registry of advanced biliary tract cancer treated with first-line cisplatin, gemcitabine and durvalumab.
The registry's development cohort (732 patients, 375 deaths) is split into five geographic nodes and analysed three
ways with the same code: pooled, federated, and one node at a time. Every resulting model (126 runs: three model
families, three feature-selection settings, two imputation methods, in the pooled, federated and five single-centre
arms) is scored on a later, non-overlapping freeze of the same registry (698 patients, 251 deaths) and the arms are
compared pairwise. This repository holds everything needed to reproduce the validation and the display items of the
paper from the logged experiments, and the exact configuration of every experiment.

## What is here

| Folder | Content |
|---|---|
| `experiments/` | The 108 configuration files that define the 126 runs of the final grid (18 shared by the pooled and federated arms, 90 single-centre), the scripts that generated them and the chain scripts that executed them on the compute instance. See `experiments/README.md`. |
| `temporal_validation/` | The validation pipeline: cohort preparation from the registry export, scoring of every logged model on the temporal cohort (discrimination, calibration, SHAP), paired comparisons between arms, and the generation of every table and figure of the paper (`paper_artifacts.py`, `paper_variants.py`, `paper_style.py`). Packaged as a three-step Azure Machine Learning pipeline (`temporal_validation/azureml/`). See `temporal_validation/README.md`. |
| `temporal_validation/resources/` | `centri_aree_federate.csv`, the table that maps the registry's centres to the five federated nodes (as in Supplementary Table 3 of the paper); `learning_curve_reference.csv`, the single-centre reference values of the learning-curve analysis. |
| `utils/` | Collection of the experiment records from MLflow (`collector_training.py`), the derivation-side analysis across experiments (`derivation_analysis.py`), data-asset helpers. |
| `config.json` | Azure Machine Learning workspace coordinates used by the helpers (no secrets; access requires an authorised identity). |

## What is not here, and why

- **The framework.** SraceFed (server controllers, client runners, the `fed_data_quality` library) lives in its own
  repository; the runs of the paper used commit `b41edce` of that code. The configuration files in `experiments/`
  are inputs to that framework. [Link and archived version to be added by the authors.]
- **Patient-level data.** The registry data (node tables, the validation export) are not public because of the
  participating centres' data-sharing agreements and data-protection regulations; requests are handled by the
  registry's steering committee through the corresponding author. The patient register used to cross-check the
  cohort (`anagrafica_pazienti_centri_aree.csv`, one row per patient with centre and inclusion status) is likewise
  not distributed: the pipeline runs without it and only skips that cross-check.
- **The experiment records.** The 126 runs are logged in MLflow on Azure Machine Learning (experiments
  `Federated Experiment Final` and `Standalone Final`); the pipeline reads them from there. Their derived tables
  (derivation and validation metrics of every run, all paired comparisons) are provided with the paper as
  Supplementary Data 1-3.

## How the validation runs

1. **Prepare**: builds the temporal cohort from the registry export (validation rows, centre to node mapping,
   removal of patients re-entered from the development cohort, follow-up recomputed from the dates, the 33 model
   predictors with the derivation recipe, coding checks).
2. **Score**: for every finished run of the source experiments, loads the logged model pipeline (imputer, encoders,
   model) and the SHAP explainer, scores the cohort overall and per area (Harrell C with bootstrap intervals, Uno C,
   time-dependent AUC, integrated Brier score, calibration-in-the-large and slope at 6, 12 and 18 months, SHAP
   importance), and extracts the derivation side of the run (trials, feature-selection tables, data-quality outputs).
3. **Compare**: paired bootstrap for every pair of runs that differ in one design factor (arm, selection,
   imputation, node alone versus federated or pooled), ranking and SHAP agreement, the single-centre matrix, and the
   paper's display items (`paper/main`, `paper/supplementary`, `paper/variants`: tables as docx and csv, figures as
   png, svg and pdf, `captions.md`, `numbers.json`).

Submission: `temporal_validation/azureml/run_pipeline.sh "Federated Experiment Final; Standalone Final"`, or the
SDK (`temporal_validation/README.md`, section 4). Everything that fixes the numbers is pinned in `pipeline.yml`
(environment, data-asset versions, seed, bootstrap sizes: 500 per-run resamples, 2,000 paired resamples).

## Reproducibility notes

- All runs: seed 42 for the data splits, the hyper-parameter sampler and the stochastic learners; 50 Optuna trials;
  five-fold cross-validation stratified on the event indicator inside each node.
- The 12 no-selection runs reproduce, to the last digit, a development run that differed only in the selection
  stage; two independent executions of the validation pipeline gave identical paired comparisons.
- The framework's privacy-preserving SHAP summary plots are the one output not reproducible bit for bit, by design
  of the private permutation seed used by every node.

## Requirements

Python 3.10 with scikit-survival 0.25.0, scikit-learn 1.7.2, mlflow 2.22.5, pandas, numpy, scipy, matplotlib,
shap, optuna, azure-ai-ml and python-docx (the pinned Azure ML environment `optuna-env:23`). The figures use Arial
when the font is available (`PAPER_FONT_DIR`), otherwise a metric-compatible sans-serif.

## Citation

See `CITATION.cff`. Licence: to be set by the authors before archiving.
