# Temporal validation of the SraceFed liver models

Two scripts, run on the Azure ML compute instance (the laptop cannot run Python) with the
framework's Python shim (`Federated/.venv/bin/python`, same container as the experiments:
scikit-survival 0.25, mlflow 2.22, azure-ai-ml, `DefaultAzureCredential` from the CLI login).

## 1. `prepare_temporal_cohort.py`

Builds the model-ready temporal cohort from the raw registry export (Azure ML data asset
`file-validation-btc-csv`, the 22/01/2026 freeze with both cohorts) and the verified centre
table `resources/centri_aree_federate.csv`:

1. keep `coorte == validation`;
2. sigla and federated area by exact match of the `center` string against
   `varianti_nei_file` (the table lists every spelling found in the files);
3. remove re-entries of training patients, matched on (dob, sex, center, cgi start date)
   against the training rows of the same file, like the UseCase-Code validation pipeline;
4. `tte` = (date of death or last fup - cgi start date) in days / (365.25/12); `event` = `death`;
   drop missing or tte <= 0;
5. the 33 model features with the derivation recipe (raw columns verbatim, `ps ecog` :=
   `ecog bin`, `location - Intrahepatic` := location == 1).

Expected result on the 22/01/2026 freeze: 779 validation rows, 80 duplicates, 1 zero follow-up,
**698 usable patients, 251 events**; per area americas 46, asia 125, europe_non_italy 129,
italy_centre_south 178, italy_north 220. `--anagrafica resources/anagrafica_pazienti_centri_aree.csv`
cross-checks the usable set and the areas patient by patient.

```
python temporal_validation/prepare_temporal_cohort.py \
  --raw-csv <file_btc_validazione_22012026.csv> \
  --centre-table temporal_validation/resources/centri_aree_federate.csv \
  --anagrafica temporal_validation/resources/anagrafica_pazienti_centri_aree.csv \
  --output out/temporal_cohort_prepared.csv
```

## 2. `score_temporal.py`

For every parent run of an MLflow experiment with a `best_optuna_model` child:

* downloads the logged pipeline (`imputer -> preprocessor -> SurvEnsModelWrapper`), reads the
  33-column signature, scores the cohort with the global model and with each ensemble member
  (one per federated node in node-name order; one member for central runs);
* metrics overall and per federated area (`--min-subset` patients): Harrell C, Uno C,
  time-dependent AUC at `--horizons` (6/12/18/24, those above tau skipped) and mean AUC,
  integrated Brier score up to `--ibs-horizon` (window: first grid point above the subset's
  shortest follow-up to the horizon), bootstrap percentile CIs (`--bootstrap` resamples;
  `*_ci_nboot` = replicates actually used) for Harrell C and IBS. Conventions are those of
  PRECISION-AI (`5_validation/lib/survival_metrics.py`): follow-up truncated at
  tau = min(`--horizon-max` 24, max derivation follow-up from `--derivation-csv`, max follow-up
  of the evaluated sample), censoring distribution of the IPCW metrics estimated in the
  evaluated sample or resample (`--ipcw-from evaluated`, default; `derivation` = sensitivity);
* derivation side from the same run: best child `weighted_c-index_validation` (reported as
  the derivation value of the global model on `all`) and the per-site CV C-index matrix of
  `cross_site_validation_results.json` (unweighted mean over folds of the validation rows);
* survival curves of the ensemble are recomputed from the members because the code bundled
  with round-1 models cannot evaluate `predict_survival_function`: the global curve follows
  the framework rule exactly (members linearly interpolated on the union of their knots,
  weighted mean, step evaluation on a 0.5-month grid), member curves are read as
  right-continuous step functions.

Known coding shift surfaced by `prepare_temporal_cohort.py` (summary key
`coding_differences_vs_training`): `stent or biliary draige` is 0/1/2/3 in the derivation rows
and 0/1 in the validation rows of the registry; the models receive the validation coding as is.

Calibration of the global model on the whole cohort at `--calibration-horizons` (6, 12 and 18
months; those above tau skipped): `calib_itl_<h>` = Kaplan-Meier observed survival at h minus the mean predicted
survival (positive = pessimistic model), `calib_slope_<h>` = coefficient of a Cox model of the
outcome (follow-up truncated at tau) on log(-log S_pred(h)) (1 = ideal, < 1 = predictions too
extreme), and observed vs predicted survival by quintile of predicted risk (`calibration.csv`,
figure).

The framework's privacy-preserving SHAP summary (beeswarm) plot is extracted as logged
(`figures/shap_summary_<run>.png`: SHAP values of the training patients on features discretised into 20 bins
with federated global boundaries, rows permuted per feature). In addition, the SHAP importance of the global
model (`shap_importance.csv`: mean |SHAP| and mean SHAP per feature on a random
subsample of `--shap-patients` cohort patients, default 100: the permutation explainer costs about
2 x features x 40 background rows model predictions per patient) with the explainer logged by the framework
(PermutationExplainer on the federated cluster centres, TreeExplainer for xgboost), applied after the run's own
imputation and scaling; the cohort is not a federated node, so there is no privacy constraint. The data-quality
outputs of the run are extracted as they are (`data_quality/`: `table1.csv`, per-node and global statistics json,
summaries, Kaplan-Meier figures).

Each run is labelled with its role from the run's `clients` parameter: `central`, `federated`
(several clients) or `standalone` + node (one non-central client; the launcher names those runs
`federated_<node>_...`, the role column is the reliable label). Model, feature-selection method
and imputation are the last three tokens of the run name. Several source experiments can be
scored in one go (`--experiment-name "A; B"`); run names must be unique across them.

Outputs under `--output-dir`: `runs/<run>/temporal_metrics.csv` (long table: model x subset x
metric, with CIs, the derivation CV value and the calibration columns on the global/all row),
`temporal_harrell_c_matrix.csv` (rows = subsets, columns = models), `derivation_cv_cindex_matrix.csv`,
`predictions.csv` (risk per model and S(h) of the global model at every AUC horizon per patient),
`calibration.csv`, `trials.csv` (every Optuna trial: metrics and parameters), the framework's
feature-selection tables when the run has them (`feature_selection_results.csv` = candidate, CV C
and list; `feature_selection_scores.csv` = global ranking of every feature; `feature_selection_site_scores.csv`;
`feature_selection_cv.csv` = per-fold CV of each candidate), `model_info.json` (role, design,
signature = the model's inputs, members, weights, parameters), figures (forest plot of Harrell C
per area, KM by tertile of the global risk, calibration); across runs `summary_runs.csv` (one row
per run: role, node, model, fs, imputation, number and list of selected features, derivation C,
temporal metrics, calibration, per-area C), `derivation_runs.csv`, `derivation_cv_cindex_all_runs.csv`,
`temporal_metrics_all_runs.csv`, `summary_derivation_vs_temporal.png`, `run_info.json`. Unless
`--no-mlflow`, everything is logged to one run of the MLflow experiment
`Temporal Validation - <first source experiment>`.

```
python temporal_validation/score_temporal.py \
  --experiment-name "Federated Experiment Final; Standalone Final" \
  --cohort out/temporal_cohort_prepared.csv \
  --derivation-csv data/data_survival/central/original/data.csv \
  --output-dir out/final_scored --bootstrap 500
```

## 3. `compare_runs.py`

One folder in (the score output), one folder out: a copy of the score output plus `tables/` and
`report.md`. The comparisons are the pairs of runs that differ in exactly one design factor and
share the others (levels ordered nofs < mi < score, basic < mice, central < federated; the pair
is "higher minus lower"): `arm` (federated - central), `fs` (mi - nofs, score - nofs, score - mi),
`imputation` (mice - basic), and for every standalone run `standalone_vs_federated` and
`standalone_vs_central` (node alone minus the federated / central model of the same
configuration, on the node's own temporal area and on the whole cohort). For each pair the
difference of the temporal Harrell C of the global models with a paired bootstrap percentile CI
(`--bootstrap` resamples of the same patients for both runs) and, when both runs log a feature
ranking, the agreement of the rankings (Spearman over all features, overlap of the top 4/8/10,
Jaccard of the selected lists). Derivation side: best-trial pivot and the per-trial difference
federated - central over the trials with identical hyper-parameters (matched on the trial parameters: the TPE
start-up trials, 10 of 50 with the framework's seed; `n_trials_matched` says how many).

`tables/`: `runs.csv` (with the SHAP top 5 and the Spearman between the SHAP order and the selection
order of the selected features), `paired_comparisons.csv`, `ranking_agreement.csv`, `rankings.csv` (long),
`ranking_model_independence.csv` (the ranking must be identical across the models of the same
role/fs/imputation), `shap_agreement.csv` (SHAP importance rankings on the same pairs: Spearman over the
common input features, top 4/8/10 overlap, Jaccard of the top-10 mean |SHAP| sets, which for a model with
<= 10 inputs is its whole input set),
`derivation_best.csv`, `derivation_trial_paired.csv`, `standalone_matrix.csv` (node alone vs federated vs
central, own area and whole cohort). `report.md` collects the key tables.

```
python temporal_validation/compare_runs.py --results out/final_scored --output-dir out/final --bootstrap 1000 \
  --data-folder data/data_survival
```

Unless `--no-paper`, the step then runs `paper_artifacts.py`, which writes `paper/` next to `tables/`: the
display items of the manuscript (`docs/PAPER_STRUCTURE.md` of the workbench) as `.docx` with caption + `.csv`
(tables) and `.png` + `.svg` + `.pdf` (figures), split into `main/` (Table 1 cohorts, Table 2 paired differences,
Fig. 2 discrimination across arms, Fig. 3 feature selection, Fig. 4 calibration, Fig. 5 SHAP) and `supplementary/`
(search spaces, centres, full Table 1, derivation and temporal metrics of every run, all paired comparisons, the
single-centre matrix, rankings and agreement, the mutual-information arm, trial-matched derivation, per-area
forest plots, calibration curves, Kaplan-Meier by node and area, SHAP beeswarm montage, missingness), plus
`captions.md` and `numbers.json` (the figures quoted in the text). Table 1 and the Kaplan-Meier curves need
`--data-folder` (node data) and the centre table (`--centres-csv`, default `resources/centri_aree_federate.csv`);
the docx output needs `python-docx` (environment `optuna-env:23`).

## 4. Azure ML pipeline (`azureml/`)

The three scripts packaged as command components (`prepare_component.yml`,
`score_component.yml`, `compare_component.yml`) and a pipeline job (`pipeline.yml`, default compute
`clusterprdwe-u6nir2`). Everything that fixes the numbers is pinned in `pipeline.yml`: environment
`azureml:optuna-env:22` (scikit-learn 1.7.2, scikit-survival 0.25.0, mlflow 2.22.5, the versions
the models were pickled with), data assets `file-validation-btc-csv:5` and
`federated-learning-data-folder:2` (their labels are logged as params), `seed`. Inputs: the source
MLflow experiment name(s) (`;`-separated), an optional list of run names, `source_nodes_version`
(v1/v2, written to every table), the metric parameters, `bootstrap` (per-run CIs) and
`bootstrap_pairs` (paired comparisons). Only FINISHED parent runs with a `best_optuna_model`
child are scored; the others are listed in `run_info.json` and in the job params.

Inside the job the scoring script attaches to the job's own MLflow run: params (source
experiment and id, runs scored/skipped/failed, cohort size, library versions, data labels), one
metric per run (`harrell_c__<run>`, `derivation_c__<run>`, `uno_c__<run>`, `ibs12__<run>`) and the
whole results folder as artifacts. Derivation-side numbers are complete: `summary_runs.csv`
(weighted CV objective per run), `derivation_cv_cindex_all_runs.csv` (per-site CV matrix),
`derivation_runs.csv` (parent and best-child run ids, model artifact URI, ensemble weights,
run config `param__*` and best-trial hyper-parameters `best_param__*`). In the long tables the
column `derivation_cv_kind` says which estimand `derivation_cv_cindex` is:
`run_weighted_objective` (pooled row, the run's sample-weighted CV C) or `site_fold_mean`
(per-area rows, unweighted mean over folds).
The compare step's output `final` holds everything (a copy of the prepared cohort, per-run
tables, predictions and figures, the cross-run summaries, `tables/`, `report.md`); the downloaded
model artifacts stay outside it. Download it from the compare step (a download addressed to the
pipeline job only returns the pipeline logs):

```
az ml job create -f temporal_validation/azureml/pipeline.yml \
  --set inputs.experiment_name="Federated Experiment Final; Standalone Final"   # all finished runs
STEP=$(az ml job list --parent-job-name <pipeline job name> --query "[?display_name=='compare'].name" -o tsv)
az ml job download -n $STEP --output-name final --download-path out/final
```

or `temporal_validation/azureml/run_pipeline.sh "<experiments>" ["run_a run_b"]` with
`DOWNLOAD_DIR=...` to wait and download. Submission needs to upload the code snapshot: the
laptop has no blob permission, so submit from the compute instance (copy `temporal_validation/`
there, strip CR line endings, `az ml job create` with the host `az`); downloads work from the
laptop. Overrides: `--set inputs.bootstrap=200`,
`--set inputs.run_names="central_gradient_nofs_basic"`, `--set inputs.ipcw_from=derivation`,
`--set inputs.source_nodes_version=v1` for experiments trained on the v1 nodes. `run_names`
takes `all` or a list; never pass an empty string (Azure ML drops empty string inputs and the
submission fails with "Assignment for parameter run_names is not specified").

Caveats: the per-area rows use the corrected geography (data asset v2); models trained on v1
nodes (Aug-6 and Oct-22 federated runs) have members that do not correspond to those areas.
Follow-up in the temporal cohort is short (median about 10 months), so horizons beyond 12
months are not meaningful.
