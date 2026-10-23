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

Outputs under `--output-dir`: `runs/<run>/temporal_metrics.csv` (long table: model x subset x
metric, with CIs and the derivation CV value), `temporal_harrell_c_matrix.csv` (rows = subsets,
columns = models), `derivation_cv_cindex_matrix.csv`, `predictions.csv` (risk per model and
S(6), S(12) of the global model per patient), `model_info.json`, figures (forest plot of Harrell C
per area, KM by tertile of the global risk); `summary_runs.csv` and
`summary_derivation_vs_temporal.png` across runs. Unless `--no-mlflow`, everything is logged to
one run of the MLflow experiment `Temporal Validation - <source experiment>`.

```
python temporal_validation/score_temporal.py \
  --experiment-name "Federated Experiment October 23rd" \
  --cohort out/temporal_cohort_prepared.csv \
  --derivation-csv data/data_survival/central/original/data.csv \
  --output-dir out/temporal_oct23 --bootstrap 500
```

## 3. Azure ML pipeline (`azureml/`)

The same two scripts packaged as command components (`prepare_component.yml`,
`score_component.yml`) and a pipeline job (`pipeline.yml`, default compute
`clusterprdwe-u6nir2`). Everything that fixes the numbers is pinned in `pipeline.yml`: environment
`azureml:optuna-env:22` (scikit-learn 1.7.2, scikit-survival 0.25.0, mlflow 2.22.5, the versions
the models were pickled with), data assets `file-validation-btc-csv:5` and
`federated-learning-data-folder:2` (their labels are logged as params), `seed`. Inputs: the source
MLflow experiment name, an optional list of run names, `source_nodes_version` (v1/v2, written to
every table), and the metric parameters. Only FINISHED parent runs with a `best_optuna_model`
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
The scoring step's output `results` holds everything (a copy of the prepared cohort, per-run
tables, predictions and figures, the cross-run summaries); the downloaded model artifacts stay
outside it. Download it from the scoring step (a download addressed to the pipeline job only
returns the pipeline logs):

```
az ml job create -f temporal_validation/azureml/pipeline.yml \
  --set inputs.experiment_name="Federated Experiment October 23rd"          # all finished runs
STEP=$(az ml job list --parent-job-name <pipeline job name> --query "[?display_name=='score'].name" -o tsv)
az ml job download -n $STEP --output-name results --download-path out/temporal_oct23
```

or `temporal_validation/azureml/run_pipeline.sh "<experiment>" ["run_a run_b"]` with
`DOWNLOAD_DIR=...` to wait and download. Overrides: `--set inputs.bootstrap=200`,
`--set inputs.run_names="central_gradient_nofs_basic"`, `--set inputs.ipcw_from=derivation`.

Caveats: the per-area rows use the corrected geography (data asset v2); models trained on v1
nodes (Aug-6 and Oct-22 federated runs) have members that do not correspond to those areas.
Follow-up in the temporal cohort is short (median about 10 months), so horizons beyond 12
months are not meaningful.
