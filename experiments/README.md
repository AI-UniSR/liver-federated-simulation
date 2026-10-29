# Experiment configurations of the final grid

The 126 runs of the paper are defined by the 108 JSON files in `configs/` and by the topology set in the framework's
launcher (`runJobSIMULATOR.py`: the list of clients). Each file is the complete configuration of one SraceFed job:
data quality, imputation, feature selection, training (Optuna, cross-validation, model family and search space),
SHAP and MLflow settings.

| Files | Runs | Topology | MLflow experiment |
|---|---|---|---|
| `<family>_<selection>_<imputation>.json` (18) | 18 pooled + 18 federated | one client `central` holding all 732 patients; or five clients, one per node | `Federated Experiment Final` (run names `central_*`, `federated_*`) |
| `solo_<node>_<family>_<selection>_<imputation>.json` (90) | 90 single-centre | one client holding one node | `Standalone Final` (run names `federated_<node>_*`: the launcher labels any single non-pooled client "federated"; the validation pipeline reads the arm from the `clients` parameter) |

Factors: family `cox`, `gradient` (gradient-boosted survival trees), `rsf` (random survival forest); selection
`nofs` (all 33 predictors), `score` (score test, 10 predictors), `mi` (mutual information, 10 predictors);
imputation `basic` (federated means), `mice` (federated chained equations, Bayesian ridge, 10 sweeps). Common to all:
seed 42, 50 Optuna trials, five-fold cross-validation stratified on the event, 40 federated cluster centres and 20
bins for the SHAP summaries.

Scripts, as executed on the Azure Machine Learning compute instance (paths are those of that machine):

- `make_configs.py`: generated the 18 main and 45 single-centre basic configurations from the previous grid's
  files, changing only the feature-selection block and the MLflow names (`fs_top_p_options = [0.31]`, that is
  `int(0.31 x 33) = 10` predictors; `fs_min_cases = 5`).
- `make_solo_mice.py`: generated the 45 single-centre MICE configurations from the basic ones.
- `run_final.sh`: the chain pooled, federated, single centre (basic), setting the launcher topology for each block;
  resumable. `run_solo_mice.sh`: the single-centre MICE block. `smoke_final.sh`: two-trial smoke runs.
- Framework code: SraceFed commit `b41edce`, data asset `federated-learning-data-folder:2` (corrected centre to node
  mapping); each parent run is tagged in MLflow with the code commit, the data asset, the grid and the feature count.
