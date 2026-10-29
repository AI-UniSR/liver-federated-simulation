"""Temporal validation of the SraceFed models of one MLflow experiment.

For every parent run of the experiment that has a ``best_optuna_model`` child, the
script downloads the logged sklearn pipeline (imputer -> preprocessor ->
SurvEnsModelWrapper), scores the prepared temporal cohort with the global model
and with each ensemble member (one per federated node, one for a central run),
and computes, overall and per federated area:

* Harrell C-index, Uno C-index (IPCW, tau = --horizon-max),
* time-dependent AUC at the requested horizons,
* integrated Brier score up to --ibs-horizon,
* bootstrap percentile CIs (patient resampling) for Harrell C and IBS.

Metric conventions follow PRECISION-AI (5_validation/lib/survival_metrics.py, 9 Oct 2026):
follow-up truncated at tau = min(--horizon-max, max derivation follow-up, max follow-up of the
evaluated sample); the censoring distribution G(t) of the IPCW metrics (Uno C, AUC, Brier)
is estimated in the evaluated sample itself (subset or bootstrap resample, --ipcw-from
evaluated, default), the derivation cohort only sets the follow-up bound; --ipcw-from
derivation is the sensitivity variant. AUC time points above tau are skipped.

Derivation-side numbers are read from the same run: the best child's
``weighted_c-index_validation`` and the per-site cross-validation C-indexes in
``cross_site_validation_results.json`` (mean over folds of the validation rows), every Optuna
trial (``trials.csv``) and the feature-selection tables of the ``feature_selection`` child
(``feature_selection_*.csv``), so that one download holds the derivation side as well.

Calibration of the global model on the whole cohort at --calibration-horizons (6, 12, 18 months):
calibration-in-the-large (Kaplan-Meier observed survival at h minus mean predicted survival at h)
and calibration slope (Cox model of the outcome on log(-log) of the predicted survival at h; 1 =
ideal), plus observed vs predicted survival in five groups of predicted risk (``calibration.csv``
and a figure).

The framework's SHAP summary (beeswarm) plot is extracted as logged (``figures/shap_summary_<run>.png``);
the SHAP importance of the global model (``shap_importance.csv``: mean |SHAP| per feature on a random
subsample of the cohort, ``--shap-patients``) is computed with the explainer logged by the framework; and the
data-quality outputs of the run (``data_quality/``: Table 1 as CSV, per-node and global statistics,
Kaplan-Meier figures), both extracted, not recomputed.

Several source experiments can be scored in one go (``--experiment-name "A; B"``): the runs of a
standalone experiment (single non-central client) are labelled role ``standalone`` with their node,
read from the run's ``clients`` parameter, not from the run name (the launcher calls any single
non-central client "federated").

Everything is written under --output-dir (one folder per run + a summary) and, unless
--no-mlflow, logged as ONE MLflow run of the experiment "Temporal Validation - <source
experiment>" so that metrics, tables and figures can be downloaded in one go.

The survival curves of the ensemble are recomputed from the members instead of calling the
wrapper (the code bundled with round-1 models lacks the StepFunction import). The global
curve replicates SurvivalEnsemble.__mean_step_functions exactly: every member is linearly
interpolated (np.interp, left fill = first value, right fill = last value) on the union of
all members' knots, the weighted mean is taken there, and the resulting step function is
evaluated on the 0.5-month grid (right-continuous; 1.0 before the first knot, where the
framework's StepFunction would raise). Member curves are evaluated as right-continuous step
functions, which is how a single sksurv estimator is read.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
LOGGER = logging.getLogger("score_temporal")

SUBSCRIPTION = os.getenv("AZURE_SUBSCRIPTION_ID", "ab211f7b-463f-4833-9605-d260e596a35a")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "1fe41577-4b8d-412a-3055-3a18d9df3ee7")
WORKSPACE = os.getenv("AZUREML_WORKSPACE_NAME", "1fe415774b8d412a30553a18d9df3ee7")
AREAS = ["americas", "asia", "europe_non_italy", "italy_centre_south", "italy_north"]


# ----------------------------------------------------------------------------- MLflow
def library_versions() -> dict:
    """Versions of the libraries that determine unpickling and scoring (recorded for lineage)."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ["scikit-learn", "scikit-survival", "mlflow", "numpy", "pandas", "scipy"]:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            try:
                out[pkg] = version(pkg + "-skinny")  # the experiment container ships mlflow-skinny
            except PackageNotFoundError:
                out[pkg] = "missing"
    return out


def in_azureml_job() -> bool:
    """True when running inside an Azure ML job (tracking URI and run token are preset)."""
    return bool(os.getenv("AZUREML_RUN_ID"))


def mlflow_client():
    """MLflow client on the workspace: the preset tracking URI inside an Azure ML job, the
    workspace URI through azure-ai-ml and DefaultAzureCredential elsewhere (compute instance)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri or not in_azureml_job():
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
        uri = ml.workspaces.get(WORKSPACE).mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    return mlflow, MlflowClient(tracking_uri=uri)


def search_all(client, experiment_id, filter_string=""):
    runs, token = [], None
    while True:
        page = client.search_runs([experiment_id], filter_string=filter_string, max_results=200, page_token=token)
        runs.extend(page)
        token = getattr(page, "token", None)
        if not token:
            return runs


def run_name(r):
    return r.data.tags.get("mlflow.runName", "")


def parent_runs(client, experiment_id):
    return [r for r in search_all(client, experiment_id) if "mlflow.parentRunId" not in r.data.tags]


def children(client, experiment_id, parent_id):
    return search_all(client, experiment_id, f"tags.mlflow.parentRunId = '{parent_id}'")


def best_child(kids):
    best = [k for k in kids if run_name(k) == "best_optuna_model"]
    if not best:
        return None
    return sorted(best, key=lambda k: k.info.start_time or 0)[-1]


def run_design(parent) -> dict:
    """Role and design factors of a run. Role from the run's ``clients`` parameter (central; one node =
    standalone; several = federated), the model / feature-selection / imputation labels from the last three
    tokens of the run name (<arm>_[<node>_]<model>_<fs>_<imputation>)."""
    name = run_name(parent)
    clients = [c.strip() for c in (parent.data.params.get("clients") or "").split(",") if c.strip()]
    if clients == ["central"]:
        role, node = "central", ""
    elif len(clients) == 1:
        role, node = "standalone", clients[0]
    elif len(clients) > 1:
        role, node = "federated", ""
    else:  # old runs without the parameter: the first token of the name
        role, node = name.split("_")[0], ""
    parts = name.split("_")
    model, fs, imputation = (parts[-3], parts[-2], parts[-1]) if len(parts) >= 4 else ("", "", "")
    return {"role": role, "node": node, "model": model, "fs": fs, "imputation": imputation}


def extract_children_tables(mlflow, kids, out: Path, cache: Path):
    """Derivation side of the run: every Optuna trial (trials.csv) and the feature-selection tables logged by
    the framework (feature_selection_{results,scores,site_scores,cv}.csv), when present."""
    trials = []
    for k in sorted(kids, key=lambda k: k.info.start_time or 0):
        if run_name(k).startswith("optuna-trial-"):
            trials.append({"trial": int(run_name(k).split("-")[-1]), "status": k.info.status,
                           **{f"metric__{m}": v for m, v in k.data.metrics.items()}, **{f"param__{p}": v for p, v in k.data.params.items()}})
    if trials:  # a trial logged twice (restarted job) keeps its latest child
        pd.DataFrame(trials).drop_duplicates("trial", keep="last").sort_values("trial").to_csv(out / "trials.csv", index=False)
    fs_runs = [k for k in kids if run_name(k) == "feature_selection"]
    if fs_runs:
        fs = sorted(fs_runs, key=lambda k: k.info.start_time or 0)[-1]
        for a in mlflow.artifacts.list_artifacts(run_id=fs.info.run_id):
            if not a.path.endswith(".json"):
                continue
            d = json.load(open(mlflow.artifacts.download_artifacts(run_id=fs.info.run_id, artifact_path=a.path, dst_path=str(cache))))
            pd.DataFrame(d["data"], columns=d["columns"]).to_csv(out / a.path.replace(".json", ".csv"), index=False)
    dq_runs = [k for k in kids if run_name(k) == "data_quality"]
    if dq_runs:
        extract_data_quality(mlflow, sorted(dq_runs, key=lambda k: k.info.start_time or 0)[-1], out / "data_quality", cache)
    return len(trials), len(fs_runs) > 0


def extract_data_quality(mlflow, dq_run, out: Path, cache: Path):
    """Data-quality stage of the run: Table 1 (table1.pkl -> table1.csv), the per-node and global statistics json
    and the Kaplan-Meier figures. Nothing is recomputed."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        local = Path(mlflow.artifacts.download_artifacts(run_id=dq_run.info.run_id, artifact_path="data_quality_results", dst_path=str(cache)))
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("data_quality artifacts not downloaded: %s", e)
        return
    for p in local.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local)
        if p.name == "table1.pkl":
            try:
                t1 = pd.read_pickle(p)
                (t1 if isinstance(t1, pd.DataFrame) else pd.DataFrame(t1)).to_csv(out / "table1.csv")
            except Exception as e:  # noqa: BLE001
                LOGGER.warning("table1.pkl not readable: %s", e)
        elif p.suffix in (".png", ".json", ".txt"):
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)


def shap_importance(mlflow, best, pipe, X: pd.DataFrame, feat_in: list, n_patients: int, seed: int, out: Path):
    """Mean |SHAP| per feature of the global model on a random subsample of the temporal cohort, with the explainer
    logged by the framework (PermutationExplainer on the federated cluster centres, or TreeExplainer), applied to
    the cohort after the run's own imputation and scaling. The cohort is not a federated node: no privacy constraint.
    Returns the table or None when the run has no explainer or it cannot be loaded."""
    try:
        uri = f"runs:/{best.info.run_id}/shap_explainer"
        if "shap_explainer" not in {a.path for a in mlflow.artifacts.list_artifacts(run_id=best.info.run_id)}:
            return None
        explainer = mlflow.shap.load_explainer(uri)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), size=min(n_patients, len(X)), replace=False))
        Xt = pipe[:-1].transform(X.iloc[idx])
        Xt = (Xt if hasattr(Xt, "columns") else pd.DataFrame(Xt, columns=feat_in))[feat_in]
        np.random.seed(seed)  # the permutation explainer draws its permutations from numpy's global RNG (reseeded at load)
        values = explainer(Xt.values)
        values = values.values if hasattr(values, "values") else np.asarray(values)
        if values.ndim == 3:  # some explainers return (n, p, outputs)
            values = values[:, :, 0]
        table = pd.DataFrame({"feature": feat_in, "mean_abs_shap": np.abs(values).mean(axis=0), "mean_shap": values.mean(axis=0),
                              "n_patients": len(idx)}).sort_values("mean_abs_shap", ascending=False)
        table["rank"] = np.arange(1, len(table) + 1)
        table.to_csv(out / "shap_importance.csv", index=False)
        return table
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("SHAP importance failed: %s", e)
        return None


# ----------------------------------------------------------------------------- survival helpers
def structured(event, time):
    from sksurv.util import Surv
    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(time, dtype=float))


def eval_step_functions(sfs, grid):
    """Evaluate a list of sksurv StepFunction on grid, constant beyond the last time, 1 before the first."""
    out = np.empty((len(sfs), len(grid)))
    for i, sf in enumerate(sfs):
        x, y = np.asarray(sf.x, dtype=float), np.asarray(sf.y, dtype=float)
        if len(x) == 0:
            out[i] = 1.0
            continue
        # StepFunction is right-continuous: value of the last knot <= t; 1 before the first knot
        idx = np.searchsorted(x, grid, side="right") - 1
        out[i] = np.where(idx < 0, 1.0, y[np.clip(idx, 0, len(y) - 1)])
    return out


def ensemble_survival_framework_rule(member_sfs, weights, grid):
    """Global survival on grid, replicating SurvivalEnsemble.__mean_step_functions.

    member_sfs: list (per member) of lists (per patient) of sksurv StepFunction.
    The framework evaluates each member at its own knots (= its y), extends it with np.interp
    on the union of all knots (left fill = y[0], right fill = y[-1]), averages with the weights
    and builds a StepFunction on the union of knots. Here that step function is evaluated on
    grid (right-continuous), with 1.0 before the first knot.
    """
    knots = [np.asarray(sfs[0].x, dtype=float) for sfs in member_sfs]
    all_x = np.unique(np.concatenate(knots))
    n = len(member_sfs[0])
    mean = np.zeros((n, len(all_x)))
    for w, sfs, x in zip(weights, member_sfs, knots):
        for i, sf in enumerate(sfs):
            y = np.asarray(sf.y, dtype=float)
            mean[i] += w * np.interp(all_x, x, y, right=y[-1])
    idx = np.searchsorted(all_x, grid, side="right") - 1
    out = np.where(idx[None, :] < 0, 1.0, mean[:, np.clip(idx, 0, len(all_x) - 1)])
    return out


def ensemble_weights(ens):
    """Replicates SurvivalEnsemble._weight() without relying on the bundled code version."""
    if getattr(ens, "weight", None):
        return list(ens.weight)
    if getattr(ens, "weighted_sample", False):
        n = [x if x is not None else -1 for x in ens.n_sample]
        tot = sum(n)
        return [x / tot if tot > 0 else 0 for x in n]
    return [1.0 / len(ens.model)] * len(ens.model)


def truncate(y, horizon):
    """Censor at horizon every patient followed beyond it (PRECISION-AI convention)."""
    y = y.copy()
    beyond = y["time"] > horizon
    y["time"][beyond] = horizon
    y["event"][beyond] = False
    return y


def evaluation_horizon(time, tau_max, max_train):
    """tau = min(tau_max, max derivation follow-up, max follow-up of the evaluated sample)."""
    return float(min(tau_max, max_train if max_train is not None else np.inf, float(np.max(time))))


def ipcw_sample(y_eval, y_deriv, ipcw_from):
    """Sample used to estimate the censoring distribution G(t) of the IPCW metrics."""
    return y_deriv if (ipcw_from == "derivation" and y_deriv is not None) else y_eval


def ibs_on_grid(y_g, y_eval, surv, grid, horizon, ibs_h, tmin):
    """IBS = trapezoid of the Brier score over the grid points in (tmin, min(horizon, ibs_h)), divided by the span."""
    from sksurv.metrics import integrated_brier_score
    keep = (grid > tmin) & (grid < np.nextafter(horizon, 0)) & (grid <= ibs_h)
    if keep.sum() < 2:
        return np.nan
    return float(integrated_brier_score(y_g, y_eval, surv[:, keep], grid[keep]))


def metrics_block(y_deriv, max_train, event, time, risk, surv, grid, horizons, tau_max, ibs_h, ipcw_from):
    """Point estimates for one model on one subset. surv: n x len(grid).

    Conventions (same as PRECISION-AI 5_validation/lib/survival_metrics.py): follow-up truncated at
    tau = min(tau_max, max derivation follow-up, max evaluated follow-up); G(t) of the IPCW metrics
    estimated in the evaluated sample (default) or in the derivation cohort (--ipcw-from derivation);
    AUC time points <= tau, a point equal to tau moved just below it.
    """
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
    out = {"n": int(len(time)), "events": int(np.sum(event))}
    ev = np.asarray(event).astype(bool)
    horizon = evaluation_horizon(time, tau_max, max_train)
    out["tau"] = horizon
    y_eval = truncate(structured(ev, time), horizon)
    y_g = ipcw_sample(y_eval, y_deriv, ipcw_from)
    gmax = np.nextafter(horizon, 0)
    try:
        out["harrell_c"] = float(concordance_index_censored(ev, time, risk)[0])
    except Exception:  # noqa: BLE001
        out["harrell_c"] = np.nan
    try:
        out["uno_c"] = float(concordance_index_ipcw(y_g, y_eval, risk, tau=horizon)[0])
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("Uno C failed (n=%d): %s", len(time), e)
        out["uno_c"] = np.nan
    pairs = [(h, h if h < horizon else gmax) for h in horizons if h <= horizon]
    try:
        if pairs:
            aucs, mean_auc = cumulative_dynamic_auc(y_g, y_eval, risk, np.array([p[1] for p in pairs], dtype=float))
            for (h, _), a in zip(pairs, aucs):
                out[f"auc_{h:g}"] = float(a)
            out["mean_auc"] = float(mean_auc)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("AUC failed (n=%d): %s", len(time), e)
    for h in horizons:
        out.setdefault(f"auc_{h:g}", np.nan)
    out.setdefault("mean_auc", np.nan)
    try:
        out[f"ibs_{ibs_h:g}"] = ibs_on_grid(y_g, y_eval, surv, grid, horizon, ibs_h, float(np.min(time)))
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("IBS failed (n=%d, events=%d, tau=%.2f): %s", len(time), int(np.sum(event)), horizon, e)
        out[f"ibs_{ibs_h:g}"] = np.nan
    return out


def bootstrap_ci(y_deriv, max_train, event, time, risk, surv, grid, tau_max, ibs_h, ipcw_from, n_boot, seed,
                 metrics=("harrell_c", "ibs")):
    """Percentile CIs by patient resampling; G(t) re-estimated in each resample (or derivation)."""
    from sksurv.metrics import concordance_index_censored
    rng = np.random.default_rng(seed)
    n = len(time)
    ev = np.asarray(event).astype(bool)
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if ev[idx].sum() < 2:
            continue
        if "harrell_c" in metrics:
            try:
                vals["harrell_c"].append(concordance_index_censored(ev[idx], time[idx], risk[idx])[0])
            except Exception:  # noqa: BLE001
                pass
        if "ibs" in metrics:
            try:
                horizon = evaluation_horizon(time[idx], tau_max, max_train)
                yt = truncate(structured(ev[idx], time[idx]), horizon)
                v = ibs_on_grid(ipcw_sample(yt, y_deriv, ipcw_from), yt, surv[idx], grid, horizon, ibs_h, float(np.min(time[idx])))
                if not np.isnan(v):
                    vals["ibs"].append(v)
            except Exception:  # noqa: BLE001
                pass
    out = {}
    for m, v in vals.items():
        key = "harrell_c" if m == "harrell_c" else f"ibs_{ibs_h:g}"
        out[f"{key}_ci_nboot"] = len(v)  # replicates actually used (others skipped: <2 events or metric failure)
        if len(v) >= 10:
            out[f"{key}_ci_low"], out[f"{key}_ci_high"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        else:
            out[f"{key}_ci_low"], out[f"{key}_ci_high"] = (np.nan, np.nan)
    return out


# ----------------------------------------------------------------------------- calibration
def km_survival_at(event, time, h):
    """Kaplan-Meier survival at h (constant beyond the last observed time, 1 before the first)."""
    from sksurv.nonparametric import kaplan_meier_estimator
    x, y = kaplan_meier_estimator(np.asarray(event, bool), np.asarray(time, float))
    i = np.searchsorted(x, h, side="right") - 1
    return float(y[i]) if i >= 0 else 1.0


def calibration_block(event, time, surv_h, h, tau, groups=5):
    """Calibration of the predicted survival at horizon h on one sample.

    calib_itl_<h>: calibration-in-the-large = Kaplan-Meier observed survival at h minus the mean predicted
    survival at h (positive = the model is pessimistic). calib_slope_<h>: coefficient of a Cox model of the
    outcome (follow-up truncated at tau) on log(-log S_pred(h)), the prognostic index implied by the prediction;
    1 = ideal, < 1 = predictions too extreme. The group rows (quintiles of predicted survival) feed the figure.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    ev, t = np.asarray(event, bool), np.asarray(time, float)
    s = np.clip(np.asarray(surv_h, float), 1e-6, 1 - 1e-6)
    out = {f"calib_itl_{h:g}": km_survival_at(ev, t, h) - float(s.mean())}
    try:
        cox = CoxPHSurvivalAnalysis(alpha=0.0).fit(np.log(-np.log(s))[:, None], truncate(structured(ev, t), tau))
        out[f"calib_slope_{h:g}"] = float(cox.coef_[0])
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("calibration slope at %g failed: %s", h, e)
        out[f"calib_slope_{h:g}"] = np.nan
    rows = []
    grp = pd.qcut(s, groups, labels=False, duplicates="drop")
    for g in sorted(set(grp)):
        m = grp == g
        rows.append({"horizon": h, "group": int(g) + 1, "n": int(m.sum()), "events_by_horizon": int((ev & (t <= h))[m].sum()),
                     "predicted_survival": float(s[m].mean()), "observed_survival_km": km_survival_at(ev[m], t[m], h)})
    return out, rows


def calibration_figure(groups: pd.DataFrame, metrics: dict, run: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = sorted(groups["horizon"].unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(3.8 * len(hs), 4), squeeze=False)
    for ax, h in zip(axes[0], hs):
        g = groups[groups["horizon"] == h]
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
        ax.plot(g["predicted_survival"], g["observed_survival_km"], "o-", color="C0")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel(f"predicted survival at {h:g} months"); ax.set_ylabel("observed (Kaplan-Meier)")
        ax.set_title(f"{h:g} months: ITL {metrics.get(f'calib_itl_{h:g}', np.nan):+.3f}, slope {metrics.get(f'calib_slope_{h:g}', np.nan):.2f}", fontsize=9)
    fig.suptitle(f"{run}: calibration of the global model, quintiles of predicted risk", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- derivation side
def derivation_matrix(cross_site_json: Path) -> pd.DataFrame:
    """Mean validation C-index over CV folds: rows = data site, columns = model."""
    d = json.load(open(cross_site_json))
    df = pd.DataFrame(d["data"], columns=d["columns"])
    df = df[(df["metric"] == "c-index") & (df["data_set_portion"] == "validation") & (df["validation_fold"] >= 0)]
    m = df.groupby(["site_name", "remote_model"])["value"].mean().unstack("remote_model")
    m.index = [s.replace("data_", "") for s in m.index]
    m.columns = [c.replace("model_", "").replace("global_model", "global") for c in m.columns]
    return m


# ----------------------------------------------------------------------------- figures
def forest_figure(table: pd.DataFrame, run: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def xerr(row):
        lo, hi = row.get("harrell_c_ci_low", np.nan), row.get("harrell_c_ci_high", np.nan)
        if pd.isna(lo) or pd.isna(hi):
            return None
        return [[row["harrell_c"] - lo], [hi - row["harrell_c"]]]

    g = table[(table["model"] == "global")].set_index("subset")
    subsets = ["all"] + [a for a in AREAS if a in g.index]
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(subsets) + 1.5))
    y = np.arange(len(subsets))[::-1]
    own_labelled = False
    for i, s in enumerate(subsets):
        r = g.loc[s]
        ax.errorbar(r["harrell_c"], y[i], xerr=xerr(r), fmt="o", color="C0", capsize=3, label="global model" if i == 0 else None)
        own = table[(table["subset"] == s) & (table["model"] == s)]
        if len(own):
            o = own.iloc[0]
            ax.errorbar(o["harrell_c"], y[i] - 0.25, xerr=xerr(o), fmt="s", color="C1", capsize=3,
                        label=None if own_labelled else "own node model")
            own_labelled = True
        ax.text(0.985, y[i] + 0.1, f"n={int(r['n'])}, ev={int(r['events'])}", va="center", ha="right", fontsize=8, transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(subsets); ax.set_xlabel("Harrell C-index on the temporal cohort (95% bootstrap CI)")
    ax.set_title(run); ax.set_xlim(0.35, 1.0); ax.set_ylim(-0.7, len(subsets) - 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def km_risk_groups_figure(time, event, risk, run: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sksurv.nonparametric import kaplan_meier_estimator

    q = np.quantile(risk, [1 / 3, 2 / 3])
    grp = np.digitize(risk, q)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for g_, name in enumerate(["low risk (T1)", "intermediate (T2)", "high risk (T3)"]):
        m = grp == g_
        if m.sum() < 5:
            continue
        t, s = kaplan_meier_estimator(np.asarray(event, bool)[m], np.asarray(time)[m])
        ax.step(t, s, where="post", label=f"{name}, n={m.sum()}, ev={int(np.sum(event[m]))}")
    ax.set_xlabel("months from CGI start"); ax.set_ylabel("overall survival"); ax.set_ylim(0, 1); ax.set_xlim(0, 24)
    ax.set_title(f"{run}\ntemporal cohort by tertile of the global risk score", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def summary_figure(summary: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary.sort_values("run_name")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(s) + 1.5))
    y = np.arange(len(s))[::-1]
    ax.scatter(s["derivation_weighted_c"], y, marker="D", color="grey", label="derivation (CV, weighted)")
    lo = (s["temporal_harrell_c"] - s["temporal_harrell_c_ci_low"]).to_numpy(float)
    hi = (s["temporal_harrell_c_ci_high"] - s["temporal_harrell_c"]).to_numpy(float)
    missing = np.isnan(lo) | np.isnan(hi)
    if missing.any():
        LOGGER.warning("summary figure: no CI for %s", s["run_name"][missing].tolist())
    xerr = [np.nan_to_num(lo), np.nan_to_num(hi)]  # runs without a CI get a zero-length bar
    ax.errorbar(s["temporal_harrell_c"], y, xerr=xerr, fmt="o", color="C0", capsize=3, label="temporal cohort (95% CI)")
    ax.set_yticks(y); ax.set_yticklabels(s["run_name"]); ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("C-index"); ax.set_xlim(0.45, 0.85); ax.set_ylim(-0.7, len(s) - 0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25 if len(s) < 4 else -0.12), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- one run
def score_run(mlflow, client, exp_id, parent, cohort, y_deriv, max_train, args, out_root: Path):
    name = run_name(parent)
    kids = children(client, exp_id, parent.info.run_id)
    best = best_child(kids)
    if best is None:
        LOGGER.warning("%s: no best_optuna_model child (aborted run), skipped", name)
        return None
    out = out_root / "runs" / name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    # model artifacts are downloaded outside the output folder: they are inputs, not results
    model_dir = Path(args.model_cache) / name if args.model_cache else Path(tempfile.mkdtemp(prefix=f"tv_{name}_"))
    n_trials, has_fs = extract_children_tables(mlflow, kids, out, model_dir)
    local = mlflow.artifacts.download_artifacts(run_id=best.info.run_id, artifact_path="model", dst_path=str(model_dir))
    pipe = mlflow.sklearn.load_model(local)
    sig_cols = [c["name"] for c in json.loads(mlflow.models.Model.load(local).signature.inputs.to_json())]
    missing = [c for c in sig_cols if c not in cohort.columns]
    if missing:
        raise RuntimeError(f"{name}: cohort lacks signature columns {missing}")
    X = cohort[sig_cols].astype(float)
    wrapper = pipe.named_steps["model"]
    ens = wrapper.model
    members = list(ens.model)
    weights = ensemble_weights(ens)
    feat_in = list(ens.feature_names_in_)

    # derivation-side info
    try:
        cs = mlflow.artifacts.download_artifacts(run_id=best.info.run_id, artifact_path="cross_site_validation_results.json", dst_path=str(model_dir))
        deriv = derivation_matrix(Path(cs))
        deriv.to_csv(out / "derivation_cv_cindex_matrix.csv")
        node_names = sorted(c for c in deriv.columns if c != "global")
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("%s: cross-site json not usable (%s)", name, e)
        deriv, node_names = None, []
    if len(node_names) != len(members):
        node_names = ["central"] if len(members) == 1 else [f"member_{i}" for i in range(len(members))]
    member_names = node_names  # ensemble members are ordered by node name (results sorted by client name)

    # the framework's privacy-preserving SHAP summary (beeswarm) plot, extracted as logged
    try:
        if "shap_summary_plot.png" in {a.path for a in mlflow.artifacts.list_artifacts(run_id=best.info.run_id)}:
            png = mlflow.artifacts.download_artifacts(run_id=best.info.run_id, artifact_path="shap_summary_plot.png", dst_path=str(model_dir))
            shutil.copyfile(png, out / "figures" / f"shap_summary_{name}.png")
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("%s: SHAP summary plot not extracted: %s", name, e)
    shap_table = shap_importance(mlflow, best, pipe, X, feat_in, args.shap_patients, args.seed, out) if args.shap_patients > 0 else None

    # predictions
    Xt = pipe[:-1].transform(X)
    if not hasattr(Xt, "columns"):
        Xt = pd.DataFrame(Xt, columns=feat_in)
    Xt = Xt[feat_in]
    risk_global = np.asarray(pipe.predict(X), dtype=float)
    risks = {"global": risk_global}
    grid = np.arange(0.5, args.horizon_max + 1e-9, 0.5)
    surv = {}
    member_sfs = []
    for mname, m in zip(member_names, members):
        risks[mname] = np.asarray(m.predict(Xt), dtype=float)
        sfs = m.predict_survival_function(Xt)
        member_sfs.append(sfs)
        surv[mname] = eval_step_functions(sfs, grid)
    surv["global"] = ensemble_survival_framework_rule(member_sfs, weights, grid)
    if len(members) == 1:
        risks = {"global": risk_global}
        surv = {"global": surv["global"]}
        member_names = []
    # sanity: global risk == weighted mean of member risks
    if member_names:
        recon = np.tensordot(np.asarray(weights), np.stack([risks[m] for m in member_names]), axes=1)
        if not np.allclose(recon, risk_global, atol=1e-6):
            LOGGER.warning("%s: global risk differs from the weighted mean of member risks (max abs %.3g)", name, np.max(np.abs(recon - risk_global)))

    pred = cohort[["patient_number", "sigla", "area_federata", "tte", "event"]].copy()
    for k, v in risks.items():
        pred[f"risk_{k}"] = v
    for h in args.horizons:
        j = int(np.argmin(np.abs(grid - h)))
        pred[f"S{h:g}_global"] = surv["global"][:, j]
    pred.to_csv(out / "predictions.csv", index=False)

    # metrics
    rows = []
    time = cohort["tte"].to_numpy(float); event = cohort["event"].to_numpy(int)
    subsets = {"all": np.ones(len(cohort), bool)}
    for a in AREAS:
        m = (cohort["area_federata"] == a).to_numpy()
        if m.sum() >= args.min_subset:
            subsets[a] = m
    for sname, mask in subsets.items():
        for mname in ["global"] + member_names:
            r = {"run_name": name, "model": mname, "subset": sname}
            r.update(metrics_block(y_deriv, max_train, event[mask], time[mask], risks[mname][mask], surv[mname][mask], grid,
                                   args.horizons, args.horizon_max, args.ibs_horizon, args.ipcw_from))
            if args.bootstrap > 0:
                r.update(bootstrap_ci(y_deriv, max_train, event[mask], time[mask], risks[mname][mask], surv[mname][mask], grid,
                                      args.horizon_max, args.ibs_horizon, args.ipcw_from, args.bootstrap, args.seed))
            # derivation-side reference for the same model on the same site. Two estimands, labelled in
            # derivation_cv_kind: 'run_weighted_objective' = the run's sample-weighted CV C-index of the global
            # model (best child metric), used for (global, all) in every arm; 'site_fold_mean' = unweighted
            # mean over CV folds of the validation rows of cross_site_validation_results.json for that site.
            r["source_nodes_version"] = args.source_nodes_version
            if sname == "all" and mname == "global" and best.data.metrics.get("weighted_c-index_validation") is not None:
                r["derivation_cv_cindex"] = float(best.data.metrics["weighted_c-index_validation"])
                r["derivation_cv_kind"] = "run_weighted_objective"
            elif deriv is not None and sname in deriv.index and mname in deriv.columns:
                r["derivation_cv_cindex"] = float(deriv.loc[sname, mname])
                r["derivation_cv_kind"] = "site_fold_mean"
            rows.append(r)
    table = pd.DataFrame(rows)
    # calibration of the global model on the whole cohort, at the calibration horizons within tau
    calib, calib_rows = {}, []
    tau_all = evaluation_horizon(time, args.horizon_max, max_train)
    for h in [h for h in args.calibration_horizons if h <= tau_all]:
        j = int(np.argmin(np.abs(grid - h)))
        m, r_ = calibration_block(event, time, surv["global"][:, j], h, tau_all)
        calib.update(m); calib_rows.extend(r_)
    if calib_rows:
        pd.DataFrame(calib_rows).to_csv(out / "calibration.csv", index=False)
        for k, v in calib.items():
            table.loc[(table["model"] == "global") & (table["subset"] == "all"), k] = v
    table.to_csv(out / "temporal_metrics.csv", index=False)
    table.pivot(index="subset", columns="model", values="harrell_c").to_csv(out / "temporal_harrell_c_matrix.csv")

    info = {
        "run_name": name, **run_design(parent), "n_trials": n_trials, "has_feature_selection_tables": has_fs,
        "shap_top_features": list(shap_table["feature"][:5]) if shap_table is not None else None,
        "parent_run_id": parent.info.run_id, "best_child_run_id": best.info.run_id,
        "model_artifact_uri": f"runs:/{best.info.run_id}/model",
        "derivation_weighted_c": best.data.metrics.get("weighted_c-index_validation"),
        "signature_columns": sig_cols, "members": [type(m).__name__ for m in members], "member_names": member_names,
        "weights": weights, "n_sample": list(getattr(ens, "n_sample", [])), "params": parent.data.params,
        "best_params": best.data.params, "best_tags": {k: v for k, v in best.data.tags.items() if not k.startswith("mlflow.")},
        "source_nodes_version": args.source_nodes_version,
    }
    json.dump(info, open(out / "model_info.json", "w"), indent=2)
    try:
        forest_figure(table, name, out / "figures" / f"forest_harrell_c_{name}.png")
        km_risk_groups_figure(time, event, risk_global, name, out / "figures" / f"km_risk_tertiles_{name}.png")
        if calib_rows:
            calibration_figure(pd.DataFrame(calib_rows), calib, name, out / "figures" / f"calibration_{name}.png")
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("%s: figure failed: %s", name, e)
    g_all = table[(table["model"] == "global") & (table["subset"] == "all")].iloc[0]
    LOGGER.info("%s: derivation C %.3f | temporal Harrell C %.3f [%.3f, %.3f], Uno C %.3f, IBS%g %.3f (n=%d, ev=%d)",
                name, info["derivation_weighted_c"] or np.nan, g_all["harrell_c"], g_all.get("harrell_c_ci_low", np.nan), g_all.get("harrell_c_ci_high", np.nan),
                g_all["uno_c"], args.ibs_horizon, g_all[f"ibs_{args.ibs_horizon:g}"], g_all["n"], g_all["events"])
    return {"table": table, "info": info}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-name", required=True, nargs="+", help="source MLflow experiment(s); several names separated by ';'")
    ap.add_argument("--cohort", required=True, help="prepared temporal cohort CSV (prepare_temporal_cohort.py)")
    ap.add_argument("--derivation-csv", default=None, help="pooled derivation data (central/original/data.csv): sets the maximum follow-up for tau; "
                    "required for --ipcw-from derivation")
    ap.add_argument("--ipcw-from", choices=["evaluated", "derivation"], default="evaluated",
                    help="sample used to estimate the censoring distribution of the IPCW metrics: the evaluated subset/resample "
                         "(PRECISION-AI convention, default) or the derivation cohort (sensitivity)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-names", nargs="*", default=None, help="restrict to these parent run names")
    ap.add_argument("--horizons", nargs="*", type=float, default=[6.0, 12.0, 18.0, 24.0], help="AUC time points (months); those above tau are skipped")
    ap.add_argument("--horizon-max", type=float, default=24.0, help="clinical horizon: tau = min(this, max derivation follow-up, max evaluated follow-up)")
    ap.add_argument("--ibs-horizon", type=float, default=12.0, help="upper end of the IBS integration window (months)")
    ap.add_argument("--calibration-horizons", nargs="*", type=float, default=[6.0, 12.0, 18.0], help="calibration of the global model at these months (those above tau skipped)")
    ap.add_argument("--shap-patients", type=int, default=100, help="patients of the cohort (random subsample) for the SHAP importance table of the global model; "
                    "0 = skip (the permutation explainer costs about 2 x features x 40 background rows predictions per patient)")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-subset", type=int, default=20, help="minimum patients for a per-area row")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--mlflow-experiment", default=None)
    ap.add_argument("--fail-on-error", action="store_true", help="exit non-zero if any run could not be scored (pipeline use)")
    ap.add_argument("--model-cache", default=None, help="folder for the downloaded model artifacts (default: a temporary directory, kept out of the results)")
    ap.add_argument("--source-nodes-version", default="v2", help="node partition the scored models were trained on (v1 = Centri sheet split, v2 = corrected geography); "
                    "recorded in every table because per-area rows only match v2 members")
    ap.add_argument("--raw-data-label", default="", help="data asset name:version of the raw registry CSV (lineage only)")
    ap.add_argument("--data-folder-label", default="", help="data asset name:version of the node data folder (lineage only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("azure", "azureml", "urllib3", "msal", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    args.horizons = sorted(set(args.horizons))
    args.calibration_horizons = sorted(set(args.calibration_horizons))
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort)
    LOGGER.info("temporal cohort %d patients, %d events, median observed time %.1f months, max %.1f",
                len(cohort), int(cohort["event"].sum()), float(cohort["tte"].median()), float(cohort["tte"].max()))
    y_deriv, max_train = None, None
    if args.derivation_csv:
        deriv = pd.read_csv(args.derivation_csv)
        y_deriv = structured(pd.to_numeric(deriv["event"]).astype(int).to_numpy(), pd.to_numeric(deriv["tte"]).to_numpy())
        max_train = float(y_deriv["time"].max())
        LOGGER.info("derivation cohort %d patients, max follow-up %.1f months", len(deriv), max_train)
    elif args.ipcw_from == "derivation":
        raise SystemExit("--ipcw-from derivation requires --derivation-csv")
    LOGGER.info("IPCW censoring distribution from the %s; tau = min(%g, %s, max evaluated follow-up)",
                "derivation cohort" if args.ipcw_from == "derivation" else "evaluated sample", args.horizon_max,
                f"{max_train:.1f}" if max_train else "no derivation bound")

    # keep a copy of the cohort next to the results so that one download holds everything
    cohort_dir = out_root / "cohort"
    cohort_dir.mkdir(exist_ok=True)
    for src in [Path(args.cohort), Path(args.cohort).with_suffix(".summary.json")]:
        if src.exists():
            shutil.copyfile(src, cohort_dir / src.name)  # data only: no chmod on a mounted output

    mlflow, client = mlflow_client()
    exp_names = [n.strip() for n in " ".join(args.experiment_name).split(";") if n.strip()]
    wanted = [n.strip() for n in " ".join(args.run_names or []).replace(",", " ").split() if n.strip()]
    if len(wanted) == 1 and wanted[0].lower() == "all":
        wanted = []  # sentinel used by the Azure ML component (an empty string input is dropped by the service)
    experiments, parents, skipped = {}, [], {}
    for exp_name in exp_names:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            raise SystemExit(f"experiment not found: {exp_name}")
        experiments[exp_name] = exp.experiment_id
        parents_all = parent_runs(client, exp.experiment_id)
        finished = [p for p in parents_all if p.info.status == "FINISHED"]
        if wanted:
            finished = [p for p in finished if run_name(p) in set(wanted)]
        LOGGER.info("experiment %r: %d parent runs, %d finished%s -> %d to score: %s", exp_name, len(parents_all),
                    sum(p.info.status == "FINISHED" for p in parents_all), f" (filtered to {len(wanted)} names)" if wanted else "",
                    len(finished), sorted(run_name(p) for p in finished))
        parents.extend((exp.experiment_id, exp_name, p) for p in sorted(finished, key=run_name))
        skipped.update({f"{exp_name}: {run_name(p)}": p.info.status for p in parents_all if p.info.status != "FINISHED"})
    names = [run_name(p) for _, _, p in parents]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise SystemExit(f"run names shared by several source experiments (one folder per run name): {duplicated}")
    if wanted:
        unknown = sorted(set(wanted) - {run_name(p) for _, _, p in parents})
        if unknown:
            LOGGER.warning("requested run names not found or not finished: %s", unknown)
    if skipped:
        LOGGER.warning("not finished, skipped: %s", skipped)

    results, failures = [], {}
    for exp_id, exp_name, p in parents:
        try:
            r = score_run(mlflow, client, exp_id, p, cohort, y_deriv, max_train, args, out_root)
            if r:
                r["info"]["source_experiment"] = exp_name
                results.append(r)
        except Exception as e:  # noqa: BLE001
            failures[run_name(p)] = f"{e}\n{traceback.format_exc()}"
            LOGGER.error("%s failed: %s", run_name(p), e)
    if not results:
        raise SystemExit("nothing scored")

    # summary across runs
    rows = []
    for r in results:
        t, info = r["table"], r["info"]
        g = t[(t["model"] == "global") & (t["subset"] == "all")].iloc[0]
        row = {"run_name": info["run_name"], "source_experiment": info["source_experiment"], "role": info["role"], "node": info["node"],
               "model": info["model"], "fs": info["fs"], "imputation": info["imputation"],
               "n_features": len(info["signature_columns"]), "selected_features": "; ".join(info["signature_columns"]),
               "derivation_weighted_c": info["derivation_weighted_c"],
               "temporal_harrell_c": g["harrell_c"], "temporal_harrell_c_ci_low": g.get("harrell_c_ci_low", np.nan),
               "temporal_harrell_c_ci_high": g.get("harrell_c_ci_high", np.nan), "temporal_uno_c": g["uno_c"]}
        for h in args.horizons:
            row[f"temporal_auc_{h:g}"] = g[f"auc_{h:g}"]
        row[f"temporal_ibs_{args.ibs_horizon:g}"] = g[f"ibs_{args.ibs_horizon:g}"]
        for k in [c for c in t.columns if c.startswith("calib_")]:
            row[f"temporal_{k}"] = g[k]
        row["shap_top5"] = "; ".join(info["shap_top_features"]) if info.get("shap_top_features") else ""
        for a in AREAS:
            ga = t[(t["model"] == "global") & (t["subset"] == a)]
            row[f"temporal_c_{a}"] = ga["harrell_c"].values[0] if len(ga) else np.nan
            for side in ("low", "high"):  # per-area bootstrap CI (TRIPOD-Cluster 13a)
                col = f"harrell_c_ci_{side}"
                row[f"temporal_c_{a}_ci_{side}"] = ga[col].values[0] if len(ga) and col in ga.columns else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("run_name")
    summary.to_csv(out_root / "summary_runs.csv", index=False)
    pd.concat([r["table"] for r in results]).to_csv(out_root / "temporal_metrics_all_runs.csv", index=False)
    # derivation side across runs: weighted objective, per-site CV matrix (long) and the run parameters
    drows, prow = [], []
    for r in results:
        info = r["info"]
        prow.append({"run_name": info["run_name"], "source_experiment": info["source_experiment"], "role": info["role"], "node": info["node"],
                     "parent_run_id": info["parent_run_id"], "best_child_run_id": info["best_child_run_id"],
                     "model_artifact_uri": info["model_artifact_uri"], "source_nodes_version": info["source_nodes_version"], "n_trials": info["n_trials"],
                     "derivation_weighted_c": info["derivation_weighted_c"], "n_members": len(info["members"]), "weights": json.dumps(info["weights"]),
                     "n_sample": json.dumps(info["n_sample"]), "n_signature_features": len(info["signature_columns"]),
                     **{f"param__{k}": v for k, v in sorted(info["params"].items())},
                     **{f"best_param__{k}": v for k, v in sorted(info["best_params"].items())}})
        m = out_root / "runs" / info["run_name"] / "derivation_cv_cindex_matrix.csv"
        if m.exists():
            dm = pd.read_csv(m, index_col=0)
            for site in dm.index:
                for model in dm.columns:
                    drows.append({"run_name": info["run_name"], "site": site, "model": model, "cv_cindex": dm.loc[site, model]})
    pd.DataFrame(prow).to_csv(out_root / "derivation_runs.csv", index=False)
    if drows:
        pd.DataFrame(drows).to_csv(out_root / "derivation_cv_cindex_all_runs.csv", index=False)
    try:
        summary_figure(summary, out_root / "summary_derivation_vs_temporal.png")
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("summary figure failed: %s", e)
    scored = [{"run_name": r["info"]["run_name"], "parent_run_id": r["info"]["parent_run_id"], "best_child_run_id": r["info"]["best_child_run_id"],
               "model_artifact_uri": r["info"]["model_artifact_uri"]} for r in results]
    done = {s["run_name"] for s in scored}
    no_final_model = {run_name(p): p.info.run_id for _, _, p in parents if run_name(p) not in done and run_name(p) not in failures}
    if no_final_model:
        LOGGER.warning("finished runs without a best_optuna_model child (aborted jobs), not scored: %s", sorted(no_final_model))
    json.dump({"source_experiments": experiments, "args": vars(args),
               "library_versions": library_versions(), "runs_scored": scored, "runs_not_finished": skipped,
               "runs_without_final_model": no_final_model, "failures": failures,
               "cohort": {"n": int(len(cohort)), "events": int(cohort["event"].sum())}}, open(out_root / "run_info.json", "w"), indent=2)
    LOGGER.info("\n%s", summary[["run_name", "derivation_weighted_c", "temporal_harrell_c", "temporal_harrell_c_ci_low", "temporal_harrell_c_ci_high", "temporal_uno_c", f"temporal_ibs_{args.ibs_horizon:g}"]].to_string(index=False))

    if failures:
        LOGGER.error("failed runs: %s", list(failures))
    if not args.no_mlflow:
        if in_azureml_job():
            # inside an Azure ML job start_run() attaches to the job's own run: metrics, params and
            # artifacts land on the job, so one job download holds everything
            ctx, exp_name = mlflow.start_run(), os.getenv("AZUREML_ARM_PROJECT_NAME", "azureml job")
        else:
            exp_name = args.mlflow_experiment or f"Temporal Validation - {exp_names[0]}"
            mlflow.set_experiment(exp_name)
            ctx = mlflow.start_run(run_name=f"temporal_validation_{len(results)}_runs")
        with ctx as run:
            params = {"source_experiment": "; ".join(exp_names), "source_experiment_id": "; ".join(experiments.values()),
                      "n_runs_scored": len(results), "n_runs_skipped": len(skipped), "n_runs_failed": len(failures),
                      "bootstrap": args.bootstrap, "seed": args.seed, "ipcw_from": args.ipcw_from,
                      "horizons": ",".join(f"{h:g}" for h in args.horizons), "horizon_max": args.horizon_max, "ibs_horizon": args.ibs_horizon,
                      "min_subset": args.min_subset, "source_nodes_version": args.source_nodes_version,
                      "raw_data_label": args.raw_data_label, "data_folder_label": args.data_folder_label,
                      "cohort_n": int(len(cohort)), "cohort_events": int(cohort["event"].sum())}
            params.update({f"lib_{k}": v for k, v in library_versions().items()})
            mlflow.log_params(params)
            for _, s in summary.iterrows():
                key = s["run_name"]
                mlflow.log_metric(f"harrell_c__{key}", float(s["temporal_harrell_c"]))
                if pd.notna(s["derivation_weighted_c"]):
                    mlflow.log_metric(f"derivation_c__{key}", float(s["derivation_weighted_c"]))
                if pd.notna(s["temporal_uno_c"]):
                    mlflow.log_metric(f"uno_c__{key}", float(s["temporal_uno_c"]))
                v = s[f"temporal_ibs_{args.ibs_horizon:g}"]
                if pd.notna(v):
                    mlflow.log_metric(f"ibs{args.ibs_horizon:g}__{key}", float(v))
            mlflow.log_artifacts(str(out_root))  # models were downloaded elsewhere: the folder is results only
            LOGGER.info("logged to MLflow (%s) run %s", exp_name, run.info.run_id)
            if failures and args.fail_on_error:
                raise SystemExit(f"{len(failures)} run(s) failed: {list(failures)}")  # inside the context: the run ends FAILED
    elif failures and args.fail_on_error:
        raise SystemExit(f"{len(failures)} run(s) failed: {list(failures)}")


if __name__ == "__main__":
    main()
