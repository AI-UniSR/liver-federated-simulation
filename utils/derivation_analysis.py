"""Derivation-side analysis of one or more SraceFed MLflow experiments (grid of 24 runs).

For each experiment: parent runs, best-child metrics (all keys, e.g. weighted C-index and IBS),
per-site cross-site matrices (validation and train rows, c-index and ibs), feature-selection
results (mode and features chosen) and every Optuna trial's metrics, so that central and
federated arms can be compared trial by trial (same seed -> same hyper-parameters per trial).

Outputs (--output-dir): runs.csv, trials.csv, cross_site_long.csv, feature_selection.csv,
site_matrix_<exp>_<run>.csv and a summary.md with the tables used in the results notes.

Run on the compute instance with the framework shim (needs mlflow + azure-ai-ml):
  .venv/bin/python utils/derivation_analysis.py --experiments "Federated Experiment October 22nd" \
      "Federated Experiment October 23rd" --labels round1 round2 --output-dir /tmp/deriv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("derivation_analysis")
SUBSCRIPTION = os.getenv("AZURE_SUBSCRIPTION_ID", "ab211f7b-463f-4833-9605-d260e596a35a")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "1fe41577-4b8d-412a-3055-3a18d9df3ee7")
WORKSPACE = os.getenv("AZUREML_WORKSPACE_NAME", "1fe415774b8d412a30553a18d9df3ee7")


def client():
    import mlflow
    from mlflow.tracking import MlflowClient
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri or not os.getenv("AZUREML_RUN_ID"):
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        uri = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE).workspaces.get(WORKSPACE).mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    return mlflow, MlflowClient(tracking_uri=uri)


def search_all(c, exp_id, flt=""):
    out, tok = [], None
    while True:
        page = c.search_runs([exp_id], filter_string=flt, max_results=500, page_token=tok)
        out.extend(page)
        tok = getattr(page, "token", None)
        if not tok:
            return out


def md(df, index=True):
    """Markdown table without the optional tabulate dependency."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.tolist()) + " |")
    return chr(10).join(lines)


def rn(r):
    return r.data.tags.get("mlflow.runName", "")


def parse_name(name):
    p = name.split("_")
    return {"arm": p[0], "model": p[1] if len(p) > 1 else "", "fs": p[2] if len(p) > 2 else "", "imputation": p[3] if len(p) > 3 else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("azure", "azureml", "urllib3", "msal", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.makedirs(args.output_dir, exist_ok=True)
    mlflow, c = client()
    tmp = tempfile.mkdtemp(prefix="deriv_")

    runs_rows, trial_rows, cs_rows, fs_rows = [], [], [], []
    for exp_name, label in zip(args.experiments, args.labels):
        exp = c.get_experiment_by_name(exp_name)
        if exp is None:
            LOGGER.error("experiment not found: %s", exp_name)
            continue
        allruns = search_all(c, exp.experiment_id)
        parents = [r for r in allruns if "mlflow.parentRunId" not in r.data.tags]
        children = {}
        for r in allruns:
            pid = r.data.tags.get("mlflow.parentRunId")
            if pid:
                children.setdefault(pid, []).append(r)
        LOGGER.info("%s: %d parents, %d children", label, len(parents), sum(len(v) for v in children.values()))
        for p in sorted(parents, key=rn):
            name = rn(p)
            kids = children.get(p.info.run_id, [])
            best = [k for k in kids if rn(k) == "best_optuna_model"]
            best = sorted(best, key=lambda k: k.info.start_time or 0)[-1] if best else None
            trials = [k for k in kids if rn(k).startswith("optuna-trial-")]
            row = {"experiment": label, "run_name": name, **parse_name(name), "parent_run_id": p.info.run_id, "status": p.info.status,
                   "n_trials": len(trials), "has_best": best is not None,
                   "best_trial": best.data.tags.get("best_trial") if best else None}
            if best:
                row.update({f"best__{k}": v for k, v in best.data.metrics.items()})
                row.update({f"best_param__{k}": v for k, v in best.data.params.items()})
                # cross-site rows of the best trial
                try:
                    f = mlflow.artifacts.download_artifacts(run_id=best.info.run_id, artifact_path="cross_site_validation_results.json", dst_path=os.path.join(tmp, best.info.run_id))
                    d = json.load(open(f))
                    df = pd.DataFrame(d["data"], columns=d["columns"])
                    df.insert(0, "run_name", name); df.insert(0, "experiment", label)
                    cs_rows.append(df)
                    v = df[(df["data_set_portion"] == "validation") & (df["validation_fold"] >= 0)]
                    for metric in v["metric"].unique():
                        m = v[v["metric"] == metric].groupby(["site_name", "remote_model"])["value"].mean().unstack("remote_model")
                        m.index = [s.replace("data_", "") for s in m.index]; m.columns = [x.replace("model_", "").replace("global_model", "global") for x in m.columns]
                        m.to_csv(os.path.join(args.output_dir, f"site_matrix_{label}_{name}_{metric}.csv"))
                    t = df[(df["data_set_portion"] == "train") & (df["validation_fold"] >= 0) & (df["remote_model"] == "model_global_model") & (df["metric"] == "c-index")]
                    row["global_train_cindex_mean"] = float(t["value"].mean()) if len(t) else np.nan
                    vg = v[(v["remote_model"] == "model_global_model") & (v["metric"] == "c-index")]
                    row["global_validation_cindex_sitemean"] = float(vg["value"].mean()) if len(vg) else np.nan
                    # fold-level weighted C (per fold, weight = n_sample) -> SD across folds
                    folds = []
                    for fold, g in vg.groupby("validation_fold"):
                        folds.append(np.average(g["value"], weights=g["n_sample"]))
                    row["global_validation_cindex_fold_sd"] = float(np.std(folds, ddof=1)) if len(folds) > 1 else np.nan
                except Exception as e:  # noqa: BLE001
                    LOGGER.warning("%s/%s: cross-site json not read (%s)", label, name, e)
            fsk = [k for k in kids if rn(k) == "feature_selection"]
            if fsk:
                try:
                    f = mlflow.artifacts.download_artifacts(run_id=fsk[0].info.run_id, artifact_path="feature_selection_results.json", dst_path=os.path.join(tmp, fsk[0].info.run_id))
                    d = json.load(open(f))
                    fs_rows.append({"experiment": label, "run_name": name, "fs_json": json.dumps(d)[:4000]})
                except Exception as e:  # noqa: BLE001
                    LOGGER.warning("%s/%s: FS json not read (%s)", label, name, e)
            for t in trials:
                tr = {"experiment": label, "run_name": name, **parse_name(name), "trial": int(rn(t).split("-")[-1]), "status": t.info.status}
                tr.update({f"metric__{k}": v for k, v in t.data.metrics.items()})
                tr.update({f"param__{k}": v for k, v in t.data.params.items()})
                trial_rows.append(tr)
            runs_rows.append(row)

    runs = pd.DataFrame(runs_rows)
    runs.to_csv(os.path.join(args.output_dir, "runs.csv"), index=False)
    trials = pd.DataFrame(trial_rows)
    trials.to_csv(os.path.join(args.output_dir, "trials.csv"), index=False)
    if cs_rows:
        pd.concat(cs_rows).to_csv(os.path.join(args.output_dir, "cross_site_long.csv"), index=False)
    pd.DataFrame(fs_rows).to_csv(os.path.join(args.output_dir, "feature_selection.csv"), index=False)

    # ---- summary tables -------------------------------------------------------------------------
    parts = []
    metric = "best__weighted_c-index_validation"
    parts.append("## Best-trial weighted validation C-index per configuration\n")
    for label in args.labels:
        r = runs[runs["experiment"] == label]
        if metric not in r.columns:
            continue
        piv = r.pivot_table(index=["model", "fs", "imputation"], columns="arm", values=metric)
        if {"central", "federated"} <= set(piv.columns):
            piv["delta_fed_minus_central"] = piv["federated"] - piv["central"]
        parts.append(f"### {label}\n\n" + piv.round(4).pipe(md) + "\n")
    ibs_cols = [x for x in runs.columns if x.startswith("best__") and "ibs" in x.lower()]
    if ibs_cols:
        parts.append("## Best-trial IBS metrics (where logged)\n")
        for label in args.labels:
            r = runs[runs["experiment"] == label]
            piv = r.pivot_table(index=["model", "fs", "imputation"], columns="arm", values=ibs_cols[0])
            parts.append(f"### {label} ({ibs_cols[0]})\n\n" + piv.round(4).pipe(md) + "\n")
    parts.append("## Train vs validation C-index of the global model (best trial, mean over folds)\n")
    cols = ["experiment", "run_name", "global_train_cindex_mean", "global_validation_cindex_sitemean", "global_validation_cindex_fold_sd", metric]
    parts.append(runs[[x for x in cols if x in runs.columns]].round(4).pipe(md, index=False) + "\n")
    # trial-paired comparison
    tm = [x for x in trials.columns if x.startswith("metric__")]
    if tm:
        key = "metric__weighted_c-index_validation" if "metric__weighted_c-index_validation" in tm else tm[0]
        parts.append(f"## Trial-paired central vs federated ({key}), per configuration\n")
        rows = []
        for (label, model, fs, imp), g in trials.groupby(["experiment", "model", "fs", "imputation"]):
            piv = g.pivot_table(index="trial", columns="arm", values=key)
            if {"central", "federated"} <= set(piv.columns):
                d = (piv["federated"] - piv["central"]).dropna()
                rows.append({"experiment": label, "model": model, "fs": fs, "imputation": imp, "n_pairs": len(d), "mean_delta": d.mean(), "sd_delta": d.std(ddof=1),
                             "frac_fed_better": float((d > 0).mean()), "central_best": piv["central"].max(), "federated_best": piv["federated"].max(),
                             "central_trial_sd": piv["central"].std(ddof=1), "federated_trial_sd": piv["federated"].std(ddof=1)})
        parts.append(pd.DataFrame(rows).round(4).pipe(md, index=False) + "\n")
    # site matrices: diagonal vs global
    if cs_rows:
        cs = pd.concat(cs_rows)
        v = cs[(cs["data_set_portion"] == "validation") & (cs["validation_fold"] >= 0) & (cs["metric"] == "c-index") & (cs["run_name"].str.startswith("federated"))]
        rows = []
        for (label, name), g in v.groupby(["experiment", "run_name"]):
            m = g.groupby(["site_name", "remote_model"])["value"].mean().unstack("remote_model")
            for site in m.index:
                own = "model_" + site.replace("data_", "")
                rows.append({"experiment": label, "run_name": name, "site": site.replace("data_", ""), "own_model": m.loc[site, own] if own in m.columns else np.nan,
                             "global": m.loc[site, "model_global_model"], "best_other": m.loc[site].drop([x for x in [own, "model_global_model"] if x in m.columns]).max()})
        parts.append("## Federated runs: per-site validation C-index, own node model vs global vs best other node (mean over folds)\n")
        parts.append(pd.DataFrame(rows).round(3).pipe(md, index=False) + "\n")
    # feature selection
    if fs_rows:
        parts.append("## Feature selection results (raw json head)\n")
        for r in fs_rows:
            parts.append(f"- **{r['experiment']} / {r['run_name']}**: `{r['fs_json'][:600]}`\n")
    with open(os.path.join(args.output_dir, "summary.md"), "w") as f:
        f.write("\n".join(parts))
    LOGGER.info("written %s", args.output_dir)


if __name__ == "__main__":
    main()
