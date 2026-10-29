"""Comparisons across the runs scored by score_temporal.py: one folder in, one folder out.

Reads the score output (summary_runs.csv, runs/<run>/{predictions,trials,feature_selection_*}.csv), copies it
into --output-dir and adds tables/ and report.md. The design of every run is read from summary_runs.csv:
role (central / federated / standalone + node), model, fs (feature selection), imputation.

Comparisons are the pairs of runs that differ in exactly one factor and share the others
(factor levels ordered nofs < mi < score, basic < mice, central < federated; the pair is "higher minus lower"):

* arm:        federated - central               (same model, fs, imputation)
* fs:         mi - nofs, score - nofs, score - mi (same role/node, model, imputation)
* imputation: mice - basic                       (same role/node, model, fs)
* standalone: node alone - federated global and node alone - central, same model/fs/imputation, on the node's
  own temporal area and on the whole cohort

For each pair: difference of the temporal Harrell C of the global models with a paired bootstrap CI (same patients,
same resamples for both runs) and, when both runs log a feature ranking, the agreement of the rankings (Spearman over
all features, overlap of the top 4/8/10, Jaccard of the selected lists). Derivation side: the best-trial pivot and the difference federated - central over
the trials with identical hyper-parameters (matched on the trial parameters: the TPE start-up trials, 10 of 50 with
the framework's seed; later trials diverge because the sampler conditions on the observed objective).

tables/: runs.csv (one row per run: design, selected features, derivation C, temporal metrics, calibration, SHAP
top 5 and the Spearman between the SHAP order and the selection order of the selected features),
paired_comparisons.csv, ranking_agreement.csv, rankings.csv (long), ranking_model_independence.csv,
shap_agreement.csv (same pairs: Spearman over the common input features, top 4/8/10 overlap and Jaccard of the
top-10 mean |SHAP| sets, which for a model with <= 10 inputs is its whole input set), derivation_best.csv, derivation_trial_paired.csv,
standalone_matrix.csv. report.md: the key tables.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("compare_runs")
LEVELS = {"role": ["central", "federated"], "fs": ["nofs", "mi", "score"], "imputation": ["basic", "mice"]}
TOPS = (4, 8, 10)
PAIR_COLS = ["comparison", "label", "model", "fs", "imputation", "role", "node", "run_a", "run_b", "subset",
             "c_a", "c_b", "delta", "ci_low", "ci_high", "n", "n_boot"]


def level_rank(factor, value):
    """Order of a factor level: the known levels first, unknown ones after them alphabetically."""
    return (LEVELS[factor].index(value), "") if value in LEVELS[factor] else (len(LEVELS[factor]), value)


def copy_tree_files(src: Path, dst: Path):
    """Copy the files of src under dst (data only: no permission or timestamp copy, which fails on a mounted output)."""
    for p in src.rglob("*"):
        if p.is_file():
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)


# ----------------------------------------------------------------------------- helpers
def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    """Markdown table without the tabulate dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def harrell(event, time, risk):
    from sksurv.metrics import concordance_index_censored
    return float(concordance_index_censored(event, time, risk)[0])


def paired_bootstrap(event, time, risk_a, risk_b, idx, n_boot, seed):
    """Difference of Harrell C (a - b) on the patients idx with a percentile CI from n_boot resamples of idx,
    the same resample for both models."""
    ev, t = event[idx], time[idx]
    ra, rb = risk_a[idx], risk_b[idx]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(idx), len(idx))
        if ev[i].sum() < 2:
            continue
        d.append(harrell(ev[i], t[i], ra[i]) - harrell(ev[i], t[i], rb[i]))
    d = np.array(d)
    return {"c_a": harrell(ev, t, ra), "c_b": harrell(ev, t, rb), "delta": harrell(ev, t, ra) - harrell(ev, t, rb),
            "ci_low": float(np.quantile(d, 0.025)) if len(d) >= 10 else np.nan,
            "ci_high": float(np.quantile(d, 0.975)) if len(d) >= 10 else np.nan, "n": int(len(idx)), "n_boot": int(len(d))}


def read_ranking(run_dir: Path, fs: str):
    """Feature ranking logged by the framework's feature-selection stage, best first: mutual information by the
    score column (FCMI - aFFMI), score_test by the forward order (row order). None when the run has no FS tables."""
    f = run_dir / "feature_selection_scores.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    s = s[s["run_name"] == s["run_name"].iloc[0]]
    if "score" in s.columns:
        s = s.sort_values("score", ascending=False, kind="stable")
    return list(s["features"])


def agreement(rank_a, rank_b, sel_a, sel_b):
    """Spearman over the features present in both rankings (ranked within the common set), top-k overlaps and the
    Jaccard index of the selected lists."""
    common = set(rank_a) & set(rank_b)
    ca, cb = [f for f in rank_a if f in common], [f for f in rank_b if f in common]
    ra, rb = {f: i for i, f in enumerate(ca)}, {f: i for i, f in enumerate(cb)}
    n = len(common)
    rho = 1 - 6 * sum((ra[f] - rb[f]) ** 2 for f in common) / (n * (n * n - 1)) if n > 1 else np.nan
    out = {"spearman": rho, "n_features": n}
    for k in TOPS:
        out[f"top{k}_overlap"] = len(set(rank_a[:k]) & set(rank_b[:k]))
    out["jaccard_selected"] = len(set(sel_a) & set(sel_b)) / len(set(sel_a) | set(sel_b)) if (set(sel_a) | set(sel_b)) else np.nan
    out["top4_a"], out["top4_b"] = ", ".join(rank_a[:4]), ", ".join(rank_b[:4])
    return out


# ----------------------------------------------------------------------------- pairs
def design_pairs(runs: pd.DataFrame):
    """Pairs of run names differing in exactly one factor (role among central/federated, fs, imputation) with the
    same model and node; and standalone runs against the federated and central run of the same configuration."""
    pairs = []
    for a, b in combinations(runs.itertuples(index=False), 2):
        if a.model != b.model or a.node != b.node:
            continue
        diff = [f for f in ("role", "fs", "imputation") if getattr(a, f) != getattr(b, f)]
        if len(diff) != 1 or diff[0] == "role" and {a.role, b.role} != {"central", "federated"}:
            continue
        f = diff[0]
        hi, lo = (a, b) if level_rank(f, getattr(a, f)) > level_rank(f, getattr(b, f)) else (b, a)
        pairs.append({"comparison": f if f != "role" else "arm", "label": f"{getattr(hi, f)} - {getattr(lo, f)}",
                      "model": a.model, "fs": a.fs if f != "fs" else "", "imputation": a.imputation if f != "imputation" else "",
                      "role": a.role if f != "role" else "", "node": a.node, "run_a": hi.run_name, "run_b": lo.run_name, "subset": "all"})
    for s in runs[runs["role"] == "standalone"].itertuples(index=False):
        for ref_role in ("federated", "central"):
            ref = runs[(runs["role"] == ref_role) & (runs["model"] == s.model) & (runs["fs"] == s.fs) & (runs["imputation"] == s.imputation)]
            if len(ref) != 1:
                continue
            for subset in (s.node, "all"):
                pairs.append({"comparison": f"standalone_vs_{ref_role}", "label": f"node alone - {ref_role}", "model": s.model, "fs": s.fs,
                              "imputation": s.imputation, "role": "standalone", "node": s.node, "run_a": s.run_name, "run_b": ref.iloc[0]["run_name"],
                              "subset": subset})
    return pairs


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="output folder of score_temporal.py")
    ap.add_argument("--output-dir", required=True, help="final folder: copy of --results plus tables/ and report.md")
    ap.add_argument("--bootstrap", type=int, default=1000, help="paired bootstrap resamples per comparison")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel workers for the bootstrap (joblib)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src, out = Path(args.results), Path(args.output_dir)
    if src.resolve() != out.resolve():
        copy_tree_files(src, out)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)

    runs = pd.read_csv(out / "summary_runs.csv").fillna({"node": ""})
    runs["node"] = runs["node"].astype(str)
    runs["selected"] = runs["selected_features"].apply(lambda s: [x.strip() for x in str(s).split(";")])
    LOGGER.info("%d runs: %s", len(runs), runs.groupby(["role"]).size().to_dict())

    # patient-level predictions, same patients in every run
    pred = {r: pd.read_csv(out / "runs" / r / "predictions.csv") for r in runs["run_name"]}
    ref = next(iter(pred.values()))
    pid = ref["patient_number"].to_numpy()
    for r, p in pred.items():
        if not np.array_equal(p["patient_number"].to_numpy(), pid):
            raise SystemExit(f"{r}: predictions are not on the same patients as {runs['run_name'][0]}")
    event, time, area = ref["event"].to_numpy(bool), ref["tte"].to_numpy(float), ref["area_federata"].astype(str).to_numpy()
    risk = {r: p["risk_global"].to_numpy(float) for r, p in pred.items()}
    rankings = {r: read_ranking(out / "runs" / r, fs) for r, fs in zip(runs["run_name"], runs["fs"])}
    selected = dict(zip(runs["run_name"], runs["selected"]))
    shap_rank = {}
    for r in runs["run_name"]:
        f = out / "runs" / r / "shap_importance.csv"
        if f.exists():
            shap_rank[r] = list(pd.read_csv(f).sort_values("mean_abs_shap", ascending=False)["feature"])

    # --- paired comparisons (temporal Harrell C) and ranking agreement, same pairs
    pairs = design_pairs(runs)
    LOGGER.info("%d paired comparisons", len(pairs))
    from joblib import Parallel, delayed

    def one(p):
        idx = np.arange(len(pid)) if p["subset"] == "all" else np.flatnonzero(area == p["subset"])
        return {**p, **paired_bootstrap(event, time, risk[p["run_a"]], risk[p["run_b"]], idx, args.bootstrap, args.seed)}
    rows = Parallel(n_jobs=args.n_jobs)(delayed(one)(p) for p in pairs)
    paired = pd.DataFrame(rows, columns=PAIR_COLS).astype({"delta": float, "ci_low": float, "ci_high": float})
    paired["ci_excludes_0"] = (paired["ci_low"] > 0) | (paired["ci_high"] < 0)
    paired.to_csv(tables / "paired_comparisons.csv", index=False)

    agree = []
    for p in pairs:
        if p["subset"] != "all" or rankings[p["run_a"]] is None or rankings[p["run_b"]] is None:
            continue
        agree.append({k: p[k] for k in ("comparison", "label", "model", "fs", "imputation", "role", "node", "run_a", "run_b")}
                     | agreement(rankings[p["run_a"]], rankings[p["run_b"]], selected[p["run_a"]], selected[p["run_b"]]))
    agree = pd.DataFrame(agree)
    agree.to_csv(tables / "ranking_agreement.csv", index=False)
    long = [{"run_name": r, "rank": i + 1, "feature": f} for r, rk in rankings.items() if rk for i, f in enumerate(rk)]
    pd.DataFrame(long).to_csv(tables / "rankings.csv", index=False)
    # the ranking must not depend on the model: same role/node/fs/imputation -> identical ranking across models
    indep = []
    for key, g in runs[runs["fs"] != "nofs"].groupby(["role", "node", "fs", "imputation"]):
        rk = [rankings[r] for r in g["run_name"] if rankings[r]]
        if len(rk) >= 2:
            indep.append(dict(zip(["role", "node", "fs", "imputation"], key)) | {"n_models": len(rk), "identical_ranking": all(x == rk[0] for x in rk),
                         "identical_selected": all(set(selected[r]) == set(selected[g["run_name"].iloc[0]]) for r in g["run_name"])})
    indep = pd.DataFrame(indep)
    indep.to_csv(tables / "ranking_model_independence.csv", index=False)
    # SHAP importance (global model on the temporal cohort): agreement on the same pairs, and within a run the
    # agreement between the SHAP order of the selected features and their selection order
    srows = []
    for p in pairs:
        if p["subset"] != "all" or p["run_a"] not in shap_rank or p["run_b"] not in shap_rank:
            continue
        srows.append({k: p[k] for k in ("comparison", "label", "model", "fs", "imputation", "role", "node", "run_a", "run_b")}
                     | {("shap_jaccard_top10" if k == "jaccard_selected" else f"shap_{k}"): v
                        for k, v in agreement(shap_rank[p["run_a"]], shap_rank[p["run_b"]], shap_rank[p["run_a"]][:10], shap_rank[p["run_b"]][:10]).items()})
    shap_agree = pd.DataFrame(srows)
    shap_agree.to_csv(tables / "shap_agreement.csv", index=False)
    runs["shap_vs_selection_spearman"] = [agreement(shap_rank[r], [f for f in rankings[r] if f in set(shap_rank[r])], [], [])["spearman"]
                                          if r in shap_rank and rankings[r] else np.nan for r in runs["run_name"]]

    runs.drop(columns=["selected"]).to_csv(tables / "runs.csv", index=False)

    # --- derivation side: best-trial pivot and federated - central over the trials with identical hyper-parameters
    main = runs[runs["role"].isin(["central", "federated"])]
    best = main.pivot_table(index=["model", "fs", "imputation"], columns="role", values="derivation_weighted_c")
    if {"central", "federated"} <= set(best.columns):
        best["federated_minus_central"] = best["federated"] - best["central"]
    best.to_csv(tables / "derivation_best.csv")
    trows, metric = [], "metric__weighted_c-index_validation"
    for (model, fs, imp), g in main.groupby(["model", "fs", "imputation"]):
        tr = {}
        for r in g.itertuples(index=False):
            f = out / "runs" / r.run_name / "trials.csv"
            if f.exists():
                t = pd.read_csv(f)
                if metric in t.columns:
                    tr[r.role] = t
        if {"central", "federated"} <= set(tr):
            keys = sorted(k for k in set(tr["central"].columns) & set(tr["federated"].columns) if k.startswith("param__"))
            m = tr["central"][keys + [metric]].merge(tr["federated"][keys + [metric]], on=keys, suffixes=("_central", "_federated"))
            d = (m[f"{metric}_federated"] - m[f"{metric}_central"]).dropna()
            trows.append({"model": model, "fs": fs, "imputation": imp, "n_trials": int(len(tr["central"])), "n_trials_matched": int(len(d)),
                          "mean_delta": d.mean(), "sd_delta": d.std(ddof=1), "frac_federated_better": float((d > 0).mean()) if len(d) else np.nan,
                          "central_best": tr["central"][metric].max(), "federated_best": tr["federated"][metric].max()})
    trial_paired = pd.DataFrame(trows)
    trial_paired.to_csv(tables / "derivation_trial_paired.csv", index=False)

    # --- standalone matrix: node alone vs the federated and central models, on the node's area and on the whole cohort
    srows = []
    for s in runs[runs["role"] == "standalone"].itertuples(index=False):
        row = {"node": s.node, "model": s.model, "fs": s.fs, "imputation": s.imputation, "alone_own_area": getattr(s, f"temporal_c_{s.node}"),
               "alone_whole": s.temporal_harrell_c}
        for ref_role in ("federated", "central"):
            ref = runs[(runs["role"] == ref_role) & (runs["model"] == s.model) & (runs["fs"] == s.fs) & (runs["imputation"] == s.imputation)]
            if len(ref) == 1:
                row[f"{ref_role}_own_area"] = ref.iloc[0][f"temporal_c_{s.node}"]
                row[f"{ref_role}_whole"] = ref.iloc[0]["temporal_harrell_c"]
        srows.append(row)
    standalone = pd.DataFrame(srows)
    standalone.to_csv(tables / "standalone_matrix.csv", index=False)

    # --- report
    parts = ["# Comparison report\n", f"{len(runs)} runs scored; roles {runs.groupby('role').size().to_dict()}; paired bootstrap {args.bootstrap} resamples, seed {args.seed}.\n"]
    piv = main.pivot_table(index=["model", "imputation"], columns=["role", "fs"], values="temporal_harrell_c")
    parts.append("## Temporal Harrell C of the global models (whole cohort)\n\n" + md(piv.reset_index()) + "\n")
    if not best.empty:
        parts.append("## Derivation: best-trial weighted validation C\n\n" + md(best.reset_index()) + "\n")
    if not trial_paired.empty:
        parts.append("## Derivation: federated - central over the trials with identical hyper-parameters\n\n" + md(trial_paired) + "\n")
    if not paired.empty:
        summ = paired[paired["subset"] == "all"].groupby(["comparison", "label"]).agg(n_pairs=("delta", "size"), mean_delta=("delta", "mean"),
                                                                                      min_delta=("delta", "min"), max_delta=("delta", "max"),
                                                                                      n_ci_excludes_0=("ci_excludes_0", "sum")).reset_index()
        parts.append("## Paired comparisons of the temporal Harrell C (whole cohort): summary\n\n" + md(summ) + "\n")
        show = paired[paired["comparison"].isin(["arm", "fs", "imputation"])][["comparison", "label", "model", "fs", "imputation", "role", "delta", "ci_low", "ci_high"]]
        parts.append("## Paired comparisons, main grid\n\n" + md(show) + "\n")
    if not standalone.empty and not paired.empty:
        own = paired[(paired["comparison"] == "standalone_vs_federated") & (paired["subset"] != "all")]
        parts.append("## Node alone vs federated global on the node's own area\n\n" +
                     f"federated better in {(own['delta'] < 0).sum()}/{len(own)} pairs, CI excludes 0 in {own['ci_excludes_0'].sum()}; mean (alone - federated) {own['delta'].mean():+.3f}\n\n" +
                     md(standalone) + "\n")
    if not agree.empty:
        parts.append("## Feature-ranking agreement (same pairs)\n\n" + md(agree.drop(columns=["run_a", "run_b"])) + "\n")
    if not indep.empty:
        parts.append("## Ranking independence from the model\n\n" + md(indep) + "\n")
    if not shap_agree.empty:
        parts.append("## SHAP importance agreement (global models on the temporal cohort, same pairs)\n\n" +
                     md(shap_agree.drop(columns=["run_a", "run_b", "shap_top4_a", "shap_top4_b"])) + "\n")
    cal = [c for c in runs.columns if c.startswith("temporal_calib_")]
    if cal:
        parts.append("## Calibration of the global models (whole cohort)\n\n" + md(runs[["run_name", "temporal_harrell_c"] + cal]) + "\n")
    (out / "report.md").write_text("\n".join(parts), encoding="utf-8")
    json.dump({"args": vars(args), "n_runs": int(len(runs)), "n_pairs": len(pairs)}, open(tables / "compare_info.json", "w"), indent=2)
    LOGGER.info("written %s", out)


if __name__ == "__main__":
    main()
