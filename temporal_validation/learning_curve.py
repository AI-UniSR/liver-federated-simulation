"""Learning curve of the pooled (central) model on the temporal cohort.

Question: is the temporal parity between the federated ensemble (0.736), the pooled model on 732 patients
(0.734) and the italy_north standalone model on 288 patients (0.735) a saturation of discrimination with
training size, or does the pooled model keep improving with more patients?

Design: random subsets of the pooled derivation data (data asset federated-learning-data-folder:2,
central/original/data.csv) of size --sizes, --repeats draws per size (different subset seeds), each fitted with
the framework's basic preprocessing replicated here (mean imputation of the 33 numeric features computed on the
subset; scaling omitted because tree ensembles are invariant to it) and the hyper-parameters selected by the
central round-2 run (fixed, not re-tuned per size: stated as a limitation). Every model is scored on the same
prepared temporal cohort with the metric code and conventions of score_temporal.py.

Anchor: the full-size fit must reproduce the temporal Harrell C of the logged central model (0.734 for
central_rsf_nofs_basic); --anchor-c checks it.

Outputs (--output-dir): learning_curve_runs.csv (one row per fit), learning_curve_summary.csv (per size),
learning_curve.png (mean +- SD and range per size, with the standalone node models at their own size and the
federated / central reference lines when --reference-csv is given).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_temporal import eval_step_functions, metrics_block, structured  # noqa: E402

LOGGER = logging.getLogger("learning_curve")
FEATURES = [
    "sex", "age", "viral", "surgery", "stent or biliary draige", "lung mets", "lymph mets", "peri mets",
    "liver mets", "bone mets", "other mets", "number of met sites", "locally advanced/metastatic", "bmi",
    "ps ecog", "ca199", "cea", "neutrophils", "nlr", "lymphocytes", "eosinophils", "basophils", "monocytes",
    "albumin", "albumin bin", "platelets", "bilirubin", "bilirubin bin", "creatinine", "astGOT", "altGPT",
    "ggt", "location - Intrahepatic",
]


def fit_rsf(X: pd.DataFrame, y, params: dict, seed: int):
    from sksurv.ensemble import RandomSurvivalForest
    means = X.mean(skipna=True)
    Xi = X.fillna(means)
    model = RandomSurvivalForest(**params, random_state=seed, n_jobs=1).fit(Xi, y)
    return model, means


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derivation-csv", required=True, help="central/original/data.csv (732 patients)")
    ap.add_argument("--cohort", required=True, help="prepared temporal cohort (temporal_cohort_prepared.csv)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[75, 150, 300, 450, 585, 732])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--params", default='{"n_estimators": 230, "max_depth": 9, "min_samples_split": 3, "min_samples_leaf": 8, "max_features": "sqrt"}',
                    help="RSF hyper-parameters (JSON); default = central_rsf_nofs_basic round 2 best trial")
    ap.add_argument("--model-seed", type=int, default=42, help="RSF random_state (framework seed)")
    ap.add_argument("--anchor-c", type=float, default=0.7343, help="temporal Harrell C of the logged central model")
    ap.add_argument("--reference-csv", default=None, help="optional CSV label,n,harrell_c,kind for the figure")
    ap.add_argument("--horizons", nargs="*", type=float, default=[6.0, 12.0])
    ap.add_argument("--ibs-horizon", type=float, default=12.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads(args.params)

    der = pd.read_csv(args.derivation_csv)
    Xd = der[FEATURES].apply(pd.to_numeric, errors="coerce")
    ev_d = pd.to_numeric(der["event"]).astype(int).to_numpy()
    t_d = pd.to_numeric(der["tte"]).to_numpy(float)
    coh = pd.read_csv(args.cohort)
    Xt = coh[FEATURES].astype(float)
    ev_t = coh["event"].astype(int).to_numpy()
    t_t = coh["tte"].to_numpy(float)
    y_deriv = structured(ev_d, t_d)
    max_train = float(t_d.max())
    grid = np.arange(0.5, 24.0 + 1e-9, 0.5)
    LOGGER.info("derivation %d patients (%d events), temporal %d (%d events); params %s",
                len(der), ev_d.sum(), len(coh), ev_t.sum(), params)

    rows = []
    for size in sorted(set(args.sizes)):
        if size > len(der):
            continue
        reps = 1 if size == len(der) else args.repeats
        for rep in range(reps):
            rng = np.random.default_rng(1000 + rep)
            idx = np.arange(len(der)) if size == len(der) else np.sort(rng.choice(len(der), size=size, replace=False))
            y = structured(ev_d[idx], t_d[idx])
            model, means = fit_rsf(Xd.iloc[idx], y, params, args.model_seed)
            Xti = Xt.fillna(means)
            risk = model.predict(Xti)
            surv = eval_step_functions(model.predict_survival_function(Xti), grid)
            m = metrics_block(y_deriv, max_train, ev_t, t_t, risk, surv, grid, args.horizons, 24.0, args.ibs_horizon, "evaluated")
            row = {"n_train": size, "repeat": rep, "events_train": int(ev_d[idx].sum()), **{k: m[k] for k in m if k not in ("n", "events")}}
            rows.append(row)
            LOGGER.info("n=%3d rep=%d events=%3d  Harrell C %.4f  Uno C %.4f  IBS12 %.4f", size, rep, row["events_train"],
                        m["harrell_c"], m["uno_c"], m[f"ibs_{args.ibs_horizon:g}"])
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "learning_curve_runs.csv", index=False)
    agg = runs.groupby("n_train").agg(n_fits=("harrell_c", "size"), events_mean=("events_train", "mean"),
                                       harrell_c_mean=("harrell_c", "mean"), harrell_c_sd=("harrell_c", "std"),
                                       harrell_c_min=("harrell_c", "min"), harrell_c_max=("harrell_c", "max"),
                                       uno_c_mean=("uno_c", "mean"), ibs_mean=(f"ibs_{args.ibs_horizon:g}", "mean")).reset_index()
    agg.to_csv(out / "learning_curve_summary.csv", index=False)
    LOGGER.info("\n%s", agg.round(4).to_string(index=False))

    full = runs[runs["n_train"] == len(der)]
    if len(full):
        diff = float(full["harrell_c"].iloc[0]) - args.anchor_c
        LOGGER.info("anchor: full-size fit Harrell C %.4f vs logged central model %.4f (diff %+.4f)",
                    float(full["harrell_c"].iloc[0]), args.anchor_c, diff)
        json.dump({"anchor_logged": args.anchor_c, "anchor_refit": float(full["harrell_c"].iloc[0]), "diff": diff},
                  open(out / "anchor_check.json", "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.fill_between(agg["n_train"], agg["harrell_c_min"], agg["harrell_c_max"], color="C0", alpha=0.15, label="pooled model, range over subsets")
        ax.errorbar(agg["n_train"], agg["harrell_c_mean"], yerr=agg["harrell_c_sd"].fillna(0), fmt="o-", color="C0", capsize=3,
                    label="pooled model, random subsets (mean +- SD)")
        if args.reference_csv and os.path.exists(args.reference_csv):
            ref = pd.read_csv(args.reference_csv)
            nodes = ref[ref["kind"] == "standalone"]
            ax.scatter(nodes["n"], nodes["harrell_c"], marker="s", color="C1", zorder=3, label="standalone node models (own tuning)")
            for _, r in nodes.iterrows():
                ax.annotate(r["label"], (r["n"], r["harrell_c"]), textcoords="offset points", xytext=(4, -10), fontsize=7)
            for _, r in ref[ref["kind"] == "line"].iterrows():
                ax.axhline(r["harrell_c"], ls="--", lw=0.9, color="grey")
                ax.text(ax.get_xlim()[0] if False else 20, r["harrell_c"] + 0.002, r["label"], fontsize=7, color="grey")
        ax.set_xlabel("training patients (pooled derivation data)")
        ax.set_ylabel("Harrell C on the temporal cohort (698 patients)")
        ax.set_title("Learning curve of the pooled RSF (fixed hyper-parameters)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "learning_curve.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("figure failed: %s", e)


if __name__ == "__main__":
    main()
