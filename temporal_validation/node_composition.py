"""Why does the italy_north standalone model transfer as well as the federated ensemble?

Part 1 (descriptive, derivation v2 nodes vs the temporal cohort): size, events, follow-up, event timing,
number and concentration of centres, case-mix, missingness, standardized distance to the temporal cohort.

Part 2 (targeted fits, RSF, scored on the 698 temporal patients with the pipeline's metric code):
  A. every node with the SAME fixed hyper-parameters (central round-2 best) -> nodes on an equal footing,
     separates "own tuning" from "own data";
  B. italy_north subsampled to the size of the other nodes (130) -> composition vs size;
  C. random pooled subsets of 288 matched on 166 events -> is it the number of events?
  D. random subsets of 288 from the Italian nodes only (north + centre-south) -> is it Italian case-mix?
  E. random subsets of 288 from the four nodes other than italy_north -> pooled data without italy_north;
  F. italy_north restricted to its centres with >= 20 patients / to its smaller centres -> within-node mix.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learning_curve import FEATURES, fit_rsf  # noqa: E402
from score_temporal import eval_step_functions, metrics_block, structured  # noqa: E402

LOGGER = logging.getLogger("node_composition")
NODES = ["americas", "asia", "europe_non_italy", "italy_centre_south", "italy_north"]
CENTRAL_PARAMS = {"n_estimators": 230, "max_depth": 9, "min_samples_split": 3, "min_samples_leaf": 8, "max_features": "sqrt"}


def km_median(t, e):
    order = np.argsort(t)
    t, e = t[order], e[order]
    s, n = 1.0, len(t)
    for i in range(n):
        if e[i]:
            s *= 1 - 1 / (n - i)
            if s <= 0.5:
                return float(t[i])
    return float("nan")


def describe(name, df, temporal=None):
    t = df["tte"].to_numpy(float)
    e = df["event"].to_numpy(int)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    row = {"group": name, "n": len(df), "events": int(e.sum()), "event_rate": e.mean(), "median_observed_tte": np.median(t),
           "km_median_os": km_median(t, e), "max_tte": t.max(), "events_within_6m_share": float(((t <= 6) & (e == 1)).sum() / max(e.sum(), 1)),
           "missing_cells": float(X.isna().mean().mean()),
           "ecog1_share": float((X["ps ecog"] == 1).mean()), "metastatic_share": float((X["locally advanced/metastatic"] == 2).mean()),
           "surgery_share": float((X["surgery"] == 1).mean()), "intrahepatic_share": float((X["location - Intrahepatic"] == 1).mean()),
           "ca199_median": float(X["ca199"].median()), "albumin_median": float(X["albumin"].median()), "nlr_median": float(X["nlr"].median()),
           "age_median": float(X["age"].median())}
    if "centre" in df.columns:
        p = df["centre"].value_counts(normalize=True)
        row.update({"centres": int(p.size), "effective_centres": float(1 / (p ** 2).sum()), "largest_centre_share": float(p.iloc[0]),
                    "largest_centre": p.index[0]})
    if temporal is not None:
        Xt = temporal[FEATURES].astype(float)
        smd = []
        for c in FEATURES:
            a, b = X[c].dropna(), Xt[c].dropna()
            sd = np.sqrt((a.var() + b.var()) / 2)
            if sd > 0:
                smd.append(abs(a.mean() - b.mean()) / sd)
        row["mean_abs_smd_vs_temporal"] = float(np.mean(smd))
        row["share_smd_gt_0.2"] = float(np.mean(np.array(smd) > 0.2))
    return row


def score(train, coh, params, seed, y_deriv, max_train, grid):
    X = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = structured(train["event"].astype(int).to_numpy(), train["tte"].to_numpy(float))
    model, means = fit_rsf(X, y, params, seed)
    Xt = coh[FEATURES].astype(float).fillna(means)
    risk = model.predict(Xt)
    surv = eval_step_functions(model.predict_survival_function(Xt), grid)
    m = metrics_block(y_deriv, max_train, coh["event"].astype(int).to_numpy(), coh["tte"].to_numpy(float), risk, surv, grid, [6.0, 12.0], 24.0, 12.0, "evaluated")
    per_area = {}
    for a in NODES:
        mask = (coh["area_federata"] == a).to_numpy()
        from sksurv.metrics import concordance_index_censored
        per_area[a] = float(concordance_index_censored(coh["event"].to_numpy(bool)[mask], coh["tte"].to_numpy(float)[mask], risk[mask])[0])
    return m["harrell_c"], m["uno_c"], per_area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="data_survival folder (v2 nodes + central)")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--anagrafica", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--draws", type=int, default=10)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ana = pd.read_csv(args.anagrafica, sep=";", dtype=str)
    ana = ana[(ana["coorte"] == "training") & (ana["stato"] == "incluso")]
    centre_of = dict(zip(ana["patient_number"].astype(int), ana["sigla"]))
    frames = []
    for n in NODES:
        d = pd.read_csv(Path(args.data_dir) / n / "original" / "data.csv")
        d["node"] = n
        d["pn"] = d["id"].astype(float).astype(int)
        d["centre"] = d["pn"].map(centre_of)
        frames.append(d)
    pooled = pd.concat(frames, ignore_index=True)
    if pooled["centre"].isna().any():
        raise SystemExit(f"{pooled['centre'].isna().sum()} derivation patients without a centre")
    coh = pd.read_csv(args.cohort)
    y_deriv = structured(pooled["event"].astype(int).to_numpy(), pooled["tte"].to_numpy(float))
    max_train = float(pooled["tte"].max())
    grid = np.arange(0.5, 24.0 + 1e-9, 0.5)

    # ---- part 1: description
    rows = [describe(n, pooled[pooled["node"] == n], coh) for n in NODES]
    rows.append(describe("pooled_732", pooled, coh))
    tc = coh.rename(columns={"sigla": "centre"})
    rows.append(describe("temporal_698", tc))
    desc = pd.DataFrame(rows)
    desc.to_csv(out / "node_description.csv", index=False)
    pd.set_option("display.width", 250)
    LOGGER.info("\n%s", desc.round(3).T.to_string())
    cen = pooled.groupby(["node", "centre"]).agg(n=("event", "size"), events=("event", "sum"), median_tte=("tte", "median")).reset_index()
    cen.to_csv(out / "node_centres.csv", index=False)

    # ---- part 2: targeted fits
    fits = []

    def add(exp, label, train, seed=42, params=CENTRAL_PARAMS):
        c, u, pa = score(train, coh, params, seed, y_deriv, max_train, grid)
        fits.append({"experiment": exp, "label": label, "n_train": len(train), "events_train": int(train["event"].sum()),
                     "harrell_c": c, "uno_c": u, **{f"c_{a}": v for a, v in pa.items()}})
        LOGGER.info("%-28s %-34s n=%3d ev=%3d C=%.4f", exp, label, len(train), int(train["event"].sum()), c)

    north = pooled[pooled["node"] == "italy_north"]
    for n in NODES:  # A
        add("A_node_fixed_params", n, pooled[pooled["node"] == n])
    for r in range(args.draws):  # B
        add("B_north_subsampled_130", f"draw{r}", north.sample(n=130, random_state=2000 + r))
    ev = pooled[pooled["event"] == 1]
    ce = pooled[pooled["event"] == 0]
    for r in range(args.draws):  # C
        tr = pd.concat([ev.sample(n=166, random_state=3000 + r), ce.sample(n=288 - 166, random_state=3100 + r)])
        add("C_random288_166events", f"draw{r}", tr)
    italian = pooled[pooled["node"].isin(["italy_north", "italy_centre_south"])]
    for r in range(args.draws):  # D
        add("D_random288_italian_only", f"draw{r}", italian.sample(n=288, random_state=4000 + r))
    others = pooled[pooled["node"] != "italy_north"]
    for r in range(args.draws):  # E
        add("E_random288_without_north", f"draw{r}", others.sample(n=288, random_state=5000 + r))
    big = north.groupby("centre")["event"].transform("size") >= 20  # F
    add("F_north_large_centres", f"centres>=20 ({north[big]['centre'].nunique()})", north[big])
    add("F_north_small_centres", f"centres<20 ({north[~big]['centre'].nunique()})", north[~big])
    for r in range(args.draws):
        add("F_random_same_size_as_north_large", f"draw{r}", pooled.sample(n=int(big.sum()), random_state=6000 + r))

    fdf = pd.DataFrame(fits)
    fdf.to_csv(out / "node_composition_fits.csv", index=False)
    summ = fdf.groupby("experiment").agg(fits=("harrell_c", "size"), n_train=("n_train", "mean"), events=("events_train", "mean"),
                                         c_mean=("harrell_c", "mean"), c_sd=("harrell_c", "std"), c_min=("harrell_c", "min"), c_max=("harrell_c", "max"),
                                         uno_mean=("uno_c", "mean")).reset_index()
    summ.to_csv(out / "node_composition_summary.csv", index=False)
    LOGGER.info("\n%s", summ.round(4).to_string(index=False))
    LOGGER.info("\n%s", fdf[fdf["experiment"] == "A_node_fixed_params"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
