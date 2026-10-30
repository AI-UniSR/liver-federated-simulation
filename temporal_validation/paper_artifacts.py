"""Paper display items from the validation pipeline output (docs/PAPER_STRUCTURE.md, section B).

Reads the final folder written by compare_runs.py (summary_runs.csv, tables/, runs/<run>/) plus the node data folder
and the centre table, and writes under <final>/paper/:
  main/            Table1_cohorts, Table2_paired_differences, Fig2_discrimination, Fig3_feature_selection,
                   Fig4_calibration, Fig5_shap (a/b = framework beeswarms as logged, c = importance agreement)
  supplementary/   TableS2_search_spaces, TableS3_centres, TableS4_table1_full, TableS5_derivation, TableS6_temporal,
                   TableS7_paired_all, TableS8_single_centre, TableS9_rankings, TableS10_mutual_information,
                   TableS11_trials_matched, FigS1_per_area, FigS2_discrimination_mice, FigS3_calibration_curves,
                   FigS4_kaplan_meier, FigS5_shap_beeswarms, FigS7_missingness
  captions.md      one caption per item; numbers.json: the numbers quoted in the text.
Tables: .docx (caption + table, python-docx) and .csv; figures: .png (300 dpi), .svg and .pdf.
Main text uses basic imputation and the selections none (33 features) vs score test (10); MI only in Table S10.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("paper_artifacts")
AREAS = ["americas", "asia", "europe_non_italy", "italy_centre_south", "italy_north"]
AREA_LABEL = {"americas": "Americas", "asia": "Asia", "europe_non_italy": "Europe outside Italy",
              "italy_centre_south": "Italy centre-south", "italy_north": "Italy north"}
MODEL_LABEL = {"cox": "Cox", "gradient": "Gradient boosting", "rsf": "Random survival forest"}
FS_LABEL = {"nofs": "33 features", "score": "10 features (score test)", "mi": "10 features (mutual information)"}
ARM_COLOR = {"central": "#1f4e79", "federated": "#c0504d", "standalone": "#8c8c8c"}
# Okabe-Ito colours for the five areas (colour-blind safe), used wherever areas are colour-coded
AREA_COLOR = {"americas": "#E69F00", "asia": "#56B4E9", "europe_non_italy": "#009E73", "italy_centre_south": "#D55E00",
              "italy_north": "#000000"}
MAIN_MODEL = ("cox", "score", "basic")  # recommended parsimonious model
# display names of the registry variables (paper vocabulary)
FEATURE_LABEL = {
    "sex": "Sex (male)", "age": "Age", "viral": "Viral hepatitis", "surgery": "Previous surgery",
    "stent or biliary draige": "Stent or biliary drainage", "lung mets": "Lung metastases", "lymph mets": "Lymph node metastases",
    "peri mets": "Peritoneal metastases", "liver mets": "Liver metastases", "bone mets": "Bone metastases", "other mets": "Other metastases",
    "number of met sites": "Number of metastatic sites", "locally advanced/metastatic": "Metastatic disease", "bmi": "BMI",
    "ps ecog": "ECOG PS ≥ 1", "ca199": "CA19-9", "cea": "CEA", "neutrophils": "Neutrophils", "nlr": "NLR", "lymphocytes": "Lymphocytes",
    "eosinophils": "Eosinophils", "basophils": "Basophils", "monocytes": "Monocytes", "albumin": "Albumin", "albumin bin": "Albumin (low)",
    "platelets": "Platelets", "bilirubin": "Bilirubin", "bilirubin bin": "Bilirubin (high)", "creatinine": "Creatinine", "astGOT": "AST",
    "altGPT": "ALT", "ggt": "GGT", "location - Intrahepatic": "Intrahepatic location",
}


def flabel(name) -> str:
    return FEATURE_LABEL.get(str(name), str(name))


# ----------------------------------------------------------------------------- output helpers
class Writer:
    def __init__(self, root: Path):
        self.root = root
        (root / "main").mkdir(parents=True, exist_ok=True)
        (root / "supplementary").mkdir(parents=True, exist_ok=True)
        self.captions = []
        try:
            import docx  # noqa: F401
            self.docx = True
        except ImportError:
            self.docx = False
            LOGGER.warning("python-docx not installed: tables as csv and md only")

    def table(self, df: pd.DataFrame, name: str, caption: str, where="main", decimals=3, index=False):
        out = self.root / where / name
        d = df.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].round(decimals)
        d.to_csv(out.with_suffix(".csv"), index=index)
        self.captions.append(f"**{name.replace('_', ' ')}.** {caption}")
        if self.docx:
            from docx import Document
            from docx.shared import Pt
            doc = Document()
            style = doc.styles["Normal"]; style.font.name = "Arial"; style.font.size = Pt(8)
            doc.add_paragraph(caption)
            dd = d.reset_index() if index else d
            t = doc.add_table(rows=1, cols=len(dd.columns)); t.style = "Table Grid"
            for j, c in enumerate(dd.columns):
                t.rows[0].cells[j].text = str(c)
            for _, r in dd.iterrows():
                cells = t.add_row().cells
                for j, v in enumerate(r.tolist()):
                    cells[j].text = "" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.{decimals}f}" if isinstance(v, float) else str(v))
            doc.save(out.with_suffix(".docx"))

    def figure(self, fig, name: str, caption: str, where="main"):
        out = self.root / where / name
        try:
            from paper_style import npj_panels
            npj_panels(fig)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("figure style not applied to %s: %s", name, e)
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        self.captions.append(f"**{name.replace('_', ' ')}.** {caption}")

    def finish(self, numbers: dict):
        (self.root / "captions.md").write_text("# Captions\n\n" + "\n\n".join(self.captions) + "\n", encoding="utf-8")
        json.dump(numbers, open(self.root / "numbers.json", "w"), indent=2, default=float)


def fmt_ci(d, lo, hi):
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


# ----------------------------------------------------------------------------- cohorts
def km_median(time, event):
    from sksurv.nonparametric import kaplan_meier_estimator
    x, y = kaplan_meier_estimator(np.asarray(event, bool), np.asarray(time, float))
    below = np.flatnonzero(y <= 0.5)
    return float(x[below[0]]) if len(below) else np.nan


def cohort_stats(df: pd.DataFrame) -> dict:
    """One column of Table 1: counts, follow-up and the key characteristics (median (Q1-Q3) or n (%))."""
    x = df.apply(pd.to_numeric, errors="coerce")
    n = len(x)

    def med(col):
        v = x[col].dropna()
        return f"{v.median():.1f} ({v.quantile(.25):.1f}-{v.quantile(.75):.1f})" if len(v) else ""

    def pct(col, value=1):
        v = x[col].dropna()
        return f"{int((v == value).sum())} ({100 * (v == value).mean():.1f}%)" if len(v) else ""
    period = ""
    if "treatment_start_month" in df.columns and df["treatment_start_month"].notna().any():
        p = df["treatment_start_month"].dropna().astype(str)
        period = f"{p.min()} to {p.max()}"
    n_pred = len([c for c in FEATURES if c in x.columns])
    s = {"Patients": str(n), "Deaths": f"{int(x['event'].sum())} ({100 * x['event'].mean():.1f}%)",
         "Events per predictor": f"{x['event'].sum() / n_pred:.1f}" if n_pred else "",
         "Treatment start, first to last (year-month)": period,
         "Median observed follow-up, months": f"{x['tte'].median():.1f}", "Median OS (Kaplan-Meier), months": f"{km_median(x['tte'], x['event']):.1f}",
         "Age, years": med("age"), "Male sex": pct("sex", 1), "ECOG >= 1": pct("ps ecog", 1),
         "Metastatic disease": pct("locally advanced/metastatic", 2),
         "Intrahepatic location": pct("location - Intrahepatic", 1), "Previous surgery": pct("surgery", 1),
         "Number of metastatic sites": med("number of met sites"), "CA19-9, U/mL": med("ca199"), "CEA, ng/mL": med("cea"),
         "Albumin, g/dL": med("albumin"), "Neutrophils, x10^9/L": med("neutrophils"), "NLR": med("nlr"),
         "Missing values, % of predictor cells": f"{100 * x[[c for c in FEATURES if c in x.columns]].isna().mean().mean():.1f}"}
    return s


FEATURES = ["sex", "age", "viral", "surgery", "stent or biliary draige", "lung mets", "lymph mets", "peri mets", "liver mets", "bone mets",
            "other mets", "number of met sites", "locally advanced/metastatic", "bmi", "ps ecog", "ca199", "cea", "neutrophils", "nlr",
            "lymphocytes", "eosinophils", "basophils", "monocytes", "albumin", "albumin bin", "platelets", "bilirubin", "bilirubin bin",
            "creatinine", "astGOT", "altGPT", "ggt", "location - Intrahepatic"]


def full_table1_row(df: pd.DataFrame) -> dict:
    x = df.apply(pd.to_numeric, errors="coerce")
    out = {}
    coded = {"sex": 1, "locally advanced/metastatic": 2}  # n (%) of the code: male; metastatic
    for c in FEATURES:
        if c not in x.columns:
            continue
        v = x[c].dropna()
        if len(v) and c in coded:
            out[flabel(c)] = f"{int((v == coded[c]).sum())} ({100 * (v == coded[c]).mean():.1f}%); missing {x[c].isna().sum()}"
        elif len(v) and set(v.unique()) <= {0, 1}:
            out[flabel(c)] = f"{int(v.sum())} ({100 * v.mean():.1f}%); missing {x[c].isna().sum()}"
        elif len(v):
            out[flabel(c)] = f"{v.median():.2f} ({v.quantile(.25):.2f}-{v.quantile(.75):.2f}); missing {x[c].isna().sum()}"
    return out


def patient_flow(final: Path, centres_csv) -> dict:
    """Numbers of the patient flow (TRIPOD-Cluster 10a): validation side from the prepare step's summary,
    derivation side from the patient register next to the centre table when available (restricted file)."""
    flow = {}
    s = final / "cohort" / "temporal_cohort_prepared.summary.json"
    if s.exists():
        try:
            j = json.load(open(s))
            sc = j.get("status_counts", {})
            flow.update(validation_rows=int(j.get("n_validation_rows", 0)), validation_usable=int(j.get("n_usable", 0)),
                        validation_duplicates=int(sc.get("duplicate_of_training", 0)),
                        validation_other_excluded=int(sum(v for k, v in sc.items() if k not in ("ok", "duplicate_of_training"))))
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("flow: prepare summary not readable: %s", e)
    reg = Path(centres_csv).with_name("anagrafica_pazienti_centri_aree.csv") if centres_csv else None
    if reg is not None and reg.exists():
        try:
            ana = pd.read_csv(reg, sep=";", dtype=str)
            tr = ana[ana["coorte"] == "training"]
            flow.update(derivation_rows=int(len(tr)), derivation_included=int((tr["stato"] == "incluso").sum()),
                        derivation_excluded=int((tr["stato"] != "incluso").sum()))
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("flow: patient register not readable: %s", e)
    return flow


def load_nodes(data_folder: Path) -> dict:
    nodes = {}
    for a in AREAS + ["central"]:
        f = data_folder / a / "original" / "data.csv"
        if f.exists():
            nodes[a] = pd.read_csv(f, dtype=str)
    return nodes


def cohorts_tables(w: Writer, nodes: dict, cohort: pd.DataFrame, centres: pd.DataFrame | None, numbers: dict):
    cols = {}
    for a in AREAS:
        if a in nodes:
            cols[f"Derivation: {AREA_LABEL[a]}"] = cohort_stats(nodes[a])
    if "central" in nodes:
        cols["Derivation: pooled"] = cohort_stats(nodes["central"])
    for a in AREAS:  # temporal cohort per area (per-cluster characteristics of the validation data, TRIPOD-Cluster 10b)
        sub = cohort[cohort["area_federata"] == a]
        if len(sub):
            cols[f"Temporal: {AREA_LABEL[a]}"] = cohort_stats(sub)
    cols["Temporal: all"] = cohort_stats(cohort)
    t1 = pd.DataFrame(cols)
    if centres is not None:
        row_c, row_k = {}, {}
        for prefix, ncol in (("Derivation", "n_derivazione"), ("Temporal", "n_validazione")):
            for a in AREAS:
                sub = centres[centres["area_federata"] == a]
                key = f"{prefix}: {AREA_LABEL[a]}"
                if key in cols:
                    row_c[key] = str(int((sub[ncol].astype(int) > 0).sum()))
                    row_k[key] = str(sub.loc[sub[ncol].astype(int) > 0, "paese"].nunique())
            key = "Derivation: pooled" if prefix == "Derivation" else "Temporal: all"
            row_c[key] = str(int((centres[ncol].astype(int) > 0).sum()))
            row_k[key] = str(centres.loc[centres[ncol].astype(int) > 0, "paese"].nunique())
        t1 = pd.concat([pd.DataFrame({"Centres": row_c, "Countries": row_k}).T, t1])
    t1.index.name = "Characteristic"
    w.table(t1, "Table1_cohorts", "Table 1. Derivation cohort by federated node, pooled derivation cohort and temporal validation cohort. "
            "Continuous variables as median (Q1-Q3), categorical as n (%). Missing values are the share of empty cells over the 33 predictors. "
            "The five nodes are the geographic areas of the registry used as federation units.", index=True)
    numbers["table1"] = t1.to_dict()
    # temporal cohort by area, full predictor list (Table S4)
    rows = {}
    for a in AREAS:
        if a in nodes:
            rows[f"Derivation: {AREA_LABEL[a]}"] = full_table1_row(nodes[a])
    if "central" in nodes:
        rows["Derivation: pooled"] = full_table1_row(nodes["central"])
    for a in AREAS:
        sub = cohort[cohort["area_federata"] == a]
        if len(sub):
            rows[f"Temporal: {AREA_LABEL[a]}"] = full_table1_row(sub)
    rows["Temporal: all"] = full_table1_row(cohort)
    s4 = pd.DataFrame(rows); s4.index.name = "Predictor"
    w.table(s4, "TableS4_table1_full", "Table S4. All 33 predictors in the derivation nodes, the pooled derivation cohort and the temporal "
            "validation cohort by area: median (Q1-Q3) for continuous variables, n (%) for binary ones, with the number of missing values.",
            where="supplementary", index=True)
    if centres is not None:
        s3 = centres[["nome_completo", "citta", "paese", "area_federata", "n_derivazione", "n_validazione"]].copy()
        s3.columns = ["Centre", "City", "Country", "Node (area)", "Derivation patients", "Temporal validation patients"]
        s3["Node (area)"] = s3["Node (area)"].map(AREA_LABEL)
        s3 = s3.sort_values(["Node (area)", "Derivation patients"], ascending=[True, False])
        w.table(s3, "TableS3_centres", "Table S3. Participating centres by federated node, with the number of patients contributed to the "
                "derivation cohort and to the temporal validation cohort.", where="supplementary", decimals=0)


# ----------------------------------------------------------------------------- figures
def fig1_design(w: Writer, nodes: dict, cohort: pd.DataFrame, centres: pd.DataFrame | None, numbers: dict):
    """Framework and study design: (a) pipeline stages, (b) the five nodes and the three arms, (c) the evaluation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.9], width_ratios=[1.15, 1])
    # (a) pipeline
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    stages = ["Data quality", "Federated\nimputation", "Federated\nfeature selection", "Federated tuning\n(Optuna, 50 trials)",
              "Node models\n+ weighted ensemble", "Cross-site\nevaluation", "Privacy-preserving\nSHAP", "MLflow\nlogging"]
    xs = np.linspace(6, 94, len(stages))
    for i, (x, s) in enumerate(zip(xs, stages)):
        ax.add_patch(FancyBboxPatch((x - 5.6, 3), 11.2, 4.4, boxstyle="round,pad=0.3", fc="#eaf0f7", ec="#1f4e79", lw=1))
        ax.text(x, 5.2, s, ha="center", va="center", fontsize=8)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 5.9, 5.2), (xs[i + 1] - 5.9, 5.2), arrowstyle="-|>", mutation_scale=10, color="#1f4e79", lw=1))
    ax.text(0.5, 9.3, "(a) The federated pipeline (NVFlare; each node keeps its patients, the server sees aggregates only)", fontsize=10, ha="left")
    # (b) nodes and arms
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.text(0.5, 97, "(b) One registry, five geographic nodes, three arms", fontsize=10)
    sizes = {a: len(nodes[a]) for a in AREAS if a in nodes}
    events = {a: int(pd.to_numeric(nodes[a]["event"]).sum()) for a in AREAS if a in nodes}
    ncent = {}
    if centres is not None:
        for a in AREAS:
            sub = centres[(centres["area_federata"] == a) & (centres["n_derivazione"].astype(int) > 0)]
            ncent[a] = (len(sub), sub["paese"].nunique())
    pos = {"americas": (14, 70), "asia": (86, 70), "europe_non_italy": (14, 28), "italy_centre_south": (86, 28), "italy_north": (50, 14)}
    ax.add_patch(Circle((50, 55), 7, fc="#c0504d", ec="none", alpha=0.9)); ax.text(50, 55, "server", ha="center", va="center", color="white", fontsize=9, weight="bold")
    for a in AREAS:
        if a not in sizes:
            continue
        x, y = pos[a]; r = 3 + 9 * np.sqrt(sizes[a] / max(sizes.values()))
        ax.add_patch(Circle((x, y), r, fc="#eaf0f7", ec="#1f4e79", lw=1.2))
        ax.add_patch(FancyArrowPatch((x, y), (50, 55), arrowstyle="<|-|>", mutation_scale=9, color="#999999", lw=0.8, shrinkA=r * 3.2, shrinkB=24))
        lab = f"{AREA_LABEL[a]}\n{sizes[a]} patients, {events[a]} deaths"
        if a in ncent:
            lab += f"\n{ncent[a][0]} centres, {ncent[a][1]} countr{'y' if ncent[a][1] == 1 else 'ies'}"
        ax.text(x, y - r - 2.5, lab, ha="center", va="top", fontsize=7.5)
    total = sum(sizes.values()); tev = sum(events.values())
    flow = numbers.get("flow", {})
    deriv_flow = (f"Development cohort of the registry: {flow['derivation_rows']} patients, {flow['derivation_excluded']} excluded "
                  f"(centre not assignable to a node)\n" if flow.get("derivation_rows") else "")
    ax.text(50, 84, f"{deriv_flow}Pooled arm: all {total} patients, {tev} deaths, in one place\nFederated arm: the five nodes + server (ensemble of node models)\n"
            "Single-centre arm: each node runs the pipeline alone", ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=0.8))
    # (c) evaluation
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(0.5, 97, "(c) Evaluation on a later registry freeze", fontsize=10)
    per_area = cohort.groupby("area_federata").agg(n=("patient_number", "size"), deaths=("event", "sum"))
    tflow = ""
    if flow.get("validation_rows"):
        tflow = (f"\n{flow['validation_rows']} records in the validation export; removed {flow.get('validation_duplicates', 0)} re-entries\n"
                 f"of development patients and {flow.get('validation_other_excluded', 0)} with unusable follow-up")
    ax.add_patch(FancyBboxPatch((8, 54 if tflow else 58), 84, 30 if tflow else 26, boxstyle="round,pad=0.5", fc="#fbf1f0", ec="#c0504d", lw=1))
    ax.text(50, 76 if tflow else 78, f"Temporal validation cohort: {len(cohort)} patients, {int(cohort['event'].sum())} deaths\n"
            f"later data freeze, no patient in common with derivation{tflow}", ha="center", va="center", fontsize=8 if tflow else 8.5)
    xs = np.linspace(16, 84, len(AREAS))
    for x, a in zip(xs, AREAS):
        if a in per_area.index:
            ax.text(x, 63, f"{AREA_LABEL[a]}\n{int(per_area.loc[a, 'n'])} / {int(per_area.loc[a, 'deaths'])}", ha="center", va="center", fontsize=7)
    items = ["every final model of the three arms scored on the same 698 patients",
             "Harrell C (95% bootstrap CI), Uno C, time-dependent AUC, integrated Brier score",
             "calibration-in-the-large and slope at 6, 12, 18 months",
             "paired bootstrap differences: federated vs pooled, 10 vs 33 features, node alone vs federated (own area, whole cohort)",
             "feature-ranking agreement pooled / federated / nodes; SHAP agreement"]
    for i, t in enumerate(items):
        ax.text(6, 48 - 9 * i, "â¢ " + t, fontsize=8, va="center", wrap=True)
    fig.tight_layout()
    w.figure(fig, "Fig1_framework_and_design", "Framework and study design. The patient flow of both cohorts is printed in panels b and c. (a) The stages of the federated pipeline: data quality checks, "
             "federated imputation, federated feature selection (summed site-stratified Cox score statistics), hyper-parameter tuning with a "
             "federated objective, node models combined into a sample-weighted ensemble, cross-site evaluation, privacy-preserving SHAP and "
             "MLflow logging. (b) The derivation cohort of the registry split into five geographic nodes (circle area proportional to the "
             "number of patients) and the three arms run with the same code: pooled, federated, single centre. (c) The temporal validation "
             "cohort from a later registry freeze, mapped to the same areas, on which every final model is scored and compared pairwise.")
    numbers["fig1"] = {"nodes": sizes, "events": events, "temporal_per_area": per_area.to_dict()}


def fig2_discrimination(w: Writer, runs: pd.DataFrame, imp: str, name: str, where: str, numbers: dict):
    import matplotlib.pyplot as plt
    rows = [(m, f) for m in ("cox", "gradient", "rsf") for f in ("nofs", "score")]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    y = np.arange(len(rows))[::-1]
    for yi, (m, f) in zip(y, rows):
        sel = runs[(runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == imp)]
        for role, marker, dy in (("central", "s", 0.18), ("federated", "D", -0.18)):
            r = sel[sel["role"] == role]
            if len(r):
                r = r.iloc[0]
                ax.errorbar(r["temporal_harrell_c"], yi + dy, xerr=[[r["temporal_harrell_c"] - r["temporal_harrell_c_ci_low"]], [r["temporal_harrell_c_ci_high"] - r["temporal_harrell_c"]]],
                            fmt=marker, color=ARM_COLOR[role], ms=7, capsize=3, label={"central": "pooled", "federated": "federated"}[role] if yi == y[0] else None)
                ax.plot(r["derivation_weighted_c"], yi + dy, marker=marker, mfc="white", mec=ARM_COLOR[role], ms=6, ls="none",
                        label=f"{'pooled' if role == 'central' else 'federated'}, derivation CV" if yi == y[0] else None)
        solo = sel[sel["role"] == "standalone"]
        for _, r in solo.iterrows():
            big = r["node"] == "italy_north"
            ax.plot(r["temporal_harrell_c"], yi, marker="o", ms=5 if big else 4, color="#444444" if big else "#aaaaaa", ls="none",
                    label=("node alone (Italy north)" if big else "node alone") if yi == y[0] and (big or r["node"] == "americas") else None)
    ax.set_yticks(y); ax.set_yticklabels([f"{MODEL_LABEL[m]}\n{FS_LABEL[f]}" for m, f in rows], fontsize=8)
    ax.axvline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Harrell C-index: temporal validation cohort (filled, 95% CI) and derivation cross-validation (hollow)")
    ax.set_xlim(0.45, 0.80); ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.legend(loc="lower left", fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    w.figure(fig, name, f"Discrimination across the three arms ({imp} imputation). For each model family and feature set, the Harrell C-index "
             "of the pooled model (square), the federated ensemble (diamond) and each of the five single-centre models (grey circles; "
             "italy_north dark) on the 698-patient temporal validation cohort, with 95% bootstrap confidence intervals for the pooled and "
             "federated models. Hollow markers: the sample-weighted cross-validated C-index of the same models in derivation, where the "
             "federated arm is below the pooled arm in every configuration.", where=where)


def table2_paired(w: Writer, paired: pd.DataFrame, runs: pd.DataFrame, numbers: dict):
    p = paired[paired["subset"] == "all"]
    rows = []

    def get(comparison, **kw):
        q = p[p["comparison"] == comparison]
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.iloc[0] if len(q) == 1 else None
    deriv = runs.pivot_table(index=["model", "fs", "imputation"], columns="role", values="derivation_weighted_c")
    for m in ("cox", "gradient", "rsf"):
        for f in ("nofs", "score"):
            r = get("arm", model=m, fs=f, imputation="basic")
            if r is not None:
                d = deriv.loc[(m, f, "basic")]
                rows.append({"Comparison": "Federated - pooled", "Model": MODEL_LABEL[m], "Features": FS_LABEL[f], "Temporal C difference [95% CI]": fmt_ci(r["delta"], r["ci_low"], r["ci_high"]),
                             "Derivation CV difference": f"{d['federated'] - d['central']:+.3f}"})
    for m in ("cox", "gradient", "rsf"):
        for role in ("central", "federated"):
            r = get("fs", model=m, role=role, imputation="basic", label="score - nofs")
            if r is not None:
                rows.append({"Comparison": f"10 - 33 features, {'pooled' if role == 'central' else 'federated'}", "Model": MODEL_LABEL[m], "Features": "",
                             "Temporal C difference [95% CI]": fmt_ci(r["delta"], r["ci_low"], r["ci_high"]), "Derivation CV difference": ""})
    for m in ("cox", "gradient", "rsf"):
        for f in ("nofs", "score"):
            for subset_label, sub in (("own area", "own"), ("whole cohort", "all")):
                q = paired[(paired["comparison"] == "standalone_vs_federated") & (paired["model"] == m) & (paired["fs"] == f) & (paired["imputation"] == "basic")]
                q = q[q["subset"] == "all"] if sub == "all" else q[q["subset"] != "all"]
                if len(q):
                    rows.append({"Comparison": f"Node alone - federated, {subset_label}", "Model": MODEL_LABEL[m], "Features": FS_LABEL[f],
                                 "Temporal C difference [95% CI]": f"{q['delta'].mean():+.3f} (mean of {len(q)} nodes; {int((q['delta'] < 0).sum())} below, {int((q['ci_high'] < 0).sum())} CI < 0)",
                                 "Derivation CV difference": ""})
    t2 = pd.DataFrame(rows)
    w.table(t2, "Table2_paired_differences", "Table 2. Paired differences of the Harrell C-index on the temporal validation cohort (basic imputation; "
            "2000 bootstrap resamples of the same patients for both models; 95% percentile CI). Federated - pooled for the 33-feature and the "
            "10-feature models; 10 - 33 features within each arm; single-centre model minus the federated model on the node's own temporal area "
            "and on the whole cohort (mean over the five nodes, with the number of nodes below the federated model and with a CI excluding 0). "
            "Last column: the same difference in derivation cross-validation.")
    numbers["table2"] = rows


def read_rank(final: Path, run: str):
    f = final / "runs" / run / "feature_selection_scores.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f); s = s[s["run_name"] == s["run_name"].iloc[0]]
    if "score" in s.columns:
        s = s.sort_values("score", ascending=False, kind="stable")
    return list(s["features"])


def fig3_selection(w: Writer, final: Path, runs: pd.DataFrame, paired: pd.DataFrame, agree: pd.DataFrame, numbers: dict):
    import matplotlib.pyplot as plt
    ranks = {}
    for role in ("central", "federated"):
        for imp in ("basic", "mice"):
            r = runs[(runs["role"] == role) & (runs["model"] == "cox") & (runs["fs"] == "score") & (runs["imputation"] == imp)]
            if len(r):
                ranks[(role, imp)] = read_rank(final, r.iloc[0]["run_name"])
    for a in AREAS:
        r = runs[(runs["role"] == "standalone") & (runs["node"] == a) & (runs["model"] == "cox") & (runs["fs"] == "score") & (runs["imputation"] == "basic")]
        if len(r):
            ranks[(a, "basic")] = read_rank(final, r.iloc[0]["run_name"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    # (a) bump chart pooled vs federated, top 12, basic and mice
    ax = axes[0, 0]
    cols = [k for k in [("central", "basic"), ("federated", "basic"), ("central", "mice"), ("federated", "mice")] if ranks.get(k)]
    top = 12
    feats = sorted({f for k in cols for f in ranks[k][:top]})
    for f in feats:
        ys = [ranks[k].index(f) + 1 if f in ranks[k] else np.nan for k in cols]
        ax.plot(range(len(cols)), ys, "-o", ms=4, lw=1, alpha=0.8)
        if not np.isnan(ys[0]) and ys[0] <= top:
            ax.text(-0.1, ys[0], flabel(f), ha="right", va="center", fontsize=7)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{'pooled' if k[0] == 'central' else 'federated'}\n{k[1]}" for k in cols], fontsize=8)
    ax.set_ylim(top + 0.5, 0.5); ax.set_ylabel("rank (score test, forward order)"); ax.set_title("(a) Pooled vs federated ranking", fontsize=10, loc="left")
    ax.axhline(10.5, color="grey", lw=0.6, ls="--"); ax.set_xlim(-1.6, len(cols) - 0.5)
    # (b) selected-set heat strip: pooled, federated, nodes (basic)
    ax = axes[0, 1]
    order = ranks.get(("central", "basic")) or []
    show = order[:14]
    groups = [("central", "basic", "pooled"), ("federated", "basic", "federated")] + [(a, "basic", AREA_LABEL[a]) for a in AREAS]
    mat = np.full((len(groups), len(show)), np.nan)
    for i, (k1, k2, _) in enumerate(groups):
        rk = ranks.get((k1, k2))
        if rk:
            for j, f in enumerate(show):
                mat[i, j] = 1 if f in rk[:10] else 0
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.4, aspect="auto")
    ax.set_xticks(range(len(show))); ax.set_xticklabels([flabel(s) for s in show], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[2] for g in groups], fontsize=8)
    ax.set_title("(b) The 10 selected features (dark) by arm and node", fontsize=10, loc="left")
    # (c) agreement with the pooled ranking
    ax = axes[1, 0]
    a = agree[(agree["comparison"].isin(["arm", "standalone_vs_central"])) & (agree["model"] == "cox") & (agree["fs"] == "score") & (agree["imputation"] == "basic")]
    labels, vals, cols_ = [], [], []
    for _, r in a.iterrows():
        if r["comparison"] == "arm":
            labels.append("federated"); vals.append(r["spearman"]); cols_.append(ARM_COLOR["federated"])
        else:
            labels.append(AREA_LABEL.get(r["node"], r["node"])); vals.append(r["spearman"]); cols_.append(ARM_COLOR["standalone"])
    ax.barh(range(len(vals)), vals, color=cols_); ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("Spearman correlation with the pooled ranking (33 features)")
    ax.set_title("(c) Agreement with the pooled selection", fontsize=10, loc="left")
    # (d) 10 vs 33 features: paired temporal differences
    ax = axes[1, 1]
    p = paired[(paired["comparison"] == "fs") & (paired["label"] == "score - nofs") & (paired["imputation"] == "basic") & (paired["subset"] == "all") & (paired["role"].isin(["central", "federated"]))]
    yy = 0
    ticks, tl = [], []
    for m in ("cox", "gradient", "rsf"):
        for role in ("central", "federated"):
            r = p[(p["model"] == m) & (p["role"] == role)]
            if len(r):
                r = r.iloc[0]
                ax.errorbar(r["delta"], yy, xerr=[[r["delta"] - r["ci_low"]], [r["ci_high"] - r["delta"]]], fmt="D" if role == "federated" else "s", color=ARM_COLOR[role], capsize=3)
                ticks.append(yy); tl.append(f"{MODEL_LABEL[m]}, {'pooled' if role == 'central' else 'federated'}"); yy += 1
    ax.axvline(0, color="grey", lw=0.6, ls="--"); ax.set_yticks(ticks); ax.set_yticklabels(tl, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Harrell C, 10 features - 33 features (temporal cohort, paired 95% CI)")
    ax.set_title("(d) Cost of parsimony", fontsize=10, loc="left")
    fig.tight_layout()
    w.figure(fig, "Fig3_feature_selection", "Federated feature selection. (a) Ranks of the leading features in the pooled and federated score-test "
             "rankings (forward order of the site-stratified Cox score statistic) under basic and MICE imputation; the same features lead both arms. "
             "(b) The ten selected features in the pooled arm, the federated arm and each node selecting on its own data. (c) Spearman correlation "
             "of the federated and of each single-node ranking with the pooled ranking over the 33 predictors. (d) Paired difference of the "
             "temporal Harrell C-index between the 10-feature and the 33-feature model, by model family and arm.")
    numbers["fig3_spearman"] = dict(zip(labels, [float(v) for v in vals]))


def fig4_calibration(w: Writer, final: Path, runs: pd.DataFrame, numbers: dict):
    import matplotlib.pyplot as plt
    main = runs[(runs["imputation"] == "basic") & (runs["fs"].isin(["nofs", "score"]))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    roles = [("central", "pooled"), ("federated", "federated"), ("standalone", "node alone")]
    # (a) calibration-in-the-large at 6/12/18 by arm
    ax = axes[0]
    for i, (role, lab) in enumerate(roles):
        sub = main[main["role"] == role]
        for j, h in enumerate((6, 12, 18)):
            col = f"temporal_calib_itl_{h}"
            if col in sub.columns:
                x = np.full(len(sub), i + (j - 1) * 0.22) + np.random.default_rng(j).uniform(-0.05, 0.05, len(sub))
                ax.scatter(x, sub[col], s=14, color=ARM_COLOR[role], alpha=0.5 + 0.25 * j, label=f"{h} months" if i == 0 else None)
    ax.axhline(0, color="grey", lw=0.6, ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels([r[1] for r in roles])
    ax.set_ylabel("observed - mean predicted survival"); ax.set_title("(a) Calibration-in-the-large", fontsize=10, loc="left"); ax.legend(fontsize=7, frameon=False)
    # (b) slope at 12 months by arm, log scale
    ax = axes[1]
    for i, (role, lab) in enumerate(roles):
        sub = main[main["role"] == role]
        x = np.full(len(sub), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(sub))
        ax.scatter(x, sub["temporal_calib_slope_12"], s=16, color=ARM_COLOR[role])
    ax.set_yscale("log"); ax.axhline(1, color="grey", lw=0.6, ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels([r[1] for r in roles])
    ax.set_yticks([0.2, 0.5, 1, 2, 5]); ax.set_yticklabels(["0.2", "0.5", "1", "2", "5"]); ax.minorticks_off()
    ax.set_ylabel("calibration slope at 12 months (log scale)"); ax.set_title("(b) Calibration slope", fontsize=10, loc="left")
    # (c) calibration curves of the recommended model in the three arms
    ax = axes[2]
    m, f, imp = MAIN_MODEL
    for role, lab in roles[:2]:
        r = runs[(runs["role"] == role) & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == imp)]
        if len(r):
            c = final / "runs" / r.iloc[0]["run_name"] / "calibration.csv"
            if c.exists():
                g = pd.read_csv(c); g = g[g["horizon"] == 12]
                ax.plot(g["predicted_survival"], g["observed_survival_km"], "-o", color=ARM_COLOR[role], label=lab)
    for a in AREAS:
        r = runs[(runs["role"] == "standalone") & (runs["node"] == a) & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == imp)]
        if len(r):
            c = final / "runs" / r.iloc[0]["run_name"] / "calibration.csv"
            if c.exists():
                g = pd.read_csv(c); g = g[g["horizon"] == 12]
                ax.plot(g["predicted_survival"], g["observed_survival_km"], "-", color=ARM_COLOR["standalone"], lw=0.8, alpha=0.7, label="node alone" if a == AREAS[0] else None)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted survival at 12 months"); ax.set_ylabel("observed (Kaplan-Meier)"); ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"(c) {MODEL_LABEL[m]}, {FS_LABEL[f]}: quintiles", fontsize=10, loc="left")
    fig.tight_layout()
    w.figure(fig, "Fig4_calibration", "Calibration on the temporal validation cohort across the three arms (basic imputation; 33- and "
             "10-feature models of the three families). (a) Calibration-in-the-large at 6, 12 and 18 months: Kaplan-Meier observed survival "
             "minus mean predicted survival (positive = pessimistic model). (b) Calibration slope at 12 months (coefficient of a Cox model of "
             "the outcome on log(-log) predicted survival; 1 = ideal, above 1 = risk estimates too moderate). (c) Observed versus predicted "
             "12-month survival by quintile of predicted risk for the 10-feature Cox model in the pooled arm, the federated arm and each node alone.")
    numbers["slope12_by_role"] = main.groupby("role")["temporal_calib_slope_12"].agg(["mean", "min", "max"]).to_dict()


def fig5_shap(w: Writer, final: Path, runs: pd.DataFrame, numbers: dict):
    import matplotlib.pyplot as plt
    m, f, imp = MAIN_MODEL
    names = {}
    for role in ("central", "federated"):
        r = runs[(runs["role"] == role) & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == imp)]
        if len(r):
            names[role] = r.iloc[0]["run_name"]
            png = final / "runs" / names[role] / "figures" / f"shap_summary_{names[role]}.png"
            if png.exists():
                shutil.copyfile(png, w.root / "main" / f"Fig5{'a' if role == 'federated' else 'b'}_shap_beeswarm_{role}.png")
    imp_tabs = {}
    for role, run in names.items():
        t = final / "runs" / run / "shap_importance.csv"
        if t.exists():
            imp_tabs[role] = pd.read_csv(t).set_index("feature")["mean_abs_shap"]
    if len(imp_tabs) == 2:
        both = pd.concat(imp_tabs, axis=1).fillna(0)  # union of the two models' inputs, ordered by the larger importance
        feats = list(both.max(axis=1).sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(feats) + 1.2))
        y = np.arange(len(feats))
        ax.barh(y - 0.18, [imp_tabs["central"].get(x, 0) for x in feats], height=0.36, color=ARM_COLOR["central"], label="pooled")
        ax.barh(y + 0.18, [imp_tabs["federated"].get(x, 0) for x in feats], height=0.36, color=ARM_COLOR["federated"], label="federated")
        ax.set_yticks(y); ax.set_yticklabels([flabel(f) for f in feats], fontsize=8); ax.invert_yaxis(); ax.set_xlabel("mean |SHAP| on the temporal cohort (100 patients)")
        ax.legend(fontsize=8, frameon=False); fig.tight_layout()
        w.figure(fig, "Fig5c_shap_importance", "Mean absolute SHAP value per feature of the 10-feature Cox model, pooled and federated arms, computed "
                 "with the logged explainers on 100 patients of the temporal cohort.")
        rk_c, rk_f = list(imp_tabs["central"].sort_values(ascending=False).index), list(imp_tabs["federated"].sort_values(ascending=False).index)
        common = [x for x in rk_c if x in rk_f]; n = len(common)
        rc, rf = {x: i for i, x in enumerate([x for x in rk_c if x in common])}, {x: i for i, x in enumerate([x for x in rk_f if x in common])}
        numbers["fig5_shap_spearman"] = 1 - 6 * sum((rc[x] - rf[x]) ** 2 for x in common) / (n * (n * n - 1)) if n > 1 else None
    w.captions.append("**Fig5a/b shap beeswarm.** The framework's privacy-preserving SHAP summary plots of the 10-feature Cox model, federated arm (a) "
                      "and pooled arm (b): SHAP values of the training patients against their feature values discretised into 20 bins with "
                      "federated global boundaries (5th-95th percentiles), rows permuted per feature; no patient-level value leaves a node.")


def supplementary_tables(w: Writer, final: Path, runs: pd.DataFrame, paired: pd.DataFrame, agree: pd.DataFrame, numbers: dict):
    # S2 search spaces and run settings from the logged params, in paper vocabulary
    def params_of(role, m, fs, imp):
        r = runs[(runs["role"] == role) & (runs["model"] == m) & (runs["fs"] == fs) & (runs["imputation"] == imp)]
        if not len(r):
            return {}
        return json.load(open(final / "runs" / r.iloc[0]["run_name"] / "model_info.json")).get("params", {})
    rows = []
    for m in ("cox", "gradient", "rsf"):
        p = params_of("central", m, "nofs", "basic")
        for k, v in sorted(p.items()):
            if "model_params_space" in k:
                rows.append({"Section": f"Search space, {MODEL_LABEL[m]}", "Parameter": k.split(".")[-1], "Value": v})
    p = params_of("central", "cox", "nofs", "basic")
    settings = [("Tuning", "trials per run", p.get("train.num_tuning_round")), ("Tuning", "sampler", "TPE, seed 42"),
                ("Tuning", "cross-validation", f"{p.get('train.cross_val_n_fold')}-fold, {p.get('train.cross_val_type')} on the event indicator"),
                ("Tuning", "objective", "sample-weighted mean over sites of the validation C-index of the global model"),
                ("Imputation", "basic", p.get("imputation.imputation_algorithm")),
                ("Imputation", "MICE", f"{params_of('central', 'cox', 'nofs', 'mice').get('imputation.imputation_algorithm')}, "
                                       f"{params_of('central', 'cox', 'nofs', 'mice').get('imputation.regression_model')}, "
                                       f"{params_of('central', 'cox', 'nofs', 'mice').get('imputation.maxit')} sweeps"),
                ("Feature selection", "score test", f"method {params_of('central', 'cox', 'score', 'basic').get('feature_selection.fs_method')}, "
                                                   f"mode {params_of('central', 'cox', 'score', 'basic').get('feature_selection.fs_selected_modes')}, "
                                                   f"fraction {params_of('central', 'cox', 'score', 'basic').get('feature_selection.fs_top_p_options')} "
                                                   f"(10 of 33), minimum events per site {params_of('central', 'cox', 'score', 'basic').get('feature_selection.fs_min_cases')}"),
                ("Feature selection", "mutual information", f"method {params_of('central', 'cox', 'mi', 'basic').get('feature_selection.fs_method')}, "
                                                           f"mode {params_of('central', 'cox', 'mi', 'basic').get('feature_selection.fs_selected_modes')}, "
                                                           f"score {params_of('central', 'cox', 'mi', 'basic').get('feature_selection.fs_divided')}, "
                                                           f"fraction {params_of('central', 'cox', 'mi', 'basic').get('feature_selection.fs_top_p_options')}"),
                ("Explanation", "background", f"{p.get('shap.clustering_num_clusters')} fuzzy c-means centres, at most {p.get('shap.clustering_max_number_rounds')} rounds, "
                                              f"tolerance {p.get('shap.clustering_epsilon')}, fuzziness {p.get('shap.clustering_lambda_factor')}"),
                ("Explanation", "feature binning", f"{p.get('shap.shap_privacy_bins')} bins between the {p.get('shap.shap_outlier_percentile')}th and "
                                                   f"{100 - float(p.get('shap.shap_outlier_percentile', 5))}th percentiles; sites with fewer than "
                                                   f"{p.get('shap.shap_min_samples_for_boundaries')} records excluded from the boundaries")]
    rows += [{"Section": s, "Parameter": k, "Value": str(v)} for s, k, v in settings if v is not None]
    if rows:
        w.table(pd.DataFrame(rows), "TableS2_search_spaces", "Table S2. Hyper-parameter search spaces (Optuna, 50 trials, seed 42, 5-fold stratified "
                "cross-validation) and run settings, as logged by the framework.", where="supplementary")
    # S12 data-quality checks per node (from the data-quality stage of one federated run; identical in every run)
    r = runs[(runs["role"] == "federated") & (runs["model"] == "cox") & (runs["fs"] == "nofs") & (runs["imputation"] == "basic")]
    dq_root = final / "runs" / r.iloc[0]["run_name"] / "data_quality" if len(r) else None
    if dq_root is not None and dq_root.exists():
        dq_rows = []
        for a in AREAS:
            try:
                cols_t = json.load(open(dq_root / a / "columns_tests_results.json"))
                rows_t = json.load(open(dq_root / a / "rows_tests_results.json"))
                ct = pd.DataFrame(cols_t["data"], columns=cols_t["columns"])
                ct = ct[ct["columns_names"].isin(FEATURES)]
                rt = pd.DataFrame(rows_t["data"], columns=rows_t["columns"]).iloc[0]
                n = int(rt["row_count"])
                miss = ct.set_index("columns_names")["missing_counts"].astype(float)
                worst = miss.idxmax()
                dq_rows.append({"Node": AREA_LABEL[a], "Records": n, "Duplicate records": "none" if rt["duplicate_rows"] == "pass" else str(rt["duplicate_rows"]),
                                "Predictors present with the expected type": f"{int((ct['expected_columns'] == 'pass').sum())} of {len(ct)}",
                                "Predictors with values outside the admissible range": int((ct["check_rules"] != "pass").sum()),
                                "Predictors with missing values": int((miss > 0).sum()),
                                "Missing cells, % of predictor cells": round(100 * miss.sum() / (n * len(ct)), 1),
                                "Predictor with most missing values": f"{flabel(worst)} ({100 * miss[worst] / n:.0f}%)"})
            except Exception as e:  # noqa: BLE001
                LOGGER.warning("data-quality table, node %s: %s", a, e)
        if dq_rows:
            w.table(pd.DataFrame(dq_rows), "TableS12_data_quality", "Table S12. Data-quality checks of the federated data-quality stage, per node: "
                    "records, duplicates, presence and type of the 33 predictors, admissible-range violations and missing values.",
                    where="supplementary", decimals=1)
    keep = ["run_name", "role", "node", "model", "fs", "imputation", "n_features", "derivation_weighted_c", "selected_features"]
    w.table(runs[keep].sort_values(["role", "node", "model", "fs", "imputation"]), "TableS5_derivation", "Table S5. Derivation results of every run: "
            "best-trial sample-weighted cross-validated C-index and the features used by the final model.", where="supplementary")
    tcols = [c for c in runs.columns if c.startswith("temporal_")]
    w.table(runs[["run_name", "role", "node", "model", "fs", "imputation"] + tcols], "TableS6_temporal", "Table S6. Temporal validation metrics of every run: "
            "Harrell C with 95% CI, Uno C (tau 24 months), time-dependent AUC at 6/12/18/24 months, integrated Brier score to 12 months, "
            "calibration-in-the-large and slope at 6/12/18 months, Harrell C per area.", where="supplementary")
    w.table(paired, "TableS7_paired_all", "Table S7. All paired bootstrap comparisons (2000 resamples): comparison type, runs, subset, "
            "C-index of each run, difference and 95% CI.", where="supplementary")
    sm = final / "tables" / "standalone_matrix.csv"
    if sm.exists():
        w.table(pd.read_csv(sm), "TableS8_single_centre", "Table S8. Single-centre models: Harrell C on the node's own temporal area and on the whole "
                "cohort, next to the federated and pooled models of the same configuration.", where="supplementary")
    rk = final / "tables" / "rankings.csv"
    if rk.exists():
        r = pd.read_csv(rk)
        r = r.merge(runs[["run_name", "role", "node", "model", "fs", "imputation"]], on="run_name")
        r = r[(r["model"] == "cox") & (r["fs"] == "score")]
        piv = r.pivot_table(index="rank", columns=["role", "node", "imputation"], values="feature", aggfunc="first")
        piv.columns = ["_".join(str(x) for x in c if str(x) not in ("", "nan")) for c in piv.columns]
        piv = piv.apply(lambda col: col.map(flabel))
        w.table(piv, "TableS9_rankings", "Table S9. Score-test feature rankings (forward order) in the pooled arm, the federated arm and each node "
                "alone, both imputations; identical across the three model families.", where="supplementary", index=True)
    a = agree[agree["comparison"].isin(["arm", "fs", "standalone_vs_central", "standalone_vs_federated", "imputation"])]
    w.table(a.drop(columns=[c for c in ("run_a", "run_b") if c in a.columns]), "TableS9b_ranking_agreement", "Table S9b. Agreement of the feature "
            "rankings for every pair of runs differing in one factor: Spearman over the 33 features, overlap of the top 4/8/10, Jaccard of the selected lists.",
            where="supplementary")
    mi = runs[runs["fs"] == "mi"][["run_name", "role", "node", "model", "imputation", "derivation_weighted_c", "temporal_harrell_c", "temporal_harrell_c_ci_low", "temporal_harrell_c_ci_high", "selected_features"]]
    pm = paired[(paired["label"].isin(["mi - nofs", "score - mi"])) & (paired["subset"] == "all")][["label", "model", "role", "node", "imputation", "delta", "ci_low", "ci_high"]]
    w.table(pd.concat([mi.assign(section="runs"), pm.assign(section="paired differences")], ignore_index=True), "TableS10_mutual_information",
            "Table S10. The mutual-information feature-selection arm (10 features): runs and paired temporal differences against no selection and "
            "against the score test. The federated MI ranking diverges from the pooled one and costs discrimination; it motivated the score-test method.",
            where="supplementary")
    tm = final / "tables" / "derivation_trial_paired.csv"
    if tm.exists():
        w.table(pd.read_csv(tm), "TableS11_trials_matched", "Table S11. Derivation comparison federated - pooled over the Optuna trials with identical "
                "hyper-parameters (the TPE start-up trials; later trials diverge).", where="supplementary")


def supplementary_figures(w: Writer, final: Path, runs: pd.DataFrame, nodes: dict, cohort: pd.DataFrame):
    import matplotlib.pyplot as plt
    from sksurv.nonparametric import kaplan_meier_estimator
    # S1 per-area forest: temporal C by area for pooled, federated and node alone (basic, nofs + score)
    main = runs[(runs["imputation"] == "basic") & (runs["fs"].isin(["nofs", "score"]))]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), sharey=True)
    rows = [(m, f) for m in ("cox", "gradient", "rsf") for f in ("nofs", "score")]
    y = np.arange(len(rows))[::-1]
    n_area = cohort.groupby("area_federata").size() if "area_federata" in cohort.columns else {}
    for ax, a in zip(axes, AREAS):
        for yi, (m, f) in zip(y, rows):
            for role, marker, dy in (("central", "s", 0.2), ("federated", "D", 0), ("standalone", "o", -0.2)):
                sel = main[(main["model"] == m) & (main["fs"] == f) & (main["role"] == role)]
                if role == "standalone":
                    sel = sel[sel["node"] == a]
                if len(sel):
                    r = sel.iloc[0]
                    lo, hi = r.get(f"temporal_c_{a}_ci_low", np.nan), r.get(f"temporal_c_{a}_ci_high", np.nan)
                    if pd.notna(lo) and pd.notna(hi):  # forest plot: per-area bootstrap CI (TRIPOD-Cluster 13a)
                        ax.plot([lo, hi], [yi + dy, yi + dy], color=ARM_COLOR[role], lw=1, solid_capstyle="butt")
                    ax.plot(r[f"temporal_c_{a}"], yi + dy, marker=marker, color=ARM_COLOR[role], ls="none", ms=5,
                            label={"central": "pooled", "federated": "federated", "standalone": "node alone"}[role] if (a == AREAS[0] and yi == y[0]) else None)
        title = f"{AREA_LABEL[a]} (n = {int(n_area[a])})" if a in n_area else AREA_LABEL[a]
        ax.set_title(title, fontsize=9); ax.axvline(0.5, color="grey", lw=0.5, ls=":"); ax.set_xlim(0.4, 0.9); ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels([f"{MODEL_LABEL[m]}, {FS_LABEL[f]}" for m, f in rows], fontsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left"); axes[2].set_xlabel("Harrell C-index on the temporal patients of the area (95% bootstrap CI)")
    fig.tight_layout()
    w.figure(fig, "FigS1_per_area", "Figure S1. Harrell C-index on the temporal patients of each area for the pooled model, the federated ensemble "
             "and the node's own model (basic imputation), with 95% bootstrap confidence intervals.", where="supplementary")
    # S3 calibration curves of every main-grid model (basic + mice, nofs + score)
    sel = runs[(runs["role"].isin(["central", "federated"])) & (runs["fs"].isin(["nofs", "score"]))].sort_values(["model", "fs", "imputation", "role"])
    n = len(sel); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow), squeeze=False)
    for ax, (_, r) in zip(axes.flat, sel.iterrows()):
        c = final / "runs" / r["run_name"] / "calibration.csv"
        if c.exists():
            g = pd.read_csv(c)
            for h, col in ((6, "#9ecae1"), (12, "#3182bd"), (18, "#08519c")):
                gg = g[g["horizon"] == h]
                ax.plot(gg["predicted_survival"], gg["observed_survival_km"], "-o", ms=2.5, lw=0.8, color=col, label=f"{h} m")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.5, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(r["run_name"].replace("central", "pooled"), fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6, frameon=False); fig.tight_layout()
    w.figure(fig, "FigS3_calibration_curves", "Figure S3. Observed versus predicted survival at 6, 12 and 18 months by quintile of predicted risk "
             "for every pooled and federated model (33- and 10-feature, basic and MICE imputation).", where="supplementary")
    # S4 Kaplan-Meier per node and per temporal area
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (title, groups) in zip(axes, (("Derivation cohort by node", [(a, nodes[a]) for a in AREAS if a in nodes]),
                                          ("Temporal validation cohort by area", [(a, cohort[cohort["area_federata"] == a]) for a in AREAS]))):
        for a, d in groups:
            if len(d) == 0:
                continue
            t = pd.to_numeric(d["tte"]).to_numpy(float); e = pd.to_numeric(d["event"]).to_numpy(int).astype(bool)
            x, s = kaplan_meier_estimator(e, t)
            ax.step(x, s, where="post", label=f"{AREA_LABEL[a]} (n={len(d)}, deaths={int(e.sum())})")
        ax.set_xlim(0, 36); ax.set_ylim(0, 1); ax.set_xlabel("months from treatment start"); ax.set_ylabel("overall survival"); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    w.figure(fig, "FigS4_kaplan_meier", "Figure S4. Kaplan-Meier overall survival of the derivation cohort by federated node and of the temporal "
             "validation cohort by area.", where="supplementary")
    # S5 montage of the framework's SHAP beeswarms (main grid, basic)
    import matplotlib.image as mpimg
    sel = runs[(runs["role"].isin(["central", "federated"])) & (runs["imputation"] == "basic") & (runs["fs"].isin(["nofs", "score"]))].sort_values(["model", "fs", "role"])
    pngs = [(r["run_name"], final / "runs" / r["run_name"] / "figures" / f"shap_summary_{r['run_name']}.png") for _, r in sel.iterrows()]
    pngs = [(n_, p) for n_, p in pngs if p.exists()]
    if pngs:
        ncol = 4; nrow = int(np.ceil(len(pngs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.4 * nrow), squeeze=False)
        for ax, (n_, p) in zip(axes.flat, pngs):
            ax.imshow(mpimg.imread(p)); ax.set_title(n_.replace("central", "pooled"), fontsize=7); ax.axis("off")
        for ax in axes.flat[len(pngs):]:
            ax.axis("off")
        fig.tight_layout()
        w.figure(fig, "FigS5_shap_beeswarms", "Figure S5. The framework's privacy-preserving SHAP summary plots of every pooled and federated model "
                 "(basic imputation, 33- and 10-feature models).", where="supplementary")
    # S7 missingness per node
    if nodes:
        miss = pd.DataFrame({AREA_LABEL.get(a, a): nodes[a][[c for c in FEATURES if c in nodes[a].columns]].isna().mean() * 100 for a in nodes})
        miss["Temporal"] = cohort[[c for c in FEATURES if c in cohort.columns]].isna().mean() * 100
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(miss.values, cmap="Reds", vmin=0, vmax=max(30, float(np.nanmax(miss.values))), aspect="auto")
        ax.set_xticks(range(miss.shape[1])); ax.set_xticklabels(miss.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(miss.shape[0])); ax.set_yticklabels(miss.index, fontsize=7); fig.colorbar(im, ax=ax, label="% missing")
        fig.tight_layout()
        w.figure(fig, "FigS7_missingness", "Figure S7. Share of missing values per predictor in each derivation node, the pooled derivation cohort "
                 "and the temporal validation cohort.", where="supplementary")


# ----------------------------------------------------------------------------- entry point
def build(final: Path, data_folder: Path | None, centres_csv: Path | None):
    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from paper_style import setup_fonts
        LOGGER.info("figure font: %s", setup_fonts())
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("figure fonts not set: %s", e)
    w = Writer(final / "paper")
    numbers = {}
    runs = pd.read_csv(final / "tables" / "runs.csv").fillna({"node": ""})
    runs["node"] = runs["node"].astype(str)
    paired = pd.read_csv(final / "tables" / "paired_comparisons.csv").fillna({"node": "", "role": "", "fs": "", "imputation": ""})
    agree = pd.read_csv(final / "tables" / "ranking_agreement.csv").fillna({"node": "", "role": "", "fs": "", "imputation": ""}) if (final / "tables" / "ranking_agreement.csv").exists() else pd.DataFrame()
    cohort = pd.read_csv(final / "cohort" / "temporal_cohort_prepared.csv")
    nodes = load_nodes(data_folder) if data_folder else {}
    centres = pd.read_csv(centres_csv, sep=";") if centres_csv and Path(centres_csv).exists() else None
    numbers["flow"] = patient_flow(final, centres_csv)
    for step in (lambda: cohorts_tables(w, nodes, cohort, centres, numbers),
                 lambda: fig1_design(w, nodes, cohort, centres, numbers),
                 lambda: fig2_discrimination(w, runs, "basic", "Fig2_discrimination", "main", numbers),
                 lambda: fig2_discrimination(w, runs, "mice", "FigS2_discrimination_mice", "supplementary", numbers),
                 lambda: table2_paired(w, paired, runs, numbers),
                 lambda: fig3_selection(w, final, runs, paired, agree, numbers),
                 lambda: fig4_calibration(w, final, runs, numbers),
                 lambda: fig5_shap(w, final, runs, numbers),
                 lambda: supplementary_tables(w, final, runs, paired, agree, numbers),
                 lambda: supplementary_figures(w, final, runs, nodes, cohort)):
        try:
            step()
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("paper item failed: %s", e, exc_info=True)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from paper_variants import build_variants
        build_variants(final, runs, paired, agree, w)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("figure variants failed: %s", e, exc_info=True)
    w.finish(numbers)
    LOGGER.info("paper artifacts written to %s", w.root)
