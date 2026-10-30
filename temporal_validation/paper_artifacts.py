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

    def med(col, nd=1):
        v = x[col].dropna()
        return f"{v.median():.{nd}f} ({v.quantile(.25):.{nd}f}-{v.quantile(.75):.{nd}f})" if len(v) else ""

    def pct(col, value=1):
        v = x[col].dropna()
        return f"{int((v == value).sum())} ({100 * (v == value).mean():.1f}%)" if len(v) else ""
    period = ""
    if "treatment_start_month" in df.columns and df["treatment_start_month"].notna().any():
        p = df["treatment_start_month"].dropna().astype(str)
        period = f"{p.min()} to {p.max()}"
    feats = [c for c in FEATURES if c in x.columns]
    n_pred = len(feats)
    miss = x[feats].isna().mean()
    s = {"Patients": str(n), "Deaths": f"{int(x['event'].sum())} ({100 * x['event'].mean():.1f}%)",
         "Events per predictor": f"{x['event'].sum() / n_pred:.1f}" if n_pred else "",
         "Treatment start, first to last (year-month)": period,
         "Median observed follow-up, months": f"{x['tte'].median():.1f}", "Median OS (Kaplan-Meier), months": f"{km_median(x['tte'], x['event']):.1f}",
         "Age, years": med("age"), "Male sex": pct("sex", 1), "ECOG >= 1": pct("ps ecog", 1),
         "Metastatic disease": pct("locally advanced/metastatic", 2),
         "Intrahepatic location": pct("location - Intrahepatic", 1), "Previous surgery": pct("surgery", 1),
         "Number of metastatic sites": med("number of met sites", 0), "CA19-9, U/mL": med("ca199", 0), "CEA, ng/mL": med("cea"),
         "Albumin, g/dL": med("albumin"), "Neutrophils, x10^9/L": med("neutrophils"), "NLR": med("nlr"),
         # missing data by node/area (TRIPOD+AI 11, TRIPOD-Cluster 10b): share of empty cells, complete records, worst predictor
         "Missing values, % of predictor cells": f"{100 * miss.mean():.1f}",
         "Patients with all 33 predictors": f"{int(x[feats].notna().all(axis=1).sum())} ({100 * x[feats].notna().all(axis=1).mean():.1f}%)" if n_pred else "",
         "Predictor with most missing values": f"{flabel(miss.idxmax())} ({100 * miss.max():.0f}%)" if n_pred and miss.max() > 0 else "none"}
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
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
    from paper_style import FULL_WIDTH_IN
    # One axes covering the whole figure, data units = inches at the printed size (6.5 in wide): every position below is
    # a printed distance, so text sizes (6.5-7.5 pt) and geometry can be reasoned about together.
    # savefig(bbox_inches="tight") keeps the whole axes and pads it by 0.1 in per side, so the canvas is 0.2 in narrower
    # than the printed width; the height is trimmed to the content at the end
    W, H = FULL_WIDTH_IN - 0.2, 5.0
    LH = 6.5 * 1.22 / 72  # one line of 6.5 pt text, in inches
    GRID, INK, SOFT = "#5a6b7b", "#222222", "#555555"
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    L, R = 0.0, W  # outer content edges

    # (a) pipeline: eight stages in one row; the heading names the federation, so the box labels stay short
    top = H - 0.05
    ax.text(L, top, "(a) Federated pipeline", fontsize=7.5, va="top", ha="left")
    ax.text(R, top, "run on NVFlare: each node keeps its patients, the server sees aggregates only", fontsize=6.5,
            va="top", ha="right", color=SOFT, style="italic")
    stages = ["Data\nquality", "Imputation", "Feature\nselection", "Tuning\n(Optuna,\n50 trials)",
              "Node models\n+ weighted\nensemble", "Cross-site\nevaluation", "Privacy-\npreserving\nSHAP", "MLflow\nlogging"]
    bw, bh = 0.63, 0.44
    gap = (R - L - len(stages) * bw) / (len(stages) - 1)
    yb = top - 0.2 - bh / 2
    for i, s in enumerate(stages):
        x = L + bw / 2 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x - bw / 2, yb - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#eef1f4", ec=GRID, lw=0.8))
        ax.text(x, yb, s, ha="center", va="center", fontsize=6.5, linespacing=1.15)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + bw / 2 + 0.025, yb), (x + bw / 2 + gap - 0.025, yb), arrowstyle="-|>",
                                         mutation_scale=7, color=GRID, lw=0.9, shrinkA=0, shrinkB=0))
    top = yb - bh / 2 - 0.22

    # (b) derivation: patient flow, the three arms, the five nodes around the server
    xb0, xb1 = L, 3.4
    ax.text(xb0, top, "(b) Derivation cohort: five nodes, three arms", fontsize=7.5, va="top", ha="left")
    sizes = {a: len(nodes[a]) for a in AREAS if a in nodes}
    events = {a: int(pd.to_numeric(nodes[a]["event"]).sum()) for a in AREAS if a in nodes}
    ncent = {}
    if centres is not None:
        for a in AREAS:
            sub = centres[(centres["area_federata"] == a) & (centres["n_derivazione"].astype(int) > 0)]
            ncent[a] = (len(sub), sub["paese"].nunique())
    total = sum(sizes.values()); tev = sum(events.values())
    flow = numbers.get("flow", {})
    y = top - 0.2
    if flow.get("derivation_rows"):
        ax.text(xb0, y, f"{flow['derivation_rows']} patients in the development cohort of the registry", fontsize=6.5, va="top")
        ax.text(xb0, y - LH, f"− {flow['derivation_excluded']} excluded (centre not assignable to a node)", fontsize=6.5,
                va="top", color=SOFT)
        y -= 2 * LH + 0.05
    arms = [("central", f"Pooled arm: all {total} patients, {tev} deaths, in one place"),
            ("federated", "Federated arm: the five nodes + server (ensemble of node models)"),
            ("standalone", "Single-centre arm: each node runs the pipeline alone")]
    for arm, t in arms:
        ax.add_patch(FancyBboxPatch((xb0, y - LH * 0.78), 0.075, 0.075, boxstyle="square,pad=0", fc=ARM_COLOR[arm], ec="none"))
        ax.text(xb0 + 0.13, y, t, fontsize=6.5, va="top")
        y -= LH
    # node network: circle area proportional to the patients; top labels above the circles, the others below
    rmax = 0.28
    rad = {a: rmax * np.sqrt(sizes[a] / max(sizes.values())) for a in sizes}
    xc, xl, xr = (xb0 + xb1) / 2, xb0 + 0.55, xb1 - 0.55
    rtop = max([rad.get("americas", 0), rad.get("asia", 0)])
    ytop = y - 0.08 - 3 * LH - 0.04 - rtop
    ys = ytop - 0.5
    pos = {"americas": (xl, ytop), "asia": (xr, ytop), "europe_non_italy": (xl, ys - 0.5), "italy_centre_south": (xr, ys - 0.5),
           "italy_north": (xc, ys - 0.78)}
    rs = 0.17
    ybot = H
    ax.add_patch(Circle((xc, ys), rs, fc=ARM_COLOR["federated"], ec="none"))
    ax.text(xc, ys, "server", ha="center", va="center", color="white", fontsize=6.5, weight="bold")
    for a in AREAS:
        if a not in sizes:
            continue
        x, yy = pos[a]; r = rad[a]
        ax.add_patch(Circle((x, yy), r, fc=AREA_COLOR[a], alpha=0.22, ec="none"))
        ax.add_patch(Circle((x, yy), r, fc="none", ec=AREA_COLOR[a], lw=1.1))
        ax.add_patch(FancyArrowPatch((x, yy), (xc, ys), arrowstyle="<|-|>", mutation_scale=6, color="#8c8c8c", lw=0.7,
                                     shrinkA=r * 72 + 2, shrinkB=rs * 72 + 2))
        lab = f"{AREA_LABEL[a]}\n{sizes[a]} patients, {events[a]} deaths"
        if a in ncent:
            lab += f"\n{ncent[a][0]} centres, {ncent[a][1]} countr{'y' if ncent[a][1] == 1 else 'ies'}"
        above = a in ("americas", "asia")
        ax.text(x, yy + rtop + 0.04 if above else yy - r - 0.04, lab, ha="center", va="bottom" if above else "top", fontsize=6.5,
                linespacing=1.15)
        if not above:
            ybot = min(ybot, yy - r - 0.04 - 2.85 * LH)

    # (c) temporal validation: patient flow and per-area counts in a neutral box, then the evaluation
    xc0, xc1 = 3.65, R
    ax.text(xc0, top, "(c) Temporal validation on a later registry freeze", fontsize=7.5, va="top", ha="left")
    per_area = cohort.groupby("area_federata").agg(n=("patient_number", "size"), deaths=("event", "sum"))
    nt, dt = len(cohort), int(cohort["event"].sum())
    lines = [("Temporal validation cohort: later data freeze,", INK), ("no patient in common with derivation", INK)]
    if flow.get("validation_rows"):
        lines += [(f"{flow['validation_rows']} records in the validation export", INK),
                  (f"− {flow.get('validation_duplicates', 0)} re-entries of development patients", SOFT),
                  (f"− {flow.get('validation_other_excluded', 0)} with unusable follow-up", SOFT)]
    lines += [(f"= {nt} patients, {dt} deaths", INK)]
    pad = 0.08
    nrow = sum(a in per_area.index for a in AREAS)
    btop = top - 0.2
    bhh = pad + len(lines) * LH + 0.08 + (nrow + 1) * LH + pad
    ax.add_patch(FancyBboxPatch((xc0, btop - bhh), xc1 - xc0, bhh, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#f2f2f2", ec=SOFT, lw=0.8))
    y = btop - pad
    for t, col in lines:
        ax.text(xc0 + pad, y, t, fontsize=6.5, va="top", color=col, weight="bold" if t.startswith("=") else "normal")
        y -= LH
    y -= 0.08
    xp, xd = xc1 - 0.75, xc1 - pad  # right edges of the two count columns
    ax.text(xc0 + pad, y, "Area", fontsize=6.5, va="top", color=SOFT, style="italic")
    ax.text(xp, y, "Patients", fontsize=6.5, va="top", ha="right", color=SOFT, style="italic")
    ax.text(xd, y, "Deaths", fontsize=6.5, va="top", ha="right", color=SOFT, style="italic")
    y -= LH
    for a in AREAS:
        if a not in per_area.index:
            continue
        ax.add_patch(Circle((xc0 + pad + 0.03, y - LH * 0.45), 0.03, fc=AREA_COLOR[a], ec="none"))
        ax.text(xc0 + pad + 0.1, y, AREA_LABEL[a], fontsize=6.5, va="top")
        ax.text(xp, y, f"{int(per_area.loc[a, 'n'])}", fontsize=6.5, va="top", ha="right")
        ax.text(xd, y, f"{int(per_area.loc[a, 'deaths'])}", fontsize=6.5, va="top", ha="right")
        y -= LH
    items = ["every final model of the three arms scored on the same 698 patients",
             "Harrell C (95% bootstrap CI), Uno C, time-dependent AUC, integrated Brier score",
             "calibration-in-the-large and slope at 6, 12, 18 months",
             "paired bootstrap differences: federated vs pooled, 10 vs 33 features, node alone vs federated (own area, whole cohort)",
             "feature-ranking agreement pooled / federated / nodes; SHAP agreement"]
    y = btop - bhh - 0.12
    for t in items:
        # no-break spaces keep "10 vs 33" on one line (textwrap splits on ASCII whitespace only)
        wrapped = textwrap.fill(t.replace("10 vs 33", "10 vs 33"), 54, break_on_hyphens=False)
        ax.text(xc0 + 0.02, y, "•", fontsize=6.5, va="top")
        ax.text(xc0 + 0.12, y, wrapped, fontsize=6.5, va="top", linespacing=1.15)
        y -= (wrapped.count("\n") + 1) * LH * 0.95 + 0.05
    ybot = min(ybot, y) - 0.02
    # trim the canvas to the content, keeping one data unit = one printed inch
    fig.set_size_inches(W, H - ybot); ax.set_ylim(ybot, H)
    w.figure(fig, "Fig1_framework_and_design", "Framework and study design. (a) The stages of the federated pipeline, run on NVFlare "
             "(each node keeps its patients; the server sees aggregates only): data quality checks, federated imputation, federated "
             "feature selection (summed site-stratified Cox score statistics), hyper-parameter tuning with a federated objective "
             "(Optuna, 50 trials), node models combined into a sample-weighted ensemble, cross-site evaluation, privacy-preserving SHAP "
             "and MLflow logging. (b) Derivation cohort: patient flow from the registry, the three arms run with the same code (pooled, "
             "federated, single centre; colours as in the other figures) and the five geographic nodes around the federation server, "
             "with patients, deaths, centres and countries per node (circle area proportional to the number of patients). (c) Temporal "
             "validation cohort from a later registry freeze: patient flow, patients and deaths per area (the same areas as the nodes), "
             "and the evaluation applied to every final model, compared pairwise.")
    numbers["fig1"] = {"nodes": sizes, "events": events, "temporal_per_area": per_area.to_dict()}


def fig2_discrimination(w: Writer, runs: pd.DataFrame, imp: str, name: str, where: str, numbers: dict):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from paper_style import FULL_WIDTH_IN
    rows = [(m, f) for m in ("cox", "gradient", "rsf") for f in ("nofs", "score")]
    # 0.12 in less than the page width: savefig(bbox_inches="tight") adds 0.1 in of padding on each side
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN - 0.12, 4.3), layout="constrained")
    y = np.arange(len(rows))[::-1]
    # temporal marker at +/-0.17 around the row centre, the derivation (hollow) marker of the same arm a little further
    # out, so that the two never hide each other; the single-centre models sit on the row centre
    arms = (("central", "s", 0.17), ("federated", "D", -0.17))
    shown = []
    for k, (yi, (m, f)) in enumerate(zip(y, rows)):
        if k % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="0.95", lw=0, zorder=0)
        sel = runs[(runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == imp)]
        for role, marker, dy in arms:
            r = sel[sel["role"] == role]
            if len(r):
                r = r.iloc[0]
                ax.errorbar(r["temporal_harrell_c"], yi + dy, xerr=[[r["temporal_harrell_c"] - r["temporal_harrell_c_ci_low"]], [r["temporal_harrell_c_ci_high"] - r["temporal_harrell_c"]]],
                            fmt=marker, color=ARM_COLOR[role], ms=5, capsize=2, elinewidth=0.9, capthick=0.9, zorder=3)
                ax.plot(r["derivation_weighted_c"], yi + 1.75 * dy, marker=marker, mfc="white", mec=ARM_COLOR[role], mew=0.9, ms=4.5,
                        ls="none", zorder=4)
                shown += [r["temporal_harrell_c_ci_low"], r["temporal_harrell_c_ci_high"], r["derivation_weighted_c"]]
        solo = sel[sel["role"] == "standalone"]
        for _, r in solo.iterrows():
            big = r["node"] == "italy_north"
            ax.plot(r["temporal_harrell_c"], yi, marker="o", ms=4 if big else 3.5, color="#333333" if big else "#a6a6a6", ls="none",
                    zorder=3 if big else 2)
            shown.append(r["temporal_harrell_c"])
    ax.set_yticks(y); ax.set_yticklabels([f"{MODEL_LABEL[m]}\n{FS_LABEL[f]}" for m, f in rows], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    # x range from the plotted values (rounded to 0.05), never narrower than 0.50-0.80
    vals = [v for v in shown if pd.notna(v)]
    lo = np.floor((min(vals) - 0.01) * 20) / 20 if vals else 0.45
    hi = np.ceil((max(vals) + 0.01) * 20) / 20 if vals else 0.80
    lo, hi = min(lo, 0.50), max(hi, 0.80)
    if lo < 0.5:
        ax.axvline(0.5, color="grey", lw=0.6, ls=":", zorder=1)
    ax.set_xlim(lo, hi); ax.grid(axis="x", lw=0.3, alpha=0.6, zorder=1)
    ax.set_xlabel("Harrell C-index")
    handles = [
        Line2D([], [], marker="s", color=ARM_COLOR["central"], ms=5, lw=0.9, label="pooled, temporal cohort (95% CI)"),
        Line2D([], [], marker="D", color=ARM_COLOR["federated"], ms=5, lw=0.9, label="federated, temporal cohort (95% CI)"),
        Line2D([], [], marker="o", color="#333333", ms=4, ls="none", label="node alone: Italy north, temporal cohort"),
        Line2D([], [], marker="s", mfc="white", mec=ARM_COLOR["central"], mew=0.9, ms=4.5, ls="none", label="pooled, derivation CV"),
        Line2D([], [], marker="D", mfc="white", mec=ARM_COLOR["federated"], mew=0.9, ms=4.5, ls="none", label="federated, derivation CV"),
        Line2D([], [], marker="o", color="#a6a6a6", ms=3.5, ls="none", label="node alone: other four nodes, temporal cohort"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, columnspacing=1.6, handlelength=1.8)
    w.figure(fig, name, f"Discrimination across the three arms ({imp} imputation). For each model family and feature set, the Harrell C-index "
             "of the pooled model (square), the federated ensemble (diamond) and each of the five single-centre models (circles; "
             "Italy north in dark grey, the other four nodes in light grey) on the 698-patient temporal validation cohort, with 95% "
             "bootstrap confidence intervals for the pooled and federated models. Hollow markers, drawn just outside the filled marker of "
             "the same arm: the sample-weighted cross-validated C-index of the same models in derivation, where the federated arm is below "
             "the pooled arm in every configuration.", where=where)


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
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from paper_style import FULL_WIDTH_IN
    arm_name = {"central": "Pooled", "federated": "Federated"}
    # savefig(bbox_inches="tight") adds 0.1 in on each side: drawn 0.2 in narrower, the file is FULL_WIDTH_IN wide
    fig = plt.figure(figsize=(FULL_WIDTH_IN - 0.2, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1.35, 1])
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1))
    # (a) bump chart pooled vs federated, top 12, basic and MICE; ranks below 12 share one row ("> 12")
    ax = ax_a
    cols = [k for k in [("central", "basic"), ("federated", "basic"), ("central", "mice"), ("federated", "mice")] if ranks.get(k)]
    top, n_sel = 12, 10
    below = top + 1
    feats = sorted({f for k in cols for f in ranks[k][:top]}, key=lambda f: (ranks[cols[0]].index(f) if f in ranks[cols[0]] else 99, str(f)))
    pos = {f: [min(ranks[k].index(f) + 1, below) if f in ranks[k] else np.nan for k in cols] for f in feats}
    # colour only the features whose rank moves by 3 or more or that cross the selection cut-off; the stable ones stay grey
    moved = [f for f in feats if np.nanmax(pos[f]) - np.nanmin(pos[f]) >= 3 or (np.nanmin(pos[f]) <= n_sel < np.nanmax(pos[f]))]
    palette = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    fcol = {f: palette[i % len(palette)] for i, f in enumerate(moved)}
    fls = {f: "-" if i < len(palette) else "--" for i, f in enumerate(moved)}
    x = np.arange(len(cols))
    ax.axhline(n_sel + 0.5, color="0.35", lw=0.6, ls="--", zorder=1)
    ax.axhspan(top + 0.5, below + 0.5, color="0.93", lw=0, zorder=0)
    for f in feats:
        if f in fcol:
            ax.plot(x, pos[f], ls=fls[f], marker="o", color=fcol[f], ms=3, lw=1.1, zorder=3, clip_on=False)
        else:
            ax.plot(x, pos[f], "-o", color="0.62", ms=2.5, lw=0.8, zorder=2, clip_on=False)
    axr = ax.twinx()
    for side, k in ((ax, cols[0]), (axr, cols[-1])):
        side.set_ylim(below + 0.5, 0.5)
        side.set_yticks(list(range(1, below + 1)))
        side.set_yticklabels([flabel(f) for f in ranks[k][:top]] + [f"> {top}"])
        for t, f in zip(side.get_yticklabels(), ranks[k][:top] + [None]):
            t.set_color(fcol.get(f, "0.15") if f is not None else "0.45")
        side.tick_params(axis="y", length=2)
    labelled = set(ranks[cols[0]][:top]) | set(ranks[cols[-1]][:top])
    for f in feats:  # a feature in the top 12 of a middle column only: label it at its best rank
        if f not in labelled:
            j = int(np.nanargmin(pos[f]))
            ax.annotate(flabel(f), (x[j], pos[f][j]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=5.5, color=fcol.get(f, "0.3"))
    ax.set_xlim(-0.15, len(cols) - 0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"{arm_name[k[0]]}\n{'basic' if k[1] == 'basic' else 'MICE'}" for k in cols])
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=2)
    for xv in x[1:] - 0.5:
        if cols[int(xv + 0.5)][1] != cols[int(xv - 0.5)][1]:
            ax.axvline(xv, color="0.8", lw=0.6, zorder=0)
    for s in ("top",):
        ax.spines[s].set_visible(False); axr.spines[s].set_visible(False)
    ax.set_title("(a) Score-test ranks, top 12", loc="left")
    # (b) selection matrix: features (pooled order) x pooled, federated and each node alone (basic imputation)
    ax = ax_b
    order = ranks.get(("central", "basic")) or []
    show = order[:14]
    groups = [("central", "basic", "Pooled"), ("federated", "basic", "Federated")] + [(a, "basic", AREA_LABEL[a]) for a in AREAS]
    mat = np.full((len(show), len(groups)), np.nan)
    for j, (k1, k2, _) in enumerate(groups):
        rk = ranks.get((k1, k2))
        if rk:
            for i, f in enumerate(show):
                mat[i, j] = 1 if f in rk[:n_sel] else 0
    c_sel, c_not = "#3d3d3d", "#e4e4e4"
    cmap = ListedColormap([c_not, c_sel]); cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, mat.shape[1]), minor=True); ax.set_yticks(np.arange(-0.5, mat.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=1); ax.tick_params(which="minor", length=0)
    ax.axvline(1.5, color="white", lw=3.5)
    ax.set_yticks(range(len(show))); ax.set_yticklabels([flabel(s) for s in show])
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[2] for g in groups], rotation=90, ha="center", va="bottom")
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=2)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(fc=c_sel, label=f"selected (top {n_sel})"), Patch(fc=c_not, label="not selected")], loc="upper center",
              bbox_to_anchor=(0.5, -0.01), ncol=2, handlelength=1.0, handleheight=1.0, columnspacing=1.2, borderaxespad=0.2)
    ax.set_title("(b) Selected features", loc="left")
    # (c) agreement with the pooled ranking
    ax = ax_c
    a = agree[(agree["comparison"].isin(["arm", "standalone_vs_central"])) & (agree["model"] == "cox") & (agree["fs"] == "score") & (agree["imputation"] == "basic")]
    labels, shown, vals, cols_ = [], [], [], []
    for _, r in a.iterrows():
        if r["comparison"] == "arm":
            labels.append("federated"); shown.append("Federated"); vals.append(r["spearman"]); cols_.append(ARM_COLOR["federated"])
        else:
            lab = AREA_LABEL.get(r["node"], r["node"])
            labels.append(lab); shown.append(lab); vals.append(r["spearman"]); cols_.append(ARM_COLOR["standalone"])
    ax.barh(range(len(vals)), vals, color=cols_, height=0.65)
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", ha="left", fontsize=6)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(shown); ax.invert_yaxis()
    ax.set_xlim(0, 1.1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("Spearman ρ with the pooled ranking (33 features)")
    ax.legend(handles=[Patch(fc=ARM_COLOR["federated"], label="federated"), Patch(fc=ARM_COLOR["standalone"], label="node alone")],
              loc="center right", handlelength=1.0, handleheight=1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("(c) Agreement with the pooled ranking", loc="left")
    # (d) 10 vs 33 features: paired temporal differences, one row per model family, pooled and federated side by side
    ax = ax_d
    p = paired[(paired["comparison"] == "fs") & (paired["label"] == "score - nofs") & (paired["imputation"] == "basic") & (paired["subset"] == "all") & (paired["role"].isin(["central", "federated"]))]
    short = {"cox": "Cox", "gradient": "Gradient\nboosting", "rsf": "Random\nsurvival forest"}
    ticks, tl = [], []
    for gi, m in enumerate(("cox", "gradient", "rsf")):
        any_m = False
        for ri, role in enumerate(("central", "federated")):
            r = p[(p["model"] == m) & (p["role"] == role)]
            if len(r):
                r = r.iloc[0]
                ax.errorbar(r["delta"], gi + (ri - 0.5) * 0.36, xerr=[[r["delta"] - r["ci_low"]], [r["ci_high"] - r["delta"]]],
                            fmt="D" if role == "federated" else "s", color=ARM_COLOR[role], ms=3.5, capsize=2, elinewidth=0.9,
                            label=("pooled" if role == "central" else "federated") if gi == 0 else None)
                any_m = True
        if any_m:
            ticks.append(gi); tl.append(short[m])
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(ticks); ax.set_yticklabels(tl); ax.set_ylim(2.6, -1.5); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ΔC-index, 10 − 33 features\n(temporal cohort)")
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.1, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("(d) Cost of parsimony", loc="left")
    w.figure(fig, "Fig3_feature_selection", "Federated feature selection. (a) Ranks of the leading features in the pooled and federated score-test "
             "rankings (forward order of the site-stratified Cox score statistic) under basic and MICE imputation; left labels give the pooled "
             "(basic) order, right labels the federated (MICE) order, ranks below 12 are shown in the shaded '> 12' row, and the dashed line is "
             "the selection cut-off (10 features). Features whose rank moves by three or more places, or that cross the cut-off, are coloured; "
             "the others are grey. The same features lead both arms. (b) Selection of the 14 leading pooled features (rows) in the pooled arm, "
             "the federated arm and each node selecting on its own data (columns): dark, among the ten selected; light, not selected. "
             "(c) Spearman correlation of the federated and of each single-node ranking with the pooled ranking over the 33 predictors. "
             "(d) Paired difference of the temporal Harrell C-index between the 10-feature and the 33-feature model (95% bootstrap CI), "
             "by model family and arm (square, pooled; diamond, federated).")
    numbers["fig3_spearman"] = dict(zip(labels, [float(v) for v in vals]))


def fig4_calibration(w: Writer, final: Path, runs: pd.DataFrame, numbers: dict):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from paper_style import FULL_WIDTH_IN
    main = runs[(runs["imputation"] == "basic") & (runs["fs"].isin(["nofs", "score"]))]
    # a and b on top; c (square) bottom left with its key in the bottom-right cell
    # 0.12 in less than the page width: savefig(bbox_inches="tight") adds 0.1 in of padding on each side
    fig = plt.figure(figsize=(FULL_WIDTH_IN - 0.12, 5.2), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1.2])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    key = fig.add_subplot(gs[1, 1]); key.axis("off")
    roles = [("central", "pooled"), ("federated", "federated"), ("standalone", "node alone")]
    horizon_marker = {6: "o", 12: "s", 18: "^"}
    # (a) calibration-in-the-large at 6/12/18 by arm; horizon = marker shape and position inside the arm (left to right)
    ax = axes[0]
    for i, (role, lab) in enumerate(roles):
        sub = main[main["role"] == role]
        for j, h in enumerate((6, 12, 18)):
            col = f"temporal_calib_itl_{h}"
            if col in sub.columns:
                x = np.full(len(sub), i + (j - 1) * 0.22) + np.random.default_rng(j).uniform(-0.05, 0.05, len(sub))
                ax.scatter(x, sub[col], s=11, marker=horizon_marker[h], color=ARM_COLOR[role], alpha=0.8, edgecolors="none", zorder=3)
    ax.axhline(0, color="grey", lw=0.6, ls="--", zorder=1); ax.set_xticks(range(3)); ax.set_xticklabels([r[1] for r in roles])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("observed − mean predicted survival"); ax.set_title("(a) Calibration-in-the-large", loc="left")
    ax.legend(handles=[Line2D([], [], marker=horizon_marker[h], color="0.3", ls="none", ms=3.5, label=f"{h} months") for h in (6, 12, 18)],
              loc="upper left", ncol=3, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.3)
    # (b) slope at 12 months by arm, log scale
    ax = axes[1]
    for i, (role, lab) in enumerate(roles):
        sub = main[main["role"] == role]
        x = np.full(len(sub), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(sub))
        ax.scatter(x, sub["temporal_calib_slope_12"], s=12, color=ARM_COLOR[role], alpha=0.85, edgecolors="none", zorder=3)
    ax.set_yscale("log"); ax.axhline(1, color="grey", lw=0.6, ls="--", zorder=1); ax.set_xticks(range(3)); ax.set_xticklabels([r[1] for r in roles])
    ax.set_xlim(-0.5, 2.5)
    ax.set_yticks([0.2, 0.5, 1, 2, 5]); ax.set_yticklabels(["0.2", "0.5", "1", "2", "5"]); ax.minorticks_off()
    ax.set_ylabel("calibration slope at 12 months\n(log scale)"); ax.set_title("(b) Calibration slope", loc="left")
    # (c) calibration curves of the recommended model in the three arms; each node alone in its area colour, dashed
    ax = axes[2]
    m, f, imp = MAIN_MODEL
    handles = []
    for role, lab in roles[:2]:
        r = runs[(runs["role"] == role) & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == imp)]
        if len(r):
            c = final / "runs" / r.iloc[0]["run_name"] / "calibration.csv"
            if c.exists():
                g = pd.read_csv(c); g = g[g["horizon"] == 12]
                handles += ax.plot(g["predicted_survival"], g["observed_survival_km"], "-o", color=ARM_COLOR[role], lw=1.3, ms=3.5,
                                   zorder=4, label=lab)
    for a in AREAS:
        r = runs[(runs["role"] == "standalone") & (runs["node"] == a) & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == imp)]
        if len(r):
            c = final / "runs" / r.iloc[0]["run_name"] / "calibration.csv"
            if c.exists():
                g = pd.read_csv(c); g = g[g["horizon"] == 12]
                handles += ax.plot(g["predicted_survival"], g["observed_survival_km"], ls=(0, (3, 1.5)), marker="o", ms=2,
                                   color=AREA_COLOR[a], lw=0.9, zorder=3, label=f"{AREA_LABEL[a]} alone")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls=":", zorder=1); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("predicted survival at 12 months"); ax.set_ylabel("observed survival at 12 months\n(Kaplan–Meier)")
    ax.set_title(f"(c) {MODEL_LABEL[m]}, {FS_LABEL[f]}", loc="left")
    key.legend(handles=handles, loc="center left", title=r"$\bf{c}$  calibration curves", alignment="left", handlelength=2.6)
    w.figure(fig, "Fig4_calibration", "Calibration on the temporal validation cohort across the three arms (basic imputation; 33- and "
             "10-feature models of the three families). (a) Calibration-in-the-large at 6, 12 and 18 months (circle, square, triangle, "
             "left to right within each arm): Kaplan-Meier observed survival minus mean predicted survival (positive = pessimistic model). "
             "(b) Calibration slope at 12 months (coefficient of a Cox model of the outcome on log(−log) predicted survival; 1 = ideal, "
             "above 1 = risk estimates too moderate). (c) Observed versus predicted 12-month survival by quintile of predicted risk for "
             "the 10-feature Cox model in the pooled arm, the federated arm and each node alone (dashed, one colour per node); dotted "
             "diagonal: perfect calibration.")
    numbers["slope12_by_role"] = main.groupby("role")["temporal_calib_slope_12"].agg(["mean", "min", "max"]).to_dict()


def _shap_raster(png: Path) -> np.ndarray:
    """The framework's SHAP beeswarm (shap default 800x550 px) without its header line, which is truncated and reads
    'Federated Robust' for every run, pooled included; trimmed to its ink with a small margin. Returns an RGB array."""
    from PIL import Image
    im = np.asarray(Image.open(png).convert("RGB")).copy()
    ink = im.min(axis=2) < 200
    h, wd = ink.shape
    rows = np.nonzero(ink.any(axis=1))[0]
    if not len(rows):
        return im
    # colour bar = the long vertical run of ink in the right third; its 'High' label sits right under the header line
    tall = [c for c in range(wd * 2 // 3, wd) if ink[:, c].sum() > h * 0.5]
    x_cb = (min(tall) - 4) if tall else wd

    def first_blank(region, start):
        r = start
        while r < h and region[r].any():
            r += 1
        return r
    y0 = rows[0]
    end_right = first_blank(ink[:, x_cb:], y0) if x_cb < wd else y0
    end_left = first_blank(ink[:, :x_cb], y0)
    im[:end_right, :] = 255
    im[:end_left, :x_cb] = 255
    ink = im.min(axis=2) < 200
    ys, xs = np.nonzero(ink)
    pad = 6
    return im[max(ys.min() - pad, 0):min(ys.max() + pad + 1, h), max(xs.min() - pad, 0):min(xs.max() + pad + 1, wd)]


def _shap_bars(ax, imp_tabs: dict, feats: list, n_pat, legend=True):
    """Mean |SHAP| of the pooled and the federated model per feature; a feature outside one arm's ten is marked 'not selected'."""
    from matplotlib.transforms import blended_transform_factory
    y = np.arange(len(feats))
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for role, dy, lab in (("central", -0.19, "pooled"), ("federated", 0.19, "federated")):
        s = imp_tabs[role]
        has = [x in s.index for x in feats]
        ax.barh([yy + dy for yy, ok in zip(y, has) if ok], [s[x] for x, ok in zip(feats, has) if ok], height=0.38, color=ARM_COLOR[role], label=lab)
        for yy, ok in zip(y, has):
            if not ok:
                ax.text(0.01, yy + dy, "not selected", transform=tr, va="center", ha="left", fontsize=5.5, style="italic", color=ARM_COLOR[role])
    ax.set_yticks(y); ax.set_yticklabels([flabel(x) for x in feats]); ax.set_ylim(len(feats) - 0.45, -0.55)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"mean |SHAP|, temporal cohort (n = {n_pat})")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(loc="lower right", handlelength=1.0, handleheight=1.0)


def fig5_shap(w: Writer, final: Path, runs: pd.DataFrame, numbers: dict):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    from paper_style import FULL_WIDTH_IN
    m, f, imp = MAIN_MODEL
    names, rasters = {}, {}
    for role in ("central", "federated"):
        r = runs[(runs["role"] == role) & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == imp)]
        if len(r):
            names[role] = r.iloc[0]["run_name"]
            png = final / "runs" / names[role] / "figures" / f"shap_summary_{names[role]}.png"
            if png.exists():
                try:
                    rasters[role] = _shap_raster(png)
                    Image.fromarray(rasters[role]).save(w.root / "main" / f"Fig5{'a' if role == 'federated' else 'b'}_shap_beeswarm_{role}.png", dpi=(100, 100))
                except Exception as e:  # noqa: BLE001
                    LOGGER.warning("SHAP beeswarm not cropped (%s): copied as logged", e)
                    shutil.copyfile(png, w.root / "main" / f"Fig5{'a' if role == 'federated' else 'b'}_shap_beeswarm_{role}.png")
    imp_tabs, n_pat = {}, {}
    for role, run in names.items():
        t = final / "runs" / run / "shap_importance.csv"
        if t.exists():
            d = pd.read_csv(t)
            imp_tabs[role] = d.set_index("feature")["mean_abs_shap"]
            n_pat[role] = int(d["n_patients"].iloc[0]) if "n_patients" in d.columns else 100
    # raw registry codes printed by the framework in a and b, with their paper names (only where they differ beyond case)
    codes = []
    for role in ("federated", "central"):
        for x in (imp_tabs[role].index if role in imp_tabs else []):
            if str(x).lower() != flabel(x).lower() and x not in [c for c, _ in codes]:
                codes.append((x, flabel(x)))
    code_text = "; ".join(f"{c} = {lab}" for c, lab in codes)
    if len(imp_tabs) == 2:
        both = pd.concat(imp_tabs, axis=1).fillna(0)  # union of the two models' inputs, ordered by the larger importance (ordering only)
        feats = list(both.max(axis=1).sort_values(ascending=False).index)
        npt = n_pat["central"] if n_pat["central"] == n_pat["federated"] else f"{n_pat['central']}/{n_pat['federated']}"
        h_c = 0.2 * len(feats) + 0.75
        # savefig(bbox_inches="tight") adds 0.1 in per side: widths are drawn 0.2 in narrower than printed
        fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN * 2 / 3 - 0.2, h_c), layout="constrained")
        _shap_bars(ax, imp_tabs, feats, npt)
        w.figure(fig, "Fig5c_shap_importance", "Mean absolute SHAP value per feature of the 10-feature Cox model, pooled and federated arms, computed "
                 f"with the logged explainers on {npt} patients of the temporal cohort. 'not selected': the feature is not among the ten "
                 "selected by that arm, so the model has no SHAP value for it.")
        rk_c, rk_f = list(imp_tabs["central"].sort_values(ascending=False).index), list(imp_tabs["federated"].sort_values(ascending=False).index)
        common = [x for x in rk_c if x in rk_f]; n = len(common)
        rc, rf = {x: i for i, x in enumerate([x for x in rk_c if x in common])}, {x: i for i, x in enumerate([x for x in rk_f if x in common])}
        numbers["fig5_shap_spearman"] = 1 - 6 * sum((rc[x] - rf[x]) ** 2 for x in common) / (n * (n * n - 1)) if n > 1 else None
        # composite: beeswarms a (federated) and b (pooled) side by side, c below at two thirds of the width, key on the right
        if len(rasters) == 2:
            col_w = (FULL_WIDTH_IN - 0.2) / 2 - 0.05
            h_top = col_w * max(r.shape[0] / r.shape[1] for r in rasters.values()) + 0.25
            fig = plt.figure(figsize=(FULL_WIDTH_IN - 0.2, h_top + h_c + 0.15), layout="constrained")
            gs = fig.add_gridspec(2, 1, height_ratios=[h_top, h_c + 0.15])
            top = gs[0].subgridspec(1, 2); bot = gs[1].subgridspec(1, 2, width_ratios=[2, 1])
            for k, (role, head) in enumerate((("federated", "(a) Federated"), ("central", "(b) Pooled"))):
                ax = fig.add_subplot(top[0, k])
                ax.imshow(rasters[role], interpolation="lanczos"); ax.set_axis_off()
                ax.set_title(head, loc="left")
            ax = fig.add_subplot(bot[0, 0])
            _shap_bars(ax, imp_tabs, feats, npt, legend=False)
            ax.set_title("(c) Importance on the temporal cohort", loc="left")
            key = fig.add_subplot(bot[0, 1]); key.set_axis_off()
            key.legend(handles=[Patch(fc=ARM_COLOR["central"], label="pooled"), Patch(fc=ARM_COLOR["federated"], label="federated")],
                       loc="upper left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0, handlelength=1.0, handleheight=1.0)
            if codes:
                key.text(0.0, 0.74, "Labels in a and b", transform=key.transAxes, va="top", ha="left", fontsize=6.5)
                key.text(0.0, 0.67, "\n".join(f"{c}  →  {lab}" for c, lab in codes), transform=key.transAxes, va="top", ha="left",
                         fontsize=6, linespacing=1.45, color="0.2")
            w.figure(fig, "Fig5_shap", "Explanations of the 10-feature Cox model in the federated and the pooled arm. (a, b) The framework's "
                     "privacy-preserving SHAP summary plots of the federated (a) and the pooled (b) model, as logged: SHAP values of the training "
                     "patients (x axis; the two panels have different x ranges) against their feature values (colour, low to high) discretised into "
                     "20 bins with federated global boundaries (5th-95th percentiles), rows permuted per feature; no patient-level value leaves a node. "
                     "The plots print the registry codes, mapped to the paper names in the key. (c) Mean absolute SHAP value per feature of the same "
                     f"two models, computed with the logged explainers on {npt} patients of the temporal cohort; 'not selected': the feature is not "
                     "among the ten selected by that arm.")
    w.captions.append("**Fig5a/b shap beeswarm.** The framework's privacy-preserving SHAP summary plots of the 10-feature Cox model, federated arm (a) "
                      "and pooled arm (b): SHAP values of the training patients against their feature values discretised into 20 bins with "
                      "federated global boundaries (5th-95th percentiles), rows permuted per feature; no patient-level value leaves a node. "
                      "The framework's header line was removed (it is truncated and reads 'Federated Robust' for both arms); the x ranges of a "
                      "and b differ" + (f"; registry codes in the plots: {code_text}." if code_text else "."))


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
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sksurv.nonparametric import kaplan_meier_estimator
    try:
        from paper_style import FULL_WIDTH_IN
    except ImportError:
        FULL_WIDTH_IN = 6.5
    # S1 per-area forest: temporal C by area for pooled, federated and node alone (basic, nofs + score)
    main = runs[(runs["imputation"] == "basic") & (runs["fs"].isin(["nofs", "score"]))]
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH_IN, 4.9), sharey=True)
    rows = [(m, f) for m in ("cox", "gradient", "rsf") for f in ("nofs", "score")]
    y = np.arange(len(rows))[::-1]
    n_area = cohort.groupby("area_federata").size() if "area_federata" in cohort.columns else {}
    arms = (("central", "s", 0.25), ("federated", "D", 0), ("standalone", "o", -0.25))
    xs = []
    for k, (ax, a) in enumerate(zip(axes.flat, AREAS)):
        for yy in (1.5, 3.5):  # thin separators between the model families
            ax.axhline(yy, color="0.85", lw=0.4, zorder=0)
        for yi, (m, f) in zip(y, rows):
            for role, marker, dy in arms:
                sel = main[(main["model"] == m) & (main["fs"] == f) & (main["role"] == role)]
                if role == "standalone":
                    sel = sel[sel["node"] == a]
                if len(sel):
                    r = sel.iloc[0]
                    lo, hi = r.get(f"temporal_c_{a}_ci_low", np.nan), r.get(f"temporal_c_{a}_ci_high", np.nan)
                    if pd.notna(lo) and pd.notna(hi):  # forest plot: per-area bootstrap CI (TRIPOD-Cluster 13a)
                        ax.plot([lo, hi], [yi + dy, yi + dy], color=ARM_COLOR[role], lw=0.9, solid_capstyle="butt")
                        xs += [lo, hi]
                    ax.plot(r[f"temporal_c_{a}"], yi + dy, marker=marker, color=ARM_COLOR[role], ls="none", ms=3.6,
                            mec="white", mew=0.4)
                    xs.append(r[f"temporal_c_{a}"])
        title = f"{AREA_LABEL[a]} (n = {int(n_area[a])})" if a in n_area else AREA_LABEL[a]
        ax.set_title(f"({'abcde'[k]}) {title}", loc="left")
        ax.axvline(0.5, color="grey", lw=0.6, ls=":", zorder=0); ax.grid(axis="x", lw=0.3, alpha=0.5)
    xs = [v for v in xs if pd.notna(v)]
    x_lo = min(0.4, np.floor(min(xs) * 20) / 20) if xs else 0.4  # never cut a confidence interval
    x_hi = max(0.9, np.ceil(max(xs) * 20) / 20) if xs else 0.9
    for ax in axes.flat[:len(AREAS)]:
        ticks = np.arange(x_lo, x_hi + 1e-9, 0.1)
        ax.set_xlim(x_lo, x_hi); ax.set_xticks(ticks)  # end labels left blank so neighbouring panels' labels do not touch
        ax.set_xticklabels(["" if k in (0, len(ticks) - 1) else f"{v:.1f}" for k, v in enumerate(ticks)])
    axes[0, 0].set_ylim(-0.6, len(rows) - 0.4)
    axes[0, 0].set_yticks(y); axes[0, 0].set_yticklabels([f"{MODEL_LABEL[m]}\n{FS_LABEL[f]}" for m, f in rows])
    leg_ax = axes.flat[len(AREAS)]
    leg_ax.axis("off")
    handles = [Line2D([0], [0], color=ARM_COLOR[role], marker=mk, ms=3.6, mec="white", mew=0.4, lw=0.9) for role, mk, _ in arms]
    handles.append(Line2D([0], [0], color="grey", lw=0.6, ls=":"))
    leg_ax.legend(handles, ["Pooled model", "Federated model", "Node alone (the area's\nown node model)", "C-index 0.5 (chance)"],
                  loc="center left", handlelength=2.2, labelspacing=1.0, title="Point estimate and 95% CI", alignment="left")
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.095, top=0.95, wspace=0.1, hspace=0.3)
    fig.supxlabel("Harrell C-index on the temporal patients of the area (95% bootstrap CI)", x=0.59, y=0.015, fontsize=7)
    w.figure(fig, "FigS1_per_area", "Figure S1. Harrell C-index on the temporal patients of each area (a-e; n = temporal patients of the area) "
             "for the pooled model, the federated ensemble and the area's own node model (node alone), basic imputation, with 95% "
             "bootstrap confidence intervals; rows are the three model families with 33 features and with the 10 features of the score test.",
             where="supplementary")
    # S3 calibration curves of every main-grid model (basic + mice, nofs + score)
    fams = ("cox", "gradient", "rsf")
    cfgs = [("nofs", "basic"), ("nofs", "mice"), ("score", "basic"), ("score", "mice")]
    horizons = ((6, ":", "o"), (12, "-", "s"), (18, "--", "^"))
    fig, axes = plt.subplots(len(fams), len(cfgs), figsize=(FULL_WIDTH_IN, 5.15), sharex=True, sharey=True, squeeze=False)
    for i, m in enumerate(fams):
        for j, (f, imp) in enumerate(cfgs):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color="0.55", lw=0.5, ls=(0, (3, 2)), zorder=0)
            for role in ("central", "federated"):
                r = runs[(runs["role"] == role) & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == imp)]
                c = final / "runs" / r.iloc[0]["run_name"] / "calibration.csv" if len(r) else None
                if c is not None and c.exists():
                    g = pd.read_csv(c)
                    for h, ls, mk in horizons:
                        gg = g[g["horizon"] == h]
                        ax.plot(gg["predicted_survival"], gg["observed_survival_km"], ls=ls, marker=mk, ms=2.4, lw=0.75,
                                color=ARM_COLOR[role], mec="white", mew=0.3, alpha=0.9)
            ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
            if i == 0:
                ax.set_title(f"{'33' if f == 'nofs' else '10'} features\n{'basic' if imp == 'basic' else 'MICE'} imputation")
            if j == len(cfgs) - 1:
                ax.text(1.07, 0.5, textwrap.fill(MODEL_LABEL[m], 16), transform=ax.transAxes, rotation=270, va="center", ha="left",
                        fontsize=7.5, multialignment="center")
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.84, wspace=0.08, hspace=0.1)
    fig.supxlabel("Predicted survival (mean of the quintile of predicted risk)", x=0.5, y=0.015, fontsize=7)
    fig.supylabel("Observed survival (Kaplan–Meier)", x=0.04, y=0.46, fontsize=7)
    handles = [Line2D([0], [0], color=ARM_COLOR["central"], lw=1.2), Line2D([0], [0], color=ARM_COLOR["federated"], lw=1.2)]
    handles += [Line2D([0], [0], color="0.3", ls=ls, marker=mk, ms=2.8, lw=0.8) for _, ls, mk in horizons]
    handles.append(Line2D([0], [0], color="0.55", lw=0.5, ls=(0, (3, 2))))
    fig.legend(handles, ["Pooled", "Federated", "6 months", "12 months", "18 months", "Ideal calibration"], loc="lower center",
               bbox_to_anchor=(0.5, 0.9), ncol=6, handlelength=2.4, columnspacing=1.4)
    w.figure(fig, "FigS3_calibration_curves", "Figure S3. Observed (Kaplan-Meier) versus predicted survival at 6, 12 and 18 months by quintile of "
             "predicted risk on the temporal validation cohort, pooled (blue) and federated (red) model of the same configuration overlaid in "
             "each panel; rows are the model families, columns the feature sets (33 features; 10 features of the score test) and imputation "
             "methods (basic, MICE); line style and marker give the horizon, the dashed diagonal is ideal calibration.", where="supplementary")
    # S4 Kaplan-Meier per node and per temporal area
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.5), sharey=True)
    for ax, (title, groups) in zip(axes, (("(a) Derivation cohort, by node", [(a, nodes[a]) for a in AREAS if a in nodes]),
                                          ("(b) Temporal validation cohort, by area", [(a, cohort[cohort["area_federata"] == a]) for a in AREAS]))):
        for a, d in groups:
            if len(d) == 0:
                continue
            t = pd.to_numeric(d["tte"]).to_numpy(float); e = pd.to_numeric(d["event"]).to_numpy(int).astype(bool)
            x, s = kaplan_meier_estimator(e, t)
            x, s = np.r_[0.0, x], np.r_[1.0, s]  # every curve starts at (0, 1)
            ax.step(x, s, where="post", color=AREA_COLOR[a], lw=1.0, label=f"{AREA_LABEL[a]} (n = {len(d)}, deaths = {int(e.sum())})")
        ax.set_xlim(0, 36); ax.set_ylim(0, 1.02); ax.set_xticks(range(0, 37, 6)); ax.grid(lw=0.3, alpha=0.4)
        ax.set_xlabel("Months from treatment start"); ax.set_title(title, loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=1, handlelength=1.6, borderaxespad=0)
    axes[0].set_ylabel("Overall survival")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.36, top=0.93, wspace=0.1)
    w.figure(fig, "FigS4_kaplan_meier", "Figure S4. Kaplan-Meier overall survival of (a) the derivation cohort by federated node and (b) the "
             "temporal validation cohort by area; the legends give the patients and deaths of each group.", where="supplementary")
    # S5 montage of the framework's SHAP beeswarms (main grid, basic)
    import matplotlib.image as mpimg
    # layout: columns = model family, rows = (pooled, federated) x (33, 10 features); pooled and federated of the same
    # configuration are stacked. Three columns, because six rows of two rasters only fit a page at a smaller scale.
    def without_header(img):
        # blank the framework's clipped "SHAP Analysis - Federated Robust [...]" line (rows 0-33 at 800 px width; its descenders
        # reach the top of the colour bar's "High" label, which starts at x > 0.85 W) and crop the blank band
        img = img.copy(); s = img.shape[1] / 800
        img[:int(30 * s)] = 1.0
        img[int(30 * s):int(34 * s), :int(0.85 * img.shape[1])] = 1.0
        return img[int(28 * s):]
    grid = {}
    for f in ("nofs", "score"):
        for role in ("central", "federated"):
            for m in fams:
                r = runs[(runs["role"] == role) & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == "basic")]
                p = final / "runs" / r.iloc[0]["run_name"] / "figures" / f"shap_summary_{r.iloc[0]['run_name']}.png" if len(r) else None
                if p is not None and p.exists():
                    grid[(f, role, m)] = without_header(mpimg.imread(p))
    if grid:
        row_keys = [(f, role) for f in ("nofs", "score") for role in ("central", "federated")]
        aspect = [max([grid[(f, role, m)].shape[0] / grid[(f, role, m)].shape[1] for m in fams if (f, role, m) in grid], default=0.65)
                  for f, role in row_keys]
        left_in, right_in, top_in, bottom_in, gap_w, gap_h, gap_block = 0.34, 0.02, 0.2, 0.02, 0.04, 0.06, 0.16
        pw = (FULL_WIDTH_IN - left_in - right_in - 2 * gap_w) / len(fams)
        heights = [pw * a for a in aspect]
        gaps = [0, gap_h, gap_block, gap_h]
        fig_h = top_in + sum(heights) + sum(gaps) + bottom_in
        fig = plt.figure(figsize=(FULL_WIDTH_IN, fig_h))
        y_top = fig_h - top_in
        for i, ((f, role), h) in enumerate(zip(row_keys, heights)):
            y_top -= gaps[i]
            for j, m in enumerate(fams):
                ax = fig.add_axes([(left_in + j * (pw + gap_w)) / FULL_WIDTH_IN, (y_top - h) / fig_h, pw / FULL_WIDTH_IN, h / fig_h])
                ax.axis("off")
                if (f, role, m) in grid:
                    ax.imshow(grid[(f, role, m)], interpolation="none", aspect="equal")
                    ax.set_anchor("N")
                if i == 0:
                    ax.set_title(MODEL_LABEL[m], pad=3)
                if j == 0:
                    ax.text(-0.015, 0.5, f"({'abcd'[i]}) {'Pooled' if role == 'central' else 'Federated'}\n{FS_LABEL[f]}",
                            transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=7.5, multialignment="center")
            y_top -= h
        w.figure(fig, "FigS5_shap_beeswarms", "Figure S5. The framework's privacy-preserving SHAP summary plots (beeswarms) of the pooled and "
                 "federated models, basic imputation: (a) pooled and (b) federated with 33 features, (c) pooled and (d) federated with the 10 "
                 "features of the score test; columns are the model families. Rasters as logged by the framework (its title line cropped): "
                 "feature names are the registry variable codes (e.g. ps ecog = ECOG performance status, astGOT = AST, altGPT = ALT, nlr = "
                 "neutrophil-to-lymphocyte ratio, 'stent or biliary draige' = stent or biliary drainage), the 33-feature panels show the 20 most "
                 "important features (SHAP default), and feature values are binned into 20 bins with federated global boundaries (5th-95th "
                 "percentiles) in both arms.", where="supplementary")
    # S7 missingness per node
    if nodes:
        def pct_missing(d):
            return d[[c for c in FEATURES if c in d.columns]].isna().mean() * 100
        cols = [("Derivation", AREA_LABEL[a], pct_missing(nodes[a])) for a in AREAS if a in nodes]
        if "central" in nodes:
            cols.append(("Derivation", "Pooled", pct_missing(nodes["central"])))
        n_der = len(cols)
        if "area_federata" in cohort.columns:
            cols += [("Temporal validation", AREA_LABEL[a], pct_missing(cohort[cohort["area_federata"] == a]))
                     for a in AREAS if (cohort["area_federata"] == a).any()]
        cols.append(("Temporal validation", "All areas", pct_missing(cohort)))
        miss = pd.concat([s for _, _, s in cols], axis=1)
        miss = miss.reindex([c for c in FEATURES if c in miss.index])
        vmax = max(30, float(np.nanmax(miss.values)))
        keep = (miss.fillna(0) > 0).any(axis=1)
        dropped = [flabel(c) for c in miss.index[~keep]]
        miss = miss[keep]
        fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 0.155 * len(miss) + 1.5))
        im = ax.imshow(miss.values, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        for i, j in np.ndindex(miss.shape):
            v = miss.iat[i, j]
            if pd.notna(v) and v >= 5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.55 * vmax else "black")
        ax.set_xticks(range(miss.shape[1])); ax.set_xticklabels([lab for _, lab, _ in cols], rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(miss.shape[0])); ax.set_yticklabels([flabel(c) for c in miss.index])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axvline(n_der - 0.5, color="white", lw=4)  # derivation | temporal
        for xx in (n_der - 1.5, miss.shape[1] - 1.5):  # nodes | pooled, areas | all areas
            ax.axvline(xx, color="white", lw=1.2)
        for blk, x0, x1 in (("Derivation cohort (nodes, pooled)", 0, n_der - 1), ("Temporal validation cohort (areas, all)", n_der, miss.shape[1] - 1)):
            ax.annotate("", xy=(x0 - 0.4, 1.035), xytext=(x1 + 0.4, 1.035), xycoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6))
            ax.text((x0 + x1) / 2, 1.05, blk, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
        cb.set_label("Missing values (%)"); cb.outline.set_visible(False)
        cap = ("Figure S7. Share of missing values (%) per predictor in each derivation node, the pooled derivation cohort, the temporal "
               "validation cohort of each area and the whole temporal cohort; cells with at least 5% missing values are annotated.")
        if dropped:
            cap += " Predictors with no missing values in any cohort (not shown): " + ", ".join(dropped) + "."
        w.figure(fig, "FigS7_missingness", cap, where="supplementary")


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
