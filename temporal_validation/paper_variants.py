"""Alternative versions of the three-arm figure and two original figures, written to <final>/paper/variants/
(docs/PAPER_STRUCTURE.md, section F). All from the pipeline tables and the patient-level predictions; basic imputation
unless stated. The main-text choice is made by the authors; the others go to the supplement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_artifacts import AREAS, AREA_LABEL, ARM_COLOR, FS_LABEL, MODEL_LABEL, Writer

AREA_COLOR = {"americas": "#e6ab02", "asia": "#7570b3", "europe_non_italy": "#1b9e77", "italy_centre_south": "#d95f02", "italy_north": "#222222"}
ROWS12 = [(m, f, i) for m in ("cox", "gradient", "rsf") for f in ("nofs", "score") for i in ("basic", "mice")]
ROWS6 = [(m, f, "basic") for m in ("cox", "gradient", "rsf") for f in ("nofs", "score")]


def row_label(m, f, i, with_imp=True):
    s = f"{MODEL_LABEL[m]}, {'33' if f == 'nofs' else '10'} features"
    return s + (f", {i}" if with_imp else "")


def pick(runs, role, m, f, i, node=""):
    q = runs[(runs["role"] == role) & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == i)]
    if node:
        q = q[q["node"] == node]
    return q.iloc[0] if len(q) else None


def paired_row(paired, comparison, **kw):
    q = paired[paired["comparison"] == comparison]
    for k, v in kw.items():
        q = q[q[k] == v]
    return q


# ----------------------------------------------------------------------------- V1 ladder
def v1_ladder(w: Writer, runs, paired):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.5, 1, 1]}, sharey=True)
    y = np.arange(len(ROWS12))[::-1]
    for yi, (m, f, i) in zip(y, ROWS12):
        c, fe = pick(runs, "central", m, f, i), pick(runs, "federated", m, f, i)
        solo = runs[(runs["role"] == "standalone") & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == i)]
        ax = axes[0]
        if len(solo):
            ax.plot([solo["temporal_harrell_c"].min(), solo["temporal_harrell_c"].max()], [yi, yi], color="#bbbbbb", lw=5, solid_capstyle="butt", alpha=0.8, zorder=1)
        for r, role, mk, dy in ((c, "central", "s", 0.18), (fe, "federated", "o", -0.18)):
            if r is not None:
                ax.errorbar(r["temporal_harrell_c"], yi + dy, xerr=[[r["temporal_harrell_c"] - r["temporal_harrell_c_ci_low"]], [r["temporal_harrell_c_ci_high"] - r["temporal_harrell_c"]]],
                            fmt=mk, color=ARM_COLOR[role], ms=6, capsize=2.5, zorder=3)
        ax = axes[1]
        p = paired_row(paired, "arm", model=m, fs=f, imputation=i, subset="all")
        if len(p):
            p = p.iloc[0]
            ax.errorbar(p["delta"], yi, xerr=[[p["delta"] - p["ci_low"]], [p["ci_high"] - p["delta"]]], fmt="o", color=ARM_COLOR["federated"], ms=5, capsize=2.5)
        ax = axes[2]
        for a in AREAS:
            q = paired_row(paired, "standalone_vs_federated", model=m, fs=f, imputation=i, node=a, subset="all")
            if len(q):
                d = -q.iloc[0]["delta"]  # federated - node alone
                ax.plot([0, d], [yi, yi], color=AREA_COLOR[a], lw=2.2 if a == "italy_north" else 1.4, alpha=0.9, solid_capstyle="round")
                ax.plot(d, yi, "o", color=AREA_COLOR[a], ms=4 if a == "italy_north" else 3)
    axes[0].set_yticks(y); axes[0].set_yticklabels([row_label(*r) for r in ROWS12], fontsize=7.5)
    axes[0].set_xlim(0.50, 0.78); axes[0].set_xlabel("Harrell C, temporal cohort"); axes[0].set_title("(a) Pooled (square), federated (circle), range of the 5 node models (bar)", fontsize=9, loc="left")
    axes[1].axvline(0, color="grey", lw=0.6); axes[1].axvspan(-0.02, 0.02, color="#eeeeee", zorder=0); axes[1].set_xlim(-0.06, 0.06)
    axes[1].set_xlabel("federated - pooled (paired 95% CI)"); axes[1].set_title("(b) Federated vs pooled", fontsize=9, loc="left")
    axes[2].axvline(0, color="grey", lw=0.6); axes[2].set_xlim(-0.05, 0.25); axes[2].set_xlabel("federated - node alone, whole cohort")
    axes[2].set_title("(c) Federated vs each node alone", fontsize=9, loc="left")
    for a in AREAS:
        axes[2].plot([], [], color=AREA_COLOR[a], lw=2, label=AREA_LABEL[a])
    axes[2].legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axes:
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout()
    w.figure(fig, "V1_configuration_ladder", "Variant 1, configuration ladder. Twelve configurations (family x feature set x imputation). "
             "(a) Harrell C on the temporal cohort: pooled model (square, 95% CI), federated ensemble (circle, 95% CI) and the range of the five "
             "single-centre models (grey bar). (b) Paired bootstrap difference federated minus pooled with 95% CI; the grey band marks +-0.02. "
             "(c) Federated minus each single-centre model on the whole cohort, one colour per node (italy_north black).", where="variants")


# ----------------------------------------------------------------------------- V2 differences + gain map
def v2_differences(w: Writer, runs, paired):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    y = np.arange(len(ROWS12))[::-1]
    for yi, (m, f, i) in zip(y, ROWS12):
        c, fe = pick(runs, "central", m, f, i), pick(runs, "federated", m, f, i)
        p = paired_row(paired, "arm", model=m, fs=f, imputation=i, subset="all")
        if c is None or fe is None or not len(p):
            continue
        dd = fe["derivation_weighted_c"] - c["derivation_weighted_c"]; p = p.iloc[0]
        ax.annotate("", xy=(p["delta"], yi), xytext=(dd, yi), arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.9, shrinkA=4, shrinkB=4))
        ax.plot(dd, yi, "o", mfc="white", mec=ARM_COLOR["federated"], ms=6)
        ax.errorbar(p["delta"], yi, xerr=[[p["delta"] - p["ci_low"]], [p["ci_high"] - p["delta"]]], fmt="o", color=ARM_COLOR["federated"], ms=6, capsize=2.5)
    ax.set_yticks(y); ax.set_yticklabels([row_label(*r) for r in ROWS12], fontsize=7.5)
    ax.axvline(0, color="grey", lw=0.6); ax.axvspan(-0.02, 0.02, color="#eeeeee", zorder=0); ax.set_xlim(-0.06, 0.06)
    ax.set_xlabel("federated - pooled\nderivation CV (hollow) -> temporal cohort (filled, paired 95% CI)")
    ax.set_title("(a) The derivation gap does not travel", fontsize=10, loc="left")
    ax = axes[1]
    node_n = {}
    for a in AREAS:
        q = paired_row(paired, "standalone_vs_federated", node=a, subset=a)
        node_n[a] = int(q["n"].max()) if len(q) else 0  # temporal patients of the node's own area
    order = sorted(AREAS, key=lambda a: -node_n.get(a, 0))
    yy = {a: k for k, a in enumerate(order[::-1])}
    markers = {"cox": "o", "gradient": "s", "rsf": "^"}
    for a in order:
        for (m, f, i) in ROWS6:
            for subset, dx, alpha in (("own", 0, 1.0), ("all", 0, 0.45)):
                q = paired_row(paired, "standalone_vs_federated", model=m, fs=f, imputation=i, node=a)
                q = q[q["subset"] == "all"] if subset == "all" else q[q["subset"] != "all"]
                if len(q):
                    q = q.iloc[0]; d = -q["delta"]
                    jitter = {"cox": -0.22, "gradient": 0, "rsf": 0.22}[m] + (0.08 if f == "score" else -0.08)
                    ax.errorbar(d, yy[a] + jitter, xerr=[[q["ci_high"] - q["delta"]], [q["delta"] - q["ci_low"]]], fmt=markers[m], mfc=ARM_COLOR["federated"] if f == "score" else "white",
                                mec=ARM_COLOR["federated"], ecolor="#cccccc", ms=5, capsize=0, alpha=alpha, lw=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{AREA_LABEL[a]}\n(n = {node_n[a]} temporal)" for a in order[::-1]], fontsize=8)
    ax.axvline(0, color="grey", lw=0.6); ax.set_xlim(-0.06, 0.20); ax.set_xlabel("federated - node alone (paired 95% CI)\nown area (opaque), whole cohort (faded)")
    ax.set_title("(b) What the federation gives each node", fontsize=10, loc="left")
    for m, mk in markers.items():
        ax.plot([], [], mk, mec=ARM_COLOR["federated"], mfc="white", label=f"{MODEL_LABEL[m]} (hollow 33, filled 10 features)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axes:
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout()
    w.figure(fig, "V2_differences_and_gain_map", "Variant 2, differences only. (a) Federated minus pooled Harrell C for the twelve configurations: "
             "hollow = derivation cross-validation, filled = temporal cohort with paired 95% CI; the arrow joins the two deficits (two cohorts, "
             "not one trajectory); grey band +-0.02. (b) Federated minus each node's own model (basic imputation, six configurations per node) "
             "on the node's own temporal area (opaque) and on the whole cohort (faded); nodes ordered by size.", where="variants")


# ----------------------------------------------------------------------------- V3 per-area dumbbells
def v3_dumbbells(w: Writer, runs, paired):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    rows = AREAS + ["all"]
    for ax, m in zip(axes, ("cox", "gradient", "rsf")):
        f = "score" if m == "cox" else "nofs"  # best feature set per family on the temporal cohort (stated in the caption)
        c, fe = pick(runs, "central", m, f, "basic"), pick(runs, "federated", m, f, "basic")
        for k, a in enumerate(rows):
            yk = len(rows) - 1 - k
            col = "temporal_harrell_c" if a == "all" else f"temporal_c_{a}"
            if c is not None:
                ax.plot(c[col], yk, "|", color=ARM_COLOR["central"], ms=14, mew=2)
            if fe is not None:
                ax.plot(fe[col], yk, "o", color=ARM_COLOR["federated"], ms=6, zorder=3)
            if a != "all":
                s = pick(runs, "standalone", m, f, "basic", node=a)
                if s is not None and fe is not None:
                    ax.plot([s[col], fe[col]], [yk, yk], color="#999999", lw=1.2, zorder=1)
                    ax.plot(s[col], yk, "o", mfc="white", mec="#555555", ms=6, zorder=2)
                    q = paired_row(paired, "standalone_vs_federated", model=m, fs=f, imputation="basic", node=a, subset=a)
                    if len(q):
                        q = q.iloc[0]
                        ax.text(0.99, yk, f"{-q['delta']:+.3f} [{-q['ci_high']:+.3f}, {-q['ci_low']:+.3f}]", fontsize=6.5, va="center", ha="right", color="#333333",
                                transform=ax.get_yaxis_transform())
        ax.set_title(f"{MODEL_LABEL[m]}, {FS_LABEL[f]}", fontsize=9)
        ax.set_xlim(0.45, 0.98); ax.set_xticks([0.5, 0.6, 0.7, 0.8]); ax.axvline(0.5, color="grey", lw=0.5, ls=":"); ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.set_xlabel("Harrell C on the temporal patients of the area")
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([AREA_LABEL.get(a, "whole cohort") for a in rows[::-1]], fontsize=8)
    axes[0].plot([], [], "o", mfc="white", mec="#555555", label="node alone"); axes[0].plot([], [], "o", color=ARM_COLOR["federated"], label="federated")
    axes[0].plot([], [], "|", color=ARM_COLOR["central"], ms=12, mew=2, label="pooled"); axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    w.figure(fig, "V3_per_area_dumbbells", "Variant 3, per-area dumbbells. For each model family (its best feature set on the temporal cohort, basic "
             "imputation) and each temporal area: the node's own model (hollow), the federated ensemble (filled) and the pooled model (tick) on the "
             "area's patients; the printed value is the paired difference federated minus node alone with its 95% CI.", where="variants")


# ----------------------------------------------------------------------------- E1 federation ledger
def e1_ledger(w: Writer, final: Path, runs, paired, agree, m="cox", f="score"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True, gridspec_kw={"width_ratios": [1.2, 1, 1]})
    node_n = {a: int(paired_row(paired, "standalone_vs_federated", node=a, subset=a)["n"].max()) if len(paired_row(paired, "standalone_vs_federated", node=a, subset=a)) else 0 for a in AREAS}
    order = sorted(AREAS, key=lambda a: -node_n[a])
    fe, c = pick(runs, "federated", m, f, "basic"), pick(runs, "central", m, f, "basic")
    for k, a in enumerate(order):
        yk = len(order) - 1 - k
        s = pick(runs, "standalone", m, f, "basic", node=a)
        if s is None or fe is None:
            continue
        # (A) C on own area
        ax = axes[0]; col = f"temporal_c_{a}"
        ax.plot([s[col], fe[col]], [yk, yk], color="#999999", lw=1.2); ax.plot(s[col], yk, "o", mfc="white", mec="#555555", ms=6); ax.plot(fe[col], yk, "o", color=ARM_COLOR["federated"], ms=6)
        if c is not None:
            ax.plot(c[col], yk, "|", color=ARM_COLOR["central"], ms=12, mew=2)
        # (B) slope at 12 months
        ax = axes[1]
        ax.plot([s["temporal_calib_slope_12"], fe["temporal_calib_slope_12"]], [yk, yk], color="#999999", lw=1.2)
        ax.plot(s["temporal_calib_slope_12"], yk, "o", mfc="white", mec="#555555", ms=6); ax.plot(fe["temporal_calib_slope_12"], yk, "o", color=ARM_COLOR["federated"], ms=6)
        if c is not None:
            ax.plot(c["temporal_calib_slope_12"], yk, "|", color=ARM_COLOR["central"], ms=12, mew=2)
        # (C) ranking agreement with pooled
        ax = axes[2]
        qa = agree[(agree["comparison"] == "standalone_vs_central") & (agree["node"] == a) & (agree["model"] == m) & (agree["fs"] == f) & (agree["imputation"] == "basic")]
        qf = agree[(agree["comparison"] == "arm") & (agree["model"] == m) & (agree["fs"] == f) & (agree["imputation"] == "basic")]
        if len(qa) and len(qf):
            ax.plot([qa.iloc[0]["spearman"], qf.iloc[0]["spearman"]], [yk, yk], color="#999999", lw=1.2)
            ax.plot(qa.iloc[0]["spearman"], yk, "o", mfc="white", mec="#555555", ms=6); ax.plot(qf.iloc[0]["spearman"], yk, "o", color=ARM_COLOR["federated"], ms=6)
            ax.text(1.02, yk, f"top-4: {int(qa.iloc[0]['top4_overlap'])}/4 -> {int(qf.iloc[0]['top4_overlap'])}/4", fontsize=6.5, va="center")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([f"{AREA_LABEL[a]}\n(n = {node_n[a]} temporal)" for a in order[::-1]], fontsize=8)
    axes[0].set_xlabel("Harrell C on the node's own area"); axes[0].set_title("(A) Discrimination at home", fontsize=9, loc="left"); axes[0].set_xlim(0.4, 0.85)
    axes[1].set_xscale("log"); axes[1].set_xticks([0.2, 0.5, 1, 2, 5]); axes[1].set_xticklabels(["0.2", "0.5", "1", "2", "5"]); axes[1].minorticks_off()
    axes[1].axvline(1, color="grey", lw=0.6, ls="--"); axes[1].set_xlabel("calibration slope at 12 months (whole cohort)"); axes[1].set_title("(B) Calibration scale", fontsize=9, loc="left")
    axes[2].set_xlim(0, 1.0); axes[2].set_xlabel("Spearman with the pooled feature ranking"); axes[2].set_title("(C) Selection agreement", fontsize=9, loc="left")
    axes[0].plot([], [], "o", mfc="white", mec="#555555", label="node alone"); axes[0].plot([], [], "o", color=ARM_COLOR["federated"], label="federated")
    axes[0].plot([], [], "|", color=ARM_COLOR["central"], ms=12, mew=2, label="pooled"); axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    for ax in axes:
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.suptitle(f"What the federation gives each node: {MODEL_LABEL[m]}, {FS_LABEL[f]}, basic imputation", fontsize=10)
    fig.tight_layout()
    w.figure(fig, f"E1_federation_ledger_{m}_{f}", "Federation ledger. For each node (rows ordered by size), the move from its own model (hollow) to "
             "the federated ensemble (filled), with the pooled model as a tick: (A) Harrell C on the node's own temporal area; (B) calibration "
             "slope at 12 months on the whole cohort (log scale, 1 = ideal); (C) Spearman agreement of the feature ranking with the pooled "
             "ranking (node alone vs federated), with the top-4 overlap printed.", where="variants")


# ----------------------------------------------------------------------------- E2 model kinship map
def e2_kinship(w: Writer, final: Path, runs):
    import matplotlib.pyplot as plt
    sel = []
    for role in ("central", "federated"):
        for (m, f, i) in ROWS6:
            r = pick(runs, role, m, f, i)
            if r is not None:
                sel.append((r["run_name"], f"{'pooled' if role == 'central' else 'federated'}: {MODEL_LABEL[m]} {'33' if f == 'nofs' else '10'}", role))
    for a in AREAS:
        for (m, f, i) in ROWS6:
            r = pick(runs, "standalone", m, f, i, node=a)
            if r is not None:
                sel.append((r["run_name"], f"{AREA_LABEL[a]}: {MODEL_LABEL[m]} {'33' if f == 'nofs' else '10'}", a))
    risks = []
    for name, _, _ in sel:
        p = final / "runs" / name / "predictions.csv"
        if not p.exists():
            return
        risks.append(pd.read_csv(p)["risk_global"].rank().to_numpy(float))
    R = np.corrcoef(np.vstack(risks))
    labels = [s[1] for s in sel]; groups = [s[2] for s in sel]
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(R, cmap="viridis", vmin=0.4, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=5.5); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5.5)
    bounds = [k for k in range(1, len(groups)) if groups[k] != groups[k - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=2); ax.axvline(b - 0.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Spearman correlation of patient risk scores (698 temporal patients)")
    # block means
    starts = [0] + bounds + [len(groups)]
    for gi in range(len(starts) - 1):
        for gj in range(len(starts) - 1):
            blk = R[starts[gi]:starts[gi + 1], starts[gj]:starts[gj + 1]]
            if gi == gj:
                v = blk[np.triu_indices_from(blk, 1)].mean() if blk.shape[0] > 1 else 1.0
            else:
                v = blk.mean()
            ax.text(starts[gj], starts[gi] - 0.6 if gi > 0 else -0.6, f"{v:.2f}", fontsize=6, color="black", ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title("Model kinship: rank agreement of the risk scores of every model on the same patients (block means printed)", fontsize=9, pad=16)
    fig.tight_layout()
    w.figure(fig, "E2_model_kinship_map", "Model kinship map. Spearman correlation between the risk scores assigned to the same 698 temporal patients by "
             "every final model (basic imputation): six pooled, six federated and thirty single-centre models, grouped by arm and node (white "
             "separators; block means printed). Pooled and federated models of the same family agree with each other as closely as two pooled "
             "models of different families do; single-centre models form isolated blocks.", where="variants")


# ----------------------------------------------------------------------------- E3 discrimination-calibration plane
def e3_plane(w: Writer, runs):
    import matplotlib.pyplot as plt
    main = runs[runs["fs"].isin(["nofs", "score"])]
    markers = {"cox": "o", "gradient": "s", "rsf": "^"}
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for role in ("standalone", "central", "federated"):
        sub = main[main["role"] == role]
        for m, mk in markers.items():
            s = sub[sub["model"] == m]
            ax.scatter(s["temporal_harrell_c"], s["temporal_calib_slope_12"], marker=mk, s=28 if role != "standalone" else 16, color=ARM_COLOR[role],
                       alpha=0.9 if role != "standalone" else 0.6, label=f"{ {'central': 'pooled', 'federated': 'federated', 'standalone': 'node alone'}[role] }, {MODEL_LABEL[m]}" if True else None)
    ax.set_yscale("log"); ax.set_yticks([0.2, 0.5, 1, 2, 5]); ax.set_yticklabels(["0.2", "0.5", "1", "2", "5"]); ax.minorticks_off()
    ax.axhline(1, color="grey", lw=0.6, ls="--"); ax.axhspan(0.8, 1.25, color="#eeeeee", zorder=0)
    ax.set_xlabel("Harrell C on the temporal cohort"); ax.set_ylabel("calibration slope at 12 months (1 = ideal; > 1 = risks too moderate)")
    ax.set_title("Discrimination and calibration of every model: arms as colours, families as shapes", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    w.figure(fig, "E3_discrimination_calibration_plane", "Discrimination-calibration plane. Every model of the grid (33- and 10-feature, both imputations) "
             "by its temporal Harrell C and its 12-month calibration slope; colour = arm, shape = family; grey band = slope within 0.8-1.25. "
             "The federated ensembles sit at pooled discrimination with slopes above 1; single-centre models scatter in both directions.", where="variants")


def build_variants(final: Path, runs: pd.DataFrame, paired: pd.DataFrame, agree: pd.DataFrame, w: Writer):
    (w.root / "variants").mkdir(exist_ok=True)
    for step in (lambda: v1_ladder(w, runs, paired), lambda: v2_differences(w, runs, paired), lambda: v3_dumbbells(w, runs, paired),
                 lambda: e1_ledger(w, final, runs, paired, agree, "cox", "score"), lambda: e1_ledger(w, final, runs, paired, agree, "rsf", "nofs"),
                 lambda: e2_kinship(w, final, runs), lambda: e3_plane(w, runs)):
        try:
            step()
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger("paper_variants").warning("variant failed: %s", e, exc_info=True)
