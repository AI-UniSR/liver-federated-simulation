"""Alternative versions of the three-arm figure and two original figures, written to <final>/paper/variants/
(docs/PAPER_STRUCTURE.md, section F). All from the pipeline tables and the patient-level predictions; basic imputation
unless stated. The main-text choice is made by the authors; the others go to the supplement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_artifacts import AREAS, AREA_COLOR, AREA_LABEL, ARM_COLOR, FS_LABEL, MODEL_LABEL, Writer

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
AREA_MARKER = {"americas": "^", "asia": "D", "europe_non_italy": "s", "italy_centre_south": "v", "italy_north": "o"}
IMP_LABEL = {"basic": "basic", "mice": "MICE"}
FAMILY_2LINE = {"cox": "Cox", "gradient": "Gradient\nboosting", "rsf": "Random\nsurvival forest"}
MINUS = "−"


def _node_sizes(paired):
    """Temporal patients of each node's own area, and the nodes ordered by it (largest first)."""
    n = {}
    for a in AREAS:
        q = paired_row(paired, "standalone_vs_federated", node=a, subset=a)
        n[a] = int(q["n"].max()) if len(q) else 0
    return n, sorted(AREAS, key=lambda a: -n[a])


def _grouped_config_axis(ax, y, rows=ROWS12, offset=36):
    """Short tick labels (features, imputation), the family name written once per block, thin separators."""
    from matplotlib.transforms import blended_transform_factory
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'33' if f == 'nofs' else '10'}, {IMP_LABEL[i]}" for (m, f, i) in rows])
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for m in dict.fromkeys(r[0] for r in rows):
        ys = [yi for yi, r in zip(y, rows) if r[0] == m]
        ax.annotate(FAMILY_2LINE[m], xy=(0, float(np.mean(ys))), xycoords=tr, xytext=(-offset, 0), textcoords="offset points",
                    ha="right", va="center", fontweight="bold", fontsize=6.5)
    ax.annotate("features, imputation", xy=(0, float(max(y)) + 0.55), xycoords=tr, xytext=(-4, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=6, color="#555555", annotation_clip=False)


def _family_separators(ax, y, rows=ROWS12):
    for k in range(1, len(rows)):
        if rows[k][0] != rows[k - 1][0]:
            ax.axhline((y[k] + y[k - 1]) / 2, color="#cccccc", lw=0.5, zorder=0.5)


def v1_ladder(w: Writer, runs, paired):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from paper_style import FULL_WIDTH_IN
    node_n, order = _node_sizes(paired)
    off = {a: 0.28 - 0.14 * k for k, a in enumerate(order)}  # within a row, the largest node on top
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 4.4), gridspec_kw={"width_ratios": [1.4, 1, 1]}, sharey=True)
    y = np.arange(len(ROWS12))[::-1]
    for yi, (m, f, i) in zip(y, ROWS12):
        c, fe = pick(runs, "central", m, f, i), pick(runs, "federated", m, f, i)
        solo = runs[(runs["role"] == "standalone") & (runs["model"] == m) & (runs["fs"] == f) & (runs["imputation"] == i)]
        ax = axes[0]
        if len(solo):
            ax.plot([solo["temporal_harrell_c"].min(), solo["temporal_harrell_c"].max()], [yi, yi], color="#c4c4c4", lw=3.5, solid_capstyle="butt", zorder=1)
        for r, role, mk, dy in ((c, "central", "s", 0.22), (fe, "federated", "o", -0.22)):
            if r is not None:
                ax.errorbar(r["temporal_harrell_c"], yi + dy, xerr=[[r["temporal_harrell_c"] - r["temporal_harrell_c_ci_low"]], [r["temporal_harrell_c_ci_high"] - r["temporal_harrell_c"]]],
                            fmt=mk, color=ARM_COLOR[role], ms=3.5, capsize=1.5, elinewidth=0.8, capthick=0.8, zorder=3)
        ax = axes[1]
        p = paired_row(paired, "arm", model=m, fs=f, imputation=i, subset="all")
        if len(p):
            p = p.iloc[0]
            ax.errorbar(p["delta"], yi, xerr=[[p["delta"] - p["ci_low"]], [p["ci_high"] - p["delta"]]], fmt="o", color=ARM_COLOR["federated"], ms=3.5,
                        capsize=1.5, elinewidth=0.8, capthick=0.8)
        ax = axes[2]
        for a in AREAS:
            q = paired_row(paired, "standalone_vs_federated", model=m, fs=f, imputation=i, node=a, subset="all")
            if len(q):
                d = -q.iloc[0]["delta"]  # federated - node alone
                ax.plot(d, yi + off[a], AREA_MARKER[a], color=AREA_COLOR[a], ms=3.2, mew=0, zorder=3)
    _grouped_config_axis(axes[0], y)
    for ax in axes:
        _family_separators(ax, y)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].set_ylim(-0.6, len(ROWS12) - 0.4)
    axes[0].set_xlim(0.50, 0.78); axes[0].set_xlabel("Harrell C, temporal cohort (95% CI)"); axes[0].set_title("(a) Temporal Harrell C", loc="left")
    axes[0].legend(handles=[Line2D([], [], marker="s", ls="none", color=ARM_COLOR["central"], ms=3.5, label="pooled"),
                            Line2D([], [], marker="o", ls="none", color=ARM_COLOR["federated"], ms=3.5, label="federated"),
                            Line2D([], [], color="#c4c4c4", lw=3.5, solid_capstyle="butt", label="range of the 5 node-alone models")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1, handlelength=1.6)
    axes[1].axvline(0, color="grey", lw=0.6); axes[1].axvspan(-0.02, 0.02, color="#eeeeee", zorder=0); axes[1].set_xlim(-0.06, 0.06)
    axes[1].set_xlabel(f"ΔC federated {MINUS} pooled\n(paired 95% CI)"); axes[1].set_title(f"(b) Federated {MINUS} pooled", loc="left")
    axes[2].axvline(0, color="grey", lw=0.6); axes[2].set_xlim(-0.04, 0.22)
    axes[2].set_xlabel(f"ΔC federated {MINUS} node alone\n(whole temporal cohort)"); axes[2].set_title(f"(c) Federated {MINUS} node alone", loc="left")
    axes[2].legend(handles=[Line2D([], [], marker=AREA_MARKER[a], ls="none", color=AREA_COLOR[a], mew=0, ms=3.5, label=AREA_LABEL[a]) for a in order],
                   loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, handlelength=1, columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    w.figure(fig, "V1_configuration_ladder", "Variant 1, configuration ladder. Twelve configurations (model family; 33 or 10 features; basic or MICE "
             "imputation). (a) Harrell C on the temporal cohort: pooled model (square, 95% CI), federated ensemble (circle, 95% CI) and the range of "
             "the five single-centre models (grey bar). (b) Paired bootstrap difference federated minus pooled with 95% CI; the grey band marks "
             "±0.02. (c) Federated minus each single-centre model on the whole temporal cohort, one colour and marker per node (Italy north black "
             "circle); within each row the nodes are stacked by size, largest on top.", where="variants")


# ----------------------------------------------------------------------------- V2 differences + gain map
def v2_differences(w: Writer, runs, paired):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from paper_style import FULL_WIDTH_IN
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 4.4), gridspec_kw={"width_ratios": [1, 1.05]})
    fed = ARM_COLOR["federated"]
    ax = axes[0]
    y = np.arange(len(ROWS12))[::-1]
    for yi, (m, f, i) in zip(y, ROWS12):
        c, fe = pick(runs, "central", m, f, i), pick(runs, "federated", m, f, i)
        p = paired_row(paired, "arm", model=m, fs=f, imputation=i, subset="all")
        if c is None or fe is None or not len(p):
            continue
        dd = fe["derivation_weighted_c"] - c["derivation_weighted_c"]; p = p.iloc[0]
        ax.annotate("", xy=(p["delta"], yi), xytext=(dd, yi), arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.7, shrinkA=3, shrinkB=3, mutation_scale=6))
        ax.plot(dd, yi, "o", mfc="white", mec=fed, ms=3.5, mew=0.8, zorder=3)
        ax.errorbar(p["delta"], yi, xerr=[[p["delta"] - p["ci_low"]], [p["ci_high"] - p["delta"]]], fmt="o", color=fed, ms=3.5, capsize=1.5,
                    elinewidth=0.8, capthick=0.8, zorder=3)
    _grouped_config_axis(ax, y)
    _family_separators(ax, y)
    ax.set_ylim(-0.6, len(ROWS12) - 0.4)
    ax.axvline(0, color="grey", lw=0.6); ax.axvspan(-0.02, 0.02, color="#eeeeee", zorder=0); ax.set_xlim(-0.06, 0.06)
    ax.set_xlabel(f"ΔC federated {MINUS} pooled")
    ax.set_title(f"(a) Federated {MINUS} pooled", loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec=fed, ms=3.5, label="derivation cross-validation"),
                       Line2D([], [], marker="o", ls="-", color=fed, ms=3.5, lw=0.8, label="temporal cohort (paired 95% CI)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1, handlelength=1.6)
    ax = axes[1]
    node_n, order = _node_sizes(paired)
    yy = {a: k for k, a in enumerate(order[::-1])}
    markers = {"cox": "o", "gradient": "s", "rsf": "^"}
    slots = [(m, f) for m in ("cox", "gradient", "rsf") for f in ("nofs", "score")]
    xlo, xhi = -0.10, 0.25
    for a in order:
        for (m, f, i) in ROWS6:
            q = paired_row(paired, "standalone_vs_federated", model=m, fs=f, imputation=i, node=a)
            q = q[q["subset"] != "all"]  # the node's own temporal area (whole-cohort differences: Supplementary Fig. 9c)
            if not len(q):
                continue
            q = q.iloc[0]; d = -q["delta"]; lo, hi = -q["ci_high"], -q["ci_low"]
            yk = yy[a] + 0.3 - 0.12 * slots.index((m, f))
            ax.plot([max(lo, xlo), min(hi, xhi)], [yk, yk], color=fed, lw=0.6, alpha=0.75, solid_capstyle="butt", zorder=2)
            if lo < xlo:
                ax.plot(xlo, yk, "<", color=fed, ms=2.5, mew=0, clip_on=False, zorder=2)
            if hi > xhi:
                ax.plot(xhi, yk, ">", color=fed, ms=2.5, mew=0, clip_on=False, zorder=2)
            ax.plot(d, yk, markers[m], mfc=fed if f == "score" else "white", mec=fed, ms=3.5, mew=0.8, zorder=3)
    for k in range(1, len(order)):
        ax.axhline(k - 0.5, color="#cccccc", lw=0.5, zorder=0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{AREA_LABEL[a]}\nn = {node_n[a]}" for a in order[::-1]])
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.axvline(0, color="grey", lw=0.6); ax.set_xlim(xlo, xhi)
    ax.set_xlabel(f"ΔC federated {MINUS} node alone, node's own area\n(paired 95% CI)")
    ax.set_title(f"(b) Federated {MINUS} node alone", loc="left")
    handles = [Line2D([], [], marker=mk, ls="none", mfc="white", mec="#555555", ms=3.5, mew=0.8, label=MODEL_LABEL[m]) for m, mk in markers.items()]
    handles += [Line2D([], [], marker="o", ls="none", mfc="white", mec=fed, ms=3.5, mew=0.8, label="33 features (hollow)"),
                Line2D([], [], marker="o", ls="none", mfc=fed, mec=fed, ms=3.5, mew=0.8, label="10 features (filled)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, handlelength=1, columnspacing=1.0)
    for ax in axes:
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout(w_pad=1.0)
    w.figure(fig, "V2_differences_and_gain_map", "Variant 2, differences only. (a) Federated minus pooled Harrell C for the twelve configurations "
             "(model family; 33 or 10 features; basic or MICE imputation): hollow = derivation cross-validation, filled = temporal cohort with "
             "paired 95% CI; the arrow joins the two deficits (two cohorts, not one trajectory); grey band ±0.02. (b) Federated minus each "
             "node's own model (basic imputation, six configurations per node: shape = family, hollow = 33 features, filled = 10 features) on the "
             "node's own temporal area, with paired 95% CI; nodes ordered by size (n = temporal patients of the area); arrowheads mark intervals "
             "that continue beyond the axis. The same differences on the whole temporal cohort are in Supplementary Fig. 9c.", where="variants")


# ----------------------------------------------------------------------------- V3 per-area dumbbells
def _fmt_signed(v):
    return f"{v:+.3f}".replace("-", MINUS)


def v3_dumbbells(w: Writer, runs, paired):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from paper_style import FULL_WIDTH_IN
    node_n, order = _node_sizes(paired)
    rows = order + ["all"]  # top to bottom: nodes by size, then the whole cohort
    ypos = {a: len(rows) - 1 - k for k, a in enumerate(rows)}
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.4))
    # explicit margins (tight_layout does not handle the shared, axis-less text columns); savefig crops to the content
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.36], hspace=0.42, wspace=0.02, left=0.215, right=0.99, top=0.95, bottom=0.14)
    first = None
    for k, m in enumerate(("cox", "gradient", "rsf")):
        ax = fig.add_subplot(gs[k, 0], sharex=first)
        first = first or ax
        tx = fig.add_subplot(gs[k, 1], sharey=ax)
        tx.axis("off"); tx.set_xlim(0, 1)
        f = "score" if m == "cox" else "nofs"  # best feature set per family on the temporal cohort (stated in the caption)
        c, fe = pick(runs, "central", m, f, "basic"), pick(runs, "federated", m, f, "basic")
        for a in rows:
            yk = ypos[a]
            col = "temporal_harrell_c" if a == "all" else f"temporal_c_{a}"
            if c is not None:
                ax.plot(c[col], yk, "|", color=ARM_COLOR["central"], ms=8, mew=1.5, zorder=4)
            if fe is not None:
                ax.plot(fe[col], yk, "o", color=ARM_COLOR["federated"], ms=4, zorder=3)
            if a != "all":
                s = pick(runs, "standalone", m, f, "basic", node=a)
                if s is not None and fe is not None:
                    ax.plot([s[col], fe[col]], [yk, yk], color="#999999", lw=1.0, zorder=1)
                    ax.plot(s[col], yk, "o", mfc="white", mec="#555555", ms=4, mew=0.8, zorder=2)
                    q = paired_row(paired, "standalone_vs_federated", model=m, fs=f, imputation="basic", node=a, subset=a)
                    if len(q):
                        q = q.iloc[0]
                        tx.text(0.04, yk, f"{_fmt_signed(-q['delta'])} [{_fmt_signed(-q['ci_high'])}, {_fmt_signed(-q['ci_low'])}]",
                                fontsize=6, va="center", ha="left", color="#333333")
        if k == 0:
            tx.text(0.04, 1.0, f"ΔC federated {MINUS} node alone\n[95% CI], own area", transform=tx.transAxes, fontsize=6, va="bottom",
                    ha="left", color="#555555")
        ax.set_title(f"({'abc'[k]}) {MODEL_LABEL[m]}, {FS_LABEL[f]}", loc="left")
        ax.set_yticks([ypos[a] for a in rows])
        ax.set_yticklabels([f"{AREA_LABEL[a]}, n = {node_n[a]}" if a != "all" else "whole cohort" for a in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.axhline(0.5, color="#cccccc", lw=0.5, zorder=0)
        ax.set_xlim(0.45, 0.85); ax.set_xticks([0.5, 0.6, 0.7, 0.8]); ax.axvline(0.5, color="grey", lw=0.5, ls=":"); ax.grid(axis="x", lw=0.3, alpha=0.5)
        if k < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Harrell C on the temporal patients of the area")
            ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555", ms=4, mew=0.8, label="node alone"),
                               Line2D([], [], marker="o", ls="none", color=ARM_COLOR["federated"], ms=4, label="federated"),
                               Line2D([], [], marker="|", ls="none", color=ARM_COLOR["central"], ms=8, mew=1.5, label="pooled")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    w.figure(fig, "V3_per_area_dumbbells", "Variant 3, per-area dumbbells. For each model family ((a) Cox, (b) gradient boosting, (c) random "
             "survival forest; its best feature set on the temporal cohort, basic imputation) and each temporal area (nodes ordered by size; n = "
             "temporal patients of the area): the node's own model (hollow), the federated ensemble (filled) and the pooled model (tick) on the "
             "area's patients; the bottom row is the whole temporal cohort. The column on the right gives the paired difference federated minus "
             "node alone on the area's patients with its 95% CI.", where="variants")


# ----------------------------------------------------------------------------- E1 federation ledger
def _e1_overlap_text(qa_by_node: dict, qf) -> str:
    """Caption sentence for the top-4 overlap with the pooled ranking (one value if all nodes share it)."""
    if qf is None or not qa_by_node:
        return ""
    vals = {a: int(r["top4_overlap"]) for a, r in qa_by_node.items()}
    if len(set(vals.values())) == 1:
        node_txt = f"{next(iter(vals.values()))}/4 for each node-alone model"
    else:
        node_txt = ", ".join(f"{AREA_LABEL[a]} {v}/4" for a, v in vals.items())
    return f" Top-4 overlap with the pooled ranking: {node_txt}, {int(qf['top4_overlap'])}/4 for the federated model."


def e1_ledger(w: Writer, final: Path, runs, paired, agree, m="cox", f="score"):
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from paper_style import FULL_WIDTH_IN
    node_n = {a: int(paired_row(paired, "standalone_vs_federated", node=a, subset=a)["n"].max()) if len(paired_row(paired, "standalone_vs_federated", node=a, subset=a)) else 0 for a in AREAS}
    order = sorted(AREAS, key=lambda a: -node_n[a])
    fe, c = pick(runs, "federated", m, f, "basic"), pick(runs, "central", m, f, "basic")
    # selection agreement exists only for the selected (10-feature) sets; a 33-feature model ranks no features
    sel = agree[(agree["model"] == m) & (agree["fs"] == f) & (agree["imputation"] == "basic")] if len(agree) else agree
    qf = sel[sel["comparison"] == "arm"] if len(sel) else sel
    qf = qf.iloc[0] if len(qf) else None
    qa_by_node = {}
    for a in order:
        q = sel[(sel["comparison"] == "standalone_vs_central") & (sel["node"] == a)] if len(sel) else sel
        if len(q):
            qa_by_node[a] = q.iloc[0]
    has_sel = qf is not None and len(qa_by_node) > 0
    npan = 3 if has_sel else 2
    fig, axes = plt.subplots(1, npan, figsize=(FULL_WIDTH_IN, 2.5), sharey=True, gridspec_kw={"width_ratios": [1.25, 1, 1][:npan]})
    node_kw = dict(ls="none", marker="o", mfc="none", mec="#444444", mew=0.9, ms=5, zorder=4)  # hollow, drawn on top so it shows when it coincides
    fed_kw = dict(ls="none", marker="o", color=ARM_COLOR["federated"], ms=4, zorder=3)
    pool_kw = dict(ls="none", marker="|", color=ARM_COLOR["central"], ms=8, mew=1.6, zorder=2)
    link_kw = dict(color="#b0b0b0", lw=1.0, zorder=1)
    slope = "temporal_calib_slope_12"
    for k, a in enumerate(order):
        yk = len(order) - 1 - k
        s = pick(runs, "standalone", m, f, "basic", node=a)
        if s is None or fe is None:
            continue
        # (a) C on the node's own area: every arm is evaluated on the same area, so all three vary by row
        ax = axes[0]; col = f"temporal_c_{a}"
        ax.plot([s[col], fe[col]], [yk, yk], **link_kw); ax.plot(fe[col], yk, **fed_kw); ax.plot(s[col], yk, **node_kw)
        if c is not None:
            ax.plot(c[col], yk, **pool_kw)
        # (b) slope at 12 months on the whole cohort: only the node-alone value varies by row
        ax = axes[1]
        ax.plot([s[slope], fe[slope]], [yk, yk], **link_kw); ax.plot(s[slope], yk, **node_kw)
        # (c) ranking agreement with pooled: only the node-alone value varies by row
        if has_sel and a in qa_by_node:
            ax = axes[2]
            ax.plot([qa_by_node[a]["spearman"], qf["spearman"]], [yk, yk], **link_kw); ax.plot(qa_by_node[a]["spearman"], yk, **node_kw)
    # whole-cohort federated / pooled values: one reference line each
    if fe is not None:
        axes[1].axvline(fe[slope], color=ARM_COLOR["federated"], lw=1.2, zorder=2)
    if c is not None:
        axes[1].axvline(c[slope], color=ARM_COLOR["central"], lw=1.2, ls=(0, (3, 1.5)), zorder=2)
    axes[1].axvline(1, color="#666666", lw=0.6, ls=":", zorder=0)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([f"{AREA_LABEL[a]}\n(n = {node_n[a]})" for a in order[::-1]])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    axes[0].set_xlabel("Harrell C on the node's own\ntemporal area"); axes[0].set_title("(a) Discrimination on own area", loc="left"); axes[0].set_xlim(0.45, 0.85)
    axes[0].set_xticks([0.5, 0.6, 0.7, 0.8])
    axes[1].set_xscale("log"); axes[1].set_xlim(0.1, 5); axes[1].set_xticks([0.2, 0.5, 1, 2, 5]); axes[1].set_xticklabels(["0.2", "0.5", "1", "2", "5"]); axes[1].minorticks_off()
    axes[1].set_xlabel("calibration slope at 12 months\n(whole cohort, log scale; 1 = ideal)"); axes[1].set_title("(b) Calibration slope", loc="left")
    if has_sel:
        axes[2].axvline(qf["spearman"], color=ARM_COLOR["federated"], lw=1.2, zorder=2)
        axes[2].set_xlim(0, 1.0); axes[2].set_xlabel("Spearman correlation with the\npooled feature ranking"); axes[2].set_title("(c) Selection agreement", loc="left")
    for ax in axes:
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.tick_params(axis="y", length=0)
    h_node = Line2D([], [], **{k: v for k, v in node_kw.items() if k != "zorder"})
    h_fed = (Line2D([], [], **fed_kw), Line2D([], [], color=ARM_COLOR["federated"], lw=1.2))
    h_pool = (Line2D([], [], **pool_kw), Line2D([], [], color=ARM_COLOR["central"], lw=1.2, ls=(0, (3, 1.5))))
    fig.legend([h_node, h_fed, h_pool], ["node alone", "federated (dot: per area; line: whole cohort)", "pooled (tick: per area; line: whole cohort)"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, handlelength=2.8, handletextpad=0.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               columnspacing=1.4, borderaxespad=0.2)
    fig.tight_layout()
    panels = ("a Harrell C on the node's own temporal area: node alone (hollow), federated ensemble (filled) and pooled model (tick), all "
              "evaluated on the same patients. b Calibration slope at 12 months on the whole temporal cohort (log scale, 1 = ideal, dotted); the "
              "federated (solid) and pooled (dashed) slopes are single whole-cohort values and are drawn as vertical lines, against which each "
              "node-alone slope is compared.")
    if has_sel:
        panels += (" c Spearman correlation of each node's feature ranking with the pooled ranking (hollow); the vertical line is the agreement of "
                   "the federated ranking with the pooled one, a single value." + _e1_overlap_text(qa_by_node, qf))
    else:
        panels += " No selection-agreement panel: without feature selection there is no feature ranking to compare."
    w.figure(fig, f"E1_federation_ledger_{m}_{f}", f"Federation ledger, {MODEL_LABEL[m]}, {FS_LABEL[f]}, basic imputation. For each node "
             f"(rows ordered by the size of its temporal area, n = temporal patients), what changes when the node uses the federated model "
             f"instead of its own. {panels}", where="variants")


# ----------------------------------------------------------------------------- E2 model kinship map
MODEL_SHORT = {"cox": "Cox", "gradient": "GB", "rsf": "RSF"}
GROUP_2LINE = {"central": "pooled", "federated": "federated", "americas": "Americas", "asia": "Asia", "europe_non_italy": "Europe\noutside Italy",
               "italy_centre_south": "Italy\ncentre-south", "italy_north": "Italy\nnorth"}


def e2_kinship(w: Writer, final: Path, runs):
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    from paper_style import FULL_WIDTH_IN
    sel = []
    for role in ("central", "federated"):
        for (m, f, i) in ROWS6:
            r = pick(runs, role, m, f, i)
            if r is not None:
                sel.append((r["run_name"], f"{MODEL_SHORT[m]}{'33' if f == 'nofs' else '10'}", role))
    for a in AREAS:
        for (m, f, i) in ROWS6:
            r = pick(runs, "standalone", m, f, i, node=a)
            if r is not None:
                sel.append((r["run_name"], f"{MODEL_SHORT[m]}{'33' if f == 'nofs' else '10'}", a))
    risks = []
    for name, _, _ in sel:
        p = final / "runs" / name / "predictions.csv"
        if not p.exists():
            return
        risks.append(pd.read_csv(p)["risk_global"].rank().to_numpy(float))
    R = np.corrcoef(np.vstack(risks))
    labels = [s[1] for s in sel]; groups = [s[2] for s in sel]
    vmin = 0.0
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN - 0.15, 5.9))  # group names sit outside the axes: keeps the cropped width at 6.5 in
    im = ax.imshow(R, cmap="viridis", vmin=vmin, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(length=1.5, pad=1.5)
    bounds = [k for k in range(1, len(groups)) if groups[k] != groups[k - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1.2); ax.axvline(b - 0.5, color="white", lw=1.2)
    starts = [0] + bounds + [len(groups)]
    left = blended_transform_factory(ax.transAxes, ax.transData)
    top = blended_transform_factory(ax.transData, ax.transAxes)
    for g in range(len(starts) - 1):
        mid = (starts[g] + starts[g + 1] - 1) / 2
        name = GROUP_2LINE.get(groups[starts[g]], groups[starts[g]])
        ax.annotate(name, xy=(0, mid), xycoords=left, xytext=(-24, 0), textcoords="offset points", ha="right", va="center", fontsize=6.5, fontweight="bold")
        ax.annotate(name, xy=(mid, 1), xycoords=top, xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, extend="min" if np.nanmin(R) < vmin else "neither")
    cb.set_label("Spearman ρ of patient risk scores"); cb.ax.tick_params(labelsize=6.5)
    # block means, centred in their block
    for gi in range(len(starts) - 1):
        for gj in range(len(starts) - 1):
            blk = R[starts[gi]:starts[gi + 1], starts[gj]:starts[gj + 1]]
            if gi == gj:
                v = blk[np.triu_indices_from(blk, 1)].mean() if blk.shape[0] > 1 else 1.0
            else:
                v = blk.mean()
            ax.text((starts[gj] + starts[gj + 1] - 1) / 2, (starts[gi] + starts[gi + 1] - 1) / 2, f"{v:.2f}", fontsize=6, color="black", ha="center",
                    va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    fig.tight_layout()
    w.figure(fig, "E2_model_kinship_map", "Model kinship map. Spearman correlation between the risk scores assigned to the same 698 temporal patients by "
             "every final model (basic imputation): six pooled, six federated and thirty single-centre models, grouped by arm and node (white "
             "separators). Model labels: GB = gradient boosting, RSF = random survival forest; 33 = all features, 10 = ten features "
             "selected by the score test. The number in each block is the mean correlation of the block (diagonal blocks: between different "
             "models only). " + (f"The colour scale starts at {vmin:g}; lower values take the darkest colour (triangle on the colour bar). "
                                 if np.nanmin(R) < vmin else "") + "Pooled and federated models of the same family agree with each other as closely as two pooled models of different families do; "
             "single-centre models form isolated blocks.", where="variants")


# ----------------------------------------------------------------------------- E3 discrimination-calibration plane
def e3_plane(w: Writer, runs):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MultipleLocator
    from paper_style import FULL_WIDTH_IN
    main = runs[runs["fs"].isin(["nofs", "score"])]
    markers = {"cox": "o", "gradient": "s", "rsf": "^"}
    arm_name = {"central": "pooled", "federated": "federated", "standalone": "node alone"}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    # zoom window of panel b: the pooled and federated models
    pf = main[main["role"].isin(["central", "federated"])]
    zx = (np.floor(pf["temporal_harrell_c"].min() * 200) / 200 - 0.005, np.ceil(pf["temporal_harrell_c"].max() * 200) / 200 + 0.005) if len(pf) else (0.68, 0.75)
    zy = (pf["temporal_calib_slope_12"].min() / 1.15, pf["temporal_calib_slope_12"].max() * 1.15) if len(pf) else (0.7, 3)
    slopes = main["temporal_calib_slope_12"].dropna()
    ylo, yhi = (min(0.12, slopes.min() / 1.25), max(8.0, slopes.max() * 1.3)) if len(slopes) else (0.12, 8)
    for ax, zoom in zip(axes, (False, True)):
        for role in ("standalone", "central", "federated"):
            sub = main[main["role"] == role]
            for m, mk in markers.items():
                s = sub[sub["model"] == m]
                if role == "standalone":
                    ax.scatter(s["temporal_harrell_c"], s["temporal_calib_slope_12"], marker=mk, s=11 if not zoom else 16, color=ARM_COLOR[role], alpha=0.55,
                               lw=0, zorder=2)
                else:  # white edge so that coinciding pooled / federated markers stay separable
                    ax.scatter(s["temporal_harrell_c"], s["temporal_calib_slope_12"], marker=mk, s=20 if not zoom else 30, color=ARM_COLOR[role], alpha=0.9,
                               edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_yscale("log"); ax.minorticks_off()
        ax.axhline(1, color="#666666", lw=0.6, ls="--", zorder=1); ax.axhspan(0.8, 1.25, color="#eeeeee", zorder=0)
        ax.grid(lw=0.3, alpha=0.5)
        ax.set_xlabel("Harrell C on the temporal cohort")
    ax = axes[0]
    ax.set_ylim(ylo, yhi)
    cx = main["temporal_harrell_c"].dropna()
    if len(cx):
        ax.set_xlim(min(cx.min(), zx[0]) - 0.008, max(cx.max(), zx[1]) + 0.006)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if ylo <= t <= yhi]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_ylabel("calibration slope at 12 months, log scale\n(1 = ideal; above 1 = risks too moderate)")
    ax.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec="#333333", lw=0.7, ls=(0, (2, 1.5)), zorder=4))
    ax.set_title("(a) All models", loc="left")
    ax = axes[1]
    ax.set_xlim(*zx); ax.set_ylim(*zy)
    zt = [t for t in (0.5, 0.8, 1, 1.25, 1.5, 2, 3) if zy[0] <= t <= zy[1]]
    ax.set_yticks(zt); ax.set_yticklabels([f"{t:g}" for t in zt])
    ax.set_title("(b) Pooled and federated models (dashed box in a)", loc="left")
    # two short legends below the panels: colour = arm, shape = family
    h_arm = [Line2D([], [], ls="none", marker="o", ms=4.5, color=ARM_COLOR[r], alpha=0.55 if r == "standalone" else 0.9) for r in ("central", "federated", "standalone")]
    h_fam = [Line2D([], [], ls="none", marker=mk, ms=4.5, mfc="none", mec="#333333") for mk in markers.values()]
    fig.tight_layout()
    leg1 = fig.legend(h_arm, [arm_name[r] for r in ("central", "federated", "standalone")], title="colour: arm", loc="upper right", ncol=3,
                      bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)
    fig.legend(h_fam, [MODEL_LABEL[m] for m in markers], title="shape: model family", loc="upper left", ncol=3, bbox_to_anchor=(0.53, 0.0),
               handletextpad=0.3, columnspacing=1.0)
    fig.add_artist(leg1)
    w.figure(fig, "E3_discrimination_calibration_plane", "Discrimination-calibration plane. Every model of the grid (33- and 10-feature, both "
             "imputations) by its Harrell C on the temporal cohort and its 12-month calibration slope (log scale); colour = arm, shape = model "
             "family; dashed line = slope 1, grey band = slope within 0.8–1.25. a All models. b Enlargement of the dashed box in a, which "
             "contains every pooled and federated model. The federated ensembles sit at pooled discrimination with slopes mostly above 1 "
             "(gradient boosting close to 1); single-centre models scatter in both directions.", where="variants")


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
