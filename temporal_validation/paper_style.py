"""Journal style for the paper figures (npj Digital Medicine / Nature Portfolio).

- One sans-serif typeface for every figure: Arial when available (fonts can be supplied in a folder named by the
  environment variable PAPER_FONT_DIR, e.g. arial.ttf and arialbd.ttf), else Liberation Sans (metric-compatible),
  else DejaVu Sans. Fonts are never shipped with the code.
- Panel letters lower-case bold ("a", "b", ...) instead of "(a)" / "(A)" in panel titles and annotations.
- No figure-level titles (the title belongs in the legend).
- Long axis labels wrapped so that nothing is clipped.
- Text kept as text in vector outputs (TrueType in PDF, text in SVG) so the figures stay editable.
"""
from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

_PANEL = re.compile(r"^\(([A-Za-z])\)\s*")
_FAMILY = None


def setup_fonts() -> str:
    """Register the fonts and set the rcParams; idempotent. Returns the family in use."""
    global _FAMILY
    if _FAMILY:
        return _FAMILY
    import matplotlib
    from matplotlib import font_manager

    d = os.environ.get("PAPER_FONT_DIR")
    if d and Path(d).is_dir():
        for f in sorted(Path(d).glob("*.tt[fc]")):
            try:
                font_manager.fontManager.addfont(str(f))
            except Exception:  # noqa: BLE001
                pass
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((n for n in ("Arial", "Helvetica", "Liberation Sans") if n in names), "DejaVu Sans")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.bf": f"{family}:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.titleweight": "normal",
    })
    _FAMILY = family
    return family


def _panel_text(s: str):
    m = _PANEL.match(s or "")
    if not m:
        return None
    rest = s[m.end():].replace("$", r"\$")
    return r"$\bf{" + m.group(1).lower() + "}$  " + rest


def npj_panels(fig, xwrap: int = 52, ywrap: int = 45) -> None:
    """Restyle a finished figure in place (panel letters, no suptitle, wrapped labels)."""
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for t in list(fig.texts):
        new = _panel_text(t.get_text())
        if new:
            t.set_text(new)
    for ax in fig.axes:
        for title in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if title is not None:
                new = _panel_text(title.get_text())
                if new:
                    title.set_text(new)
        for t in ax.texts:
            new = _panel_text(t.get_text())
            if new:
                t.set_text(new)
        for lab, width in ((ax.xaxis.label, xwrap), (ax.yaxis.label, ywrap)):
            s = lab.get_text()
            if s and "\n" not in s and len(s) > width:
                lab.set_text(textwrap.fill(s, width))
