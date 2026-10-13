"""Plotting style with a portable font fallback.

Half-frame axes (left + bottom spines, no tick marks), Arial when available (else Liberation Sans,
Helvetica or DejaVu Sans), stixsans math, fixed palette, legends below the axes, PDF + PNG output.
The correlation-surface colour map is viridis_r on [vyellow, vmax] extended by a pale ramp on [vmin, vyellow).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

C = {"ref": "tab:blue", "alt": "tab:red", "muted": "grey"}
SURFACE_CMAP = "viridis_r"
BAND_ALPHA = 0.18
PALEST = "#ffffe0"


def _register_optional_fonts() -> None:
    """Register .ttf files from $HIBISCUS_FONT_DIR if the user points to a font folder."""
    font_dir = os.environ.get("HIBISCUS_FONT_DIR", "")
    if not font_dir:
        return
    for ttf in Path(font_dir).glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(ttf))
        except Exception:
            pass


def available_sans_serif() -> list[str]:
    installed = {font.name for font in fm.fontManager.ttflist}
    preferred = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
    family = [name for name in preferred if name in installed]
    return family or ["DejaVu Sans"]


def use_ss_style() -> None:
    """Set the rcParams once before creating figures."""
    _register_optional_fonts()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": available_sans_serif(),
        "mathtext.fontset": "stixsans",
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def half_frame(ax) -> None:
    """Keep the left and bottom spines only; drop tick marks, keep labels."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(length=0, width=0.9)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def grouped_legends(ax, groups, y: float = -0.13, dx: float = 0.48, **kw):
    """Side-by-side legends, one per group [(title, handles, labels), ...], each with a bold title."""
    kw.setdefault("frameon", False)
    kw.setdefault("handlelength", 2.4)
    kw.setdefault("handletextpad", 0.7)
    out = []
    for i, (title, hs, ls) in enumerate(groups):
        leg = ax.legend(hs, ls, title=title, alignment="left", loc="upper left",
                        bbox_to_anchor=(0.04 + dx * i, y), borderaxespad=0.0, **kw)
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontsize(plt.rcParams["legend.fontsize"])
        if i < len(groups) - 1:
            ax.add_artist(leg)
        out.append(leg)
    return out


def save_figure(fig, out, dpi_png: int = 200, extra_artists=None) -> Path:
    """Write the vector PDF plus a PNG (legends below the axes are kept in the tight bbox)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
    if extra_artists:
        kwargs["bbox_extra_artists"] = list(extra_artists)
    fig.savefig(out, **kwargs)
    fig.savefig(out.with_suffix(".png"), dpi=dpi_png, **kwargs)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    return out


def build_cmap(vmin: float = 0.2, vmax: float = 1.0, vyellow: float = 0.5, n: int = 256):
    """viridis_r on [vyellow, vmax], extended by a pale ramp on [vmin, vyellow)."""
    base = plt.get_cmap(SURFACE_CMAP)
    frac = (vyellow - vmin) / (vmax - vmin)
    n_lo = int(round(n * frac))
    low = LinearSegmentedColormap.from_list("pale", [PALEST, base(0.0)])(np.linspace(0, 1, max(n_lo, 1)))
    high = base(np.linspace(0, 1, n - n_lo))
    return ListedColormap(np.vstack([low, high]), name="viridis_r_pale"), Normalize(vmin=vmin, vmax=vmax)
