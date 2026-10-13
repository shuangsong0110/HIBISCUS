"""Figures: h2(t) curve with SE band and HE reference; genetic-correlation heatmap.

Shaded +-1 SE band, HE reference line and span, legends below the axes. Every figure writes
PDF + PNG + a *_data.csv with all plotted values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import BAND_ALPHA, C, build_cmap, grouped_legends, half_frame, save_figure, use_ss_style

import matplotlib.pyplot as plt  # after .style, which selects the backend


def _fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def plot_h2_curve(curve: pd.DataFrame, he: dict, out_stem, *, truth: pd.DataFrame | None = None,
                  n_subjects: int | None = None, tested: pd.DataFrame | None = None, p_h2=None, p_corr=None, p_joint=None,
                  dpi_png: int = 200) -> None:
    """curve: t, h2, h2_se_conditional; he: {h2, h2_se_conditional}; truth: t, h2_truth (optional);
    tested: t, h2_raw, the raw curve at the h2 test's ages."""
    use_ss_style()
    curve = curve.sort_values("t").reset_index(drop=True)
    t = curve["t"].to_numpy(float)
    h2 = curve["h2"].to_numpy(float)
    se = curve["h2_se_conditional"].to_numpy(float)
    lower = np.clip(h2 - se, 0.0, 1.0)
    upper = np.clip(h2 + se, 0.0, 1.0)
    he_h2 = float(he["h2"])
    he_se = float(he["h2_se_conditional"])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.fill_between(t, lower, upper, color=C["ref"], alpha=BAND_ALPHA, lw=0)
    ax.plot(t, h2, "-", color=C["ref"], lw=1.8)
    ax.axhspan(he_h2 - he_se, he_h2 + he_se, color="black", alpha=0.06, lw=0)
    ax.axhline(he_h2, color="black", ls="--", lw=1.1)
    if truth is not None:
        truth = truth.sort_values("t")
        ax.plot(truth["t"].to_numpy(float), truth["h2_truth"].to_numpy(float), ":", color=C["muted"], lw=1.6)
    if tested is not None:
        ax.plot(tested["t"].to_numpy(float), tested["h2_raw"].to_numpy(float), "o", mfc="white", mec=C["alt"], mew=1.3, ms=5.5)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("SNP heritability")
    ax.set_xlim(float(t.min()) - 1.0, float(t.max()) + 1.0)
    ymax = max(float(np.max(upper)), he_h2 + he_se, float(truth["h2_truth"].max()) if truth is not None else 0.0,
               float(tested["h2_raw"].max()) if tested is not None else 0.0)
    ax.set_ylim(0.0, min(1.0, ymax * 1.15 + 0.02))
    half_frame(ax)

    band_label = r"$\pm$ 1 SE"
    handles = [
        (Line2D([], [], color=C["ref"], lw=1.8), Patch(facecolor=C["ref"], alpha=BAND_ALPHA, lw=0)),
        (Line2D([], [], color="black", ls="--", lw=1.1), Patch(facecolor="black", alpha=0.06, lw=0)),
    ]
    labels = [
        r"$\widehat{h}^2_{\mathrm{HIBISCUS}}(\mathrm{Age})$ " + band_label,
        r"$\widehat{h}^2_{\mathrm{HE}}$ = " + f"{he_h2:.3f} " + band_label + f" ({he_se:.3f})",
    ]
    if truth is not None:
        handles.append(Line2D([], [], color=C["muted"], ls=":", lw=1.6))
        labels.append(r"true $h^2(\mathrm{Age})$")
    if tested is not None:
        handles.append(Line2D([], [], marker="o", ls="", mfc="white", mec=C["alt"], mew=1.3, ms=5.5))
        labels.append("raw curve at the tested ages")
    group_title = "Estimates" + (f" (N = {n_subjects:,})" if n_subjects else "")
    groups = [(group_title, handles, labels)]
    if p_h2 is not None or p_corr is not None or p_joint is not None:
        test_labels = [
            r"$p_{h^2}$ ($h^2$ test, $h^2(t)=c$) = " + _fmt_p(p_h2),
            r"$p_{\mathrm{corr}}$ (correlation test, $R_G=1$) = " + _fmt_p(p_corr),
            r"$p_{\mathrm{joint}}$ (ACAT) = " + _fmt_p(p_joint),
        ]
        test_handles = [Line2D([], [], color="none") for _ in test_labels]
        groups.append(("Tests", test_handles, test_labels))
    legends = grouped_legends(ax, groups, y=-0.14, dx=0.55, handlelength=2.0)
    fig.subplots_adjust(bottom=0.30)
    out_stem = Path(out_stem)
    save_figure(fig, out_stem.with_suffix(".pdf"), dpi_png=dpi_png, extra_artists=legends)
    plt.close(fig)

    data = pd.DataFrame({"t": t, "h2": h2, "h2_se_conditional": se, "band_lower": lower, "band_upper": upper,
                         "h2_he": he_h2, "h2_he_se_conditional": he_se})
    if truth is not None:
        data = data.merge(truth[["t", "h2_truth"]], on="t", how="left")
    if tested is not None:
        data = data.merge(tested[["t", "h2_raw"]].rename(columns={"h2_raw": "h2_raw_tested"}), on="t", how="left")
    data_path = out_stem.parent / (out_stem.name + "_data.csv")
    data.to_csv(data_path, index=False, float_format="%.10g")
    print(f"wrote {data_path}")


def _corr_matrix(surface: pd.DataFrame, column: str):
    grid = np.sort(surface["t_row"].unique().astype(float))
    matrix = surface.pivot(index="t_row", columns="t_col", values=column).sort_index().sort_index(axis=1).to_numpy(float)
    return grid, matrix


def plot_corr_heatmap(surface: pd.DataFrame, out_stem, *, truth_surface: pd.DataFrame | None = None,
                      vmin: float = 0.2, vmax: float = 1.0, vyellow: float = 0.5, dpi_png: int = 200) -> float:
    """surface: long t_row, t_col, corr (estimate); truth_surface: t_row, t_col, corr_truth (optional).
    Returns the minimum estimated correlation."""
    use_ss_style()
    grid, corr = _corr_matrix(surface, "corr")
    panels = [(r"$\widehat{R}_G$ (HIBISCUS)", corr)]
    if truth_surface is not None:
        _, corr_truth = _corr_matrix(truth_surface, "corr_truth")
        panels.append((r"true $R_G$", corr_truth))
    cmap, norm = build_cmap(vmin, vmax, vyellow)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels) + 0.8, 4.4), squeeze=False)
    extent = [float(grid.min()), float(grid.max()), float(grid.min()), float(grid.max())]
    image = None
    for ax, (label, matrix) in zip(axes[0], panels):
        image = ax.imshow(np.clip(matrix, vmin, vmax), origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Age (years)")
        ax.set_title(label)
        half_frame(ax)
    cbar = fig.colorbar(image, ax=list(axes[0]), fraction=0.035, pad=0.03)
    cbar.set_label(r"$R_G$")
    cbar.set_ticks(list(np.round(np.linspace(vmin, vmax, 5), 2)))
    cbar.outline.set_visible(False)
    out_stem = Path(out_stem)
    save_figure(fig, out_stem.with_suffix(".pdf"), dpi_png=dpi_png)
    plt.close(fig)

    data = surface[["t_row", "t_col", "corr"]].rename(columns={"corr": "corr_estimate"}).copy()
    if truth_surface is not None:
        data = data.merge(truth_surface[["t_row", "t_col", "corr_truth"]], on=["t_row", "t_col"], how="left")
    data_path = out_stem.parent / (out_stem.name + "_data.csv")
    data.to_csv(data_path, index=False, float_format="%.10g")
    print(f"wrote {data_path}")
    return float(np.nanmin(corr))
