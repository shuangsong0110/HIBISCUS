#!/usr/bin/env python3
"""Step 6 - figures: h2(t) curve with SE band, HE reference and the tested raw points; genetic-correlation heatmap.

Reads  results/<name>/curves_diag2.csv (+SE), curves_sparse_he_no_t.csv, surfaces_diag2.csv,
       h2_test.csv / corr_test.csv / joint_test.csv (optional), data/<name>/truth_h2.csv / truth_surface.csv (optional)
Writes figures/<name>/h2_curve.{pdf,png} + h2_curve_data.csv
       figures/<name>/corr_heatmap.{pdf,png} + corr_heatmap_data.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hibiscus.cli import common_parser, relpath, resolve
from hibiscus.io import ensure_dir, load_json
from hibiscus.plots import plot_corr_heatmap, plot_h2_curve


def first_row_value(path: Path, column: str):
    """The named column of a one-row CSV, or None if the file or column is absent."""
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0]
    return None if column not in row else float(row[column])


def main() -> int:
    parser = common_parser(__doc__)
    parser.add_argument("--no-truth", action="store_true", help="do not overlay the simulation truth")
    args = parser.parse_args()
    cfg, paths = resolve(args)
    fig_cfg = cfg["figures"]
    res_dir, data_dir = paths["results"], paths["data"]
    out_dir = ensure_dir(paths["figures"])
    curves = pd.read_csv(res_dir / "curves_diag2.csv")
    if "h2_se_conditional" not in curves.columns:
        raise SystemExit("run scripts/04_se.py first (no SE columns in curves_diag2.csv)")
    he_curve = pd.read_csv(res_dir / "curves_sparse_he_no_t.csv").iloc[0]
    he = {"h2": float(he_curve["h2"]), "h2_se_conditional": float(he_curve.get("h2_se_conditional", np.nan))}
    n_subjects = int(load_json(res_dir / "fit_manifest.json")["n_subjects_matched"])
    truth_h2 = pd.read_csv(data_dir / "truth_h2.csv") if (data_dir / "truth_h2.csv").exists() and not args.no_truth else None
    truth_surface = pd.read_csv(data_dir / "truth_surface.csv") if (data_dir / "truth_surface.csv").exists() and not args.no_truth else None
    h2 = pd.read_csv(res_dir / "h2_test.csv").iloc[0] if (res_dir / "h2_test.csv").exists() else None
    tested = None
    if h2 is not None:
        ages = [float(a) for a in cfg["grid"]["test_ages"]]
        tested = pd.DataFrame({"t": ages, "h2_raw": [float(h2[f"h2_raw_{a:g}"]) for a in ages]})

    plot_h2_curve(
        curves, he, out_dir / "h2_curve", truth=truth_h2, n_subjects=n_subjects, tested=tested,
        p_h2=None if h2 is None else float(h2["p_h2"]),
        p_corr=first_row_value(res_dir / "corr_test.csv", "p_corr"),
        p_joint=first_row_value(res_dir / "joint_test.csv", "p_joint"),
        dpi_png=int(fig_cfg.get("dpi_png", 200)),
    )
    surface = pd.read_csv(res_dir / "surfaces_diag2.csv")
    if truth_surface is not None:
        grid_est = np.sort(surface["t_row"].unique())
        grid_truth = np.sort(truth_surface["t_row"].unique())
        if len(grid_est) != len(grid_truth) or not np.allclose(grid_est, grid_truth):
            truth_surface = None
    min_corr = plot_corr_heatmap(
        surface, out_dir / "corr_heatmap", truth_surface=truth_surface,
        vmin=float(fig_cfg.get("heatmap_vmin", 0.2)), vmax=float(fig_cfg.get("heatmap_vmax", 1.0)),
        vyellow=float(fig_cfg.get("heatmap_vyellow", 0.5)), dpi_png=int(fig_cfg.get("dpi_png", 200)),
    )
    print(f"{paths['name']}: min estimated off-diagonal R_G = {min_corr:.3f}; figures in {relpath(out_dir, paths['workdir'])}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
