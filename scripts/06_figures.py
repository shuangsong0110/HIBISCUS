#!/usr/bin/env python3
"""Step 6 - figures: h2(t) curve with SE band and HE reference; genetic-correlation heatmap.

Reads  results/<name>/curves.csv (+SE), surfaces.csv, tests.csv (optional),
       data/<name>/truth_h2.csv, truth_surface.csv (optional, toy datasets)
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
from hibiscus.defaults import FIGURE
from hibiscus.io import ensure_dir, load_json
from hibiscus.plots import plot_corr_heatmap, plot_h2_curve


def main() -> int:
    parser = common_parser(__doc__)
    parser.add_argument("--no-truth", action="store_true", help="do not overlay the simulation truth")
    args = parser.parse_args()
    paths = resolve(args)
    res_dir, data_dir = paths["results"], paths["data"]
    out_dir = ensure_dir(paths["figures"])
    curves = pd.read_csv(res_dir / "curves.csv")
    if "h2_se_conditional" not in curves.columns:
        raise SystemExit("run scripts/04_se.py first (no SE columns in curves.csv)")
    he_row = curves.loc[curves["method"] == "he"].iloc[0]
    he = {"h2": float(he_row["h2"]), "h2_se_conditional": float(he_row["h2_se_conditional"])}
    n_subjects = int(load_json(res_dir / "fit_manifest.json")["n_subjects_matched"])
    truth_h2 = pd.read_csv(data_dir / "truth_h2.csv") if (data_dir / "truth_h2.csv").exists() and not args.no_truth else None
    truth_surface = pd.read_csv(data_dir / "truth_surface.csv") if (data_dir / "truth_surface.csv").exists() and not args.no_truth else None
    tests = pd.read_csv(res_dir / "tests.csv").iloc[0] if (res_dir / "tests.csv").exists() else None
    p = {k: (None if tests is None else float(tests[k])) for k in ("p_h2", "p_corr", "p_joint")}

    plot_h2_curve(curves.loc[curves["method"] == "hibiscus"], he, out_dir / "h2_curve", truth=truth_h2, n_subjects=n_subjects,
                  p_h2=p["p_h2"], p_corr=p["p_corr"], p_joint=p["p_joint"], dpi_png=int(FIGURE["dpi_png"]))
    surface = pd.read_csv(res_dir / "surfaces.csv")
    if truth_surface is not None:
        grid_est = np.sort(surface["t_row"].unique())
        grid_truth = np.sort(truth_surface["t_row"].unique())
        if len(grid_est) != len(grid_truth) or not np.allclose(grid_est, grid_truth):
            truth_surface = None
    min_corr = plot_corr_heatmap(surface, out_dir / "corr_heatmap", truth_surface=truth_surface, vmin=float(FIGURE["heatmap_vmin"]),
                                 vmax=float(FIGURE["heatmap_vmax"]), vyellow=float(FIGURE["heatmap_vyellow"]), dpi_png=int(FIGURE["dpi_png"]))
    print(f"{paths['name']}: min estimated off-diagonal R_G = {min_corr:.3f}; figures in {relpath(out_dir, paths['workdir'])}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
