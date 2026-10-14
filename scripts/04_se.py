#!/usr/bin/env python3
"""Step 4 - conditional delta-method standard errors.

Recomputes the step-3 fit (seconds), verifies it against curves.csv, then adds
  h2_se_conditional, h2_ci_lower_conditional, h2_ci_upper_conditional
to every row of results/<name>/curves.csv (the hibiscus rows and the he rows).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hibiscus.cli import common_parser, relpath, resolve, timed
from hibiscus.defaults import SE_FD_EPS_SCALE
from hibiscus.estimate import build_design, fit_hibiscus, fit_he
from hibiscus.io import atomic_to_csv, load_json, load_matched_dataset
from hibiscus.se import conditional_se

SE_COLUMNS = ["h2_se_conditional", "h2_ci_lower_conditional", "h2_ci_upper_conditional"]


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    paths = resolve(args)
    res_dir = paths["results"]
    curves_path = res_dir / "curves.csv"
    if not curves_path.exists():
        raise SystemExit("run scripts/03_estimate.py first")
    curves = pd.read_csv(curves_path)
    if all(c in curves.columns for c in SE_COLUMNS) and not args.force:
        print(f"SKIP: {relpath(curves_path, paths['workdir'])} already has SE columns (use --force)")
        return 0
    manifest = load_json(res_dir / "fit_manifest.json")
    config = manifest["config"]
    grid = np.asarray(manifest["grid_t"], float)

    data = load_matched_dataset(paths["phenotype_tsv"], paths["sample_ids_csv"], paths["edges_npz"], paths["edges_json"])
    y, t, edges = data["y"], data["t"], data["edges"]
    if data["n_subjects"] != int(manifest["n_subjects_matched"]):
        raise SystemExit("matched subject count differs from fit_manifest.json; rerun step 3")
    if abs(data["y_scale"] - float(manifest["y_scale"])) > 1e-9 * float(manifest["y_scale"]):
        raise SystemExit("phenotype scale differs from fit_manifest.json (phenotype file changed?); rerun step 3")
    design = build_design(t, edges, grid, config)
    fit = fit_hibiscus(y, t, design)
    he = fit_he(y, edges)
    saved = curves.loc[curves["method"] == "hibiscus"].sort_values("t")["h2"].to_numpy(float)
    if float(np.max(np.abs(fit["h2"] - saved))) > 1e-10:
        raise SystemExit("recomputed h2(t) differs from curves.csv; rerun step 3")

    result, seconds = timed(conditional_se, y, t, fit, design, edges, he, SE_FD_EPS_SCALE)
    se_curve = result["curve_se"]
    print(f"{paths['name']}: conditional SE in {seconds:.1f}s; SE(h2(t)) min {se_curve['h2_se_conditional'].min():.4f} max {se_curve['h2_se_conditional'].max():.4f}; "
          f"SE(h2_he) = {result['he_se']['h2_se_conditional']:.4f}")

    curves = curves.drop(columns=[c for c in SE_COLUMNS if c in curves.columns])
    hibiscus_rows = curves.loc[curves["method"] == "hibiscus"].merge(se_curve, on="t", how="left")
    he_rows = curves.loc[curves["method"] == "he"].copy()
    for column in SE_COLUMNS:
        he_rows[column] = result["he_se"][column]
    atomic_to_csv(pd.concat([hibiscus_rows, he_rows], ignore_index=True), curves_path)
    print(f"  wrote {relpath(curves_path, paths['workdir'])} (+SE columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
