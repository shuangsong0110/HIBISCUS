#!/usr/bin/env python3
"""Step 4 - conditional delta-method standard errors.

Recomputes the step-3 fit (seconds), verifies it against the saved curves, then adds
  h2_se_conditional, h2_ci_lower_conditional, h2_ci_upper_conditional
to results/<name>/curves_diag2.csv, curves_sparse_he_no_t.csv and curves_all_methods.csv, and writes
  results/<name>/se/se_summary.json
The dynamic influence is smoothed once (see hibiscus/se.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hibiscus import Z975
from hibiscus.cli import common_parser, environment, relpath, resolve, timed
from hibiscus.estimate import build_design, fit_diag2, sparse_he_no_t
from hibiscus.io import atomic_to_csv, ensure_dir, load_json, load_matched_dataset, save_json
from hibiscus.se import conditional_se

SE_COLUMNS = ["h2_se_conditional", "h2_ci_lower_conditional", "h2_ci_upper_conditional"]


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg, paths = resolve(args)
    res_dir = paths["results"]
    curves_path = res_dir / "curves_diag2.csv"
    if not curves_path.exists():
        raise SystemExit("run scripts/03_estimate.py first")
    existing = pd.read_csv(curves_path)
    if all(c in existing.columns for c in SE_COLUMNS) and not args.force:
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
    fit = fit_diag2(y, t, design)
    he = sparse_he_no_t(y, edges)
    rel = float(np.max(np.abs(fit["h2"] - existing.sort_values("t")["h2"].to_numpy(float))))
    if rel > 1e-10:
        raise SystemExit(f"recomputed h2(t) differs from curves_diag2.csv by {rel:.2e}; rerun step 3")

    result, seconds = timed(conditional_se, y, t, fit, design, edges, he, float(cfg["se"].get("fd_eps_scale", 1e-6)))
    se_curve = result["curve_se"]
    print(f"{paths['name']}: conditional SE in {seconds:.1f}s; SE(h2(t)) min {se_curve['h2_se_conditional'].min():.4f} max {se_curve['h2_se_conditional'].max():.4f}; "
          f"SE(h2_HE) = {result['he_se']['h2_se_conditional']:.4f}")

    d2 = existing.drop(columns=[c for c in SE_COLUMNS if c in existing.columns]).merge(se_curve, on="t", how="left")
    atomic_to_csv(d2, curves_path)
    he_path = res_dir / "curves_sparse_he_no_t.csv"
    he_curve = pd.read_csv(he_path).drop(columns=[c for c in SE_COLUMNS if c in existing.columns], errors="ignore")
    he_curve["h2_se_conditional"] = result["he_se"]["h2_se_conditional"]
    he_curve["h2_ci_lower_conditional"] = result["he_se"]["h2_ci_lower_conditional"]
    he_curve["h2_ci_upper_conditional"] = result["he_se"]["h2_ci_upper_conditional"]
    atomic_to_csv(he_curve, he_path)
    atomic_to_csv(pd.concat([he_curve, d2], ignore_index=True), res_dir / "curves_all_methods.csv")

    se_dir = ensure_dir(res_dir / "se")
    summary = {
        "method": "conditional delta method in y; sigma_e,i^2 = Yhat(t_i) - Ghat(t_i); dynamic influence smoothed once",
        "fd_eps_scale": float(cfg["se"].get("fd_eps_scale", 1e-6)),
        "grid_t": grid.tolist(),
        "se_dynamic": result["se_dynamic"].tolist(),
        "se_he": result["he_se"]["h2_se_conditional"],
        "curve_reproduction_relative_error": result["curve_reproduction_relative_error"],
        "total_variance_reproduction_relative_error": result["total_variance_reproduction_relative_error"],
        "he_numerator_reproduction_relative_error": result["he_se"]["A_relative_error"],
        "z975": Z975,
        "seconds": seconds,
        "environment": environment(),
    }
    save_json(se_dir / "se_summary.json", summary)
    print(f"  wrote {relpath(curves_path, paths['workdir'])} (+SE columns), {relpath(se_dir, paths['workdir'])}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
