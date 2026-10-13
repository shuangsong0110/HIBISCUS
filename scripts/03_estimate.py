#!/usr/bin/env python3
"""Step 3 - estimation of h2(t) and the genetic-correlation surface (diag2) plus the age-independent (flat-kernel) HE estimate.

Reads  data/<name>/phenotype.tsv, sample_ids.csv, grm/<name>/edges.{npz,json}
Writes results/<name>/curves_diag2.csv            t, method, sigma_g2, sigma_eps2, sigma_y2, h2
       results/<name>/curves_sparse_he_no_t.csv   same schema, constant in t
       results/<name>/surfaces_diag2.csv          t_row, t_col, method, cov, corr   (PSD surface)
       results/<name>/surfaces_sparse_he_no_t.csv
       results/<name>/surface_rows_cg_surface_diag2.csv   t_row, t_col, cg_raw, cg_psd, cg_corr, cg_denom
       results/<name>/curves_all_methods.csv, surfaces_all_methods.csv, subject_table.csv,
       dataset_info.json, diagnostics.json, fit_manifest.json, runtime.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hibiscus.cli import common_parser, environment, relpath, resolve, timed
from hibiscus.estimate import (
    build_design, diag2_curve_frame, estimator_config, fit_diag2, he_curve_frame, in_original_units,
    sparse_he_no_t, surface_frame, surface_rows_frame,
)
from hibiscus.io import atomic_to_csv, ensure_dir, load_matched_dataset, save_json
from hibiscus.kernel import grid_points


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg, paths = resolve(args)
    out_dir = ensure_dir(paths["results"])
    if (out_dir / "curves_diag2.csv").exists() and not args.force:
        print(f"SKIP: {relpath(out_dir / 'curves_diag2.csv', paths['workdir'])} exists (use --force)")
        return 0

    data, load_seconds = timed(load_matched_dataset, paths["phenotype_tsv"], paths["sample_ids_csv"], paths["edges_npz"], paths["edges_json"])
    y, t, edges = data["y"], data["t"], data["edges"]          # y is standardised (mean 0, SD 1); see io.load_matched_dataset
    n = data["n_subjects"]
    y_scale = float(data["y_scale"])
    s2 = y_scale ** 2                                          # variance-type outputs are written in the original units
    print(f"{paths['name']}: matched {n} of {data['n_samples']} genotyped subjects; edges {data['edge_count_before_filter']:,} -> {edges['edge_count']:,}; "
          f"phenotype SD {y_scale:.4g} (standardised internally)")
    grid = grid_points(cfg["grid"]["t_min"], cfg["grid"]["t_max"], cfg["grid"]["size"])
    config = estimator_config(cfg["kernel"], n)
    print(f"  grid {grid[0]:g}..{grid[-1]:g} ({len(grid)} points); target ESS = {config['target_effective_n']:.1f} (= {config['target_effective_n_fraction']} x N); "
          f"bandwidth [{config['bandwidth_min']:g}, {config['bandwidth_max']:g}]; PSD clip; post-smooth {config['cg_surface_post_smooth_bandwidth']:g}")

    he, he_seconds = timed(sparse_he_no_t, y, edges)
    print(f"  sparse_he_no_t: h2 = {he['h2']:.4f} (sigma_g2 {he['sigma_g2'] * s2:.4g}, sigma_y2 {he['sigma_y2'] * s2:.4g}); {he_seconds:.1f}s")
    design, design_seconds = timed(build_design, t, edges, grid, config)
    fit, fit_seconds = timed(fit_diag2, y, t, design)
    print(f"  diag2: h2(t) min {fit['h2'].min():.4f} max {fit['h2'].max():.4f}; bandwidths {fit['bandwidths'].min():.2f}-{fit['bandwidths'].max():.2f} y; "
          f"min off-diagonal R_G {np.min(fit['corr'][~np.eye(len(grid), dtype=bool)]):.3f}; {design_seconds + fit_seconds:.1f}s")

    he_curve = in_original_units(he_curve_frame(he, grid), y_scale)
    d2_curve = in_original_units(diag2_curve_frame(fit), y_scale)
    he_surface = in_original_units(surface_frame("sparse_he_no_t", grid, np.full((len(grid), len(grid)), he["sigma_g2"])), y_scale)
    d2_surface = in_original_units(surface_frame("diag2", grid, fit["psd"]), y_scale)
    atomic_to_csv(he_curve, out_dir / "curves_sparse_he_no_t.csv")
    atomic_to_csv(d2_curve, out_dir / "curves_diag2.csv")
    atomic_to_csv(he_surface, out_dir / "surfaces_sparse_he_no_t.csv")
    atomic_to_csv(d2_surface, out_dir / "surfaces_diag2.csv")
    atomic_to_csv(in_original_units(surface_rows_frame(fit), y_scale), out_dir / "surface_rows_cg_surface_diag2.csv")
    atomic_to_csv(pd.concat([he_curve, d2_curve], ignore_index=True), out_dir / "curves_all_methods.csv")
    atomic_to_csv(pd.concat([he_surface, d2_surface], ignore_index=True), out_dir / "surfaces_all_methods.csv")
    atomic_to_csv(data["matched"][["FID", "IID", "global_idx", "local_idx", "y", "t"]], out_dir / "subject_table.csv")

    runtime = {"load_seconds": load_seconds, "sparse_he_no_t_seconds": he_seconds, "design_seconds": design_seconds, "diag2_seconds": fit_seconds}
    he_variance_keys = ("sigma_g2", "sigma_g2_raw", "sigma_y2", "sigma_eps2", "A")
    diagnostics = {
        "sparse_he_no_t": {k: (float(v) * s2 if k in he_variance_keys else v) for k, v in he.items()},
        "diag2": {"bandwidths": fit["bandwidths"].tolist(), "edge_count": fit["edge_count"],
                  "row_weight_capture_mean": design["row_weight_capture_mean"],
                  "min_offdiag_corr": float(np.min(fit["corr"][~np.eye(len(grid), dtype=bool)])),
                  "raw_min_eigenvalue": float(np.min(np.linalg.eigvalsh(0.5 * (fit["raw"] + fit["raw"].T)))) * s2},
    }
    manifest = {
        "label": paths["name"],
        "phenotype_tsv": relpath(paths["phenotype_tsv"], paths["workdir"]),
        "sample_ids_csv": relpath(paths["sample_ids_csv"], paths["workdir"]),
        "subject_edges_npz": relpath(paths["edges_npz"], paths["workdir"]),
        "subject_edges_meta": relpath(paths["edges_json"], paths["workdir"]),
        "match_on": "IID",
        "first_entry_policy": "drop missing y/t, keep first row per IID in TSV order",
        "phenotype_transform": "standardised at load: (y - y_mean) / y_scale; variance-type outputs written in the original units",
        "y_mean": float(data["y_mean"]),
        "y_scale": y_scale,
        "n_samples_genotyped": data["n_samples"],
        "n_subjects_matched": n,
        "phenotype_rows_first_entry": data["pheno_rows_first_entry"],
        "edge_count_full": data["edge_count_before_filter"],
        "edge_count_subset": edges["edge_count"],
        "grid_t": grid.tolist(),
        "grid_size": int(len(grid)),
        "t_min": float(grid[0]),
        "t_max": float(grid[-1]),
        "config": config,
        "environment": environment(),
    }
    save_json(out_dir / "runtime.json", runtime)
    save_json(out_dir / "diagnostics.json", diagnostics)
    save_json(out_dir / "fit_manifest.json", manifest)
    save_json(out_dir / "dataset_info.json", {k: manifest[k] for k in ("label", "n_samples_genotyped", "n_subjects_matched", "edge_count_full", "edge_count_subset", "grid_t", "config")})
    print(f"  wrote {relpath(out_dir, paths['workdir'])}/ (curves_*.csv, surfaces_*.csv, surface_rows_cg_surface_diag2.csv, fit_manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
