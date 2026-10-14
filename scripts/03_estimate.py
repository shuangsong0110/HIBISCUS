#!/usr/bin/env python3
"""Step 3 - estimation of h2(t) and the genetic-correlation surface, plus the age-independent HE estimate.

Reads  data/<name>/phenotype.tsv, sample_ids.csv, grm/<name>/edges.{npz,json}
Writes results/<name>/curves.csv          t, method (hibiscus | he), sigma_g2, sigma_eps2, sigma_y2, h2
       results/<name>/surfaces.csv        t_row, t_col, cov, corr      (PSD genetic covariance and R_G)
       results/<name>/fit_manifest.json   grid, estimator settings, phenotype scale, counts
The grid is every integer age from the youngest to the oldest matched subject.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hibiscus.cli import common_parser, environment, relpath, resolve, timed
from hibiscus.defaults import KERNEL
from hibiscus.estimate import (
    build_design, hibiscus_curve_frame, estimator_config, fit_hibiscus, he_curve_frame, in_original_units,
    fit_he, surface_frame,
)
from hibiscus.io import atomic_to_csv, ensure_dir, load_matched_dataset, save_json


def age_grid(t) -> np.ndarray:
    return np.arange(np.floor(np.min(t)), np.ceil(np.max(t)) + 1.0)


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    paths = resolve(args)
    out_dir = ensure_dir(paths["results"])
    if (out_dir / "curves.csv").exists() and not args.force:
        print(f"SKIP: {relpath(out_dir / 'curves.csv', paths['workdir'])} exists (use --force)")
        return 0

    data, load_seconds = timed(load_matched_dataset, paths["phenotype_tsv"], paths["sample_ids_csv"], paths["edges_npz"], paths["edges_json"])
    y, t, edges = data["y"], data["t"], data["edges"]          # y is standardised (mean 0, SD 1); see io.load_matched_dataset
    n = data["n_subjects"]
    y_scale = float(data["y_scale"])
    print(f"{paths['name']}: matched {n} of {data['n_samples']} genotyped subjects; edges {data['edge_count_before_filter']:,} -> {edges['edge_count']:,}; "
          f"phenotype SD {y_scale:.4g} (standardised internally); {load_seconds:.1f}s")
    grid = age_grid(t)
    config = estimator_config(KERNEL, n)
    print(f"  grid {grid[0]:g}..{grid[-1]:g} ({len(grid)} points); target ESS = {config['target_effective_n']:.1f} (= {config['target_effective_n_fraction']} x N); "
          f"bandwidth [{config['bandwidth_min']:g}, {config['bandwidth_max']:g}]; PSD clip; post-smooth {config['cg_surface_post_smooth_bandwidth']:g}")

    he, he_seconds = timed(fit_he, y, edges)
    print(f"  he: h2 = {he['h2']:.4f} (sigma_g2 {he['sigma_g2'] * y_scale ** 2:.4g}, sigma_y2 {he['sigma_y2'] * y_scale ** 2:.4g}); {he_seconds:.1f}s")
    design, design_seconds = timed(build_design, t, edges, grid, config)
    fit, fit_seconds = timed(fit_hibiscus, y, t, design)
    print(f"  hibiscus: h2(t) min {fit['h2'].min():.4f} max {fit['h2'].max():.4f}; bandwidths {fit['bandwidths'].min():.2f}-{fit['bandwidths'].max():.2f} y; "
          f"min off-diagonal R_G {np.min(fit['corr'][~np.eye(len(grid), dtype=bool)]):.3f}; {design_seconds + fit_seconds:.1f}s")

    curves = pd.concat([hibiscus_curve_frame(fit), he_curve_frame(he, grid)], ignore_index=True)
    atomic_to_csv(in_original_units(curves, y_scale), out_dir / "curves.csv")
    atomic_to_csv(in_original_units(surface_frame(grid, fit["psd"]), y_scale), out_dir / "surfaces.csv")
    manifest = {
        "label": paths["name"],
        "phenotype_tsv": relpath(paths["phenotype_tsv"], paths["workdir"]),
        "sample_ids_csv": relpath(paths["sample_ids_csv"], paths["workdir"]),
        "subject_edges_npz": relpath(paths["edges_npz"], paths["workdir"]),
        "match_on": "IID",
        "first_entry_policy": "drop missing y/t, keep first row per IID in TSV order",
        "phenotype_transform": "standardised at load: (y - y_mean) / y_scale; variance-type outputs written in the original units",
        "y_mean": float(data["y_mean"]),
        "y_scale": y_scale,
        "n_samples_genotyped": data["n_samples"],
        "n_subjects_matched": n,
        "edge_count_full": data["edge_count_before_filter"],
        "edge_count_subset": edges["edge_count"],
        "grid_t": grid.tolist(),
        "config": config,
        "environment": environment(),
    }
    save_json(out_dir / "fit_manifest.json", manifest)
    print(f"  wrote {relpath(out_dir, paths['workdir'])}/ (curves.csv, surfaces.csv, fit_manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
