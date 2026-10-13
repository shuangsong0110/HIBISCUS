#!/usr/bin/env python3
"""Step 5 - tests of age variation: the h2 test (H0: h2(t) = c), the correlation test (H0: R_G = 1),
their ACAT combination p_joint, and the summary table across traits.

Reads  results/<name>/curves_diag2.csv, surface_rows_cg_surface_diag2.csv, fit_manifest.json (step 3)
       data + edges; grm/<name>/genetic_scores_<B x rank>.float32.npy (h2 test); random_scores_rank<q>.float32.npy (correlation test)
Writes results/<name>/h2_test.csv, h2_test_replicates.npz
       results/<name>/corr_test/source_<grid>_{curve,surface_rows}.csv   (source fit on the test grid)
       results/<name>/corr_test.csv, corr_test_diagnostics.npz
       results/<name>/joint_test.csv                                      (p_h2, p_corr, p_joint)
       summary/all_traits.csv   one row per results/<trait>/ with the three p-values and multiple-testing columns
"""
from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hibiscus import EPS
from hibiscus.cli import common_parser, environment, relpath, resolve, timed
from hibiscus.corr_test import run_directional_p2
from hibiscus.estimate import (
    build_design, diag2_curve_frame, estimator_config, fit_diag2, in_original_units, pivot_surface, surface_rows_frame,
)
from hibiscus.h2_test import flat_test, raw_curves_many, target_indices
from hibiscus.io import atomic_savez, atomic_to_csv, ensure_dir, load_json, load_matched_dataset, save_json
from hibiscus.kernel import grid_points, smoothing_matrix
from hibiscus.multiple_testing import acat, add_multiple_testing

GATE = 1e-10          # the h2 test must reproduce the saved step-3 fit before it is computed


def trait_seed(base: int, name: str) -> int:
    """Stable per-trait seed (Python's hash() is salted per interpreter)."""
    return int(zlib.crc32(f"{int(base)}:{name}".encode()))


def run_h2_test(cfg, paths, data) -> dict:
    res_dir = paths["results"]
    manifest = load_json(res_dir / "fit_manifest.json")
    y, t, edges = data["y"], data["t"], data["edges"]
    if data["n_subjects"] != int(manifest["n_subjects_matched"]):
        raise SystemExit("matched subject count differs from fit_manifest.json; rerun step 3")
    if abs(data["y_scale"] - float(manifest["y_scale"])) > 1e-9 * float(manifest["y_scale"]):
        raise SystemExit("phenotype scale differs from fit_manifest.json (phenotype file changed?); rerun step 3")
    grid = np.asarray(manifest["grid_t"], float)
    config = manifest["config"]
    s2 = float(data["y_scale"]) ** 2          # saved surfaces are in original units; the test runs on standardised y
    rows = pd.read_csv(res_dir / "surface_rows_cg_surface_diag2.csv")
    raw_saved, den_saved = pivot_surface(rows, "cg_raw") / s2, pivot_surface(rows, "cg_denom")
    sigma_y2_saved = pd.read_csv(res_dir / "curves_diag2.csv").sort_values("t")["sigma_y2"].to_numpy(float) / s2
    ages = [float(a) for a in cfg["grid"]["test_ages"]]
    idx = target_indices(grid, ages)

    design = build_design(t, edges, grid, config)
    # gates: the batched estimator used for the replicates reproduces the saved fit on the observed y
    smoother = smoothing_matrix(grid, float(config["cg_surface_post_smooth_bandwidth"]))
    obs = raw_curves_many(y, t, design, smoother)
    gates = {
        "den_diag": float(np.max(np.abs(np.diag(design["den_surface"]) - np.diag(den_saved)) / np.abs(np.diag(den_saved)))),
        "raw_diag": float(np.max(np.abs(obs["raw_diag"][:, 0] - np.diag(raw_saved))) / max(float(np.max(np.abs(np.diag(raw_saved)))), EPS)),
        "sigma_y2": float(np.max(np.abs(obs["sigma_y2"][:, 0] - sigma_y2_saved)) / max(float(np.max(sigma_y2_saved)), EPS)),
    }
    for name, value in gates.items():
        if not value <= GATE:
            raise SystemExit(f"h2 test: {name} does not reproduce the step-3 fit ({value:.2e} > {GATE:g}); rerun step 3")

    bank_full = np.load(paths["genetic_scores_npy"], mmap_mode="r")
    if bank_full.shape[0] != data["n_samples"]:
        raise SystemExit("genetic score bank was built for a different sample-ID file")
    bank_meta = load_json(paths["genetic_scores_npy"].with_name("genetic_scores_meta.json"))
    if abs(float(bank_meta["grm_scale"]) - 1.0) > 0.05:
        raise SystemExit(f"genetic score bank GRM scale {bank_meta['grm_scale']:.4f} is far from 1: bank and edges do not match")
    bank = np.asarray(bank_full[data["matched"]["global_idx"].to_numpy()], float)

    h2_cfg = cfg["h2_test"]
    seed = trait_seed(h2_cfg["seed"], paths["name"])
    out, seconds = timed(
        flat_test, y, t, design, raw_saved, sigma_y2_saved, idx, bank,
        n_boot=int(h2_cfg["n_boot"]), max_rank=int(h2_cfg["max_rank"]), seed=seed,
        c_star_bounds=tuple(float(v) for v in h2_cfg["c_star_bounds"]),
    )
    record = {
        "phenotype_key": paths["name"],
        "method": "h2 test, H0 h2(t) = c; raw smoothed curve; noncentral generalised chi-square tail of B bootstrap curves",
        "T": out["T"], "p_h2": out["p_h2"], "log10_p_h2": out["log10_p_h2"], "p_h2_mc": out["p_h2_mc"],
        "c_star": out["c_star"], "c_star_clipped": out["c_star_clipped"], "null_rank": out["null_rank"],
        "n_boot": out["n_boot"], "seed": seed, "bootstrap_mean_T": out["bootstrap_mean_T"],
        "law_mean_T": out["law_mean"], "law_mean_share": out["law_mean_share"],
        "grm_scale_bank": float(bank_meta["grm_scale"]), "seconds": seconds,
        **{f"gate_{k}": v for k, v in gates.items()},
    }
    for j, age in enumerate(ages):
        record[f"h2_raw_{age:g}"] = float(out["h2_raw_obs"][j])
        record[f"sd_null_{age:g}"] = float(out["sd_null"][j])
    atomic_to_csv(pd.DataFrame([record]), res_dir / "h2_test.csv")
    atomic_savez(res_dir / "h2_test_replicates.npz", test_ages=np.asarray(ages), h2_raw_obs=out["h2_raw_obs"],
                 null_curves=out["null_curves"], t_null=out["t_null"], sd_null=out["sd_null"])
    return record


def run_corr_test(cfg, paths, data) -> dict:
    res_dir = paths["results"]
    corr_dir = ensure_dir(res_dir / "corr_test")
    y, t, edges = data["y"], data["t"], data["edges"]
    n = data["n_subjects"]
    grid = grid_points(cfg["grid"]["corr_t_min"], cfg["grid"]["corr_t_max"], cfg["grid"]["corr_size"])
    tag = f"{grid[0]:g}_{grid[-1]:g}"
    config = estimator_config(cfg["kernel"], n)
    source_cfg = {k: config[k] for k in ("kernel", "target_effective_n", "bandwidth_min", "bandwidth_max")}
    design = build_design(t, edges, grid, config)
    fit = fit_diag2(y, t, design)
    y_scale = float(data["y_scale"])
    atomic_to_csv(in_original_units(diag2_curve_frame(fit, method="diag2_source_corr_test"), y_scale), corr_dir / f"source_{tag}_curve.csv")
    atomic_to_csv(in_original_units(surface_rows_frame(fit), y_scale), corr_dir / f"source_{tag}_surface_rows.csv")
    source = {"raw": fit["raw"], "den": fit["den_surface"], "sigma_y2": fit["sigma_y2"]}

    scores_full = np.load(paths["random_scores_npy"], mmap_mode="r")
    if scores_full.shape[0] != data["n_samples"]:
        raise SystemExit("random GRM scores were built for a different sample-ID file")
    random_scores = np.asarray(scores_full[data["matched"]["global_idx"].to_numpy()], float)
    corr_cfg = cfg["corr_test"]
    try:
        out, seconds = timed(
            run_directional_p2, y, t, edges, random_scores, source_cfg, grid, source,
            basis_variance_capture=float(corr_cfg.get("basis_variance_capture", 0.999)),
            max_basis_rank=int(corr_cfg.get("max_basis_rank", 12)),
            nuisance_ess_exponent=float(corr_cfg.get("nuisance_ess_exponent", 0.8)),
            edge_chunk=int(corr_cfg.get("edge_chunk_size", 200000)),
            phenotype_key=paths["name"],
        )
    except (ValueError, RuntimeError) as error:
        status = "nonregular" if isinstance(error, ValueError) else "failed"
        row = {"phenotype_key": paths["name"], "status": status, "reason": str(error), "p_corr": math.nan}
        atomic_to_csv(pd.DataFrame([row]), res_dir / "corr_test.csv")
        print(f"  correlation test: {status} ({error})")
        return row
    row = out["row"]
    row["corr_min_residual_variance"] = float(row["corr_min_residual_variance"]) * y_scale ** 2   # back to original units
    row["elapsed_seconds"] = seconds
    row["source_fit_dir"] = relpath(corr_dir, paths["workdir"])
    atomic_to_csv(pd.DataFrame([row]), res_dir / "corr_test.csv")
    diagnostics = dict(out["diagnostics"])
    for key in ("null_surface", "raw_surface", "raw_genetic_local_quadratic", "total_variance_nuisance_grid"):
        diagnostics[key] = np.asarray(diagnostics[key], float) * y_scale ** 2
    diagnostics["y_scale"] = y_scale
    atomic_savez(res_dir / "corr_test_diagnostics.npz", **diagnostics)
    print(f"  correlation test: mean departure {row['corr_mean_departure']:.4f}, Z = {row['corr_mean_z']:.3f}, "
          f"one-sided p_corr = {row['p_corr']:.3e}; {row['corr_pair_count']} pairs on {row['corr_identifiable_grid_count']} ages; "
          f"basis rank {row['corr_basis_rank']}; {seconds:.1f}s")
    return row


def run_joint_test(paths, h2_row: dict, corr_row: dict) -> dict:
    """p_joint = ACAT(p_h2, p_corr); missing when the correlation test is nonregular."""
    p_h2 = float(h2_row["p_h2"])
    p_corr = float(corr_row.get("p_corr", math.nan))
    row = {"phenotype_key": paths["name"], "p_h2": p_h2, "p_corr": p_corr, "p_joint": acat([p_h2, p_corr])}
    atomic_to_csv(pd.DataFrame([row]), paths["results"] / "joint_test.csv")
    print(f"  joint test: p_joint = ACAT(p_h2, p_corr) = {row['p_joint']:.3e}")
    return row


def build_summary(cfg, paths) -> pd.DataFrame:
    """One row per completed trait under results/ (h2_test.csv + corr_test.csv + joint_test.csv + curves)."""
    ages = [float(a) for a in cfg["grid"]["test_ages"]]
    rows = []
    results_root = paths["results"].parent
    for trait_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        files = {name: trait_dir / f"{name}.csv" for name in ("h2_test", "corr_test", "joint_test")}
        if not all(path.exists() for path in files.values()):
            continue
        h2 = pd.read_csv(files["h2_test"]).iloc[0]
        corr = pd.read_csv(files["corr_test"]).iloc[0]
        joint = pd.read_csv(files["joint_test"]).iloc[0]
        curves = pd.read_csv(trait_dir / "curves_diag2.csv").sort_values("t")
        he = pd.read_csv(trait_dir / "curves_sparse_he_no_t.csv").iloc[0]
        manifest = load_json(trait_dir / "fit_manifest.json")
        row = {
            "phenotype_key": trait_dir.name,
            "n_subjects": int(manifest["n_subjects_matched"]),
            "edge_count": int(manifest["edge_count_subset"]),
            "h2_he": float(he["h2"]),
            "se_he": float(he.get("h2_se_conditional", np.nan)),
        }
        for age in ages:
            match = curves.loc[np.isclose(curves["t"], age)]
            row[f"h2_dynamic_{age:g}"] = float(match["h2"].iloc[0]) if len(match) else np.nan
            row[f"se_dynamic_{age:g}"] = float(match["h2_se_conditional"].iloc[0]) if len(match) else np.nan
        row["min_offdiag_corr"] = float(load_json(trait_dir / "diagnostics.json")["diag2"]["min_offdiag_corr"])
        row["p_h2"] = float(h2["p_h2"])
        row["p_corr"] = float(corr.get("p_corr", np.nan))
        row["p_joint"] = float(joint["p_joint"])
        row["p_h2_mc"] = float(h2["p_h2_mc"])
        row["T_h2"] = float(h2["T"])
        row["c_star_h2"] = float(h2["c_star"])
        row["corr_status"] = str(corr.get("status", ""))
        for column in ("corr_mean_z", "corr_mean_departure", "corr_pair_count", "corr_basis_rank"):
            row[column] = corr.get(column, np.nan)
        rows.append(row)
    summary = pd.DataFrame(rows)
    alpha = float(cfg["h2_test"].get("alpha", 0.05))
    for column, suffix in (("p_h2", "h2"), ("p_corr", "corr"), ("p_joint", "joint")):
        if len(summary) and summary[column].notna().all():
            add_multiple_testing(summary, column, suffix, alpha=alpha)
    return summary


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg, paths = resolve(args)
    res_dir = paths["results"]
    outputs = [res_dir / f"{name}.csv" for name in ("h2_test", "corr_test", "joint_test")]
    if all(path.exists() for path in outputs) and not args.force:
        print(f"SKIP: test outputs exist in {relpath(res_dir, paths['workdir'])}/ (use --force)")
    else:
        data = load_matched_dataset(paths["phenotype_tsv"], paths["sample_ids_csv"], paths["edges_npz"], paths["edges_json"])
        h2 = run_h2_test(cfg, paths, data)
        print(f"{paths['name']}: h2 test T = {h2['T']:.3f}, p_h2 = {h2['p_h2']:.3e} (Monte Carlo {h2['p_h2_mc']:.4f}, B = {h2['n_boot']}); "
              f"c* = {h2['c_star']:.4f}, null rank {h2['null_rank']}; gates {h2['gate_den_diag']:.1e} / {h2['gate_raw_diag']:.1e} / "
              f"{h2['gate_sigma_y2']:.1e}; {h2['seconds']:.1f}s")
        corr = run_corr_test(cfg, paths, data)
        run_joint_test(paths, h2, corr)
    summary = build_summary(cfg, paths)
    summary_dir = ensure_dir(paths["summary"])
    atomic_to_csv(summary, summary_dir / "all_traits.csv")
    save_json(summary_dir / "all_traits_meta.json", {"n_traits": int(len(summary)), "test_ages": cfg["grid"]["test_ages"],
                                                     "alpha": cfg["h2_test"].get("alpha", 0.05), "environment": environment()})
    print(f"  wrote {relpath(res_dir, paths['workdir'])}/h2_test.csv, corr_test.csv, joint_test.csv "
          f"and {relpath(summary_dir / 'all_traits.csv', paths['workdir'])} ({len(summary)} trait(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
