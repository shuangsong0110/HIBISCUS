#!/usr/bin/env python3
"""Step 5 - tests of age variation on the test age range: the h2 test (H0: h2(t) = c), the correlation
test (H0: R_G = 1) and their ACAT combination p_joint.

Reads  results/<name>/curves.csv, fit_manifest.json (step 3); data + edges; grm/<name>/genetic_scores.npy, random_scores.npy
Writes results/<name>/tests.csv      p_h2, p_corr, p_joint
       summary/all_traits.csv        one row per trait directory under results/
--test-range T_MIN T_MAX sets the ages tested (default: the observed age range shrunk by 5 years at each
end); the h2 test uses every 5th year of the range, the correlation test a 25-point grid on it.
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
from hibiscus.acat import acat
from hibiscus.cli import common_parser, relpath, resolve, timed
from hibiscus.corr_test import run_directional_p2
from hibiscus.defaults import CORR_GRID_SIZE, CORR_TEST, H2_TEST, TEST_AGE_STEP, TEST_MARGIN
from hibiscus.estimate import build_design, fit_hibiscus
from hibiscus.h2_test import flat_test, raw_curves_many, target_indices
from hibiscus.io import atomic_to_csv, ensure_dir, load_json, load_matched_dataset
from hibiscus.kernel import smoothing_matrix

GATE = 1e-10          # the batched estimator of the h2 test must reproduce the step-3 fit


def trait_seed(base: int, name: str) -> int:
    """Stable per-trait seed (Python's hash() is salted per interpreter)."""
    return int(zlib.crc32(f"{int(base)}:{name}".encode()))


def test_range(paths: dict, grid: np.ndarray) -> tuple[float, float]:
    lo, hi = paths["test_range"] or (float(grid[0]) + TEST_MARGIN, float(grid[-1]) - TEST_MARGIN)
    if not (grid[0] <= lo < hi <= grid[-1]):
        raise SystemExit(f"test range [{lo:g}, {hi:g}] must lie inside the age grid [{grid[0]:g}, {grid[-1]:g}]")
    return float(lo), float(hi)


def load_scores(path: Path, n_samples: int, global_idx: np.ndarray) -> np.ndarray:
    full = np.load(path, mmap_mode="r")
    if full.shape[0] != n_samples:
        raise SystemExit(f"{path.name} was built for a different sample-ID file")
    return np.asarray(full[global_idx], float)


def run_h2_test(paths: dict, data: dict, grid: np.ndarray, config: dict, design: dict, fit: dict, ages: np.ndarray) -> dict:
    y, t = data["y"], data["t"]
    idx = target_indices(grid, ages)
    smoother = smoothing_matrix(grid, float(config["cg_surface_post_smooth_bandwidth"]))
    obs = raw_curves_many(y, t, design, smoother)
    gates = {
        "raw_diag": float(np.max(np.abs(obs["raw_diag"][:, 0] - np.diag(fit["raw"]))) / max(float(np.max(np.abs(np.diag(fit["raw"])))), EPS)),
        "sigma_y2": float(np.max(np.abs(obs["sigma_y2"][:, 0] - fit["sigma_y2"])) / max(float(np.max(fit["sigma_y2"])), EPS)),
    }
    for name, value in gates.items():
        if not value <= GATE:
            raise SystemExit(f"h2 test: {name} does not reproduce the step-3 fit ({value:.2e} > {GATE:g})")
    bank_meta = load_json(paths["genetic_scores_npy"].with_name("genetic_scores_meta.json"))
    if abs(float(bank_meta["grm_scale"]) - 1.0) > 0.05:
        raise SystemExit(f"genetic score bank GRM scale {bank_meta['grm_scale']:.4f} is far from 1: bank and edges do not match")
    bank = load_scores(paths["genetic_scores_npy"], data["n_samples"], data["matched"]["global_idx"].to_numpy())
    out, seconds = timed(
        flat_test, y, t, design, fit["raw"], fit["sigma_y2"], idx, bank,
        n_boot=int(H2_TEST["n_boot"]), max_rank=int(H2_TEST["max_rank"]), seed=trait_seed(H2_TEST["seed"], paths["name"]),
        c_star_bounds=tuple(H2_TEST["c_star_bounds"]),
    )
    print(f"  h2 test at ages {', '.join(f'{a:g}' for a in ages)}: T = {out['T']:.3f}, p_h2 = {out['p_h2']:.3e} "
          f"(Monte Carlo {out['p_h2_mc']:.4f}, B = {out['n_boot']}); c* = {out['c_star']:.4f}; {seconds:.1f}s")
    return out


def run_corr_test(paths: dict, data: dict, config: dict, lo: float, hi: float) -> dict:
    y, t, edges = data["y"], data["t"], data["edges"]
    corr_grid = np.linspace(lo, hi, CORR_GRID_SIZE)
    source_cfg = {k: config[k] for k in ("kernel", "target_effective_n", "bandwidth_min", "bandwidth_max")}
    design = build_design(t, edges, corr_grid, config)
    fit = fit_hibiscus(y, t, design)
    scores = load_scores(paths["random_scores_npy"], data["n_samples"], data["matched"]["global_idx"].to_numpy())
    try:
        out, seconds = timed(
            run_directional_p2, y, t, edges, scores, source_cfg, corr_grid,
            {"raw": fit["raw"], "den": fit["den_surface"], "sigma_y2": fit["sigma_y2"]},
            basis_variance_capture=float(CORR_TEST["basis_variance_capture"]), max_basis_rank=int(CORR_TEST["max_basis_rank"]),
            edge_chunk=int(CORR_TEST["edge_chunk_size"]),
        )
    except (ValueError, RuntimeError) as error:
        print(f"  correlation test: not computable ({error}); p_corr is missing")
        return {"p_corr": math.nan}
    print(f"  correlation test on [{lo:g}, {hi:g}]: Z = {out['z']:.3f}, p_corr = {out['p_corr']:.3e} "
          f"({out['pair_count']} age pairs, basis rank {out['basis_rank']}); {seconds:.1f}s")
    return out


def build_summary(paths: dict, ages: np.ndarray) -> pd.DataFrame:
    rows = []
    for trait_dir in sorted(p for p in paths["results"].parent.iterdir() if p.is_dir()):
        needed = [trait_dir / f for f in ("tests.csv", "curves.csv", "surfaces.csv", "fit_manifest.json")]
        if not all(path.exists() for path in needed):
            continue
        tests = pd.read_csv(needed[0]).iloc[0]
        curves = pd.read_csv(needed[1])
        surfaces = pd.read_csv(needed[2])
        manifest = load_json(needed[3])
        hib = curves.loc[curves["method"] == "hibiscus"].sort_values("t")
        he = curves.loc[curves["method"] == "he"].iloc[0]
        row = {
            "phenotype_key": trait_dir.name,
            "n_subjects": int(manifest["n_subjects_matched"]),
            "edge_count": int(manifest["edge_count_subset"]),
            "h2_he": float(he["h2"]),
            "se_he": float(he.get("h2_se_conditional", np.nan)),
        }
        for age in ages:
            match = hib.loc[np.isclose(hib["t"], age)]
            row[f"h2_{age:g}"] = float(match["h2"].iloc[0]) if len(match) else np.nan
            row[f"se_{age:g}"] = float(match["h2_se_conditional"].iloc[0]) if len(match) and "h2_se_conditional" in match else np.nan
        off = surfaces.loc[surfaces["t_row"] != surfaces["t_col"], "corr"]
        row["min_offdiag_corr"] = float(off.min()) if len(off) else np.nan
        for column in ("p_h2", "p_corr", "p_joint"):
            row[column] = float(tests[column])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    paths = resolve(args)
    res_dir = paths["results"]
    manifest = load_json(res_dir / "fit_manifest.json")
    grid = np.asarray(manifest["grid_t"], float)
    config = manifest["config"]
    lo, hi = test_range(paths, grid)
    ages = np.arange(lo, hi + 1e-9, TEST_AGE_STEP)
    if (res_dir / "tests.csv").exists() and not args.force:
        print(f"SKIP: {relpath(res_dir / 'tests.csv', paths['workdir'])} exists (use --force)")
    else:
        data = load_matched_dataset(paths["phenotype_tsv"], paths["sample_ids_csv"], paths["edges_npz"], paths["edges_json"])
        if data["n_subjects"] != int(manifest["n_subjects_matched"]):
            raise SystemExit("matched subject count differs from fit_manifest.json; rerun step 3")
        if abs(data["y_scale"] - float(manifest["y_scale"])) > 1e-9 * float(manifest["y_scale"]):
            raise SystemExit("phenotype scale differs from fit_manifest.json (phenotype file changed?); rerun step 3")
        design = build_design(data["t"], data["edges"], grid, config)
        fit = fit_hibiscus(data["y"], data["t"], design)
        saved = pd.read_csv(res_dir / "curves.csv")
        saved = saved.loc[saved["method"] == "hibiscus"].sort_values("t")["h2"].to_numpy(float)
        if float(np.max(np.abs(fit["h2"] - saved))) > GATE:
            raise SystemExit("recomputed h2(t) differs from curves.csv; rerun step 3")
        print(f"{paths['name']}: tests on ages [{lo:g}, {hi:g}]")
        h2 = run_h2_test(paths, data, grid, config, design, fit, ages)
        corr = run_corr_test(paths, data, config, lo, hi)
        p_joint = acat([h2["p_h2"], corr["p_corr"]])
        print(f"  joint test: p_joint = ACAT(p_h2, p_corr) = {p_joint:.3e}")
        atomic_to_csv(pd.DataFrame([{"phenotype_key": paths["name"], "p_h2": h2["p_h2"], "p_corr": corr["p_corr"], "p_joint": p_joint}]),
                      res_dir / "tests.csv")
    summary = build_summary(paths, ages)
    summary_path = ensure_dir(paths["summary"]) / "all_traits.csv"
    atomic_to_csv(summary, summary_path)
    print(f"  wrote {relpath(res_dir / 'tests.csv', paths['workdir'])} and {relpath(summary_path, paths['workdir'])} ({len(summary)} trait(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
