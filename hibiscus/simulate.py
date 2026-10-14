"""Toy data-generating process.

Truth: a low-rank genetic covariance surface
    Cg(t, s) = sum_k lambda_k phi_k(t) phi_k(s),    phi(t) = [1, z(t)]  (rank 2),
    z(t)     = clipped, rescaled sigmoid ramp,
    Y(t)     = Cg(t, t) + residual_variance,      h2(t) = Cg(t, t) / Y(t),
    R_G(t, s)   = Cg(t, s) / sqrt(Cg(t, t) Cg(s, s)).
Genotypes: X_ij ~ Binomial(2, p_j), p_j ~ U(maf_min, maf_max), column standardised.
Genetic scores a = Z theta, theta ~ N(0, 1/M);
g_i = sum_k sqrt(lambda_k) a_ik phi_k(t_i); y_i = g_i + e_i, e_i ~ N(0, residual_variance).
One visit per subject, age ~ U(age_min, age_max).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import __version__
from .estimate import surface_to_correlation
from .io import atomic_to_csv, ensure_dir, save_json, write_plink


def sigmoid(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def monotone_z(t_values, center: float, scale: float) -> np.ndarray:
    """Clipped, rescaled sigmoid ramp with an explicit centre and scale."""
    tau = (np.asarray(t_values, dtype=float) - float(center)) / float(scale)
    raw = sigmoid(1.8 * tau)
    lo = float(sigmoid(-3.6))
    hi = float(sigmoid(3.6))
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def truth_basis(t_values, truth: dict) -> np.ndarray:
    """phi(t) for the configured truth pattern."""
    t_values = np.asarray(t_values, dtype=float)
    pattern = str(truth.get("pattern", "monotone_rank2"))
    if pattern == "monotone_rank2":
        z = monotone_z(t_values, truth["z_center"], truth["z_scale"])
        return np.column_stack([np.ones_like(t_values), z])
    if pattern == "constant_rank1":
        return np.ones((t_values.size, 1), dtype=float)
    raise ValueError(f"unknown truth pattern: {pattern}")


def truth_lambdas(truth: dict) -> np.ndarray:
    lambdas = np.asarray(truth["lambdas"], dtype=float)
    if lambdas.ndim != 1 or np.any(lambdas <= 0.0):
        raise ValueError("truth lambdas must be a positive vector")
    return lambdas


def truth_covariance(t_row, t_col, truth: dict) -> np.ndarray:
    brow = truth_basis(t_row, truth)
    bcol = truth_basis(t_col, truth)
    lambdas = truth_lambdas(truth)
    if brow.shape[1] != lambdas.size:
        raise ValueError("truth basis rank and lambdas length differ")
    return (brow * lambdas[None, :]) @ bcol.T


def truth_sigma_g2(t_values, truth: dict) -> np.ndarray:
    basis = truth_basis(t_values, truth)
    return np.sum(np.square(basis) * truth_lambdas(truth)[None, :], axis=1)


def truth_h2_frame(grid_t, dataset_cfg: dict) -> pd.DataFrame:
    grid_t = np.asarray(grid_t, dtype=float)
    sg = truth_sigma_g2(grid_t, dataset_cfg["truth"])
    residual = np.repeat(float(dataset_cfg["residual_variance"]), grid_t.size)
    sy = sg + residual
    return pd.DataFrame(
        {
            "t": grid_t,
            "sigma_g2_truth": sg,
            "sigma_eps2_truth": residual,
            "sigma_y2_truth": sy,
            "h2_truth": sg / sy,
        }
    )


def truth_surface_frame(grid_t, dataset_cfg: dict) -> pd.DataFrame:
    grid_t = np.asarray(grid_t, dtype=float)
    cov = truth_covariance(grid_t, grid_t, dataset_cfg["truth"])
    corr = surface_to_correlation(cov)
    rows = []
    for i, tr in enumerate(grid_t):
        for j, tc in enumerate(grid_t):
            rows.append({"t_row": float(tr), "t_col": float(tc), "cov_truth": float(cov[i, j]), "corr_truth": float(corr[i, j])})
    return pd.DataFrame(rows)


def simulate_genotypes(n, m, maf_min, maf_max, rng):
    """Simulate column-standardised genotypes, also returning the raw 0/1/2 counts."""
    mafs = rng.uniform(float(maf_min), float(maf_max), size=int(m))
    raw = rng.binomial(2, mafs[None, :], size=(int(n), int(m)))
    x = raw.astype(float)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds < 1e-8] = 1.0
    x = (x - means[None, :]) / stds[None, :]
    return x, raw.astype(np.int8)


TOY_TRUTH = {
    "toy_dynamic": {"pattern": "monotone_rank2", "lambdas": [0.2, 1.0], "z_center": 55.0, "z_scale": 10.0},
    "toy_null": {"pattern": "constant_rank1", "lambdas": [0.4]},
}


def toy_config(name: str, n_subjects: int = 5000, n_snps: int = 1000, seed: int = 20260921) -> dict:
    """The two toy datasets: ages 40-70, MAF 0.05-0.50, residual variance 0.60; h2 rises 0.25 -> 0.65 (toy_dynamic)
    or stays at 0.40 (toy_null)."""
    if name not in TOY_TRUTH:
        raise ValueError(f"unknown toy dataset {name!r}; choose one of {sorted(TOY_TRUTH)}")
    return {
        "name": name, "seed": int(seed), "n_subjects": int(n_subjects), "n_snps": int(n_snps),
        "maf_min": 0.05, "maf_max": 0.5, "age_min": 40.0, "age_max": 70.0, "residual_variance": 0.6,
        "truth": dict(TOY_TRUTH[name]),
    }


def simulate_dataset(dataset_cfg: dict) -> dict:
    """Deterministic toy dataset. RNG draw order: ages -> genotypes -> theta -> residuals."""
    rng = np.random.RandomState(int(dataset_cfg["seed"]))
    n = int(dataset_cfg["n_subjects"])
    m = int(dataset_cfg["n_snps"])
    truth = dataset_cfg["truth"]
    ages = rng.uniform(float(dataset_cfg["age_min"]), float(dataset_cfg["age_max"]), size=n)
    z, raw = simulate_genotypes(n, m, dataset_cfg["maf_min"], dataset_cfg["maf_max"], rng)
    lambdas = truth_lambdas(truth)
    theta = rng.normal(size=(m, len(lambdas))) / math.sqrt(float(m))
    score_components = z @ theta
    basis_obs = truth_basis(ages, truth)
    g = np.sum(np.sqrt(lambdas)[None, :] * score_components * basis_obs, axis=1)
    residual_variance = float(dataset_cfg["residual_variance"])
    eps = math.sqrt(residual_variance) * rng.normal(size=n)
    return {"ages": ages, "genotypes_raw": raw, "genotypes_standardized": z, "y": g + eps}


def write_dataset(out_dir, dataset: dict, dataset_cfg: dict, grid_t) -> dict:
    """Write the toy dataset in the real input formats plus truth files."""
    out_dir = ensure_dir(out_dir)
    n, m = dataset["genotypes_raw"].shape
    iids = [str(1000001 + i) for i in range(n)]
    fam = pd.DataFrame({"FID": ["0"] * n, "IID": iids})
    bim = pd.DataFrame(
        {
            "CHR": ["1"] * m,
            "SNP": [f"rs{j + 1}" for j in range(m)],
            "CM": ["0"] * m,
            "BP": [str(1000 * (j + 1)) for j in range(m)],
            "A1": ["A"] * m,
            "A2": ["G"] * m,
        }
    )
    plink_prefix = out_dir / "geno"
    write_plink(plink_prefix, dataset["genotypes_raw"], fam, bim)
    pheno = pd.DataFrame({"FID": fam["FID"], "IID": fam["IID"], "y": dataset["y"], "t": dataset["ages"]})
    pheno.to_csv(out_dir / "phenotype.tsv", sep="\t", index=False, float_format="%.10g")
    fam[["FID", "IID"]].to_csv(out_dir / "sample_ids.csv", index=False)
    atomic_to_csv(truth_h2_frame(grid_t, dataset_cfg), out_dir / "truth_h2.csv")
    atomic_to_csv(truth_surface_frame(grid_t, dataset_cfg), out_dir / "truth_surface.csv")
    truth = dataset_cfg["truth"]
    grid_t = np.asarray(grid_t, dtype=float)
    cov_grid = truth_covariance(grid_t, grid_t, truth)
    h2_grid = truth_sigma_g2(grid_t, truth) / (truth_sigma_g2(grid_t, truth) + float(dataset_cfg["residual_variance"]))
    corr_grid = surface_to_correlation(cov_grid)
    info = {
        "hibiscus_version": __version__,
        "name": dataset_cfg["name"],
        "truth_pattern": dataset_cfg["truth"]["pattern"],
        "seed": int(dataset_cfg["seed"]),
        "n_subjects": int(n),
        "n_snps": int(m),
        "visits_per_subject": 1,
        "age_min": float(dataset_cfg["age_min"]),
        "age_max": float(dataset_cfg["age_max"]),
        "maf_min": float(dataset_cfg["maf_min"]),
        "maf_max": float(dataset_cfg["maf_max"]),
        "residual_variance": float(dataset_cfg["residual_variance"]),
        "truth": truth,
        "truth_summary": {
            "h2_at_grid_min": float(h2_grid[0]),
            "h2_at_grid_max": float(h2_grid[-1]),
            "corr_grid_min_grid_max": float(corr_grid[0, -1]),
            "grid_t_min": float(grid_t[0]),
            "grid_t_max": float(grid_t[-1]),
        },
        "files": {
            "plink_prefix": "geno",
            "phenotype_tsv": "phenotype.tsv",
            "sample_ids_csv": "sample_ids.csv",
            "truth_h2_csv": "truth_h2.csv",
            "truth_surface_csv": "truth_surface.csv",
        },
    }
    save_json(out_dir / "dataset_info.json", info)
    return info
