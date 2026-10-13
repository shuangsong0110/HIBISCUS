"""Kernel-weighted Haseman-Elston surface estimator ("diag2") and the flat-kernel HE estimate.

Estimator (single visit, r = y - mean(y), Gaussian weights w_ia):
  num[a,b] = sum_E K_ij r_i r_j (w_ia w_jb + w_ib w_ja)/2,  den[a,b] = sum_E K_ij^2 (w_ia w_jb + w_ib w_ja)/2
  C_raw = num / den  ->  PSD projection (eigenvalue clipping)  ->  sigma_g^2(t) = smooth_2(diag PSD)
  sigma_y^2(t) = local-linear weighted residual variance;  h2(t) = clip(sigma_g^2 / sigma_y^2, 0, 1)
  R_G from the UNSMOOTHED PSD surface.
The drivers pass the standardised phenotype returned by io.load_matched_dataset and convert the
variance-type outputs back to the original units with in_original_units().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import EPS
from .kernel import bandwidths_and_weights, local_sigma_y2_grid, smooth_grid_curve


# ----------------------------------------------------------------------------- configuration
def estimator_config(kernel_cfg: dict, n_matched: int) -> dict:
    """Estimator config: target ESS = fraction * N_matched, eigenvalue-clip PSD, post-smooth 2."""
    config = {
        "kernel": str(kernel_cfg.get("kernel", "gaussian")),
        "target_effective_n": float(kernel_cfg["target_effective_n_fraction"]) * float(n_matched),
        "target_effective_n_fraction": float(kernel_cfg["target_effective_n_fraction"]),
        "bandwidth_min": float(kernel_cfg.get("bandwidth_min", 3.0)),
        "bandwidth_max": float(kernel_cfg.get("bandwidth_max", 12.0)),
        "cg_surface_post_smooth_bandwidth": float(kernel_cfg.get("cg_surface_post_smooth_bandwidth", 2.0)),
        "cg_surface_single_visit_chunk_size": int(kernel_cfg.get("cg_surface_single_visit_chunk_size", 200000)),
    }
    return config


# ----------------------------------------------------------------------------- PSD projection
def project_psd(surface):
    """Project the symmetrised raw surface onto the PSD cone by eigenvalue clipping."""
    surface = 0.5 * (np.asarray(surface, dtype=float) + np.asarray(surface, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(surface)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))


def surface_to_correlation(cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    diag = np.clip(np.diag(cov), EPS, None)
    corr = cov / np.sqrt(np.outer(diag, diag))
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


# ----------------------------------------------------------------------------- edge passes
def accumulate_surfaces_single_visit(subject_weights, residuals, edges: dict, chunk_size: int = 200000):
    """One pass over the edges: numerator and denominator surfaces (L x L), symmetrised."""
    row = np.asarray(edges["row"], dtype=np.int32)
    col = np.asarray(edges["col"], dtype=np.int32)
    kij = np.asarray(edges["kij"], dtype=float)
    subject_weights = np.asarray(subject_weights, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    grid_size = int(subject_weights.shape[1])
    cg_num_surface = np.zeros((grid_size, grid_size), dtype=float)
    cg_den_surface = np.zeros((grid_size, grid_size), dtype=float)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, len(kij), chunk_size):
        stop = min(start + chunk_size, len(kij))
        row_chunk = row[start:stop]
        col_chunk = col[start:stop]
        kij_chunk = kij[start:stop]
        wi = subject_weights[row_chunk]
        wj = subject_weights[col_chunk]
        ri = residuals[row_chunk]
        rj = residuals[col_chunk]
        left_num = (kij_chunk * ri)[:, None] * wi
        right_num = rj[:, None] * wj
        num_cross = np.dot(left_num.T, right_num)
        left_den = np.square(kij_chunk)[:, None] * wi
        den_cross = np.dot(left_den.T, wj)
        cg_num_surface += 0.5 * (num_cross + num_cross.T)
        cg_den_surface += 0.5 * (den_cross + den_cross.T)
    return cg_num_surface, cg_den_surface


def accumulate_surfaces_dense_single_visit(subject_weights, residuals, grm_subject, chunk_rows: int = 512):
    """Dense-GRM counterpart (all off-diagonal pairs); used by the tests to validate the edge pass."""
    subject_weights = np.asarray(subject_weights, dtype=float)
    subject_residuals = np.asarray(residuals, dtype=float)
    grid_size = int(subject_weights.shape[1])
    grm_subject = np.asarray(grm_subject, dtype=np.float32)
    diag = np.asarray(np.diag(grm_subject), dtype=float)
    weighted_residuals = subject_weights * subject_residuals[:, None]
    cg_num_surface = np.zeros((grid_size, grid_size), dtype=float)
    cg_den_surface = np.zeros((grid_size, grid_size), dtype=float)
    chunk_rows = max(int(chunk_rows), 1)
    for start in range(0, grm_subject.shape[0], chunk_rows):
        stop = min(start + chunk_rows, grm_subject.shape[0])
        grm_block = np.asarray(grm_subject[start:stop], dtype=float)
        w_block = subject_weights[start:stop]
        rw_block = weighted_residuals[start:stop]
        cg_num_surface += np.dot(rw_block.T, np.dot(grm_block, weighted_residuals))
        cg_den_surface += np.dot(w_block.T, np.dot(np.square(grm_block), subject_weights))
    diag_num = np.dot(subject_weights.T * (diag * np.square(subject_residuals))[None, :], subject_weights)
    diag_den = np.dot(subject_weights.T * np.square(diag)[None, :], subject_weights)
    cg_num_surface = 0.5 * (cg_num_surface - diag_num)
    cg_den_surface = 0.5 * (cg_den_surface - diag_den)
    return cg_num_surface, cg_den_surface


def raw_surface_from_num_den(num_surface, den_surface):
    return np.divide(
        num_surface,
        np.maximum(den_surface, EPS),
        out=np.zeros_like(num_surface, dtype=float),
        where=np.asarray(den_surface, dtype=float) > EPS,
    )


# ----------------------------------------------------------------------------- design + fit
def build_design(t, edges: dict, grid_t, config: dict) -> dict:
    """Weights and the phenotype-free denominator surface (computed once per dataset/grid)."""
    t = np.asarray(t, dtype=float)
    grid_t = np.asarray(grid_t, dtype=float)
    bandwidths, obs_weight_matrix = bandwidths_and_weights(t, grid_t, config)
    n_subjects = int(len(t))
    row = np.asarray(edges["row"], dtype=np.int32)
    col = np.asarray(edges["col"], dtype=np.int32)
    kij = np.asarray(edges["kij"], dtype=float)
    if len(row) and (row.max() >= n_subjects or col.max() >= n_subjects):
        raise ValueError("edge indices exceed the number of matched subjects (edges must be local)")
    chunk_size = max(int(config.get("cg_surface_single_visit_chunk_size", 200000)), 1)
    den_surface = np.zeros((len(grid_t), len(grid_t)), dtype=float)
    for start in range(0, len(kij), chunk_size):
        stop = min(start + chunk_size, len(kij))
        wi = obs_weight_matrix[row[start:stop]]
        wj = obs_weight_matrix[col[start:stop]]
        left_den = np.square(kij[start:stop])[:, None] * wi
        den_cross = left_den.T @ wj
        den_surface += 0.5 * (den_cross + den_cross.T)
    return {
        "grid_t": grid_t,
        "bandwidths": np.asarray(bandwidths, dtype=float),
        "subject_weights": obs_weight_matrix,
        "row": row,
        "col": col,
        "kij": kij,
        "chunk_size": int(chunk_size),
        "den_surface": den_surface,
        "config": dict(config),
        "n_subjects": n_subjects,
        "edge_count": int(len(kij)),
        "row_weight_capture_mean": float(edges.get("row_weight_capture_mean", np.nan)),
    }


def numerator_surface(residual, design: dict) -> np.ndarray:
    row, col, kij = design["row"], design["col"], design["kij"]
    subject_weights = design["subject_weights"]
    residual = np.asarray(residual, dtype=float)
    num_surface = np.zeros_like(design["den_surface"], dtype=float)
    chunk_size = int(design["chunk_size"])
    for start in range(0, len(kij), chunk_size):
        stop = min(start + chunk_size, len(kij))
        row_chunk = row[start:stop]
        col_chunk = col[start:stop]
        wi = subject_weights[row_chunk]
        wj = subject_weights[col_chunk]
        ri = residual[row_chunk]
        rj = residual[col_chunk]
        left_num = (kij[start:stop] * ri)[:, None] * wi
        right_num = rj[:, None] * wj
        num_cross = left_num.T @ right_num
        num_surface += 0.5 * (num_cross + num_cross.T)
    return num_surface


def smooth_sigma_g2(cov: np.ndarray, grid_values: np.ndarray, config: dict) -> np.ndarray:
    sigma_g2 = np.maximum(np.diag(cov), 0.0)
    bandwidth = float(config.get("cg_surface_post_smooth_bandwidth", 0.0))
    if bandwidth > 0.0:
        sigma_g2 = np.maximum(smooth_grid_curve(grid_values, sigma_g2, bandwidth), 0.0)
    return np.asarray(sigma_g2, dtype=float)


def fit_diag2(y, t, design: dict) -> dict:
    """Point estimates on the design grid. Returns arrays; use *_frame helpers for the CSV schemas."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    grid_t = design["grid_t"]
    config = design["config"]
    residual = y - float(np.mean(y))
    num_surface = numerator_surface(residual, design)
    den_surface = design["den_surface"]
    raw = raw_surface_from_num_den(num_surface, den_surface)
    cov = project_psd(0.5 * (raw + raw.T))
    sigma_y2 = np.maximum(local_sigma_y2_grid(y, t, grid_t, config), EPS)
    sigma_g2 = smooth_sigma_g2(cov, grid_t, config)
    sigma_eps2 = np.maximum(sigma_y2 - sigma_g2, 0.0)
    h2 = np.clip(sigma_g2 / np.maximum(sigma_y2, EPS), 0.0, 1.0)
    corr = surface_to_correlation(cov)
    return {
        "grid_t": grid_t,
        "num_surface": num_surface,
        "den_surface": den_surface,
        "raw": raw,
        "psd": cov,
        "corr": corr,
        "sigma_g2": sigma_g2,
        "sigma_y2": sigma_y2,
        "sigma_eps2": sigma_eps2,
        "h2": h2,
        "bandwidths": design["bandwidths"],
        "config": dict(config),
        "edge_count": int(design["edge_count"]),
    }


def sparse_he_no_t(y, edges: dict) -> dict:
    """Flat-kernel HE on the same edges: sigma_g^2 = sum K r r / sum K^2 (clipped to [0, sigma_y^2])."""
    y = np.asarray(y, dtype=float)
    y_centered = y - np.mean(y)
    row = np.asarray(edges["row"], dtype=np.int32)
    col = np.asarray(edges["col"], dtype=np.int32)
    kij = np.asarray(edges["kij"], dtype=float)
    yij = y_centered[row] * y_centered[col]
    off_num = float(np.sum(kij * yij))
    off_den = float(np.sum(np.square(kij)))
    sigma_g2_raw = float(off_num / max(off_den, EPS))
    sigma_y2 = float(np.mean(np.square(y_centered)))
    sigma_g2 = float(np.clip(sigma_g2_raw, 0.0, sigma_y2))
    sigma_eps2 = float(max(sigma_y2 - sigma_g2, 0.0))
    h2 = float(np.clip(sigma_g2 / max(sigma_y2, EPS), 0.0, 1.0))
    return {
        "sigma_g2": sigma_g2,
        "sigma_g2_raw": sigma_g2_raw,
        "sigma_y2": sigma_y2,
        "sigma_eps2": sigma_eps2,
        "h2": h2,
        "A": off_num,
        "B": off_den,
        "edge_count": int(len(kij)),
        "n": int(len(y)),
    }


# ----------------------------------------------------------------------------- CSV schemas
VARIANCE_COLUMNS = ("sigma_g2", "sigma_eps2", "sigma_y2", "cov", "cg_raw", "cg_psd")


def in_original_units(frame: pd.DataFrame, y_scale: float) -> pd.DataFrame:
    """Multiply the variance-type columns (units of y^2) by y_scale^2; every other column is unit-free."""
    out = frame.copy()
    for column in VARIANCE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].to_numpy(float) * float(y_scale) ** 2
    return out


def curve_frame(method: str, grid_t, sigma_g2, sigma_eps2, sigma_y2, h2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": np.asarray(grid_t, dtype=float),
            "method": method,
            "sigma_g2": np.asarray(sigma_g2, dtype=float),
            "sigma_eps2": np.asarray(sigma_eps2, dtype=float),
            "sigma_y2": np.asarray(sigma_y2, dtype=float),
            "h2": np.asarray(h2, dtype=float),
        }
    )


def diag2_curve_frame(fit: dict, method: str = "diag2") -> pd.DataFrame:
    return curve_frame(method, fit["grid_t"], fit["sigma_g2"], fit["sigma_eps2"], fit["sigma_y2"], fit["h2"])


def he_curve_frame(he: dict, grid_t, method: str = "sparse_he_no_t") -> pd.DataFrame:
    size = len(np.asarray(grid_t))
    return curve_frame(
        method, grid_t, np.repeat(he["sigma_g2"], size), np.repeat(he["sigma_eps2"], size),
        np.repeat(he["sigma_y2"], size), np.repeat(he["h2"], size),
    )


def surface_frame(method: str, grid_t, cov: np.ndarray) -> pd.DataFrame:
    grid_t = np.asarray(grid_t, dtype=float)
    corr = surface_to_correlation(cov)
    rows = []
    for i, t_row in enumerate(grid_t):
        for j, t_col in enumerate(grid_t):
            rows.append({"t_row": float(t_row), "t_col": float(t_col), "method": method, "cov": float(cov[i, j]), "corr": float(corr[i, j])})
    return pd.DataFrame(rows)


def surface_rows_frame(fit: dict) -> pd.DataFrame:
    """surface_rows_cg_surface_diag2.csv: raw, PSD, correlation and denominator surfaces (long)."""
    grid_t = fit["grid_t"]
    rows = []
    for i, t_row in enumerate(grid_t):
        for j, t_col in enumerate(grid_t):
            rows.append(
                {
                    "t_row": float(t_row),
                    "t_col": float(t_col),
                    "cg_raw": float(fit["raw"][i, j]),
                    "cg_psd": float(fit["psd"][i, j]),
                    "cg_corr": float(fit["corr"][i, j]),
                    "cg_denom": float(fit["den_surface"][i, j]),
                }
            )
    return pd.DataFrame(rows)


def pivot_surface(rows: pd.DataFrame, column: str) -> np.ndarray:
    return rows.pivot(index="t_row", columns="t_col", values=column).sort_index().sort_index(axis=1).to_numpy(float)
