"""Conditional delta-method standard errors for h2(t) and for the age-independent (flat-kernel) HE estimate.

Target: Var{ h2_hat(t) | X, edges, ages, realised genetic effects } with residual variance
sigma_e,i^2 = Y_hat(t_i) - G_hat(t_i) interpolated to each subject's age.
  Var(theta_hat) = sum_i sigma_e,i^2 phi_i^2,   phi_i = d theta_hat / d y_i.

The dynamic influence uses the finite-difference sensitivity of the composite map
num -> smooth(diag PSD(num / den)), so the post-smoothing enters the derivative exactly once.
U = K_off diag(r) [W | 1] is accumulated in one edge pass. The HE influence is
((q_i - q_bar) / sum K^2) / sigma_y^2 - (sigma_g,HE^2 / sigma_y^4) (2 r_i / N) with q = K_off r.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import EPS, Z975
from .estimate import project_psd, raw_surface_from_num_den
from .kernel import grid_local_variance_and_y_derivative, smooth_grid_curve


# ----------------------------------------------------------------------------- sensitivity of the curve map
def sigma_g_curve_from_raw_surfaces(cg_num_surface, cg_den_surface, grid_t, config):
    """(num, den) -> smoothed sigma_g^2 curve (the map differentiated below)."""
    cg_raw = raw_surface_from_num_den(cg_num_surface, cg_den_surface)
    cg_psd = project_psd(0.5 * (cg_raw + cg_raw.T))
    sigma_g2_grid = np.maximum(np.diag(cg_psd), 0.0)
    post_smooth_bandwidth = float(config.get("cg_surface_post_smooth_bandwidth", 0.0))
    if post_smooth_bandwidth > 0.0:
        sigma_g2_grid = np.maximum(smooth_grid_curve(grid_t, sigma_g2_grid, post_smooth_bandwidth), 0.0)
    return sigma_g2_grid


def symmetric_basis(grid_size: int):
    basis = []
    for a in range(grid_size):
        for b in range(a, grid_size):
            mat = np.zeros((grid_size, grid_size), dtype=float)
            mat[a, b] = 1.0
            if a != b:
                mat[b, a] = 1.0
            basis.append((a, b, mat))
    return basis


def finite_difference_sensitivity(cg_num_surface, cg_den_surface, grid_t, config, epsilon_scale=1e-6):
    """J[g, a, b] = d sigma_g^2(t_g) / d num[a, b] by central differences."""
    grid_size = int(np.asarray(grid_t).size)
    sens_num = np.zeros((grid_size, grid_size, grid_size), dtype=float)
    eps_num = float(epsilon_scale * max(float(np.max(np.abs(cg_num_surface))), 1.0))
    for a, b, mat in symmetric_basis(grid_size):
        plus_num = sigma_g_curve_from_raw_surfaces(cg_num_surface + eps_num * mat, cg_den_surface, grid_t, config)
        minus_num = sigma_g_curve_from_raw_surfaces(cg_num_surface - eps_num * mat, cg_den_surface, grid_t, config)
        deriv_num = (plus_num - minus_num) / (2.0 * eps_num)
        if a == b:
            sens_num[:, a, a] = deriv_num
        else:
            sens_num[:, a, b] = 0.5 * deriv_num
            sens_num[:, b, a] = 0.5 * deriv_num
    return sens_num


# ----------------------------------------------------------------------------- influence products
def influence_products(residual, subject_weights, edges: dict, chunk_size: int = 200000):
    """One edge pass: U = K_off diag(r) W  (N x L) and q = K_off r  (N)."""
    residual = np.asarray(residual, dtype=float)
    subject_weights = np.asarray(subject_weights, dtype=float)
    row = np.asarray(edges["row"], dtype=np.int32)
    col = np.asarray(edges["col"], dtype=np.int32)
    kij = np.asarray(edges["kij"], dtype=float)
    n_subjects, grid_size = subject_weights.shape
    u = np.zeros((n_subjects, grid_size), dtype=float)
    q = np.zeros(n_subjects, dtype=float)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, len(kij), chunk_size):
        stop = min(start + chunk_size, len(kij))
        rr, cc, kk = row[start:stop], col[start:stop], kij[start:stop]
        k_rj = kk * residual[cc]
        k_ri = kk * residual[rr]
        q += np.bincount(rr, weights=k_rj, minlength=n_subjects)
        q += np.bincount(cc, weights=k_ri, minlength=n_subjects)
        for g in range(grid_size):
            u[:, g] += np.bincount(rr, weights=k_rj * subject_weights[cc, g], minlength=n_subjects)
            u[:, g] += np.bincount(cc, weights=k_ri * subject_weights[rr, g], minlength=n_subjects)
    return u, q


def dynamic_sigma_g2_influence(subject_weights, u, sens_num):
    """psi[i, g] = d sigma_g^2(t_g) / d y_i = w_i' J_g U_i, mean-centred over subjects (smoothed once)."""
    subject_weights = np.asarray(subject_weights, dtype=float)
    u = np.asarray(u, dtype=float)
    grid_size = subject_weights.shape[1]
    direct = np.empty_like(u)
    for g in range(grid_size):
        direct[:, g] = np.einsum("ij,ij->i", subject_weights @ sens_num[g], u)
    return direct - direct.mean(axis=0, keepdims=True)


def h2_influence(psi_sigma_g2, d_sigma_y2, sigma_g2_grid, sigma_y2_grid):
    """Quotient rule: d h2 / d y = psi / sigma_y^2 - (sigma_g^2 / sigma_y^4) d sigma_y^2 / d y."""
    ratio_curve = sigma_g2_grid[None, :] / np.maximum(np.square(sigma_y2_grid)[None, :], EPS)
    return psi_sigma_g2 / np.maximum(sigma_y2_grid[None, :], EPS) - ratio_curve * d_sigma_y2


def he_influence(residual, q, he: dict):
    """phi_HE,i = ((q_i - mean q) / B) / sigma_y^2 - (sigma_g^2 / sigma_y^4) (2 r_i / N)."""
    residual = np.asarray(residual, dtype=float)
    n = int(residual.size)
    b_den = max(float(he["B"]), EPS)
    sigma_y2 = max(float(he["sigma_y2"]), EPS)
    sigma_g2 = float(he["sigma_g2"])
    d_a = q - float(np.mean(q))
    return (d_a / b_den) / sigma_y2 - (sigma_g2 / sigma_y2**2) * (2.0 / n) * residual


def residual_variance_by_subject(t, grid_t, sigma_y2_grid, sigma_g2_grid):
    sigma_e2_grid = np.maximum(np.asarray(sigma_y2_grid, float) - np.asarray(sigma_g2_grid, float), EPS)
    return np.interp(np.asarray(t, float), np.asarray(grid_t, float), sigma_e2_grid,
                     left=float(sigma_e2_grid[0]), right=float(sigma_e2_grid[-1]))


# ----------------------------------------------------------------------------- full SE pass
def conditional_se(y, t, fit: dict, design: dict, edges: dict, he: dict, fd_eps_scale: float = 1e-6) -> dict:
    """Conditional SE for h2_D(t) on the grid and for h2_HE (constant residual variance)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    grid_t = np.asarray(fit["grid_t"], dtype=float)
    config = fit["config"]
    weights = np.asarray(design["subject_weights"], dtype=float)
    residual = y - float(np.mean(y))

    sigma_g2_grid = sigma_g_curve_from_raw_surfaces(fit["num_surface"], fit["den_surface"], grid_t, config)
    sigma_y2_grid, d_sigma_y2 = grid_local_variance_and_y_derivative(y, t, grid_t, weights)
    curve_error = float(np.max(np.abs(sigma_g2_grid - fit["sigma_g2"])) / max(float(np.max(np.abs(fit["sigma_g2"]))), EPS))
    total_error = float(np.max(np.abs(sigma_y2_grid - fit["sigma_y2"])) / max(float(np.max(np.abs(fit["sigma_y2"]))), EPS))
    h2_grid = np.clip(sigma_g2_grid / np.maximum(sigma_y2_grid, EPS), 0.0, 1.0)

    sens_num = finite_difference_sensitivity(fit["num_surface"], fit["den_surface"], grid_t, config, epsilon_scale=fd_eps_scale)
    u, q = influence_products(residual, weights, edges, chunk_size=int(config.get("cg_surface_single_visit_chunk_size", 200000)))
    psi = dynamic_sigma_g2_influence(weights, u, sens_num)
    d_h2 = h2_influence(psi, d_sigma_y2, sigma_g2_grid, sigma_y2_grid)

    sigma_e2 = residual_variance_by_subject(t, grid_t, sigma_y2_grid, sigma_g2_grid)
    se_dynamic = np.sqrt(np.maximum(np.sum(np.square(d_h2) * sigma_e2[:, None], axis=0), 0.0))

    phi_he = he_influence(residual, q, he)
    a_reconstructed = 0.5 * float(np.dot(residual, q))
    se_he = float(np.sqrt(max(float(he["sigma_eps2"]) * float(np.dot(phi_he, phi_he)), 0.0)))

    curve = pd.DataFrame(
        {
            "t": grid_t,
            "h2_se_conditional": se_dynamic,
            "h2_ci_lower_conditional": np.clip(h2_grid - Z975 * se_dynamic, 0.0, 1.0),
            "h2_ci_upper_conditional": np.clip(h2_grid + Z975 * se_dynamic, 0.0, 1.0),
        }
    )
    he_se = {
        "h2": float(he["h2"]),
        "h2_se_conditional": se_he,
        "h2_ci_lower_conditional": float(np.clip(he["h2"] - Z975 * se_he, 0.0, 1.0)),
        "h2_ci_upper_conditional": float(np.clip(he["h2"] + Z975 * se_he, 0.0, 1.0)),
        "A_relative_error": abs(a_reconstructed - float(he["A"])) / max(abs(float(he["A"])), 1e-300),
    }
    return {
        "curve_se": curve,
        "he_se": he_se,
        "se_dynamic": se_dynamic,
        "sigma_e2_subject": sigma_e2,
        "d_h2": d_h2,
        "phi_he": phi_he,
        "curve_reproduction_relative_error": curve_error,
        "total_variance_reproduction_relative_error": total_error,
    }
