"""Kernel weights, adaptive bandwidths, local total variance and grid smoothers."""
from __future__ import annotations

import numpy as np

from . import EPS


def grid_points(t_min: float, t_max: float, size: int) -> np.ndarray:
    return np.linspace(float(t_min), float(t_max), int(size))


def gaussian_kernel(u):
    values = np.asarray(u, dtype=float)
    return np.exp(-0.5 * np.square(values))


def effective_sample_size(w):
    sw = float(np.sum(w))
    sw2 = float(np.sum(np.square(w)))
    if sw2 <= EPS:
        return 0.0
    return sw * sw / sw2


def choose_adaptive_bandwidth(t, t_g, kernel, target_effective_n, bandwidth_min, bandwidth_max, max_iter=40):
    """Bisection on the Gaussian bandwidth so that ESS(w) reaches the target within [min, max]."""
    if kernel != "gaussian":
        raise ValueError("Only the Gaussian kernel is implemented.")

    t = np.asarray(t, dtype=float)
    target_effective_n = float(target_effective_n)
    local_bandwidth_min = min(float(bandwidth_max), float(bandwidth_min))
    bandwidth_max = float(bandwidth_max)

    def ess_for_bandwidth(h):
        w = gaussian_kernel((t - float(t_g)) / max(float(h), EPS))
        return effective_sample_size(w)

    ess_min = ess_for_bandwidth(local_bandwidth_min)
    if ess_min >= target_effective_n:
        return local_bandwidth_min, ess_min

    ess_max = ess_for_bandwidth(bandwidth_max)
    if ess_max <= target_effective_n:
        return bandwidth_max, ess_max

    lo = local_bandwidth_min
    hi = bandwidth_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ess_mid = ess_for_bandwidth(mid)
        if ess_mid >= target_effective_n:
            hi = mid
        else:
            lo = mid
    final_h = hi
    return final_h, ess_for_bandwidth(final_h)


def bandwidths_and_weights(t_obs, grid_t, config):
    """Per-grid adaptive bandwidth and the N x L Gaussian weight matrix."""
    t_obs = np.asarray(t_obs, dtype=float)
    grid_t = np.asarray(grid_t, dtype=float)
    bandwidths = []
    weights = np.zeros((len(t_obs), len(grid_t)), dtype=float)
    for g, t_g in enumerate(grid_t):
        bandwidth_used, _ = choose_adaptive_bandwidth(
            t=t_obs,
            t_g=float(t_g),
            kernel=config["kernel"],
            target_effective_n=float(config["target_effective_n"]),
            bandwidth_min=float(config["bandwidth_min"]),
            bandwidth_max=float(config["bandwidth_max"]),
        )
        bandwidths.append(float(bandwidth_used))
        weights[:, g] = gaussian_kernel((t_obs - float(t_g)) / float(bandwidth_used))
    return np.asarray(bandwidths, dtype=float), weights


def weighted_least_squares(x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    sqrt_w = np.sqrt(np.clip(w, EPS, None))
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    beta, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
    return beta


def local_linear_mean_and_residual(y, t, t_g, w):
    centered_t = np.asarray(t, dtype=float) - float(t_g)
    x = np.column_stack([np.ones_like(centered_t), centered_t])
    beta = weighted_least_squares(x=x, y=y, w=w)
    fitted = np.dot(x, beta)
    residual = np.asarray(y, dtype=float) - fitted
    return float(beta[0]), float(beta[1]), fitted, residual


def weighted_variance(y, w, mean):
    denom = np.sum(w)
    mean_arr = np.asarray(mean, dtype=float)
    if denom <= EPS or not np.all(np.isfinite(mean_arr)):
        return np.nan
    residual = np.asarray(y, dtype=float) - mean_arr
    return float(np.dot(w, np.square(residual)) / denom)


def local_sigma_y2_grid(y, t, grid_t, config):
    """Local-linear weighted residual variance of y at every grid age (the estimator's sigma_y^2)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    grid_t = np.asarray(grid_t, dtype=float)
    estimates = []
    for t_g in grid_t:
        bandwidth_used, _ = choose_adaptive_bandwidth(
            t=t,
            t_g=float(t_g),
            kernel=config["kernel"],
            target_effective_n=float(config["target_effective_n"]),
            bandwidth_min=float(config["bandwidth_min"]),
            bandwidth_max=float(config["bandwidth_max"]),
        )
        w = gaussian_kernel((t - float(t_g)) / float(bandwidth_used))
        _, _, fitted, _ = local_linear_mean_and_residual(y, t, t_g, w)
        estimates.append(weighted_variance(y, w, fitted))
    return np.asarray(estimates, dtype=float)


def grid_local_variance_and_y_derivative(y_obs, t_obs, grid_t, subject_weights):
    """sigma_y^2 on the grid and its exact derivative with respect to every y_i (N x L)."""
    y_obs = np.asarray(y_obs, dtype=float)
    t_obs = np.asarray(t_obs, dtype=float)
    grid_t = np.asarray(grid_t, dtype=float)
    n_subjects = int(y_obs.size)
    sigma_y2 = np.zeros(grid_t.size, dtype=float)
    d_sigma_y2 = np.zeros((n_subjects, grid_t.size), dtype=float)
    for g, t_g in enumerate(grid_t):
        w = np.asarray(subject_weights[:, g], dtype=float)
        sw = max(float(np.sum(w)), EPS)
        centered_t = t_obs - float(t_g)
        x = np.column_stack([np.ones_like(centered_t), centered_t])
        wx = w[:, None] * x
        xtwx = x.T @ wx
        xtwx_inv = np.linalg.pinv(xtwx)
        beta = xtwx_inv @ (x.T @ (w * y_obs))
        fitted = x @ beta
        residual = y_obs - fitted
        sigma_y2_g = float(np.dot(w, np.square(residual)) / sw)
        sigma_y2[g] = sigma_y2_g
        u = w * residual
        st_u = wx @ (xtwx_inv @ (x.T @ u))
        d_sigma_y2[:, g] = 2.0 * (u - st_u) / sw
    return np.maximum(sigma_y2, EPS), d_sigma_y2


def smooth_grid_curve(grid_t, values, bandwidth):
    """Row-normalised Gaussian smoother applied to a curve on the grid (post-smoothing)."""
    grid_t = np.asarray(grid_t, dtype=float)
    values = np.asarray(values, dtype=float)
    bandwidth = float(bandwidth)
    if bandwidth <= 0.0 or len(grid_t) <= 1:
        return np.asarray(values, dtype=float)
    smoothed = np.zeros_like(values, dtype=float)
    for idx, t_g in enumerate(grid_t):
        weights = np.asarray(gaussian_kernel((grid_t - float(t_g)) / bandwidth), dtype=float)
        smoothed[idx] = float(np.sum(weights * values) / max(np.sum(weights), EPS))
    return smoothed


def smoothing_matrix(grid, bandwidth):
    """Matrix form of smooth_grid_curve: smoothed = S @ values."""
    grid = np.asarray(grid, float)
    if bandwidth <= 0:
        return np.eye(len(grid))
    weights = np.exp(-0.5 * np.square((grid[None, :] - grid[:, None]) / float(bandwidth)))
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), EPS)
