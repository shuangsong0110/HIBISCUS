from __future__ import annotations

import numpy as np

from _toy import small_problem
from hibiscus.estimate import (accumulate_surfaces_dense_single_visit, accumulate_surfaces_single_visit, fit_diag2,
                               project_psd, sparse_he_no_t, surface_to_correlation)
from hibiscus.kernel import smooth_grid_curve, smoothing_matrix
from hibiscus.simulate import truth_h2_frame


def test_sparse_edge_pass_equals_dense_formula():
    prob = small_problem(n=300, m=200, grid_size=6, all_edges=True)
    y = prob["y"]
    residual = y - y.mean()
    weights = prob["design"]["subject_weights"]
    num_s, den_s = accumulate_surfaces_single_visit(weights, residual, prob["edges"], chunk_size=1000)
    num_d, den_d = accumulate_surfaces_dense_single_visit(weights, residual, prob["grm"])
    assert np.allclose(num_s, num_d, rtol=0, atol=1e-8 * np.max(np.abs(num_d)))
    assert np.allclose(den_s, den_d, rtol=0, atol=1e-8 * np.max(np.abs(den_d)))
    assert np.allclose(den_s, prob["design"]["den_surface"])


def test_flat_weights_reduce_to_he():
    prob = small_problem(n=300, m=200, grid_size=3, all_edges=True)
    y = prob["y"]
    residual = y - y.mean()
    ones = np.ones((len(y), 3))
    num, den = accumulate_surfaces_single_visit(ones, residual, prob["edges"])
    he = sparse_he_no_t(y, prob["edges"])
    assert np.allclose(num, he["A"]) and np.allclose(den, he["B"])
    assert abs(he["sigma_g2_raw"] * he["B"] - he["A"]) < 1e-9


def test_psd_projection_smoother_and_correlation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    sym = 0.5 * (a + a.T)
    psd = project_psd(sym)
    assert np.min(np.linalg.eigvalsh(psd)) >= -1e-12
    assert np.allclose(project_psd(psd), psd, atol=1e-12)
    grid = np.linspace(40, 70, 7)
    s = smoothing_matrix(grid, 2.0)
    assert np.allclose(s.sum(axis=1), 1.0)
    values = rng.normal(size=7)
    assert np.allclose(s @ values, smooth_grid_curve(grid, values, 2.0))
    corr = surface_to_correlation(psd + 6 * np.eye(6))
    assert np.allclose(np.diag(corr), 1.0) and np.all(np.abs(corr) <= 1.0)


def test_fit_recovers_truth_roughly():
    prob = small_problem(n=1500, m=400, grid_size=7, all_edges=False, seed=11)
    fit = fit_diag2(prob["y"], prob["t"], prob["design"])
    truth = truth_h2_frame(prob["grid"], prob["cfg"])["h2_truth"].to_numpy()
    assert fit["h2"].shape == truth.shape
    assert np.all((fit["h2"] >= 0) & (fit["h2"] <= 1))
    assert np.sqrt(np.mean((fit["h2"] - truth) ** 2)) < 0.15
    assert fit["h2"][-1] > fit["h2"][0]  # rising truth is recovered in direction
