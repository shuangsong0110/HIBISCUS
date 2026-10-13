from __future__ import annotations

import numpy as np
from scipy.stats import norm

from _toy import small_problem
from hibiscus.corr_test import (build_smooth_basis, correlation_tangent, directional_correlation_test,
                         raw_correlation, sparse_forward_null, sparse_symmetric_product,
                         upper_pairs, random_score_factor_covariance)


def test_correlation_tangent_matches_finite_differences():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 3))
    surface = a @ a.T + np.eye(5)
    tangents = rng.normal(size=(2, 5, 5))
    tangents = 0.5 * (tangents + np.transpose(tangents, (0, 2, 1)))
    analytic = correlation_tangent(surface, tangents)
    left, right = upper_pairs(5)
    eps = 1e-6
    for k in range(2):
        fd = (raw_correlation(surface + eps * tangents[k]) - raw_correlation(surface - eps * tangents[k])) / (2 * eps)
        assert np.allclose(analytic[:, k], fd[left, right], atol=1e-7)


def test_forward_null_matches_dense_expectation():
    """The sparse forward null equals sum_E K_ij (H V0 H)_ij omega_ij,ab / den with exact K."""
    prob = small_problem(n=120, m=80, grid_size=4, all_edges=True, seed=9)
    edges, grm, weights = prob["edges"], prob["grm"], prob["design"]["subject_weights"]
    n = grm.shape[0]
    rng = np.random.default_rng(4)
    amplitude = 0.5 + 0.3 * rng.random(n)
    residual = 0.4 + 0.2 * rng.random(n)
    basis = rng.normal(size=(n, 2))
    k_work = grm.copy()
    np.fill_diagonal(k_work, 1.0)                  # working kinship diagonal K_ii = 1
    ka = k_work @ amplitude
    kb = k_work @ (0.5 * amplitude[:, None] * basis)
    out = sparse_forward_null(edges["row"], edges["col"], edges["kij"], weights, prob["design"]["den_surface"],
                              amplitude, 0.5 * amplitude[:, None] * basis, residual, ka=ka, kb=kb)
    v0 = np.diag(amplitude) @ k_work @ np.diag(amplitude) + np.diag(residual)
    h = np.eye(n) - np.ones((n, n)) / n
    hvh = h @ v0 @ h
    row, col, kij = edges["row"], edges["col"], edges["kij"]
    wi, wj = weights[row], weights[col]
    cross = ((kij * hvh[row, col])[:, None] * wi).T @ wj
    expected = 0.5 * (cross + cross.T) / prob["design"]["den_surface"]
    assert np.allclose(out["null_surface"], expected, rtol=1e-9, atol=1e-12)
    # symmetric product helper agrees with the dense working kinship
    assert np.allclose(sparse_symmetric_product(row, col, kij, amplitude), ka)


def test_directional_test_is_calibrated_under_gaussian_null():
    rng = np.random.default_rng(5)
    k = 15
    a = rng.normal(size=(k, 4))
    omega = a @ a.T / 4 + 0.1 * np.eye(k)
    chol = np.linalg.cholesky(omega)
    pvals = np.array([directional_correlation_test(chol @ rng.normal(size=k), omega)["pvalue_lower"] for _ in range(4000)])
    assert abs(pvals.mean() - 0.5) < 0.03
    assert abs(np.mean(pvals < 0.05) - 0.05) < 0.015
    res = directional_correlation_test(np.full(k, -0.2), omega)
    assert res["z"] < 0 and abs(res["pvalue_lower"] - norm.cdf(res["z"])) < 1e-12


def test_smooth_basis_and_null_covariance():
    prob = small_problem(n=200, m=80, grid_size=9, all_edges=True, seed=3)
    basis = build_smooth_basis(prob["design"]["subject_weights"], 0.999, 12)
    assert 1 <= basis.rank <= 9 and basis.vectors.shape == (9, basis.rank)
    assert np.allclose(basis.vectors.T @ basis.vectors, np.eye(basis.rank), atol=1e-10)
    cov = random_score_factor_covariance(np.random.default_rng(1).normal(size=(200, 3)), np.ones(200), np.full(200, 0.5),
                                         np.random.default_rng(2).normal(size=(200, 8)))
    assert cov.shape == (3, 3) and np.allclose(cov, cov.T) and np.all(np.linalg.eigvalsh(cov) > 0)
