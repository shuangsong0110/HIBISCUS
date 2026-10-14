from __future__ import annotations

import numpy as np
from scipy import stats

from _toy import small_problem
from hibiscus.estimate import fit_hibiscus, numerator_surface
from hibiscus.h2_test import (bootstrap_law, draw_null, flat_null_model, flat_test, flatness_statistic, gchisq_logsf,
                           local_linear_variance_many, raw_curves_many, target_indices)
from hibiscus.grm import genetic_score_bank_from_standardized
from hibiscus.kernel import local_sigma_y2_grid, smoothing_matrix


def test_statistic_ignores_the_level():
    rng = np.random.default_rng(0)
    v, s = rng.normal(size=5), rng.uniform(0.5, 2.0, size=5)
    assert abs(flatness_statistic(v, s) - flatness_statistic(v + 3.7, s)) < 1e-12
    assert flatness_statistic(np.full(5, 0.4), s) == 0.0


def test_null_model_is_flat_and_keeps_the_correlation():
    grid = np.linspace(40, 70, 9)
    rng = np.random.default_rng(1)
    a = rng.normal(size=(9, 3))
    surface = a @ a.T + 0.05 * rng.normal(size=(9, 9))                 # indefinite, like a raw fit
    sy2 = np.linspace(0.8, 1.3, 9)
    basis, lam, sige2 = flat_null_model(surface, sy2, grid, grid, c_star=0.3, max_rank=4)
    cstar = (basis * lam) @ basis.T
    assert np.allclose(np.diag(cstar), 0.3 * sy2, rtol=1e-12)
    assert np.allclose(sige2, 0.7 * sy2, rtol=1e-12)
    val, vec = np.linalg.eigh(0.5 * (surface + surface.T))
    top = np.argsort(val)[::-1][: len(lam)]
    part = (vec[:, top] * val[top]) @ vec[:, top].T                    # rank <= 4 positive part
    corr = lambda m: m / np.sqrt(np.outer(np.diag(m), np.diag(m)))   
    assert np.allclose(corr(cstar), corr(part), atol=1e-12)


def test_batched_curves_equal_the_estimator():
    prob = small_problem(n=400, m=300, grid_size=7, all_edges=False, seed=5)
    design, t = prob["design"], prob["t"]
    rng = np.random.default_rng(2)
    Y = np.column_stack([prob["y"], rng.normal(size=(len(t), 3))])
    smoother = smoothing_matrix(prob["grid"], 2.0)
    out = raw_curves_many(Y, t, design, smoother)
    for q in range(Y.shape[1]):
        y = Y[:, q]
        num = numerator_surface(y - y.mean(), design)
        raw_diag = np.diag(num) / np.diag(design["den_surface"])
        assert np.allclose(out["raw_diag"][:, q], raw_diag, rtol=1e-12, atol=0)
        sy2 = local_sigma_y2_grid(y, t, prob["grid"], design["config"])
        assert np.max(np.abs(out["sigma_y2"][:, q] - sy2)) < 1e-10 * np.max(sy2)
    fit = fit_hibiscus(prob["y"], t, design)
    assert np.allclose(out["raw_diag"][:, 0], np.diag(fit["raw"]), rtol=1e-12)
    W = design["subject_weights"]
    assert np.allclose(local_linear_variance_many(Y, t, prob["grid"], W), out["sigma_y2"])


def test_tail_matches_closed_forms():
    for k in (3, 6):
        lam = np.full(k, 0.37)
        for q in (0.9, 0.5, 1e-2, 1e-4, 1e-8, 1e-12, 1e-30, 1e-100):
            x = 0.37 * stats.chi2.isf(q, k)
            assert abs(np.exp(gchisq_logsf(x, lam)) / q - 1.0) < 1e-6, (k, q)
        mu = np.full(k, 0.8)
        for q in (0.9, 0.5, 1e-2, 1e-4, 1e-8, 1e-12):
            x = 0.37 * stats.ncx2.isf(q, k, float(np.sum(mu ** 2)))
            assert abs(np.exp(gchisq_logsf(x, lam, mu)) / q - 1.0) < 1e-6, (k, q)
    # bootstrap law of a Gaussian sample reproduces the Monte Carlo tail of T
    rng = np.random.default_rng(3)
    curves = rng.normal(size=(4000, 5)) * np.array([1.0, 0.7, 0.5, 0.7, 1.0]) + np.array([0.1, 0, -0.05, 0, 0.1])
    scale = curves.std(axis=0, ddof=1)
    lam, mu = bootstrap_law(curves, scale)
    t_star = np.array([flatness_statistic(c, scale) for c in curves])
    x = np.quantile(t_star, 0.95)
    assert abs(np.exp(gchisq_logsf(x, lam, mu)) - 0.05) < 0.01


def test_score_bank_has_the_grm_covariance():
    prob = small_problem(n=60, m=400, grid_size=3, all_edges=True, seed=9)
    bank, check = genetic_score_bank_from_standardized(prob["z"], n_cols=20000, seed=4, edges=prob["edges"])
    assert abs(check["grm_scale"] - 1.0) < 1e-5
    emp = np.cov(bank.astype(float), bias=True)
    assert np.max(np.abs(emp - prob["grm"])) < 0.06                    # MC error ~ sqrt(2/20000) * 1


def test_flat_test_runs_end_to_end():
    prob = small_problem(n=600, m=300, grid_size=31, all_edges=False, seed=12)
    design, t, y = prob["design"], prob["t"], prob["y"]
    fit = fit_hibiscus(y, t, design)
    idx = target_indices(prob["grid"], [45.0, 50.0, 55.0, 60.0, 65.0])
    bank, _ = genetic_score_bank_from_standardized(prob["z"], n_cols=50 * 4, seed=1, edges=prob["edges"])
    out = flat_test(y, t, design, fit["raw"], fit["sigma_y2"], idx, bank.astype(float), n_boot=50, seed=0)
    assert 0.0 < out["p_h2"] <= 1.0 and 1.0 / 51 <= out["p_h2_mc"] <= 1.0
    assert out["null_curves"].shape == (50, 5) and np.all(out["sd_null"] > 0)
    assert abs(out["law_mean"] / out["bootstrap_mean_T"] - 1.0) < 1.0 / 49 + 1e-9
    rng = np.random.default_rng(0)
    basis, lam, sige2 = flat_null_model(fit["raw"], fit["sigma_y2"], prob["grid"], t, out["c_star"])
    y_null = draw_null(bank.astype(float), basis, lam, sige2, 50, rng)
    assert y_null.shape == (len(t), 50)
