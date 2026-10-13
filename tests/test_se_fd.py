from __future__ import annotations

import numpy as np

from _toy import small_problem
from hibiscus.estimate import fit_diag2, sparse_he_no_t
from hibiscus.se import conditional_se


def _fd_derivative(func, y, indices, step):
    out = []
    for i in indices:
        plus = y.copy()
        minus = y.copy()
        plus[i] += step
        minus[i] -= step
        out.append((func(plus) - func(minus)) / (2.0 * step))
    return np.asarray(out)


def test_dynamic_and_he_influence_match_finite_differences():
    prob = small_problem(n=400, m=300, grid_size=7, all_edges=True, seed=5)
    y, t, design, edges = prob["y"], prob["t"], prob["design"], prob["edges"]
    fit = fit_diag2(y, t, design)
    he = sparse_he_no_t(y, edges)
    result = conditional_se(y, t, fit, design, edges, he)
    assert result["curve_reproduction_relative_error"] < 1e-10
    assert result["total_variance_reproduction_relative_error"] < 1e-8
    rng = np.random.default_rng(1)
    indices = rng.choice(len(y), size=6, replace=False)
    step = 1e-5 * float(np.std(y))

    fd_dyn = _fd_derivative(lambda yy: fit_diag2(yy, t, design)["h2"], y, indices, step)   # 6 x L
    ana_dyn = result["d_h2"][indices]
    scale = np.max(np.abs(ana_dyn))
    assert np.max(np.abs(fd_dyn - ana_dyn)) < 1e-5 * scale, np.max(np.abs(fd_dyn - ana_dyn)) / scale

    fd_he = _fd_derivative(lambda yy: np.array([sparse_he_no_t(yy, edges)["h2"]]), y, indices, step)[:, 0]
    ana_he = result["phi_he"][indices]
    assert np.max(np.abs(fd_he - ana_he)) < 1e-6 * np.max(np.abs(ana_he))

    # SE is the conditional quadratic form of the influence
    manual = np.sqrt(np.sum(result["sigma_e2_subject"][:, None] * result["d_h2"] ** 2, axis=0))
    assert np.allclose(manual, result["se_dynamic"])
    assert np.all(result["se_dynamic"] > 0)
