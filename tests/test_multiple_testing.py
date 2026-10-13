from __future__ import annotations

import math

import numpy as np
from scipy.stats import cauchy

from hibiscus.multiple_testing import acat, bh_adjust


def test_acat_matches_the_cauchy_tail():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(1e-6, 1.0, size=rng.integers(1, 6))
        stat = np.mean(np.tan((0.5 - p) * np.pi))
        assert abs(acat(p) - cauchy.sf(stat)) < 1e-12
    w = np.array([1.0, 3.0])
    p = np.array([0.01, 0.4])
    stat = np.sum(w * np.tan((0.5 - p) * np.pi)) / np.sum(w)
    assert abs(acat(p, w) - cauchy.sf(stat)) < 1e-12


def test_acat_identical_pvalues_are_returned_unchanged():
    for p in (0.9, 0.3, 1e-3, 1e-5, 1e-9):
        assert abs(acat([p, p]) / p - 1.0) < 1e-9


def test_acat_edge_cases():
    assert abs(acat([1.0, 1.0]) - 1.0) < 1e-12
    assert abs(acat([1e-20, 0.5]) / 2e-20 - 1.0) < 1e-6          # tiny p: 1/(p pi) term, p_joint ~ 2 p
    assert acat([0.0, 0.3]) == 0.0
    assert math.isnan(acat([math.nan, 0.3]))
    assert abs(acat([0.5]) - 0.5) < 1e-12


def test_bh_adjust_small_example():
    q = bh_adjust(np.array([0.01, 0.04, 0.03, 0.2]))
    assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])
