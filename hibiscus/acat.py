"""ACAT combination of p-values."""
from __future__ import annotations

import numpy as np


def acat(pvalues, weights=None) -> float:
    """Aggregated Cauchy association test (ACAT): one p-value from several.

        T = sum_j w_j tan((1/2 - p_j) pi) / sum_j w_j,     p = P(Cauchy > T) = 1/2 - arctan(T) / pi.

    Equal weights by default. Each term is evaluated as cot(p_j pi) = 1 / tan(p_j pi), which keeps
    full precision for small p_j (tan((1/2 - p_j) pi) loses digits there). Any missing p-value makes
    the result missing.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("ACAT requires at least one p-value")
    if np.any(np.isnan(p)):
        return float("nan")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("ACAT inputs must be p-values in [0, 1]")
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != p.shape or np.any(w < 0.0) or not np.sum(w) > 0.0:
        raise ValueError("ACAT weights must be nonnegative, match the p-values and have a positive sum")
    w = w / np.sum(w)
    with np.errstate(divide="ignore"):
        terms = 1.0 / np.tan(p * np.pi)                      # = tan((1/2 - p) pi); p = 0 gives +inf
    stat = float(np.sum(w * terms))
    if not np.isfinite(stat):
        return 0.0 if stat > 0 else 1.0
    if stat > 0.0:
        return float(np.arctan(1.0 / stat) / np.pi)          # upper tail, accurate for large T
    return float(0.5 - np.arctan(stat) / np.pi)
