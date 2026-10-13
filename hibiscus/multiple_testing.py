"""ACAT combination of p-values; Benjamini-Hochberg and Bonferroni columns across traits."""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment, preserving the original row order."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not len(pvalues):
        raise ValueError("BH requires a nonempty one-dimensional p-value vector")
    if np.any(~np.isfinite(pvalues)) or np.any((pvalues < 0.0) | (pvalues > 1.0)):
        raise ValueError("BH inputs must be finite p-values in [0, 1]")
    order = np.argsort(pvalues, kind="mergesort")
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def add_multiple_testing(summary: pd.DataFrame, p_column: str, suffix: str, alpha: float = 0.05) -> None:
    pvalues = summary[p_column].to_numpy(dtype=float)
    summary[f"bh_q_{suffix}"] = bh_adjust(pvalues)
    summary[f"bonferroni_p_{suffix}"] = np.minimum(pvalues * len(pvalues), 1.0)
    summary[f"raw_significant_{suffix}"] = pvalues < alpha
    summary[f"bh_significant_{suffix}"] = summary[f"bh_q_{suffix}"] < alpha
    summary[f"bonferroni_significant_{suffix}"] = summary[f"bonferroni_p_{suffix}"] < alpha
