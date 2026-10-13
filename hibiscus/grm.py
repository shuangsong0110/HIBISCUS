"""GRM construction, sparse subject-edge selection, randomised low-rank GRM scores (correlation
test) and the genetic score bank (h2 test).

The default GRM route is numpy: K = Z Z' / M with Z the column-standardised genotype matrix
(missing genotypes contribute 0), which is the same quantity PLINK2 --make-rel computes.
The PLINK2 route is an option for larger data.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

from . import EPS
from .io import PlinkBedReader


# ----------------------------------------------------------------------------- standardisation / dense GRM
def standardized_block(raw: np.ndarray) -> np.ndarray:
    """(variants x subjects) int8 block -> float32 standardised rows; missing (-1) -> 0."""
    values = raw.astype(np.float32, copy=False)
    valid = values >= 0.0
    count = np.maximum(np.sum(valid, axis=1), 1)
    mean = np.sum(np.where(valid, values, 0.0), axis=1) / count
    centered = values - mean[:, None].astype(np.float32)
    centered[~valid] = 0.0
    variance = np.sum(np.square(centered), axis=1) / count
    scale = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
    centered /= scale[:, None]
    return centered


def dense_grm_from_plink(reader: PlinkBedReader, variant_block_size: int = 256) -> np.ndarray:
    """K = Z Z' / M accumulated over variant blocks (float64, n x n)."""
    n = int(reader.n_subjects)
    m = int(reader.n_variants)
    grm = np.zeros((n, n), dtype=np.float64)
    step = max(int(variant_block_size), 1)
    for start in range(0, m, step):
        stop = min(start + step, m)
        z = standardized_block(reader.read_contiguous(start, stop)).astype(np.float64)
        grm += z.T @ z
    grm /= float(m)
    return 0.5 * (grm + grm.T)


def dense_grm_from_standardized(z_subjects_by_variants: np.ndarray) -> np.ndarray:
    z = np.asarray(z_subjects_by_variants, dtype=np.float64)
    grm = z @ z.T / float(z.shape[1])
    return 0.5 * (grm + grm.T)


# ----------------------------------------------------------------------------- optional PLINK2 route
def run_plink2_make_rel(plink2_bin, plink_prefix, out_prefix, threads: int = 1) -> Path:
    """plink2 --make-rel square0 bin4; returns the .rel.bin path."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(plink2_bin), "--bfile", str(plink_prefix),
        "--make-rel", "square0", "bin4",
        "--threads", str(int(threads)),
        "--out", str(out_prefix),
    ]
    subprocess.run(command, check=True)
    rel_bin = Path(str(out_prefix) + ".rel.bin")
    if not rel_bin.is_file():
        raise FileNotFoundError(f"plink2 did not write {rel_bin}")
    return rel_bin


def read_plink2_rel_square0_bin4(rel_bin, n_subjects: int) -> np.ndarray:
    """Read the square0 float32 matrix and return a full symmetric float64 GRM (diagonal set to 1)."""
    rel = np.memmap(rel_bin, dtype=np.float32, mode="r", shape=(int(n_subjects), int(n_subjects)))
    lower = np.tril(np.asarray(rel, dtype=np.float64), k=-1)
    full = lower + lower.T
    np.fill_diagonal(full, 1.0)
    return full


# ----------------------------------------------------------------------------- sparse edges
def select_sparse_row_indices(kij_full, weight_mass_fraction, max_edges_per_row):
    """Keep the largest K_ij^2 entries of one row until `weight_mass_fraction` of the row mass, capped."""
    kij_full = np.asarray(kij_full, dtype=float)
    n_edges = len(kij_full)
    if n_edges == 0:
        return np.empty(0, dtype=int), 0.0, 0.0

    weights = np.square(kij_full)
    total_weight = float(np.sum(weights))
    if total_weight <= EPS:
        keep = min(int(max_edges_per_row), n_edges)
        if keep <= 0:
            return np.empty(0, dtype=int), 0.0, 0.0
        idx = np.argpartition(np.abs(kij_full), -keep)[-keep:]
        kept_weight = float(np.sum(weights[idx]))
        return np.asarray(idx, dtype=int), total_weight, kept_weight

    order = np.argsort(-weights, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    keep = int(np.searchsorted(cumulative, float(weight_mass_fraction) * total_weight, side="left")) + 1
    keep = min(max(1, keep), int(max_edges_per_row), n_edges)
    idx = np.asarray(order[:keep], dtype=int)
    kept_weight = float(np.sum(weights[idx]))
    return idx, total_weight, kept_weight


def select_subject_edges(grm_subject, config: dict) -> dict:
    """Dense GRM -> sparse edge set {row, col, kij} with row < col."""
    grm_subject = np.asarray(grm_subject, dtype=float)
    weight_mass_fraction = float(config.get("cg_surface_weight_mass_fraction", 0.995))
    max_edges_per_row = int(config.get("cg_surface_max_edges_per_row", 256))

    rows = []
    cols = []
    kij_list = []
    row_weight_capture = []
    n_subjects = grm_subject.shape[0]
    for i in range(n_subjects - 1):
        kij_full = np.asarray(grm_subject[i, (i + 1):], dtype=float)
        if kij_full.size == 0:
            continue
        idx_local, total_weight, kept_weight = select_sparse_row_indices(
            kij_full=kij_full,
            weight_mass_fraction=weight_mass_fraction,
            max_edges_per_row=max_edges_per_row,
        )
        if idx_local.size == 0:
            continue
        j_idx = (i + 1) + idx_local
        rows.append(np.repeat(i, len(j_idx)))
        cols.append(j_idx)
        kij_list.append(kij_full[idx_local])
        row_weight_capture.append(float(kept_weight) / max(float(total_weight), EPS))
    if not kij_list:
        raise ValueError("No sparse subject edges retained.")
    return {
        "row": np.concatenate(rows).astype(np.int32, copy=False),
        "col": np.concatenate(cols).astype(np.int32, copy=False),
        "kij": np.concatenate(kij_list).astype(float, copy=False),
        "edge_count": int(sum(len(x) for x in kij_list)),
        "row_weight_capture_mean": float(np.mean(row_weight_capture)) if row_weight_capture else np.nan,
    }


def edge_metadata(edges: dict, n_subjects: int, n_variants: int, config: dict, input_mode: str) -> dict:
    return {
        "subject_edge_input_mode": input_mode,
        "grm_definition": "K = Z Z' / M, Z column-standardised genotypes (missing -> 0); K_ii treated as 1 downstream",
        "cg_surface_weight_mass_fraction": float(config.get("cg_surface_weight_mass_fraction", 0.995)),
        "cg_surface_max_edges_per_row": int(config.get("cg_surface_max_edges_per_row", 256)),
        "edge_count": int(edges["edge_count"]),
        "row_weight_capture_mean": float(edges["row_weight_capture_mean"]),
        "n_subjects": int(n_subjects),
        "n_variants": int(n_variants),
        "storage": "npz arrays row (int32), col (int32), kij (float64); each pair once with row < col",
    }


# ----------------------------------------------------------------------------- random GRM scores
def _random_scores_from_blocks(read_block, n_subjects: int, n_variants: int, rank: int, seed: int, variant_block_size: int):
    boundaries = np.linspace(0, int(n_variants), int(rank) + 1, dtype=np.int64)
    scores = np.empty((int(n_subjects), int(rank)), dtype=np.float32)
    for component in range(int(rank)):
        first = int(boundaries[component])
        last = int(boundaries[component + 1])
        variant_count = last - first
        if variant_count <= 0:
            raise ValueError("random_score_rank exceeds the number of variants")
        rng = np.random.default_rng(int(seed) + component)
        beta = rng.standard_normal(variant_count).astype(np.float32) / math.sqrt(variant_count)
        score = np.zeros(int(n_subjects), dtype=np.float32)
        for start in range(first, last, int(variant_block_size)):
            stop = min(start + int(variant_block_size), last)
            genotype = read_block(start, stop)
            score += genotype.T @ beta[start - first: stop - first]
        score -= np.mean(score, dtype=np.float64)
        score /= max(float(np.std(score, dtype=np.float64)), 1e-12)
        scores[:, component] = score
    return scores


def random_grm_scores(reader: PlinkBedReader, rank: int = 256, seed: int = 2026081501, variant_block_size: int = 256):
    """Partitioned Hutchinson scores S (n x rank, float32): K_hat = S S' / rank."""

    def read_block(start, stop):
        return standardized_block(reader.read_contiguous(start, stop))

    scores = _random_scores_from_blocks(read_block, reader.n_subjects, reader.n_variants, rank, seed, variant_block_size)
    meta = {
        "n_subjects": int(reader.n_subjects),
        "n_variants": int(reader.n_variants),
        "rank": int(rank),
        "seed": int(seed),
        "variant_assignment": "contiguous_equal_size_disjoint_partition",
        "score_definition": "S_c = G_c z_c / sqrt(m_c), then column-centered and variance-standardized",
        "grm_approximation": "K_hat = S S' / rank",
    }
    return scores, meta


# ----------------------------------------------------------------------------- genetic score bank (h2 test)
def _genetic_score_bank_from_blocks(read_block, n_subjects: int, n_variants: int, n_cols: int, seed: int,
                                    variant_block_size: int, edges: dict, n_check_edges: int = 20000):
    """S = Z Theta / sqrt(M), Theta ~ N(0, I): every column is an exact draw from N(0, Z Z'/M).

    In the same pass the Z-GRM is evaluated on a sample of the stored edges, and the columns are
    rescaled by sqrt(c), c = sum K_ij k_ij / sum k_ij^2 with k_ij = z_i.z_j / M, so that Cov(S) matches
    the GRM the edges were built from (c = 1 for the numpy route; close to 1 for the plink2 route).
    """
    rng = np.random.default_rng(int(seed))
    bank = np.zeros((int(n_subjects), int(n_cols)), dtype=np.float64)
    row = np.asarray(edges["row"], np.int64)
    col = np.asarray(edges["col"], np.int64)
    pick = np.random.default_rng(int(seed) + 1).choice(len(row), size=min(int(n_check_edges), len(row)), replace=False)
    ri, ci, kij = row[pick], col[pick], np.asarray(edges["kij"], float)[pick]
    zz = np.zeros(len(pick), dtype=np.float64)
    for start in range(0, int(n_variants), int(variant_block_size)):
        stop = min(start + int(variant_block_size), int(n_variants))
        z = np.asarray(read_block(start, stop), dtype=np.float64)          # variants x subjects
        bank += z.T @ rng.standard_normal((stop - start, int(n_cols)))
        zz += np.einsum("vi,vi->i", z[:, ri], z[:, ci])
    bank /= math.sqrt(float(n_variants))
    zz /= float(n_variants)
    scale = float(np.dot(kij, zz) / max(float(np.dot(zz, zz)), EPS))
    fit_corr = float(np.corrcoef(kij, zz)[0, 1]) if len(pick) > 2 else math.nan
    return (bank * math.sqrt(scale)).astype(np.float32), {"grm_scale": scale, "grm_scale_check_corr": fit_corr,
                                                          "grm_scale_check_edges": int(len(pick))}


def genetic_score_bank(reader: PlinkBedReader, n_cols: int, seed: int, edges: dict, variant_block_size: int = 256):
    """Score bank for the h2 test's null draws (n x n_cols float32), with its metadata."""

    def read_block(start, stop):
        return standardized_block(reader.read_contiguous(start, stop))

    bank, check = _genetic_score_bank_from_blocks(read_block, reader.n_subjects, reader.n_variants, n_cols, seed,
                                                  variant_block_size, edges)
    meta = {
        "n_subjects": int(reader.n_subjects),
        "n_variants": int(reader.n_variants),
        "n_cols": int(n_cols),
        "seed": int(seed),
        "score_definition": "S = sqrt(c) Z Theta / sqrt(M), Theta ~ N(0, I); Cov(S[:, j]) = c Z Z'/M",
        **check,
    }
    return bank, meta


def genetic_score_bank_from_standardized(z_subjects_by_variants: np.ndarray, n_cols: int, seed: int, edges: dict,
                                         variant_block_size: int = 256):
    """Same algorithm on an in-memory standardised matrix (tests)."""
    z = np.asarray(z_subjects_by_variants, dtype=np.float32)

    def read_block(start, stop):
        return np.ascontiguousarray(z[:, start:stop].T)

    return _genetic_score_bank_from_blocks(read_block, z.shape[0], z.shape[1], n_cols, seed, variant_block_size, edges)
