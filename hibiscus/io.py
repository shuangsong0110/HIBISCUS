"""Input/output for the HIBISCUS pipeline.

File formats (see README, Step 1):
  phenotype TSV   tab-separated, header FID IID y t, one row per visit
  sample-ID CSV   header FID,IID; one row per genotyped subject in GRM (= .fam) order
  PLINK 1 binary  prefix.bed / prefix.bim / prefix.fam (SNP-major .bed)
  edge cache      edges.npz (row, col int32; kij float64; row < col; indices into sample order)
                  + edges.json metadata
Subjects are matched to the sample-ID file by IID; the first remaining row per IID is kept.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BED_HEADER = b"\x6c\x1b\x01"
_BED_LOOKUP = None


# ----------------------------------------------------------------------------- generic helpers
def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_ready(payload):
    if isinstance(payload, dict):
        return {str(k): json_ready(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_ready(x) for x in payload]
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, (np.floating,)):
        return float(payload)
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.bool_,)):
        return bool(payload)
    if isinstance(payload, Path):
        return str(payload)
    return payload


def save_json(path, payload) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True, allow_nan=True)
    temporary.replace(path)
    return path


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_to_csv(frame: pd.DataFrame, path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def atomic_save_npy(path, array) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "wb") as handle:
        np.save(handle, np.asarray(array))
    temporary.replace(path)
    return path


def atomic_savez(path, **arrays) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return path


# ----------------------------------------------------------------------------- PLINK binary
def _bed_lookup_table():
    global _BED_LOOKUP
    if _BED_LOOKUP is not None:
        return _BED_LOOKUP
    lookup = np.empty((256, 4), dtype=np.int8)
    code_map = np.array([0, -1, 1, 2], dtype=np.int8)
    for byte in range(256):
        for offset in range(4):
            code = (byte >> (2 * offset)) & 0b11
            lookup[byte, offset] = code_map[code]
    _BED_LOOKUP = lookup
    return lookup


class PlinkBedReader:
    """Minimal SNP-major PLINK 1 .bed reader (0/1/2 allele counts, -1 = missing)."""

    def __init__(self, plink_prefix):
        self.prefix = Path(plink_prefix)
        self.bed_path = Path(str(self.prefix) + ".bed")
        self.bim_path = Path(str(self.prefix) + ".bim")
        self.fam_path = Path(str(self.prefix) + ".fam")
        for path in (self.bed_path, self.bim_path, self.fam_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing PLINK file: {path}")
        self.n_subjects = sum(1 for _ in open(self.fam_path, "r", encoding="utf-8"))
        self.n_variants = sum(1 for _ in open(self.bim_path, "r", encoding="utf-8"))
        self.bytes_per_snp = (self.n_subjects + 3) // 4
        with open(self.bed_path, "rb") as handle:
            header = handle.read(3)
        if header != BED_HEADER:
            raise ValueError(f"Unexpected PLINK .bed header for {self.bed_path}: {header!r}")
        expected = 3 + self.n_variants * self.bytes_per_snp
        actual = self.bed_path.stat().st_size
        if actual != expected:
            raise ValueError(f"{self.bed_path}: size {actual} != expected {expected} (SNP-major bed?)")

    def fam_frame(self) -> pd.DataFrame:
        fam = pd.read_csv(self.fam_path, sep=r"\s+", header=None, dtype=str)
        fam = fam.iloc[:, :2]
        fam.columns = ["FID", "IID"]
        return fam

    def read_contiguous(self, start: int, stop: int) -> np.ndarray:
        """Variants [start, stop) as an int8 array of shape (stop - start, n_subjects)."""
        count = int(stop - start)
        with open(self.bed_path, "rb") as handle:
            handle.seek(3 + int(start) * self.bytes_per_snp)
            raw = np.frombuffer(handle.read(count * self.bytes_per_snp), dtype=np.uint8)
        raw = raw.reshape(count, self.bytes_per_snp)
        expanded = _bed_lookup_table()[raw].reshape(count, self.bytes_per_snp * 4)
        return expanded[:, : self.n_subjects]

    def read_variant_indices(self, variant_indices) -> np.ndarray:
        variant_indices = np.asarray(variant_indices, dtype=np.int64)
        out = np.empty((len(variant_indices), int(self.n_subjects)), dtype=np.int8)
        lookup = _bed_lookup_table()
        with open(self.bed_path, "rb") as handle:
            for out_idx, variant_idx in enumerate(variant_indices):
                handle.seek(3 + int(variant_idx) * self.bytes_per_snp)
                raw = np.frombuffer(handle.read(self.bytes_per_snp), dtype=np.uint8)
                expanded = lookup[raw].reshape(self.bytes_per_snp * 4)
                out[out_idx, :] = expanded[: self.n_subjects]
        return out

    def read_all(self) -> np.ndarray:
        """All genotypes as int8 (n_variants, n_subjects). Only for small toy data."""
        return self.read_contiguous(0, self.n_variants)


def write_plink(prefix, genotypes, fam: pd.DataFrame, bim: pd.DataFrame) -> dict:
    """Write SNP-major .bed/.bim/.fam.

    genotypes: int array (n_subjects, n_variants) with values 0/1/2 (allele counts) or -1 missing.
    fam: columns FID, IID (PID, MID, SEX, PHENO filled with 0 0 0 -9 when absent).
    bim: columns CHR, SNP, CM, BP, A1, A2.
    Encoding matches PlinkBedReader: 2-bit code 0 -> 0, 1 -> missing, 2 -> 1, 3 -> 2.
    """
    prefix = Path(prefix)
    ensure_dir(prefix.parent)
    g = np.asarray(genotypes)
    if g.ndim != 2:
        raise ValueError("genotypes must be 2-D (n_subjects, n_variants)")
    n_subjects, n_variants = g.shape
    if len(fam) != n_subjects or len(bim) != n_variants:
        raise ValueError("fam / bim row counts do not match the genotype matrix")
    code = np.empty(g.shape, dtype=np.uint8)
    code[g == 0] = 0
    code[g == 1] = 2
    code[g == 2] = 3
    code[g < 0] = 1
    bad = ~np.isin(g, (-1, 0, 1, 2))
    if np.any(bad):
        raise ValueError("genotypes must be in {-1, 0, 1, 2}")
    bytes_per_snp = (n_subjects + 3) // 4
    padded = np.zeros((n_variants, bytes_per_snp * 4), dtype=np.uint8)
    padded[:, :n_subjects] = code.T
    packed = (
        padded[:, 0::4]
        | (padded[:, 1::4] << 2)
        | (padded[:, 2::4] << 4)
        | (padded[:, 3::4] << 6)
    ).astype(np.uint8)
    bed_path = Path(str(prefix) + ".bed")
    with open(bed_path, "wb") as handle:
        handle.write(BED_HEADER)
        handle.write(np.ascontiguousarray(packed).tobytes())
    fam_out = fam.copy()
    for column, default in (("PID", "0"), ("MID", "0"), ("SEX", "0"), ("PHENO", "-9")):
        if column not in fam_out.columns:
            fam_out[column] = default
    fam_out = fam_out[["FID", "IID", "PID", "MID", "SEX", "PHENO"]]
    fam_path = Path(str(prefix) + ".fam")
    fam_out.to_csv(fam_path, sep=" ", header=False, index=False)
    bim_path = Path(str(prefix) + ".bim")
    bim[["CHR", "SNP", "CM", "BP", "A1", "A2"]].to_csv(bim_path, sep="\t", header=False, index=False)
    return {"bed": bed_path, "bim": bim_path, "fam": fam_path}


# ----------------------------------------------------------------------------- phenotype + matching
def read_phenotype_first_entry(phenotype_tsv) -> pd.DataFrame:
    """Read FID/IID/y/t, drop rows with missing y or t, keep the first row per IID (TSV order)."""
    df = pd.read_csv(phenotype_tsv, sep="\t", dtype={"FID": str, "IID": str})
    required = {"FID", "IID", "y", "t"}
    if not required.issubset(df.columns):
        raise ValueError(f"Phenotype TSV must contain columns {sorted(required)}; found {list(df.columns)}")
    df = df.loc[df["y"].notna() & df["t"].notna(), ["FID", "IID", "y", "t"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.loc[df["y"].notna() & df["t"].notna()].copy()
    return df.drop_duplicates(subset=["IID"], keep="first").reset_index(drop=True)


def read_sample_ids(sample_ids_csv) -> pd.DataFrame:
    sample_df = pd.read_csv(sample_ids_csv, dtype={"FID": str, "IID": str})
    if not {"FID", "IID"}.issubset(sample_df.columns):
        raise ValueError("sample-ID CSV must have header FID,IID")
    return sample_df[["FID", "IID"]].reset_index(drop=True)


def match_phenotype_to_samples(pheno_df: pd.DataFrame, sample_df: pd.DataFrame):
    """Align the phenotype to GRM sample order, matching on IID.

    Returns (matched_mask over sample order, matched frame with FID, IID, y, t, global_idx, local_idx).
    """
    merged = sample_df.merge(pheno_df[["IID", "y", "t"]], on="IID", how="left")
    if len(merged) != len(sample_df):
        raise ValueError("duplicate IIDs in the phenotype after first-entry filtering")
    matched_mask = merged["y"].notna().to_numpy(dtype=bool)
    matched = merged.loc[matched_mask, ["FID", "IID", "y", "t"]].copy().reset_index(drop=True)
    matched["y"] = matched["y"].astype(float)
    matched["t"] = matched["t"].astype(float)
    matched["global_idx"] = np.flatnonzero(matched_mask).astype(np.int64)
    matched["local_idx"] = np.arange(len(matched), dtype=np.int64)
    return matched_mask, matched


# ----------------------------------------------------------------------------- edge cache
def save_edges(npz_path, meta_path, row, col, kij, meta: dict) -> None:
    atomic_savez(npz_path, row=np.asarray(row, np.int32), col=np.asarray(col, np.int32), kij=np.asarray(kij, float))
    save_json(meta_path, meta)


def load_edges(npz_path, meta_path=None) -> dict:
    payload = np.load(npz_path)
    edges = {
        "row": np.asarray(payload["row"], dtype=np.int32),
        "col": np.asarray(payload["col"], dtype=np.int32),
        "kij": np.asarray(payload["kij"], dtype=float),
    }
    edges["edge_count"] = int(edges["kij"].size)
    edges["meta"] = load_json(meta_path) if meta_path is not None and Path(meta_path).is_file() else {}
    edges["row_weight_capture_mean"] = float(edges["meta"].get("row_weight_capture_mean", np.nan))
    if np.any(edges["row"] >= edges["col"]):
        raise ValueError("edge cache must store each pair once with row < col")
    return edges


def subset_edges(edges: dict, matched_mask) -> dict:
    """Keep edges whose two subjects are both matched and reindex them to local (matched) indices."""
    matched_mask = np.asarray(matched_mask, dtype=bool)
    row = np.asarray(edges["row"], dtype=np.int32)
    col = np.asarray(edges["col"], dtype=np.int32)
    kij = np.asarray(edges["kij"], dtype=float)
    new_index = np.full(int(matched_mask.shape[0]), -1, dtype=np.int32)
    new_index[matched_mask] = np.arange(int(matched_mask.sum()), dtype=np.int32)
    edge_keep = matched_mask[row] & matched_mask[col]
    return {
        "row": new_index[row[edge_keep]].astype(np.int32, copy=False),
        "col": new_index[col[edge_keep]].astype(np.int32, copy=False),
        "kij": kij[edge_keep].astype(float, copy=False),
        "edge_count": int(np.sum(edge_keep)),
        "row_weight_capture_mean": float(edges.get("row_weight_capture_mean", np.nan)),
        "n_subjects": int(matched_mask.sum()),
    }


def load_matched_dataset(phenotype_tsv, sample_ids_csv, edges_npz, edges_json) -> dict:
    """Phenotype -> sample order -> local edge set. The single entry point used by steps 3-5.

    The returned `y` is the matched phenotype standardised to mean 0 and standard deviation 1, so that
    every downstream computation is independent of the unit of y (h2, R_G, the standard errors and the
    tests are unit-free). `y_scale` is the standard deviation of the matched phenotype: variance-type
    outputs (sigma_g2, sigma_y2, sigma_eps2, covariance surfaces) are multiplied by y_scale**2 when
    written, so files are in the original units. `y_original` is the untouched phenotype.
    """
    pheno_df = read_phenotype_first_entry(phenotype_tsv)
    sample_df = read_sample_ids(sample_ids_csv)
    matched_mask, matched = match_phenotype_to_samples(pheno_df, sample_df)
    if matched_mask.sum() < 2:
        raise ValueError("fewer than two phenotyped subjects matched the sample-ID file")
    edges = load_edges(edges_npz, edges_json)
    meta = edges["meta"]
    if meta.get("n_subjects") is not None and int(meta["n_subjects"]) != len(sample_df):
        raise ValueError(
            f"edge cache built for {meta['n_subjects']} subjects but sample-ID file has {len(sample_df)} rows"
        )
    local_edges = subset_edges(edges, matched_mask)
    y_original = matched["y"].to_numpy(float)
    y_mean = float(np.mean(y_original))
    y_scale = float(np.sqrt(np.mean(np.square(y_original - y_mean))))
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        raise ValueError("the matched phenotype has zero variance")
    return {
        "pheno_rows_first_entry": int(len(pheno_df)),
        "n_samples": int(len(sample_df)),
        "matched_mask": matched_mask,
        "matched": matched,
        "y": (y_original - y_mean) / y_scale,
        "y_original": y_original,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "t": matched["t"].to_numpy(float),
        "n_subjects": int(len(matched)),
        "edges": local_edges,
        "edge_count_before_filter": int(edges["edge_count"]),
        "edge_meta": meta,
    }
