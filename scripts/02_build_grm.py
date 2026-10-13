#!/usr/bin/env python3
"""Step 2 - GRM, sparse subject edges, randomised low-rank GRM scores (correlation test), genetic score bank (h2 test).

Reads  data/<name>/geno.{bed,bim,fam} and sample_ids.csv (must be in .fam order).
Writes grm/<name>/edges.npz + edges.json      sparse edge cache (row, col, kij; row < col)
       grm/<name>/random_scores_rank<q>.float32.npy + random_scores_meta.json     (correlation test)
       grm/<name>/genetic_scores_<B x rank>.float32.npy + genetic_scores_meta.json (h2 test)
Default GRM: numpy K = Z Z'/M from column-standardised genotypes. --plink2 PATH uses
`plink2 --make-rel square0 bin4` instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hibiscus.cli import common_parser, environment, relpath, resolve, timed
from hibiscus.grm import (
    dense_grm_from_plink, edge_metadata, genetic_score_bank, random_grm_scores, read_plink2_rel_square0_bin4,
    run_plink2_make_rel, select_subject_edges,
)
from hibiscus.io import PlinkBedReader, atomic_save_npy, ensure_dir, read_sample_ids, save_edges, save_json


def main() -> int:
    parser = common_parser(__doc__)
    parser.add_argument("--plink2", type=Path, default=None, help="path to a plink2 binary (optional GRM route)")
    parser.add_argument("--threads", type=int, default=1, help="threads for plink2")
    args = parser.parse_args()
    cfg, paths = resolve(args)
    grm_cfg = cfg["grm"]
    out_dir = ensure_dir(paths["grm"])
    edges_npz = paths["edges_npz"]
    edges_json = paths["edges_json"]
    scores_npy = paths["random_scores_npy"]
    bank_npy = paths["genetic_scores_npy"]
    if edges_npz.exists() and edges_json.exists() and scores_npy.exists() and bank_npy.exists() and not args.force:
        print(f"SKIP: {relpath(out_dir, paths['workdir'])}/ already complete (use --force)")
        return 0

    reader = PlinkBedReader(paths["plink_prefix"])
    sample = read_sample_ids(paths["sample_ids_csv"])
    fam = reader.fam_frame()
    if len(fam) != len(sample) or not np.array_equal(fam["IID"].to_numpy(str), sample["IID"].to_numpy(str)):
        raise SystemExit("sample_ids.csv must list the same IIDs as the .fam file, in the same order")
    print(f"genotypes: {reader.n_subjects} subjects x {reader.n_variants} variants ({paths['plink_prefix']})")

    if args.plink2 is not None:
        rel_prefix = out_dir / "plink2_rel_square0_bin4"
        rel_bin = Path(str(rel_prefix) + ".rel.bin")
        if not rel_bin.exists() or args.force:
            _, seconds = timed(run_plink2_make_rel, args.plink2, paths["plink_prefix"], rel_prefix, threads=args.threads)
            print(f"  plink2 --make-rel square0 bin4: {seconds:.1f}s")
        grm, seconds = timed(read_plink2_rel_square0_bin4, rel_bin, reader.n_subjects)
        input_mode = "plink2_make_rel"
    else:
        grm, seconds = timed(dense_grm_from_plink, reader, int(grm_cfg.get("variant_block_size", 256)))
        input_mode = "numpy_dense_grm"
    off = grm[np.triu_indices(grm.shape[0], k=1)]
    print(f"  dense GRM ({input_mode}) in {seconds:.1f}s; off-diagonal mean {off.mean():.2e}, sd {off.std():.4f}; diagonal mean {np.mean(np.diag(grm)):.4f}")

    edges, seconds = timed(select_subject_edges, grm, grm_cfg)
    meta = edge_metadata(edges, reader.n_subjects, reader.n_variants, grm_cfg, input_mode)
    meta["plink_prefix"] = relpath(paths["plink_prefix"], paths["workdir"])
    meta["sample_ids_csv"] = relpath(paths["sample_ids_csv"], paths["workdir"])
    meta["select_seconds"] = seconds
    meta["environment"] = environment()
    save_edges(edges_npz, edges_json, edges["row"], edges["col"], edges["kij"], meta)
    print(f"  edges: {edges['edge_count']:,} retained (cap {grm_cfg['cg_surface_max_edges_per_row']} per row, "
          f"mass fraction {grm_cfg['cg_surface_weight_mass_fraction']}); mean row K_ij^2 capture {edges['row_weight_capture_mean']:.3f}; {seconds:.1f}s")
    del grm

    scores, score_meta = random_grm_scores(reader, int(grm_cfg["random_score_rank"]), int(grm_cfg["random_score_seed"]), int(grm_cfg.get("variant_block_size", 256)))
    atomic_save_npy(scores_npy, scores)
    score_meta["output_npy"] = relpath(scores_npy, paths["workdir"])
    score_meta["plink_prefix"] = meta["plink_prefix"]
    save_json(out_dir / "random_scores_meta.json", score_meta)
    print(f"  random GRM scores: {scores.shape[0]} x {scores.shape[1]} (K_hat = S S'/{scores.shape[1]})")

    h2_cfg = cfg["h2_test"]
    n_cols = int(h2_cfg["n_boot"]) * int(h2_cfg["max_rank"])
    (bank, bank_meta), seconds = timed(genetic_score_bank, reader, n_cols, int(h2_cfg["bank_seed"]), edges,
                                       int(grm_cfg.get("variant_block_size", 256)))
    atomic_save_npy(bank_npy, bank)
    bank_meta["output_npy"] = relpath(bank_npy, paths["workdir"])
    bank_meta["plink_prefix"] = meta["plink_prefix"]
    save_json(out_dir / "genetic_scores_meta.json", bank_meta)
    print(f"  genetic score bank: {bank.shape[0]} x {bank.shape[1]} (Z Theta/sqrt(M)); GRM scale on "
          f"{bank_meta['grm_scale_check_edges']:,} edges = {bank_meta['grm_scale']:.6f} (corr {bank_meta['grm_scale_check_corr']:.6f}); {seconds:.1f}s")
    print(f"  wrote {relpath(out_dir, paths['workdir'])}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
