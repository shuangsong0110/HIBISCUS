#!/usr/bin/env python3
"""Step 1 - data preparation.

Default: simulate the toy dataset described in the config and write it in the real input formats
  data/<name>/geno.{bed,bim,fam}   PLINK 1 genotypes (SNP-major .bed)
  data/<name>/phenotype.tsv        FID<TAB>IID<TAB>y<TAB>t, one row per visit
  data/<name>/sample_ids.csv       FID,IID in .fam (= GRM) order
  data/<name>/truth_h2.csv, truth_surface.csv, dataset_info.json
Validation mode (real data): --validate checks a phenotype TSV, sample-ID CSV and PLINK prefix
against the format specification without writing anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hibiscus.cli import common_parser, environment, relpath, resolve, timed
from hibiscus.io import PlinkBedReader, match_phenotype_to_samples, read_phenotype_first_entry, read_sample_ids, save_json
from hibiscus.kernel import grid_points
from hibiscus.simulate import simulate_dataset, write_dataset


def validate(paths: dict) -> int:
    problems = []
    print(f"phenotype TSV : {paths['phenotype_tsv']}")
    print(f"sample IDs    : {paths['sample_ids_csv']}")
    print(f"PLINK prefix  : {paths['plink_prefix']}")
    try:
        pheno = read_phenotype_first_entry(paths["phenotype_tsv"])
        print(f"  phenotype rows after dropping missing y/t and keeping the first entry per IID: {len(pheno)}")
        print(f"  age range: {pheno['t'].min():g} - {pheno['t'].max():g}; y mean {pheno['y'].mean():.4g}, sd {pheno['y'].std():.4g}")
    except Exception as error:
        problems.append(f"phenotype TSV: {error}")
        pheno = None
    try:
        sample = read_sample_ids(paths["sample_ids_csv"])
        print(f"  sample-ID rows: {len(sample)}")
        if sample["IID"].duplicated().any():
            problems.append("sample-ID CSV has duplicate IIDs")
    except Exception as error:
        problems.append(f"sample-ID CSV: {error}")
        sample = None
    try:
        reader = PlinkBedReader(paths["plink_prefix"])
        print(f"  PLINK: {reader.n_subjects} subjects x {reader.n_variants} variants")
        if sample is not None:
            fam = reader.fam_frame()
            if len(fam) != len(sample) or not np.array_equal(fam["IID"].to_numpy(str), sample["IID"].to_numpy(str)):
                problems.append("sample-ID CSV is not in .fam order (same IIDs, same order required)")
    except Exception as error:
        problems.append(f"PLINK files: {error}")
    if pheno is not None and sample is not None:
        _, matched = match_phenotype_to_samples(pheno, sample)
        print(f"  matched subjects: {len(matched)} of {len(sample)} genotyped; {len(pheno) - len(matched)} phenotype rows unmatched")
        if len(matched) < 2:
            problems.append("fewer than two phenotyped subjects match the sample-ID file")
    if problems:
        print("VALIDATION FAILED")
        for item in problems:
            print("  -", item)
        return 1
    print("VALIDATION OK")
    return 0


def main() -> int:
    parser = common_parser(__doc__)
    parser.add_argument("--validate", action="store_true",
                        help="validate user-supplied inputs instead of simulating (automatic when the config has no simulation block)")
    args = parser.parse_args()
    cfg, paths = resolve(args)
    if args.validate or "n_subjects" not in cfg["dataset"]:
        if not args.validate:
            print("no simulation block in the config: validating the input files instead")
        return validate(paths)

    data_dir = paths["data"]
    info_path = data_dir / "dataset_info.json"
    if info_path.exists() and not args.force:
        print(f"SKIP: {relpath(info_path, paths['workdir'])} exists (use --force to regenerate)")
        return 0
    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg["name"] = paths["name"]
    grid = grid_points(cfg["grid"]["t_min"], cfg["grid"]["t_max"], cfg["grid"]["size"])
    print(f"simulating {paths['name']}: N={dataset_cfg['n_subjects']} M={dataset_cfg['n_snps']} seed={dataset_cfg['seed']}")
    dataset, seconds = timed(simulate_dataset, dataset_cfg)
    info = write_dataset(data_dir, dataset, dataset_cfg, grid)
    info["environment"] = environment()
    info["simulate_seconds"] = seconds
    save_json(info_path, info)
    summary = info["truth_summary"]
    print(f"  truth h2: {summary['h2_at_grid_min']:.3f} at age {summary['grid_t_min']:g} -> {summary['h2_at_grid_max']:.3f} at age {summary['grid_t_max']:g}; "
          f"true R_G({summary['grid_t_min']:g},{summary['grid_t_max']:g}) = {summary['corr_grid_min_grid_max']:.3f}")
    print(f"  wrote {relpath(data_dir, paths['workdir'])}/ (geno.bed/.bim/.fam, phenotype.tsv, sample_ids.csv, truth_*.csv, dataset_info.json) in {seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
