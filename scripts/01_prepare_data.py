#!/usr/bin/env python3
"""Step 1 - input files.

Checks data/<name>/phenotype.tsv, sample_ids.csv and geno.{bed,bim,fam} (or the files given with
--phenotype-tsv, --sample-ids-csv and --plink-prefix) against the input format. The two toy datasets,
toy_dynamic and toy_null, are simulated into data/<name>/ when their files are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hibiscus.cli import common_parser, environment, relpath, resolve, timed
from hibiscus.io import PlinkBedReader, match_phenotype_to_samples, read_phenotype_first_entry, read_sample_ids, save_json
from hibiscus.simulate import TOY_TRUTH, simulate_dataset, toy_config, write_dataset


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
    parser.add_argument("--n-subjects", type=int, default=5000, help="toy datasets only (default 5000)")
    parser.add_argument("--n-snps", type=int, default=1000, help="toy datasets only (default 1000)")
    parser.add_argument("--seed", type=int, default=20260921, help="toy datasets only")
    args = parser.parse_args()
    paths = resolve(args)
    if paths["name"] in TOY_TRUTH and (args.force or not paths["phenotype_tsv"].exists()):
        cfg = toy_config(paths["name"], args.n_subjects, args.n_snps, args.seed)
        grid = np.arange(cfg["age_min"], cfg["age_max"] + 1.0)
        print(f"simulating {paths['name']}: N={cfg['n_subjects']} M={cfg['n_snps']} seed={cfg['seed']}")
        dataset, seconds = timed(simulate_dataset, cfg)
        info = write_dataset(paths["data"], dataset, cfg, grid)
        info["environment"] = environment()
        save_json(paths["data"] / "dataset_info.json", info)
        summary = info["truth_summary"]
        print(f"  truth h2: {summary['h2_at_grid_min']:.3f} at age {summary['grid_t_min']:g} -> {summary['h2_at_grid_max']:.3f} at age {summary['grid_t_max']:g}; "
              f"true R_G({summary['grid_t_min']:g},{summary['grid_t_max']:g}) = {summary['corr_grid_min_grid_max']:.3f}")
        print(f"  wrote {relpath(paths['data'], paths['workdir'])}/ (geno.bed/.bim/.fam, phenotype.tsv, sample_ids.csv, truth_*.csv, dataset_info.json) in {seconds:.1f}s")
        return 0
    return validate(paths)


if __name__ == "__main__":
    raise SystemExit(main())
