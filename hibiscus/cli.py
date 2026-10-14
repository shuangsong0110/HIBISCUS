"""Shared command-line plumbing for the step scripts."""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from . import __version__
from .defaults import H2_TEST


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workdir", type=Path, default=repo_root() / "example",
                        help="root holding data/, grm/, results/, figures/, summary/ (default: example/)")
    parser.add_argument("--dataset", default="toy_dynamic", help="trait name (default: toy_dynamic)")
    parser.add_argument("--test-range", type=float, nargs=2, metavar=("T_MIN", "T_MAX"), default=None,
                        help="age range of the tests (default: observed age range shrunk by 5 years at each end)")
    parser.add_argument("--phenotype-tsv", type=Path, default=None, help="override data/<name>/phenotype.tsv")
    parser.add_argument("--sample-ids-csv", type=Path, default=None, help="override data/<name>/sample_ids.csv")
    parser.add_argument("--plink-prefix", type=Path, default=None, help="override data/<name>/geno")
    parser.add_argument("--edges-npz", type=Path, default=None, help="override grm/<name>/edges.npz")
    parser.add_argument("--edges-json", type=Path, default=None, help="override grm/<name>/edges.json")
    parser.add_argument("--random-scores-npy", type=Path, default=None, help="override grm/<name>/random_scores.npy (correlation test)")
    parser.add_argument("--genetic-scores-npy", type=Path, default=None, help="override grm/<name>/genetic_scores.npy (h2 test)")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    return parser


def resolve(args: argparse.Namespace) -> dict:
    name = args.dataset
    workdir = Path(args.workdir).resolve()
    paths = {
        "workdir": workdir,
        "name": name,
        "test_range": None if args.test_range is None else (float(args.test_range[0]), float(args.test_range[1])),
        "data": workdir / "data" / name,
        "grm": workdir / "grm" / name,
        "results": workdir / "results" / name,
        "figures": workdir / "figures" / name,
        "summary": workdir / "summary",
    }
    paths["phenotype_tsv"] = args.phenotype_tsv or paths["data"] / "phenotype.tsv"
    paths["sample_ids_csv"] = args.sample_ids_csv or paths["data"] / "sample_ids.csv"
    paths["plink_prefix"] = args.plink_prefix or paths["data"] / "geno"
    paths["edges_npz"] = args.edges_npz or paths["grm"] / "edges.npz"
    paths["edges_json"] = args.edges_json or paths["grm"] / "edges.json"
    paths["random_scores_npy"] = args.random_scores_npy or paths["grm"] / "random_scores.npy"
    paths["genetic_scores_npy"] = args.genetic_scores_npy or paths["grm"] / "genetic_scores.npy"
    return paths


def bank_columns() -> int:
    return int(H2_TEST["n_boot"]) * int(H2_TEST["max_rank"])


def environment() -> dict:
    import matplotlib

    return {
        "hibiscus": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
    }


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, float(time.perf_counter() - start)


def relpath(path, workdir) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(workdir).resolve()))
    except ValueError:
        return str(path)
