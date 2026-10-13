from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from hibiscus.estimate import in_original_units
from hibiscus.io import (PlinkBedReader, load_edges, load_matched_dataset, match_phenotype_to_samples,
                         read_phenotype_first_entry, read_sample_ids, save_edges, subset_edges, write_plink)


def test_bed_round_trip():
    rng = np.random.default_rng(3)
    n, m = 37, 11  # n not a multiple of 4 exercises the padding
    g = rng.integers(0, 3, size=(n, m)).astype(np.int8)
    g[5, 2] = -1
    g[0, 0] = -1
    fam = pd.DataFrame({"FID": ["0"] * n, "IID": [str(1000 + i) for i in range(n)]})
    bim = pd.DataFrame({"CHR": ["1"] * m, "SNP": [f"rs{j}" for j in range(m)], "CM": ["0"] * m, "BP": [str(j) for j in range(m)], "A1": ["A"] * m, "A2": ["G"] * m})
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        prefix = Path(tmp) / "geno"
        write_plink(prefix, g, fam, bim)
        reader = PlinkBedReader(prefix)
        assert reader.n_subjects == n and reader.n_variants == m
        back = reader.read_all()
        assert back.shape == (m, n)
        assert np.array_equal(back.T, g)
        assert np.array_equal(reader.read_variant_indices([2, 0]), back[[2, 0]])
        assert list(reader.fam_frame()["IID"]) == list(fam["IID"])


def test_phenotype_first_entry_and_matching():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        tsv = Path(tmp) / "pheno.tsv"
        pd.DataFrame({
            "FID": ["0"] * 6, "IID": ["a", "b", "b", "c", "d", "e"],
            "y": [1.0, 2.0, 2.5, np.nan, 4.0, 5.0], "t": [50, 51, 60, 52, np.nan, 55],
        }).to_csv(tsv, sep="\t", index=False)
        pheno = read_phenotype_first_entry(tsv)
        assert list(pheno["IID"]) == ["a", "b", "e"]          # c and d dropped (missing), first b kept
        assert float(pheno.loc[pheno["IID"] == "b", "y"].iloc[0]) == 2.0
        csv = Path(tmp) / "ids.csv"
        pd.DataFrame({"FID": ["0"] * 4, "IID": ["e", "zz", "a", "b"]}).to_csv(csv, index=False)
        sample = read_sample_ids(csv)
        mask, matched = match_phenotype_to_samples(pheno, sample)
        assert mask.tolist() == [True, False, True, True]
        assert matched["IID"].tolist() == ["e", "a", "b"]      # sample (GRM) order is preserved
        assert matched["global_idx"].tolist() == [0, 2, 3]


def test_edge_subset_and_round_trip():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        row = np.array([0, 0, 1, 2], dtype=np.int32)
        col = np.array([1, 3, 3, 3], dtype=np.int32)
        kij = np.array([0.1, 0.2, 0.3, 0.4])
        save_edges(Path(tmp) / "e.npz", Path(tmp) / "e.json", row, col, kij, {"n_subjects": 4, "row_weight_capture_mean": 0.5})
        edges = load_edges(Path(tmp) / "e.npz", Path(tmp) / "e.json")
        assert edges["edge_count"] == 4
        local = subset_edges(edges, np.array([True, False, True, True]))
        # kept pairs: (0,3) -> (0,2), (2,3) -> (1,2)
        assert local["row"].tolist() == [0, 1] and local["col"].tolist() == [2, 2]
        assert np.allclose(local["kij"], [0.2, 0.4])
        assert local["n_subjects"] == 3


def test_load_matched_dataset_standardises_the_phenotype():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        tmp = Path(tmp)
        ids = ["a", "b", "c", "d", "e"]
        y = np.array([3.0, 10.0, 7.0, 1.0, 4.0]) * 1e-4                      # tiny units on purpose
        pd.DataFrame({"FID": ["0"] * 5, "IID": ids, "y": y, "t": [45.0, 50.0, 55.0, 60.0, 65.0]}).to_csv(tmp / "pheno.tsv", sep="\t", index=False)
        pd.DataFrame({"FID": ["0"] * 5, "IID": ids}).to_csv(tmp / "ids.csv", index=False)
        save_edges(tmp / "e.npz", tmp / "e.json", np.array([0, 1, 2], dtype=np.int32), np.array([1, 3, 4], dtype=np.int32),
                   np.array([0.1, 0.2, 0.3]), {"n_subjects": 5, "row_weight_capture_mean": 0.5})
        data = load_matched_dataset(tmp / "pheno.tsv", tmp / "ids.csv", tmp / "e.npz", tmp / "e.json")
        assert abs(float(np.mean(data["y"]))) < 1e-12 and abs(float(np.std(data["y"])) - 1.0) < 1e-12
        assert np.isclose(data["y_scale"], float(np.std(y)), rtol=1e-12, atol=0) and np.isclose(data["y_mean"], float(np.mean(y)), rtol=1e-12, atol=0)
        assert np.allclose(data["y_original"], y, rtol=1e-15, atol=0)
        frame = pd.DataFrame({"t": [40.0], "sigma_g2": [0.4], "sigma_y2": [1.0], "h2": [0.4], "cg_corr": [0.9]})
        back = in_original_units(frame, data["y_scale"])
        assert np.isclose(back["sigma_g2"][0], 0.4 * data["y_scale"] ** 2) and back["h2"][0] == 0.4 and back["cg_corr"][0] == 0.9
