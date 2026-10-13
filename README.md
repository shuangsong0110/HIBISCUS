# HIBISCUS

HIBISCUS estimates how SNP heritability changes with age. From genotypes, one quantitative phenotype
and the age at which it was measured, it estimates the age-varying SNP heritability h²(t) and the
genetic-correlation surface R_G(t₁, t₂) between ages, with standard errors, by kernel-weighted
Haseman–Elston regression on a sparse genomic-relationship (GRM) edge set. It then tests two
hypotheses: the **h² test** (H₀: h²(t) is constant over age) and the **correlation test**
(H₀: R_G(t₁, t₂) = 1 for all ages, i.e. the same genetic effects act at every age), and combines
them into a joint p-value, **p_joint = ACAT(p_h2, p_corr)**. The package is pure Python (numpy, scipy,
pandas, matplotlib); the toy example below runs in about two minutes on a laptop.

## Table of contents
* [Installation](#hammer-installation)
* [Step 1: Prepare the input files](#dna-step-1-prepare-the-input-files)
* [Step 2: Build the GRM and the sparse edge set](#scroll-step-2-build-the-grm-and-the-sparse-edge-set)
* [Step 3: Estimate h2(t) and R_G(t1, t2)](#rocket-step-3-estimate-h2t-and-r_gt1-t2)
* [Step 4: Standard errors](#bar_chart-step-4-standard-errors)
* [Step 5: Tests of age variation](#mag-step-5-tests-of-age-variation)
* [Step 6: Figures](#framed_picture-step-6-figures)
* [Output](#bulb-output)
* [A toy example](#key-a-toy-example)
* [Configuration](#wrench-configuration)
* [Maintainer](#busts_in_silhouette-maintainer)
* [References](#references)

## :hammer: Installation

HIBISCUS requires Python >= 3.10 with `numpy`, `scipy`, `pandas` and `matplotlib`. No R, PLINK
or compiled code is needed (PLINK 2 can optionally build the GRM in Step 2).

```
git clone https://github.com/shuangsong0110/HIBISCUS.git
cd HIBISCUS
pip install -r requirements.txt
```

`pip install -e .` additionally makes `import hibiscus` available outside the checkout; the
scripts below run either way.

The analysis is six scripts in `scripts/`, run in order. `run_pipeline.sh` runs all six:

```
bash run_pipeline.sh                           # the toy example (config/toy.json, work directory example/)
bash run_pipeline.sh my_config.json my_workdir # your own data
```

Every script accepts `--config PATH` (a JSON file, see [Configuration](#wrench-configuration)),
`--workdir DIR` (the root of `data/ grm/ results/ figures/ summary/`), `--dataset NAME` (the trait
name; default `dataset.name` of the config) and `--force` (recompute outputs that already exist).
Every script also accepts `--phenotype-tsv`, `--sample-ids-csv`, `--plink-prefix`, `--edges-npz`,
`--edges-json`, `--random-scores-npy` and `--genetic-scores-npy` to point at your own files
instead of the default locations `<workdir>/data/<name>/` and `<workdir>/grm/<name>/`.

## :dna: Step 1: Prepare the input files

#### Script: [`scripts/01_prepare_data.py`](scripts/01_prepare_data.py)

HIBISCUS reads three files.

| File | Format |
|---|---|
| `phenotype.tsv` | Tab-separated with a header. Columns `FID`, `IID`, `y` (the phenotype) and `t` (the age at which `y` was measured); other columns are ignored. One row per visit: rows with a missing `y` or `t` are dropped, then the **first remaining row per `IID`** is kept. Apart from the internal standardisation, no transformation is applied to `y`. |
| `sample_ids.csv` | Comma-separated with a header `FID,IID`: the genotyped subjects **in the order of the `.fam` file**. Row *i* is subject *i* in the GRM, the edge set and the score files. |
| `geno.bed`, `geno.bim`, `geno.fam` | PLINK 1 binary genotypes. Missing genotypes are set to the SNP mean. |

Synthetic example rows:

```
FID	IID	y	t
1	1	0.4817	58.3
2	2	-1.2203	44.9
3	3	0.0652	61.0
```

```
FID,IID
1,1
2,2
3,3
```

**Adjust the phenotype for covariates before running HIBISCUS**: regress `y` on an intercept, age, age²,
sex and the top genetic PCs (plus any batch covariates), and use the residuals as `y`. HIBISCUS itself
only centres and standardises `y` internally, so no result depends on its unit; variances are written
back in the original units.

Subjects are matched to the genotypes by `IID`. Check your files before running the pipeline:

```
python scripts/01_prepare_data.py --validate --phenotype-tsv my.tsv --sample-ids-csv ids.csv --plink-prefix geno
```

Without `--validate`, the script simulates the toy dataset described by the `dataset` block of the
config into `<workdir>/data/<name>/` (used by the toy example; it skips if `dataset_info.json`
already exists).

## :scroll: Step 2: Build the GRM and the sparse edge set

#### Script: [`scripts/02_build_grm.py`](scripts/02_build_grm.py)

```
python scripts/02_build_grm.py --config config/template.json --workdir mywork --dataset mytrait \
    --plink-prefix geno --sample-ids-csv ids.csv
```

- The GRM is K = Z Zᵀ / M with Z the column-standardised genotypes (`--plink2 /path/to/plink2
  [--threads 8]` uses `plink2 --make-rel square0 bin4` instead).
- For every subject i the GRM entries K_ij with j > i are sorted by K²_ij and kept until 99.5% of
  their Σ K²_ij is covered or `cg_surface_max_edges_per_row` entries are reached, whichever comes
  first (in the toy the cap of 1,024 binds and the kept pairs carry 86% of the mass). Each pair is
  stored once in `grm/<name>/edges.npz` (`row < col`, `kij`) with `edges.json`.
- Two auxiliary files are drawn from the genotypes: rank-256 randomised GRM scores
  (`random_scores_rank256.float32.npy`, used by the correlation test) and a bank of 800 genetic
  scores (`genetic_scores_800.float32.npy`, used by the h² test).

Run this step **once per cohort**: all traits measured on the same genotyped subjects share the
GRM, the edges and the score files (see [Your own data](#your-own-data) for the flags that point
later steps at them). The GRM is built as a dense N × N matrix and the script holds a few copies of
it while selecting the edges, about 24 N² bytes at peak (0.6 GB at N = 5,000, 60 GB at N = 50,000);
Steps 3–6 only touch the stored edges.

#### Output: `grm/<name>/edges.npz`, `edges.json`, `random_scores_rank256.float32.npy`, `genetic_scores_800.float32.npy` and their `*_meta.json`

## :rocket: Step 3: Estimate h2(t) and R_G(t1, t2)

#### Script: [`scripts/03_estimate.py`](scripts/03_estimate.py)

```
python scripts/03_estimate.py --config config/template.json --workdir mywork --dataset mytrait \
    --phenotype-tsv my.tsv --sample-ids-csv ids.csv
```

On a grid of ages (default 40 to 70, 31 points), each subject receives Gaussian kernel weights
around every grid age, with the bandwidth chosen per grid age so that the kernel effective sample
size is 20% of N (within 3–12 years). The genetic covariance surface C_g(t₁, t₂) is the
kernel-weighted Haseman–Elston regression of the phenotype cross-products of the stored subject
pairs on their GRM entries; it is projected to positive semi-definite, its diagonal is smoothed
(bandwidth 2 years) and divided by the local phenotypic variance σ²_y(t) to give h²(t), and its
correlation gives R_G(t₁, t₂). The age-independent HE estimate ĥ²_HE on the same edges is
reported as a reference. The matched phenotype is standardised (mean 0, SD 1) before estimation, so
h²(t), R_G, the standard errors and the tests do not depend on the unit of `y`; σ²_g(t), σ²_y(t), σ²_ε(t)
and the covariance surfaces are converted back to the original units when written (the SD is recorded as
`y_scale` in `fit_manifest.json`).

#### Output: `results/<name>/curves_diag2.csv` (h²(t), σ²_g(t), σ²_y(t) on the grid), `curves_sparse_he_no_t.csv` (ĥ²_HE), `surfaces_diag2.csv` (C_g and R_G), `surface_rows_cg_surface_diag2.csv`, `fit_manifest.json`

In the `method` column, `diag2` labels the HIBISCUS kernel-weighted estimator and `sparse_he_no_t`
the age-independent HE estimate. The step also writes `surfaces_sparse_he_no_t.csv`, the combined
`curves_all_methods.csv` and `surfaces_all_methods.csv`, `subject_table.csv` (the matched subjects)
and the run metadata `diagnostics.json`, `runtime.json` and `dataset_info.json`.

## :bar_chart: Step 4: Standard errors

#### Script: [`scripts/04_se.py`](scripts/04_se.py)

```
python scripts/04_se.py --config config/template.json --workdir mywork --dataset mytrait \
    --phenotype-tsv my.tsv --sample-ids-csv ids.csv
```

Conditional standard errors of ĥ²(t) and ĥ²_HE (given genotypes, edges and ages) from the
influence function of each estimator, with the residual variance σ²_y(t) − σ²_g(t). The step adds
`h2_se_conditional`, `h2_ci_lower_conditional` and `h2_ci_upper_conditional` (± 1.96 SE, clipped to
[0, 1]) to the curve files; the figure of Step 6 draws the ± 1 SE band.

#### Output: SE columns in `results/<name>/curves_diag2.csv`, `curves_sparse_he_no_t.csv` and `curves_all_methods.csv`; `results/<name>/se/se_summary.json`

## :mag: Step 5: Tests of age variation

#### Script: [`scripts/05_tests.py`](scripts/05_tests.py)

```
python scripts/05_tests.py --config config/template.json --workdir mywork --dataset mytrait \
    --phenotype-tsv my.tsv --sample-ids-csv ids.csv
```

### Step 5.1: The h² test

H₀: h²(t) = c for some constant c. The test uses the raw curve ĥ²_raw(t), the smoothed kernel
estimate before the positive semi-definite projection of Step 3; it is stored as `h2_raw_<age>`
and drawn as open circles in the figure. The
statistic compares ĥ²_raw at the tested ages (default 45, 50, 55, 60, 65) with its own mean, each
deviation scaled by its null standard deviation. The null distribution is obtained by simulating
B = 200 phenotypes from the fitted model with its heritability flattened to c*, the mean of ĥ²_raw
at the tested ages clipped to `c_star_bounds` (genetic effects redrawn from the score bank of
Step 2, correlations kept), and passing each through the same estimator. Two p-values are reported:
`p_h2`, from the fitted law of the null statistic, and `p_h2_mc`, the Monte Carlo p-value, whose
floor is 1/(B + 1).

#### Output: `results/<name>/h2_test.csv` (`T`, `p_h2`, `p_h2_mc`, `c_star`, `c_star_clipped`, the raw curve `h2_raw_<age>` and null SD `sd_null_<age>` at the tested ages, the per-trait `seed`), `h2_test_replicates.npz`

### Step 5.2: The correlation test

H₀: R_G(t₁, t₂) = 1 for all pairs of ages (on a 25-point grid from 45 to 65 by default), i.e.
C_g(t₁, t₂) = a(t₁) a(t₂). A rank-one null with a smooth amplitude a(t) is fitted and forwarded
through the same estimator; the mean departure of the observed correlations from the null
correlations, standardised by its analytic variance, gives Z and the one-sided `p_corr` = Φ(Z):
the alternative is that the correlations fall below 1 (Z < 0), so Z > 0 gives `p_corr` > 0.5. If
the null model cannot be fitted the row is written with `status = nonregular` (degenerate inputs)
or `status = failed` (the amplitude fit did not converge), the `reason`, and a missing `p_corr`;
the pipeline continues.

#### Output: `results/<name>/corr_test.csv` (`status`, `p_corr`, `corr_mean_z`, `corr_mean_departure`, `corr_pair_count`, `corr_basis_rank` and further diagnostics), `corr_test_diagnostics.npz`, `corr_test/source_45_65_*.csv`

### Step 5.3: The joint test

`p_joint = ACAT(p_h2, p_corr)`, the equal-weight Cauchy combination of the two p-values
(Liu & Xie 2020); it is missing when `p_corr` is missing.

#### Output: `results/<name>/joint_test.csv` (`phenotype_key`, `p_h2`, `p_corr`, `p_joint`)

The script finally rebuilds `summary/all_traits.csv` (with `all_traits_meta.json`): one row per
trait directory under `results/` that holds all three test files, with N, the number of edges,
ĥ²_HE ± SE, ĥ²(t) ± SE at the tested ages, the minimum off-diagonal R̂_G, the three p-values and,
across traits, Benjamini–Hochberg q-values and Bonferroni p-values for each of `p_h2`, `p_corr`
and `p_joint` (columns `bh_q_*`, `bonferroni_p_*`, `*_significant_*` at level `alpha`). The
multiple-testing columns of a p-value are added only when it is present for every trait. With
several traits on one cohort, run Step 2 once and Steps 3–6 per trait (see
[Your own data](#your-own-data)); the summary then covers all of them.

## :framed_picture: Step 6: Figures

#### Script: [`scripts/06_figures.py`](scripts/06_figures.py)

```
python scripts/06_figures.py --config config/template.json --workdir mywork --dataset mytrait --no-truth
```

- `h2_curve.pdf/.png`: ĥ²(t) with its ± 1 SE band, the HE reference (dashed, ± 1 SE), the raw curve
  at the tested ages (open circles) and the three p-values in the legend.
- `corr_heatmap.pdf/.png`: R̂_G(t₁, t₂).

`--no-truth` omits the simulation truth, which is drawn only when `data/<name>/truth_*.csv` exist.
Each figure comes with a `*_data.csv` holding every plotted value.

#### Output: `figures/<name>/h2_curve.{pdf,png}`, `h2_curve_data.csv`, `corr_heatmap.{pdf,png}`, `corr_heatmap_data.csv`

## :bulb: Output

Variance columns (`sigma_g2`, `sigma_eps2`, `sigma_y2`, `cov`, `cg_raw`, `cg_psd`) are in the units of `y`²;
every other quantity is unit-free.

| File | Columns |
|---|---|
| `results/<name>/curves_diag2.csv` | `t`, `method`, `sigma_g2`, `sigma_eps2`, `sigma_y2`, `h2`, `h2_se_conditional`, `h2_ci_lower_conditional`, `h2_ci_upper_conditional` (31 rows, one per grid age) |
| `results/<name>/curves_sparse_he_no_t.csv` | the same columns for the age-independent HE estimate (constant over `t`) |
| `results/<name>/surfaces_diag2.csv` | `t_row`, `t_col`, `method`, `cov` (C_g) and `corr` (R_G) for every pair of grid ages |
| `results/<name>/h2_test.csv` | `T`, `p_h2`, `log10_p_h2`, `p_h2_mc`, `c_star`, `c_star_clipped`, `null_rank`, `n_boot`, `seed` (derived from `h2_test.seed` and the trait name), `h2_raw_<age>`, `sd_null_<age>`, the gates `gate_*` |
| `results/<name>/corr_test.csv` | `status`, `p_corr`, `corr_mean_z`, `corr_mean_departure`, `corr_pair_count`, `corr_basis_rank`, `corr_observed_min_correlation`, `corr_forward_null_min_correlation`, … (`reason` when the status is not `complete`) |
| `results/<name>/joint_test.csv` | `phenotype_key`, `p_h2`, `p_corr`, `p_joint` |
| `summary/all_traits.csv` | `phenotype_key`, `n_subjects`, `edge_count`, `h2_he`, `se_he`, `h2_dynamic_<age>`, `se_dynamic_<age>`, `min_offdiag_corr`, `p_h2`, `p_corr`, `p_joint`, `p_h2_mc`, `T_h2`, `c_star_h2`, `corr_status`, `corr_mean_z`, …, `bh_q_{h2,corr,joint}`, `bonferroni_p_{h2,corr,joint}`, `raw_significant_*`, `bh_significant_*`, `bonferroni_significant_*` |

## :key: A toy example

The repository ships a simulated dataset in `example/data/toy_dynamic/` (N = 5,000 subjects,
M = 1,000 SNPs, one visit per subject at ages 40–70) whose true h²(t) rises from 0.25 at age 40 to
0.65 at age 70 with R_G(40, 70) = 0.50. The files are in exactly the input format of Step 1.

```
bash run_pipeline.sh                           # steps 1-6 on example/data/toy_dynamic, about 2 minutes
bash run_pipeline.sh config/toy_null.json      # the same on a simulated null: h2 = 0.40 at every age
python tests/run_tests.py                      # 22 unit tests (plain asserts, no pytest needed)
```

`PYTHON_BIN=/path/to/python bash run_pipeline.sh` selects the interpreter and `FORCE=1` recomputes
everything. Results are written under `example/`; reference outputs of both configurations are in
`example/expected/`.

| Quantity | `config/toy.json` (h² 0.25 → 0.65) | `config/toy_null.json` (h² ≡ 0.40) |
|---|---|---|
| ĥ²_HE ± SE | 0.372 ± 0.016 | 0.407 ± 0.016 |
| ĥ²(45) / ĥ²(55) / ĥ²(65) ± SE | 0.283 ± 0.037 / 0.436 ± 0.029 / 0.608 ± 0.029 | 0.393 ± 0.030 / 0.446 ± 0.028 / 0.421 ± 0.036 |
| h² test: T, c* | 9.08, 0.441 | 0.306, 0.408 |
| `p_h2` (`p_h2_mc`) | 3.8 × 10⁻⁸ (0.0050, the Monte Carlo floor) | 0.62 (0.62) |
| `p_corr` (Z) | 3.3 × 10⁻⁶ (−4.5) | 0.66 (+0.41) |
| `p_joint` | 7.6 × 10⁻⁸ | 0.64 |
| min off-diagonal R̂_G | 0.51 (truth 0.50) | 0.88 (truth 1) |

<p align="center">
<img src="example/expected/toy_dynamic/h2_curve.png" width="520">
</p>
<p align="center">
<img src="example/expected/toy_dynamic/corr_heatmap.png" width="600">
</p>

### Your own data

```
python scripts/01_prepare_data.py --validate --phenotype-tsv my.tsv --sample-ids-csv ids.csv --plink-prefix geno
python scripts/02_build_grm.py   --config config/template.json --workdir mywork --dataset mytrait --plink-prefix geno --sample-ids-csv ids.csv
python scripts/03_estimate.py    --config config/template.json --workdir mywork --dataset mytrait --phenotype-tsv my.tsv --sample-ids-csv ids.csv
python scripts/04_se.py          --config config/template.json --workdir mywork --dataset mytrait --phenotype-tsv my.tsv --sample-ids-csv ids.csv
python scripts/05_tests.py       --config config/template.json --workdir mywork --dataset mytrait --phenotype-tsv my.tsv --sample-ids-csv ids.csv
python scripts/06_figures.py     --config config/template.json --workdir mywork --dataset mytrait --no-truth
```

`config/template.json` holds the default parameters without a simulation block; copy it and edit
the grid or the tested ages to match the age range of your cohort (the tested ages must be grid
points). The same run as one command: put the three files in `mywork/data/mytrait/` as
`phenotype.tsv`, `sample_ids.csv` and `geno.bed/.bim/.fam`, set `dataset.name` to `mytrait` in your
config, and run `bash run_pipeline.sh my_config.json mywork` (Step 1 validates the files instead of
simulating when the config has no simulation block).

Several traits on one cohort: run Step 2 once with `--dataset cohort`, then Steps 3–6 per trait with
`--dataset <trait>` and the shared files passed explicitly, e.g.
`--edges-npz mywork/grm/cohort/edges.npz --edges-json mywork/grm/cohort/edges.json
--random-scores-npy mywork/grm/cohort/random_scores_rank256.float32.npy
--genetic-scores-npy mywork/grm/cohort/genetic_scores_800.float32.npy`.

## :wrench: Configuration

All parameters live in one JSON file (`config/toy.json`, `config/toy_null.json`, `config/template.json`).

| Block | Keys | Default |
|---|---|---|
| `dataset` | `name`; for simulation only: `seed`, `n_subjects`, `n_snps`, `maf_min`, `maf_max`, `age_min`, `age_max`, `residual_variance`, `truth{pattern, lambdas, z_center, z_scale}` | `toy_dynamic`: N = 5000, M = 1000, `monotone_rank2`, λ = (0.2, 1.0) |
| `grm` | `cg_surface_weight_mass_fraction`, `cg_surface_max_edges_per_row`, `variant_block_size`, `random_score_rank`, `random_score_seed` | 0.995, 1024, 256, 256, 2026081501 |
| `kernel` | `kernel`, `target_effective_n_fraction`, `bandwidth_min`, `bandwidth_max`, `cg_surface_post_smooth_bandwidth`, `cg_surface_single_visit_chunk_size` | gaussian, 0.2, 3, 12, 2.0, 200000 |
| `grid` | `t_min`, `t_max`, `size`, `test_ages`, `corr_t_min`, `corr_t_max`, `corr_size` | 40, 70, 31; 45 … 65; 45, 65, 25 |
| `se` | `fd_eps_scale` | 1e-6 |
| `h2_test` | `n_boot`, `max_rank`, `seed`, `bank_seed`, `c_star_bounds`, `alpha` | 200, 4, 20260923, 2026092301, [0.01, 0.99], 0.05 |
| `corr_test` | `basis_variance_capture`, `max_basis_rank`, `edge_chunk_size`, `nuisance_ess_exponent` | 0.999, 12, 200000, 0.8 |
| `figures` | `heatmap_vmin`, `heatmap_vmax`, `heatmap_vyellow`, `dpi_png` | 0.2, 1.0, 0.5, 200 |

## :busts_in_silhouette: Maintainer

Please contact Shuang Song (shuangsong@hsph.harvard.edu) if there are any problems or questions.

## References

* Haseman JK, Elston RC (1972). The investigation of linkage between a quantitative trait and a marker locus. *Behavior Genetics* 2, 3–19.
* Fan J, Gijbels I (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
* Imhof JP (1961). Computing the distribution of quadratic forms in normal variables. *Biometrika* 48, 419–426.
* Liu Y, Xie J (2020). Cauchy combination test: a powerful test with analytic p-value calculation under arbitrary dependency structures. *Journal of the American Statistical Association* 115, 393–402.
