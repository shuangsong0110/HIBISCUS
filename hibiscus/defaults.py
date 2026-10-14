"""Fixed settings of the pipeline (not user inputs)."""
from __future__ import annotations

GRM = {
    "cg_surface_weight_mass_fraction": 0.995,     # per subject: keep pairs until this share of sum K_ij^2 ...
    "cg_surface_max_edges_per_row": 1024,         # ... or this many pairs, whichever comes first
    "variant_block_size": 256,
    "random_score_rank": 256,                     # rank of the randomised GRM approximation (correlation test)
    "random_score_seed": 2026081501,
}
KERNEL = {
    "kernel": "gaussian",
    "target_effective_n_fraction": 0.2,           # bandwidth per grid age: kernel effective sample size = 0.2 N
    "bandwidth_min": 3.0,                         # years
    "bandwidth_max": 12.0,
    "cg_surface_post_smooth_bandwidth": 2.0,      # smoothing of the genetic-variance diagonal, years
    "cg_surface_single_visit_chunk_size": 200000,
}
TEST_MARGIN = 5.0                                 # default test range: observed age range shrunk by this many years at each end
TEST_AGE_STEP = 5.0                               # h2 test: tested ages every 5 years within the test range
CORR_GRID_SIZE = 25                               # correlation test: grid points within the test range
SE_FD_EPS_SCALE = 1e-6                            # relative finite-difference step of the standard errors
H2_TEST = {"n_boot": 200, "max_rank": 4, "seed": 20260923, "bank_seed": 2026092301, "c_star_bounds": (0.01, 0.99)}
CORR_TEST = {"basis_variance_capture": 0.999, "max_basis_rank": 12, "edge_chunk_size": 200000}
FIGURE = {"heatmap_vmin": 0.2, "heatmap_vmax": 1.0, "heatmap_vyellow": 0.5, "dpi_png": 200}
