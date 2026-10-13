"""Small in-memory problems shared by the tests."""
from __future__ import annotations

from hibiscus.estimate import build_design, estimator_config
from hibiscus.grm import dense_grm_from_standardized, select_subject_edges
from hibiscus.kernel import grid_points
from hibiscus.simulate import simulate_dataset

KERNEL_CFG = {
    "kernel": "gaussian", "target_effective_n_fraction": 0.2, "bandwidth_min": 3.0, "bandwidth_max": 12.0,
    "cg_surface_post_smooth_bandwidth": 2.0, "cg_surface_single_visit_chunk_size": 50000,
}


def dataset_cfg(n=400, m=300, seed=7, pattern="monotone_rank2", lambdas=(0.3, 0.9)):
    return {
        "name": "test", "seed": seed, "n_subjects": n, "n_snps": m, "maf_min": 0.05, "maf_max": 0.5,
        "age_min": 40.0, "age_max": 70.0, "residual_variance": 0.6,
        "truth": {"pattern": pattern, "lambdas": list(lambdas), "z_center": 55.0, "z_scale": 10.0},
    }


def small_problem(n=400, m=300, seed=7, grid_size=7, all_edges=True, **kw):
    cfg = dataset_cfg(n=n, m=m, seed=seed, **kw)
    data = simulate_dataset(cfg)
    z = data["genotypes_standardized"]
    grm = dense_grm_from_standardized(z)
    edge_cfg = {"cg_surface_weight_mass_fraction": 1.0 if all_edges else 0.995,
                "cg_surface_max_edges_per_row": n if all_edges else 64}
    edges = select_subject_edges(grm, edge_cfg)
    grid = grid_points(40.0, 70.0, grid_size)
    config = estimator_config(KERNEL_CFG, n)
    design = build_design(data["ages"], edges, grid, config)
    return {"cfg": cfg, "data": data, "z": z, "grm": grm, "edges": edges, "grid": grid, "config": config, "design": design,
            "y": data["y"], "t": data["ages"]}
