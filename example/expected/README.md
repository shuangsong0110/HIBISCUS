# Reference outputs of the toy example

Produced by `bash run_pipeline.sh` (dataset `toy_dynamic`) and `bash run_pipeline.sh --dataset toy_null`
with Python 3.12, numpy 2.4, scipy 1.17, pandas 3.0 and matplotlib 3.10.

    <name>/curves.csv          h2(t), sigma_g2, sigma_eps2, sigma_y2 and the conditional SE / CI (method hibiscus and he)
    <name>/surfaces.csv        genetic covariance and correlation R_G for every pair of grid ages
    <name>/tests.csv           p_h2, p_corr, p_joint
    all_traits.csv             the summary over both datasets
    toy_dynamic/h2_curve.png   the two figures of toy_dynamic (shown in the README)
    toy_dynamic/corr_heatmap.png

A rerun reproduces these files bit for bit with the same BLAS thread count. Across thread counts
or BLAS libraries the estimates agree to about 1e-15 relative, and the standard errors and p-values
to about 1e-8 relative (floating-point reduction order).
