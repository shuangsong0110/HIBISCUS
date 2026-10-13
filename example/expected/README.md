# Reference outputs of the toy example

Produced by `bash run_pipeline.sh` (config/toy.json, dataset `toy_dynamic`) and
`bash run_pipeline.sh config/toy_null.json` (dataset `toy_null`) with Python 3.12, numpy 2.4,
scipy 1.17, pandas 3.0 and matplotlib 3.10.

    <name>/curves_diag2.csv           h2(t), sigma_g2, sigma_y2 and the conditional SE / CI on the 31-point grid
    <name>/curves_sparse_he_no_t.csv  age-independent HE estimate with its SE
    <name>/h2_test.csv                the h2 test (T, p_h2, p_h2_mc, c*, raw curve and null SD at the tested ages, gates)
    <name>/corr_test.csv              the correlation test (p_corr, Z and its diagnostics)
    <name>/joint_test.csv             p_h2, p_corr and p_joint = ACAT(p_h2, p_corr)
    all_traits.csv                    the summary over both datasets
    toy_dynamic/h2_curve.png          the two figures of toy_dynamic (shown in the README)
    toy_dynamic/corr_heatmap.png

A rerun reproduces these files bit for bit with the same BLAS thread count. Across thread counts
or BLAS libraries the estimates agree to about 1e-15 relative, and the standard errors and p-values
to about 1e-8 relative (floating-point reduction order).
