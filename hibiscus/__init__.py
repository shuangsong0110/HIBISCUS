"""HIBISCUS: age-varying SNP heritability h2(t) and the genetic-correlation surface R_G(t1, t2).

Kernel-weighted Haseman-Elston estimation on a sparse genomic-relationship edge set, conditional
delta-method standard errors, the h2 test of age variation (H0: h2(t) = c; hibiscus/h2_test.py),
the correlation test of unit genetic correlation (H0: R_G = 1; hibiscus/corr_test.py) and their
ACAT combination p_joint. Everything runs from a relative-path checkout with numpy, scipy, pandas
and matplotlib.
"""
from __future__ import annotations

__version__ = "0.1.0"

EPS = 1e-12
Z975 = 1.959963984540054
