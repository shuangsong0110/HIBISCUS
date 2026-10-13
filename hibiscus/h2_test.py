"""The h2 test of age variation in heritability: H0 h2(t) = c for some constant c.

Statistic (K tested ages):  T = (1/K) sum_k ((h2(t_k) - mean_j h2(t_j)) / s_k)^2

Null: B phenotypes simulated from the fitted model flattened to h2 = c*, each pushed through the
same estimator as the data:
    c*        = mean of the raw fitted curve at the tested ages, clipped to [0.01, 0.99]
    Cg*(s, t) = positive part (rank <= 4) of the raw fitted surface, rescaled by a diagonal
                congruence so that Cg*(t, t) = c* sigma_y2(t)   (correlations unchanged)
    sigma_e2* = (1 - c*) sigma_y2(t)
    y*        = g* + e*,  g*_i = sum_m sqrt(lambda_m) b_m(t_i) s_mi,  s_m ~ N(0, K) from the score bank
s_k is the SD of the B replicate curves at t_k.

p_h2 is the upper tail of T under the Gaussian law N(m, S) of the centred replicate curves, a
noncentral generalised chi-square inverted exactly on the saddle-point contour; the Monte Carlo
p (1 + #{T* >= T}) / (B + 1) is reported alongside. The tested curve is the raw smoothed diagonal
S diag(num/den) / sigma_y2, without the PSD projection.
"""
from __future__ import annotations

import cmath
import math
import warnings

import numpy as np
from scipy import integrate, optimize

from . import EPS
from .kernel import smoothing_matrix


# ----------------------------------------------------------------------------- helpers
def target_indices(grid_t, ages, atol: float = 1e-9) -> np.ndarray:
    grid_t = np.asarray(grid_t, dtype=float)
    idx = []
    for age in np.asarray(ages, dtype=float):
        hits = np.flatnonzero(np.abs(grid_t - age) <= atol)
        if hits.size != 1:
            raise ValueError(f"age {age} is not a grid point of the estimation grid")
        idx.append(int(hits[0]))
    return np.asarray(idx, dtype=int)


def flatness_statistic(curve_at_ages, scale) -> float:
    """Standardised variance of the curve about its own mean. Invariant to the level c."""
    v = np.asarray(curve_at_ages, float)
    return float(np.mean(np.square((v - v.mean()) / np.maximum(np.asarray(scale, float), EPS))))


# ----------------------------------------------------------------------------- the estimator, batched
def local_linear_variance_many(Y, t, grid, W) -> np.ndarray:
    """Local-linear residual variance on the grid for many phenotypes at once (L x Q).

    sigma_y2(g) = [S_yy - b' M^-1 b] / S0 with M = [[S0, S1], [S1, S2]], b = [S_y, S_dy], every
    sum weighted by W[:, g] with dt = t - t_g. Equals kernel.local_sigma_y2_grid column by column.
    """
    Y = np.asarray(Y, float)
    t = np.asarray(t, float)
    grid = np.asarray(grid, float)
    Wt = np.asarray(W, float).T
    S0 = Wt.sum(axis=1)
    St = Wt @ t
    Stt = Wt @ (t * t)
    S1 = St - grid * S0
    S2 = Stt - 2.0 * grid * St + grid * grid * S0
    Sy = Wt @ Y
    Sdy = Wt @ (t[:, None] * Y) - grid[:, None] * Sy
    Syy = Wt @ (Y * Y)
    det = S0 * S2 - S1 * S1
    quad = (S2[:, None] * Sy * Sy - 2.0 * S1[:, None] * Sy * Sdy + S0[:, None] * Sdy * Sdy) / det[:, None]
    return np.maximum((Syy - quad) / np.maximum(S0, EPS)[:, None], EPS)


def raw_curves_many(Y, t, design: dict, smoother: np.ndarray) -> dict:
    """Raw smoothed h2(t) on the design grid for Q phenotypes (columns of Y), one edge pass.

    num_diag[b, q] = sum_E K_ij w_ib w_jb r_iq r_jq  (the diagonal of estimate.numerator_surface).
    """
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    R = Y - Y.mean(axis=0, keepdims=True)
    row, col, kij = design["row"], design["col"], design["kij"]
    W = design["subject_weights"]
    num_diag = np.zeros((W.shape[1], Y.shape[1]), dtype=float)
    chunk = int(design["chunk_size"])
    for start in range(0, len(kij), chunk):
        stop = min(start + chunk, len(kij))
        rr, cc = row[start:stop], col[start:stop]
        num_diag += (kij[start:stop, None] * W[rr] * W[cc]).T @ (R[rr] * R[cc])
    den_diag = np.diag(design["den_surface"])
    raw_diag = np.where(den_diag[:, None] > EPS, num_diag / np.maximum(den_diag, EPS)[:, None], 0.0)
    sigma_g2 = smoother @ raw_diag
    sigma_y2 = local_linear_variance_many(Y, t, design["grid_t"], W)
    return {"h2": sigma_g2 / sigma_y2, "sigma_g2": sigma_g2, "sigma_y2": sigma_y2, "raw_diag": raw_diag}


# ----------------------------------------------------------------------------- the null model
def flat_null_model(raw_surface, sigma_y2_grid, grid, t_subject, c_star, max_rank: int = 4, tol: float = 1e-8):
    """Basis expansion of Cg*: the fitted correlation structure with a FLAT h2 diagonal.

    Returns (basis N x r, lambdas r, sigma_e2 N) with
        Cov(g*_i, g*_j) = K_ij sum_m lambda_m basis[i, m] basis[j, m],   Cov(g*_i, g*_i) = K_ii c* sigma_y2(t_i).
    The per-subject rescaling is a diagonal congruence A -> D A D, so correlations are unchanged.
    """
    sym = 0.5 * (np.asarray(raw_surface, float) + np.asarray(raw_surface, float).T)
    values, vectors = np.linalg.eigh(sym)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    keep = np.flatnonzero(values > tol * max(float(values[0]), np.finfo(float).tiny))[:max_rank]
    if len(keep) == 0:                                              # degenerate fit: rank-1 fallback
        keep = np.array([0])
        values = np.maximum(values, np.finfo(float).tiny)
    lambdas = values[keep]
    basis = np.column_stack([np.interp(t_subject, grid, vectors[:, j]) for j in keep])
    target = np.maximum(np.interp(t_subject, grid, float(c_star) * np.asarray(sigma_y2_grid, float)), 0.0)
    implied = np.maximum((lambdas[None, :] * np.square(basis)).sum(axis=1), np.finfo(float).tiny)
    basis = basis * np.sqrt(target / implied)[:, None]
    sigma_e2 = np.maximum(np.interp(t_subject, grid, (1.0 - float(c_star)) * np.asarray(sigma_y2_grid, float)), EPS)
    return basis, lambdas, sigma_e2


def draw_null(bank, basis, lambdas, sigma_e2, n_boot: int, rng) -> np.ndarray:
    """N x B null phenotypes. Genetic part from bank columns [0, B*r): column b*r + m feeds replicate b, factor m."""
    n, rank = basis.shape
    if bank.shape[0] != n or bank.shape[1] < n_boot * rank:
        raise ValueError(f"score bank {bank.shape} cannot supply {n_boot} replicates x rank {rank} for {n} subjects")
    scores = np.asarray(bank[:, : n_boot * rank], dtype=float).reshape(n, n_boot, rank)
    g = np.einsum("nbm,m,nm->nb", scores, np.sqrt(np.maximum(lambdas, 0.0)), basis)
    return g + rng.standard_normal((n, n_boot)) * np.sqrt(sigma_e2)[:, None]


# ----------------------------------------------------------------------------- tail of the null law
def _cgf(s, lam, d2):
    a = 1.0 - 2.0 * lam * s
    k0 = float(np.sum(-0.5 * np.log(a) + d2 * lam * s / a))
    k1 = float(np.sum(lam / a + d2 * lam / a ** 2))
    k2 = float(np.sum(2.0 * lam ** 2 / a ** 2 + 4.0 * d2 * lam ** 2 / a ** 3))
    return k0, k1, k2


def _imhof_sf(x, lam, d2):
    """Imhof (1961) on the imaginary axis; used only in the body of the distribution."""
    def integrand(u):
        if u == 0.0:
            return 0.5 * float(np.sum(lam * (1.0 + d2))) - 0.5 * x
        lu = lam * u
        q = 1.0 + lu * lu
        theta = 0.5 * float(np.sum(np.arctan(lu) + d2 * lu / q)) - 0.5 * x * u
        log_rho = 0.25 * float(np.sum(np.log(q))) + 0.5 * float(np.sum(d2 * lu * lu / q))
        return math.sin(theta) / u * math.exp(-log_rho)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        total, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-15, epsrel=1e-11, limit=2000)
    return float(0.5 + total / math.pi)


def gchisq_logsf(x, lam, mu=None) -> float:
    """log P(Q > x), Q = sum_j lam_j (Z_j + mu_j)^2, by inversion on the saddle-point contour.

    Same integral as Imhof's with the Bromwich line moved through the saddle point s^, so the
    phase is stationary and the factor exp(K(s^) - s^ x) is taken out analytically: the error is
    relative, <= 1e-7 for three or more weights from p = 0.5 down to p = 1e-100 (1e-3 for a single weight).
    """
    lam = np.asarray(lam, float).ravel()
    mu = np.zeros_like(lam) if mu is None else np.asarray(mu, float).ravel()
    keep = lam > 1e-10 * max(float(lam.max()), np.finfo(float).tiny)
    lam, mu = lam[keep], mu[keep]
    scale = float(lam.max())
    lam, x, d2 = lam / scale, float(x) / scale, np.square(mu)
    mean = float(np.sum(lam * (1.0 + d2)))
    f = lambda s: _cgf(s, lam, d2)[1] - x
    if x > mean:
        s0 = optimize.brentq(f, 0.0, 0.5 / float(lam.max()) * (1.0 - 1e-15), xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
    else:
        lo = -1.0
        while f(lo) > 0:
            lo *= 2.0
        s0 = optimize.brentq(f, lo, 0.0, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
    k0, _, k2 = _cgf(s0, lam, d2)
    if abs(s0) * math.sqrt(k2) < 0.05:                                  # body of the distribution
        return math.log(max(_imhof_sf(x, lam, d2), 1e-300))
    pairs = list(zip(lam.tolist(), d2.tolist()))

    def log_mgf(s):
        acc = 0j
        for l_, e_ in pairs:
            a = 1.0 - 2.0 * l_ * s
            acc += -0.5 * cmath.log(a) + e_ * l_ * s / a
        return acc

    def integrand(y):
        s = complex(s0, y)
        return (cmath.exp(log_mgf(s) - k0 - 1j * y * x) / s).real

    def envelope(y):
        s = complex(s0, y)
        return math.exp((log_mgf(s) - k0).real) / abs(s)

    width = 1.0 / math.sqrt(k2)
    total, a_ = 0.0, 0.0
    # far segments contribute nothing but raise IntegrationWarnings; the total is checked in tests/test_h2_test.py
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        for jseg in range(40):                                          # geometric segments
            b_ = width * 4.0 ** jseg
            total += integrate.quad(integrand, a_, b_, epsabs=1e-13, epsrel=1e-9, limit=400)[0]
            a_ = b_
            if jseg >= 2 and envelope(b_) * b_ * 4.0 < 1e-10 * abs(total):
                break
        else:
            total += integrate.quad(integrand, a_, np.inf, epsabs=1e-13, epsrel=1e-9, limit=400)[0]
    val = total / math.pi
    if s0 > 0:                                                          # right of the pole: P(Q > x)
        return float(k0 - s0 * x + math.log(max(val, 1e-300)))
    return float(math.log(max(1.0 - math.exp(k0 - s0 * x) * (-val), 1e-300)))


def bootstrap_law(null_curves, scale):
    """(lam, mu) with T* = sum_j lam_j (Z_j + mu_j)^2 for centred curves ~ N(m, S) (mean included)."""
    curves = np.asarray(null_curves, float)
    k = curves.shape[1]
    delta = curves - curves.mean(axis=1, keepdims=True)
    m = delta.mean(axis=0)
    s = np.cov(delta, rowvar=False, ddof=1)
    dh = 1.0 / (np.maximum(np.asarray(scale, float), EPS) * math.sqrt(k))   # D^(1/2), T = delta' D delta
    omega = dh[:, None] * s * dh[None, :]
    val, vec = np.linalg.eigh(0.5 * (omega + omega.T))
    keep = val > 1e-10 * val.max()
    lam = val[keep]
    mu = (vec[:, keep].T @ (dh * m)) / np.sqrt(lam)
    return lam, mu


# ----------------------------------------------------------------------------- the test
def flat_test(y, t, design: dict, raw_surface, sigma_y2_grid, test_idx, bank, *, n_boot: int = 200,
              max_rank: int = 4, seed: int = 0, c_star_bounds=(0.01, 0.99)) -> dict:
    """Run the h2 test on one trait. `raw_surface` and `sigma_y2_grid` are the step-3 fit on design['grid_t']."""
    grid = np.asarray(design["grid_t"], float)
    smoother = smoothing_matrix(grid, float(design["config"]["cg_surface_post_smooth_bandwidth"]))
    obs = raw_curves_many(y, t, design, smoother)
    h2_obs = obs["h2"][test_idx, 0]
    c_star = float(np.mean(h2_obs))
    c_star_clipped = bool(not (c_star_bounds[0] <= c_star <= c_star_bounds[1]))
    c_star = float(np.clip(c_star, *c_star_bounds))

    basis, lambdas, sigma_e2 = flat_null_model(raw_surface, sigma_y2_grid, grid, t, c_star, max_rank=max_rank)
    rng = np.random.default_rng(int(seed))
    y_null = draw_null(bank, basis, lambdas, sigma_e2, int(n_boot), rng)
    null_curves = raw_curves_many(y_null, t, design, smoother)["h2"][test_idx].T          # B x K

    scale = null_curves.std(axis=0, ddof=1)
    t_obs = flatness_statistic(h2_obs, scale)
    t_null = np.array([flatness_statistic(row, scale) for row in null_curves])
    p_mc = float((1.0 + np.sum(t_null >= t_obs)) / (len(t_null) + 1.0))
    lam, mu = bootstrap_law(null_curves, scale)
    log_p = gchisq_logsf(t_obs, lam, mu)
    law_mean = float(np.sum(lam * (1.0 + mu ** 2)))
    return {
        "T": t_obs,
        "p_h2": math.exp(log_p),
        "log10_p_h2": log_p / math.log(10.0),
        "p_h2_mc": p_mc,
        "c_star": c_star,
        "c_star_clipped": c_star_clipped,
        "null_rank": int(len(lambdas)),
        "n_boot": int(n_boot),
        "h2_raw_obs": h2_obs,
        "sd_null": scale,
        "null_curves": null_curves,
        "t_null": t_null,
        "law_mean": law_mean,
        "law_mean_share": float(np.sum(lam * mu ** 2) / law_mean),
        "bootstrap_mean_T": float(t_null.mean()),
    }
