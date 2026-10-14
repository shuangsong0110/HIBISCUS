"""The correlation test: a directional test of unit genetic correlation across age.

H0: R_G(t_a, t_b) = 1 for all a < b  <=>  C_g(s, t) = a(s) a(t) with a nuisance amplitude a(t).
Pipeline:
  1. source fit on the test grid: raw pre-PSD surface C_raw, denominator D, total variance Y (ESS 0.2N);
  2. local-quadratic HE diagonal G_a and local-linear Y_loc with y-gradients, on the same kernel weights;
  3. smooth basis from the normalised kernel-weight Gram matrix (capture 0.999, rank <= 12) and the
     logistic genetic-fraction fit  G_a = Y_loc,a expit(b_a' eta)  with its implicit influence;
  4. subject-level rank-one null  a_i = sqrt(Y_i q_i), e_i = Y_i (1 - q_i);  K_hat = S S'/q random scores;
     forward the null through the same sparse estimator -> C^F and its eta / Y tangents;
  5. d = vech_<{R(C_raw) - R(C^F)};  Psi = G_rho,raw - Phi_eta J_eta' - G_Y J_Y';
     Omega = 1/2 Psi' V_0 Psi;  Z = mean(d) / sqrt(w' Omega w);  p_corr = Phi(Z).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.stats import norm

from . import EPS
from .kernel import bandwidths_and_weights, grid_local_variance_and_y_derivative

TINY = np.finfo(float).tiny


# ============================================================================= sparse operators
def _as_matrix(values: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, float)
    vector = values.ndim == 1
    if vector:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("sparse operator input must be a vector or matrix")
    return values, vector


def sparse_symmetric_product(row, col, value, matrix, *, chunk_size: int = 1_000_000) -> np.ndarray:
    """Apply I + the selected symmetric off-diagonal edges."""
    matrix, vector = _as_matrix(matrix)
    row = np.asarray(row, np.int32)
    col = np.asarray(col, np.int32)
    value = np.asarray(value, float)
    if not (len(row) == len(col) == len(value)):
        raise ValueError("edge arrays have different lengths")
    n = matrix.shape[0]
    out = matrix.copy()
    step = max(int(chunk_size), 1)
    for start in range(0, len(value), step):
        stop = min(start + step, len(value))
        rr, cc, vv = row[start:stop], col[start:stop], value[start:stop]
        for column_index in range(matrix.shape[1]):
            source = matrix[:, column_index]
            out[:, column_index] += np.bincount(rr, weights=vv * source[cc], minlength=n)
            out[:, column_index] += np.bincount(cc, weights=vv * source[rr], minlength=n)
    return out[:, 0] if vector else out


def sparse_offdiag_product(row, col, value, matrix, *, chunk_size: int = 1_000_000) -> np.ndarray:
    matrix, vector = _as_matrix(matrix)
    row = np.asarray(row, np.int32)
    col = np.asarray(col, np.int32)
    value = np.asarray(value, float)
    n = matrix.shape[0]
    out = np.zeros_like(matrix, float)
    step = max(int(chunk_size), 1)
    for start in range(0, len(value), step):
        stop = min(start + step, len(value))
        rr, cc, vv = row[start:stop], col[start:stop], value[start:stop]
        for column_index in range(matrix.shape[1]):
            source = matrix[:, column_index]
            out[:, column_index] += np.bincount(rr, weights=vv * source[cc], minlength=n)
            out[:, column_index] += np.bincount(cc, weights=vv * source[rr], minlength=n)
    return out[:, 0] if vector else out


def sparse_k2_and_k_offdiag_products(row, col, value, k2_matrix, k_matrix, *, chunk_size: int = 1_000_000):
    """Apply K_off^2 and K_off in one edge scan."""
    k2_matrix, _ = _as_matrix(k2_matrix)
    k_matrix, _ = _as_matrix(k_matrix)
    if k2_matrix.shape[0] != k_matrix.shape[0]:
        raise ValueError("K and K-squared inputs have different row counts")
    row = np.asarray(row, np.int32)
    col = np.asarray(col, np.int32)
    value = np.asarray(value, float)
    n = k_matrix.shape[0]
    k2_out = np.zeros_like(k2_matrix, float)
    k_out = np.zeros_like(k_matrix, float)
    step = max(int(chunk_size), 1)
    for start in range(0, len(value), step):
        stop = min(start + step, len(value))
        rr, cc, vv = row[start:stop], col[start:stop], value[start:stop]
        vv2 = np.square(vv)
        for column_index in range(k2_matrix.shape[1]):
            source = k2_matrix[:, column_index]
            k2_out[:, column_index] += np.bincount(rr, weights=vv2 * source[cc], minlength=n)
            k2_out[:, column_index] += np.bincount(cc, weights=vv2 * source[rr], minlength=n)
        for column_index in range(k_matrix.shape[1]):
            source = k_matrix[:, column_index]
            k_out[:, column_index] += np.bincount(rr, weights=vv * source[cc], minlength=n)
            k_out[:, column_index] += np.bincount(cc, weights=vv * source[rr], minlength=n)
    return k2_out, k_out


def diagonal_local_quadratic_sparse(y, ages, grid, weights, bandwidths, row, col, kij, *, chunk_size: int = 1_000_000) -> dict:
    """Sparse-K local-quadratic HE diagonal G(t_a) = beta_0a with its subject-level gradient."""
    y = np.asarray(y, float)
    ages = np.asarray(ages, float)
    grid = np.asarray(grid, float)
    weights = np.asarray(weights, float)
    bandwidths = np.asarray(bandwidths, float)
    residual = y - float(np.mean(y))
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1))
    terms = [((a, b),) if a == b else ((a, b), (b, a)) for a, b in pairs]
    product_pairs = tuple((a, b) for a in range(3) for b in range(a, 3))
    feature_index = {pair: index for index, pair in enumerate(product_pairs)}

    curve = np.full(len(grid), np.nan, float)
    gradient = np.full((len(y), len(grid)), np.nan, float)
    k2_weights = np.full((len(y), len(grid)), np.nan, float)
    condition = np.full(len(grid), np.nan, float)

    for grid_index, point in enumerate(grid):
        bandwidth = max(float(bandwidths[grid_index]), TINY)
        x = (ages - float(point)) / bandwidth
        z = np.column_stack((np.ones(len(ages)), x, np.square(x)))
        w = weights[:, grid_index]
        feature = np.column_stack([w * z[:, left] * z[:, right] for left, right in product_pairs])
        residual_feature = residual[:, None] * w[:, None] * z
        k2_feature, k_residual_feature = sparse_k2_and_k_offdiag_products(
            row, col, kij, feature, residual_feature, chunk_size=chunk_size
        )
        k2_weights[:, grid_index] = k2_feature[:, feature_index[(0, 0)]]

        normal = np.empty((len(pairs), len(pairs)), float)
        for p, pterms in enumerate(terms):
            for q, qterms in enumerate(terms):
                value_sum = 0.0
                for u, v in pterms:
                    for a, b in qterms:
                        ua = (min(u, a), max(u, a))
                        vb = (min(v, b), max(v, b))
                        value_sum += float(feature[:, feature_index[ua]] @ k2_feature[:, feature_index[vb]])
                normal[p, q] = 0.5 * value_sum
        normal = 0.5 * (normal + normal.T)
        inverse = np.linalg.pinv(normal, rcond=1e-10, hermitian=True)
        singular = np.linalg.eigvalsh(normal)
        maximum = max(float(np.max(np.abs(singular))), TINY)
        retained = np.abs(singular) > 1e-10 * maximum
        condition[grid_index] = (
            float(np.max(np.abs(singular[retained])) / np.min(np.abs(singular[retained])))
            if np.any(retained)
            else np.inf
        )

        score = np.empty(len(pairs), float)
        score_gradient = np.empty((len(y), len(pairs)), float)
        for p, ((left, right), *rest) in enumerate(terms):
            if not rest:
                score[p] = 0.5 * float(residual_feature[:, left] @ k_residual_feature[:, right])
                score_gradient[:, p] = w * z[:, left] * k_residual_feature[:, right]
            else:
                score[p] = float(residual_feature[:, left] @ k_residual_feature[:, right])
                score_gradient[:, p] = (
                    w * z[:, left] * k_residual_feature[:, right]
                    + w * z[:, right] * k_residual_feature[:, left]
                )
        curve[grid_index] = float(inverse[0] @ score)
        gradient[:, grid_index] = score_gradient @ inverse[0]
        gradient[:, grid_index] -= float(np.mean(gradient[:, grid_index]))

    surface_denominator = 0.5 * (weights.T @ k2_weights)
    surface_denominator = 0.5 * (surface_denominator + surface_denominator.T)
    denominator_scale = max(float(np.median(np.abs(np.diag(surface_denominator)))), TINY)
    return {
        "genetic_curve": curve,
        "genetic_gradient": gradient,
        "normal_condition": condition,
        "surface_valid": surface_denominator > 1e-12 * denominator_scale,
    }


def sparse_forward_null(
    row, col, kij, weights, den_surface, amplitude, amplitude_basis, residual_variance,
    total_variance_basis=None, total_amplitude_basis=None, ka=None, kb=None, k_total_amplitude_basis=None,
    *, chunk_size: int = 1_000_000,
) -> dict:
    """Forward the rank-one null through selected sparse edges (with eta and Y tangents)."""
    weights = np.asarray(weights, float)
    den = np.asarray(den_surface, float)
    amplitude = np.asarray(amplitude, float)
    amplitude_basis = np.asarray(amplitude_basis, float)
    residual_variance = np.asarray(residual_variance, float)
    n, grid_size = weights.shape
    total_basis = np.zeros((n, 0), float) if total_variance_basis is None else np.asarray(total_variance_basis, float)
    total_amp_basis = np.zeros_like(total_basis) if total_amplitude_basis is None else np.asarray(total_amplitude_basis, float)
    if total_basis.shape != total_amp_basis.shape:
        raise ValueError("total variance and amplitude bases differ")

    ka = sparse_symmetric_product(row, col, kij, amplitude, chunk_size=chunk_size) if ka is None else np.asarray(ka, float)
    kb = sparse_symmetric_product(row, col, kij, amplitude_basis, chunk_size=chunk_size) if kb is None else np.asarray(kb, float)
    k_total_amp = (
        sparse_symmetric_product(row, col, kij, total_amp_basis, chunk_size=chunk_size)
        if k_total_amplitude_basis is None and total_basis.shape[1]
        else total_basis.copy() if not total_basis.shape[1]
        else np.asarray(k_total_amplitude_basis, float)
    )
    row_sum = amplitude * ka + residual_variance
    total_sum = float(np.sum(row_sum))
    d_residual = -2.0 * amplitude[:, None] * amplitude_basis
    d_row_sum = amplitude_basis * ka[:, None] + amplitude[:, None] * kb + d_residual
    d_total_sum = np.sum(d_row_sum, axis=0)
    d_total_residual = total_basis - 2.0 * amplitude[:, None] * total_amp_basis
    d_total_row_sum = total_amp_basis * ka[:, None] + amplitude[:, None] * k_total_amp + d_total_residual
    d_total_total_sum = np.sum(d_total_row_sum, axis=0)

    null_num = np.zeros((grid_size, grid_size), float)
    tangent_num = np.zeros((amplitude_basis.shape[1], grid_size, grid_size), float)
    total_tangent_num = np.zeros((total_basis.shape[1], grid_size, grid_size), float)
    step = max(int(chunk_size), 1)
    for start in range(0, len(kij), step):
        stop = min(start + step, len(kij))
        rr, cc, kk = row[start:stop], col[start:stop], kij[start:stop]
        wi, wj = weights[rr], weights[cc]
        covariance_edge = kk * amplitude[rr] * amplitude[cc]
        centered = covariance_edge - row_sum[rr] / n - row_sum[cc] / n + total_sum / (n * n)
        cross = ((kk * centered)[:, None] * wi).T @ wj
        null_num += 0.5 * (cross + cross.T)

        d_covariance = kk[:, None] * (amplitude_basis[rr] * amplitude[cc, None] + amplitude[rr, None] * amplitude_basis[cc])
        d_centered = d_covariance - d_row_sum[rr] / n - d_row_sum[cc] / n + d_total_sum / (n * n)
        for basis_index in range(amplitude_basis.shape[1]):
            cross = ((kk * d_centered[:, basis_index])[:, None] * wi).T @ wj
            tangent_num[basis_index] += 0.5 * (cross + cross.T)

        if total_basis.shape[1]:
            d_total_covariance = kk[:, None] * (total_amp_basis[rr] * amplitude[cc, None] + amplitude[rr, None] * total_amp_basis[cc])
            d_total_centered = d_total_covariance - d_total_row_sum[rr] / n - d_total_row_sum[cc] / n + d_total_total_sum / (n * n)
            for basis_index in range(total_basis.shape[1]):
                cross = ((kk * d_total_centered[:, basis_index])[:, None] * wi).T @ wj
                total_tangent_num[basis_index] += 0.5 * (cross + cross.T)

    den_scale = max(float(np.median(np.abs(den))), TINY)
    if np.any(den <= 1e-12 * den_scale):
        raise ValueError("nonpositive sparse HE denominator")
    return {
        "null_surface": null_num / den,
        "tangent_surfaces": tangent_num / den[None, :, :],
        "total_variance_tangent_surfaces": total_tangent_num / den[None, :, :],
    }


def random_score_kinship_product(matrix, scores) -> np.ndarray:
    """Apply the rank-q approximation K_hat = S S' / q."""
    matrix, vector = _as_matrix(matrix)
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or scores.shape[0] != matrix.shape[0]:
        raise ValueError("random GRM scores and matrix have incompatible dimensions")
    out = scores @ (scores.T @ matrix) / max(float(scores.shape[1]), 1.0)
    return out[:, 0] if vector else out


def random_score_factor_covariance(matrix, genetic_factors, residual_variance, scores) -> np.ndarray:
    """0.5 G' V0 G with V0 = diag(a) K_hat diag(a) + diag(e), K_hat = S S'/q at the full score rank."""
    matrix = np.asarray(matrix, float)
    factors = np.asarray(genetic_factors, float)
    residual = np.asarray(residual_variance, float)
    scores = np.asarray(scores, float)
    if factors.ndim == 1:
        factors = factors[:, None]
    if matrix.ndim != 2 or factors.shape[0] != matrix.shape[0] or residual.shape != (matrix.shape[0],):
        raise ValueError("random-score covariance dimensions do not agree")
    if scores.ndim != 2 or scores.shape[0] != matrix.shape[0]:
        raise ValueError("random GRM scores have the wrong subject dimension")
    if np.any(residual <= 0.0):
        raise ValueError("null residual variance is not positive")
    rank = int(scores.shape[1])
    covariance = matrix.T @ (residual[:, None] * matrix)
    for factor_index in range(factors.shape[1]):
        projected = matrix.T @ (factors[:, factor_index, None] * scores)
        covariance += projected[:, :rank] @ projected[:, :rank].T / float(rank)
    covariance *= 0.5
    return 0.5 * (covariance + covariance.T)


# ============================================================================= smooth basis
@dataclass
class SmoothBasis:
    vectors: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    capture_1d: float
    capture_2d: float


def build_smooth_basis(weights, variance_capture: float = 0.999, max_rank: int = 0) -> SmoothBasis:
    weights = np.asarray(weights, dtype=float)
    norms = np.sqrt(np.maximum(np.sum(np.square(weights), axis=0), EPS))
    normalized = weights / norms[None, :]
    gram = normalized.T @ normalized
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    total = max(float(np.sum(values)), EPS)
    target_1d = math.sqrt(float(variance_capture))
    rank = int(np.searchsorted(np.cumsum(values) / total, target_1d) + 1)
    if int(max_rank) > 0:
        rank = min(rank, int(max_rank))
    rank = max(1, min(rank, vectors.shape[1]))
    vectors = vectors[:, :rank]
    for column in range(rank):
        if float(np.sum(vectors[:, column])) < 0.0:
            vectors[:, column] *= -1.0
    capture_1d = float(np.sum(values[:rank]) / total)
    return SmoothBasis(vectors=vectors, eigenvalues=values[:rank], rank=rank, capture_1d=capture_1d, capture_2d=float(capture_1d * capture_1d))


# ============================================================================= nuisance genetic fraction
def fit_local_diagonal_genetic_fraction(genetic_curve, total_curve, basis_grid, rank_tolerance: float = 1e-10) -> dict:
    """Fit G(t)=Y(t) expit{b(t)'eta} using every raw local diagonal (least squares + exact Newton)."""
    genetic = np.asarray(genetic_curve, float)
    total = np.asarray(total_curve, float)
    basis = np.asarray(basis_grid, float)
    if genetic.ndim != 1 or total.shape != genetic.shape:
        raise ValueError("local genetic-fraction curves do not agree")
    if basis.ndim != 2 or basis.shape[0] != len(genetic):
        raise ValueError("local genetic-fraction basis has incompatible dimensions")
    total_scale = max(float(np.median(np.abs(total))), TINY)
    if np.any(total <= 1e-10 * total_scale):
        raise ValueError("local genetic-fraction fit requires positive total variance")
    scale = max(float(np.median(np.abs(genetic))), float(np.std(genetic)), TINY)
    initial_fraction = np.clip(genetic / total, 1e-4, 1.0 - 1e-4)
    initial_logit = np.log(initial_fraction / (1.0 - initial_fraction))
    eta0 = np.linalg.lstsq(basis, initial_logit, rcond=float(rank_tolerance))[0]

    def components(eta):
        fraction = expit(basis @ eta)
        return fraction, total * fraction

    def residual(eta):
        _, fitted = components(eta)
        return (fitted - genetic) / scale

    def jacobian(eta):
        fraction, _ = components(eta)
        return total[:, None] * fraction[:, None] * (1.0 - fraction[:, None]) * basis / scale

    fit = least_squares(residual, eta0, jac=jacobian, method="trf", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=500)
    if not fit.success:
        raise RuntimeError(f"local genetic-fraction fit failed: {fit.message}")
    eta = np.asarray(fit.x, float)

    def objective_score_hessian(coefficient):
        fraction_value, fitted_value = components(coefficient)
        residual_value = fitted_value - genetic
        jacobian_value = total[:, None] * fraction_value[:, None] * (1.0 - fraction_value[:, None]) * basis
        second_scale = total * fraction_value * (1.0 - fraction_value) * (1.0 - 2.0 * fraction_value)
        hessian_value = jacobian_value.T @ jacobian_value
        for grid_index in range(len(genetic)):
            hessian_value += residual_value[grid_index] * second_scale[grid_index] * np.outer(basis[grid_index], basis[grid_index])
        hessian_value = 0.5 * (hessian_value + hessian_value.T)
        return (
            0.5 * float(np.dot(residual_value, residual_value)),
            jacobian_value.T @ residual_value,
            hessian_value,
            fraction_value,
            fitted_value,
            jacobian_value,
        )

    for _ in range(21):
        objective_value, score, hessian, fraction, fitted, jacobian_unscaled = objective_score_hessian(eta)
        score_scale = max(float(np.linalg.norm(jacobian_unscaled)) * float(np.linalg.norm(genetic)), TINY)
        if float(np.linalg.norm(score) / score_scale) <= 1e-10:
            break
        eigenvalues = np.linalg.eigvalsh(hessian)
        eigen_scale = max(float(np.max(np.abs(eigenvalues))), TINY)
        if np.min(eigenvalues) <= 1e-12 * eigen_scale:
            raise ValueError("local genetic-fraction Newton Hessian is not positive definite")
        direction = np.linalg.solve(hessian, score)
        accepted = False
        step = 1.0
        for _ in range(30):
            candidate = eta - step * direction
            candidate_objective = objective_score_hessian(candidate)[0]
            if candidate_objective <= objective_value - 1e-4 * step * float(score @ direction):
                eta = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            relative_score = float(np.linalg.norm(score) / score_scale)
            if relative_score <= 1e-8:
                break
            raise RuntimeError(f"local genetic-fraction Newton refinement line search failed: relative_score={relative_score:g}")
    objective_value, score, hessian, fraction, fitted, jacobian_unscaled = objective_score_hessian(eta)
    singular = np.linalg.svd(jacobian_unscaled, compute_uv=False)
    maximum = max(float(np.max(singular)), TINY)
    rank = int(np.sum(singular > float(rank_tolerance) * maximum))
    if rank < basis.shape[1]:
        raise ValueError(f"local genetic-fraction Jacobian is rank deficient: {rank} < {basis.shape[1]}")
    score_scale = max(float(np.linalg.norm(jacobian_unscaled)) * float(np.linalg.norm(genetic)), TINY)
    relative_score = float(np.linalg.norm(score) / score_scale)
    if relative_score > 1e-8:
        raise RuntimeError(f"local genetic-fraction score did not converge: relative_score={relative_score:g}")
    return {"eta": eta, "genetic_fraction_grid": fraction}


def local_diagonal_genetic_fraction_influence(genetic_gradient, total_gradient, genetic_curve, total_curve, fitted_fraction, basis_grid, rank_tolerance: float = 1e-10) -> np.ndarray:
    """Implicit influence (subject x coefficient) of the logistic genetic-fraction fit."""
    genetic_derivative = np.asarray(genetic_gradient, float)
    total_derivative = np.asarray(total_gradient, float)
    genetic = np.asarray(genetic_curve, float)
    total = np.asarray(total_curve, float)
    fraction = np.asarray(fitted_fraction, float)
    basis = np.asarray(basis_grid, float)
    if genetic_derivative.shape != total_derivative.shape:
        raise ValueError("genetic and total gradients have different dimensions")
    if genetic_derivative.shape[1] != len(genetic) or total.shape != genetic.shape:
        raise ValueError("local fraction influence curve dimensions do not agree")
    fitted = total * fraction
    jacobian = total[:, None] * fraction[:, None] * (1.0 - fraction[:, None]) * basis
    fit_residual = fitted - genetic
    second_scale = total * fraction * (1.0 - fraction) * (1.0 - 2.0 * fraction)
    hessian = jacobian.T @ jacobian
    for grid_index in range(len(genetic)):
        hessian += fit_residual[grid_index] * second_scale[grid_index] * np.outer(basis[grid_index], basis[grid_index])
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    eigen_scale = max(float(np.max(np.abs(eigenvalues))), TINY)
    material_negative = int(np.sum(eigenvalues < -float(rank_tolerance) * eigen_scale))
    retained_rank = int(np.sum(eigenvalues > float(rank_tolerance) * eigen_scale))
    if material_negative or retained_rank < basis.shape[1]:
        raise ValueError(
            "local genetic-fraction objective Hessian is not positive definite:"
            f"negative={material_negative}, rank={retained_rank}/{basis.shape[1]}"
        )
    inverse = np.linalg.pinv(hessian, rcond=float(rank_tolerance), hermitian=True)
    total_score_jacobian = fraction[:, None] * (1.0 - fraction[:, None]) * (2.0 * fitted - genetic)[:, None] * basis
    influence = (genetic_derivative @ jacobian - total_derivative @ total_score_jacobian) @ inverse
    return influence - np.mean(influence, axis=0, keepdims=True)


# ============================================================================= correlation residual and gradients
def raw_correlation(surface) -> np.ndarray:
    surface = 0.5 * (np.asarray(surface, float) + np.asarray(surface, float).T)
    diagonal = np.diag(surface)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError("raw correlation requires a finite, strictly positive diagonal")
    return surface / np.sqrt(np.outer(diagonal, diagonal))


def upper_pairs(size: int):
    return np.triu_indices(int(size), k=1)


def correlation_residual(observed, forwarded_null):
    obs = raw_correlation(observed)
    null = raw_correlation(forwarded_null)
    left, right = upper_pairs(obs.shape[0])
    return obs[left, right] - null[left, right], obs, null


def correlation_tangent(surface, tangent_surfaces) -> np.ndarray:
    """Derivative of upper-triangle correlations along surface tangents."""
    surface = 0.5 * (np.asarray(surface, float) + np.asarray(surface, float).T)
    tangents = np.asarray(tangent_surfaces, float)
    diagonal = np.diag(surface)
    if np.any(diagonal <= 0.0):
        raise ValueError("correlation tangent requires a positive diagonal")
    left, right = upper_pairs(surface.shape[0])
    scale = np.sqrt(diagonal[left] * diagonal[right])
    rho = surface[left, right] / scale
    result = tangents[:, left, right].T / scale[:, None]
    result -= 0.5 * rho[:, None] * (tangents[:, left, left].T / diagonal[left, None] + tangents[:, right, right].T / diagonal[right, None])
    return result


def raw_pair_gradient_block(weights, k_residual_weights, den_surface, reference_surface) -> np.ndarray:
    """Delta-method gradients of upper correlations for one subject block (N x K)."""
    weights = np.asarray(weights, float)
    krw = np.asarray(k_residual_weights, float)
    den = np.asarray(den_surface, float)
    ref = np.asarray(reference_surface, float)
    diagonal = np.diag(ref)
    if np.any(diagonal <= 0.0):
        raise ValueError("reference surface has a nonpositive diagonal")
    left, right = upper_pairs(ref.shape[0])
    q_diag = weights * krw / np.maximum(np.diag(den), EPS)[None, :]
    q_cross = 0.5 * (weights[:, left] * krw[:, right] + weights[:, right] * krw[:, left]) / np.maximum(den[left, right], EPS)[None, :]
    scale = np.sqrt(diagonal[left] * diagonal[right])
    rho = ref[left, right] / scale
    gradients = q_cross / scale[None, :]
    gradients -= 0.5 * rho[None, :] * (q_diag[:, left] / diagonal[left][None, :] + q_diag[:, right] / diagonal[right][None, :])
    return gradients


# ============================================================================= tests
def directional_correlation_test(residual, covariance, weights=None) -> dict:
    """One-sided test for the correlation-cone direction R_G < 1 (equal weights by default)."""
    residual = np.asarray(residual, float)
    covariance = 0.5 * (np.asarray(covariance, float) + np.asarray(covariance, float).T)
    if residual.ndim != 1 or covariance.shape != (len(residual), len(residual)):
        raise ValueError("correlation test: residual and covariance dimensions do not agree")
    contrast = np.repeat(1.0 / len(residual), len(residual)) if weights is None else np.asarray(weights, float)
    if contrast.shape != residual.shape or np.any(contrast < 0.0):
        raise ValueError("correlation test: weights must be nonnegative and match the residual")
    weight_sum = float(np.sum(contrast))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("correlation test: weights must have a positive finite sum")
    contrast = contrast / weight_sum
    departure = float(contrast @ residual)
    variance = float(contrast @ covariance @ contrast)
    covariance_scale = max(float(np.max(np.abs(np.diag(covariance)))), TINY)
    if not np.isfinite(variance) or variance <= 1e-12 * covariance_scale:
        raise ValueError("correlation test: contrast variance is nonregular")
    z_score = departure / math.sqrt(variance)
    return {"departure": departure, "variance": variance, "z": z_score, "pvalue_lower": float(norm.cdf(z_score))}


# ============================================================================= orchestration
def interpolation_basis(source_grid, target) -> np.ndarray:
    source_grid = np.asarray(source_grid, float)
    basis = np.eye(len(source_grid))
    return np.column_stack([np.interp(target, source_grid, basis[:, j]) for j in range(len(source_grid))])


def run_directional_p2(y, ages, edges: dict, random_scores, source_cfg: dict, grid, source: dict, *,
                       basis_variance_capture: float = 0.999, max_basis_rank: int = 12, edge_chunk: int = 200000) -> dict:
    """The correlation test on one trait. `source` = {raw, den, sigma_y2} from the estimator on `grid`."""
    y = np.asarray(y, float)
    ages = np.asarray(ages, float)
    grid = np.asarray(grid, float)
    row = np.asarray(edges["row"], np.int32)
    col = np.asarray(edges["col"], np.int32)
    kij = np.asarray(edges["kij"], float)
    random_scores = np.asarray(random_scores, float)
    n = len(y)
    if random_scores.ndim != 2 or random_scores.shape[0] != n:
        raise ValueError("random GRM scores must be (n_subjects x rank) for the matched subjects")
    source_raw = 0.5 * (np.asarray(source["raw"], float) + np.asarray(source["raw"], float).T)
    source_den = np.asarray(source["den"], float)
    source_sigma_y2 = np.asarray(source["sigma_y2"], float)
    if source_raw.shape != (len(grid), len(grid)) or source_sigma_y2.shape != (len(grid),):
        raise ValueError("source fit and test grid dimensions differ")

    # --- kernel weights, total variance and its gradient ---
    source_bandwidths, source_weights = bandwidths_and_weights(ages, grid, source_cfg)
    source_total, source_total_gradient = grid_local_variance_and_y_derivative(y, ages, grid, source_weights)
    source_total_error = float(np.max(np.abs(source_total - source_sigma_y2)) / max(np.max(np.abs(source_sigma_y2)), TINY))
    if source_total_error > 1e-8:
        raise RuntimeError(f"source total variance does not match the step-3 fit: {source_total_error:g}")

    # --- local-quadratic genetic diagonal on the same kernel, and the smooth logistic genetic fraction ---
    local = diagonal_local_quadratic_sparse(y, ages, grid, source_weights, source_bandwidths, row, col, kij, chunk_size=edge_chunk)
    raw_genetic = np.asarray(local["genetic_curve"], float)
    raw_genetic_gradient = np.asarray(local["genetic_gradient"], float)
    smooth_basis = build_smooth_basis(source_weights, basis_variance_capture, max_basis_rank)
    basis_subject = np.column_stack([np.interp(ages, grid, smooth_basis.vectors[:, j]) for j in range(smooth_basis.rank)])
    amplitude_fit = fit_local_diagonal_genetic_fraction(raw_genetic, source_total, smooth_basis.vectors)
    source_total_subject = np.interp(ages, grid, source_sigma_y2)
    genetic_fraction_subject = expit(basis_subject @ amplitude_fit["eta"])
    amplitude_subject = np.sqrt(source_total_subject * genetic_fraction_subject)
    amplitude_basis_subject = 0.5 * amplitude_subject[:, None] * (1.0 - genetic_fraction_subject[:, None]) * basis_subject
    residual_variance_p2 = source_total_subject * (1.0 - genetic_fraction_subject)
    residual_floor = max(1e-10 * float(np.median(np.abs(source_total_subject))), TINY)
    if np.any(residual_variance_p2 <= residual_floor):
        raise ValueError("null_residual_variance_nonregular")
    total_basis_subject = interpolation_basis(grid, ages)
    total_amplitude_basis = 0.5 * amplitude_subject[:, None] / source_total_subject[:, None] * total_basis_subject

    # --- forward null through the sparse estimator (full-GRM row sums via K_hat) ---
    score_ka = random_score_kinship_product(amplitude_subject, random_scores)
    score_kb = random_score_kinship_product(amplitude_basis_subject, random_scores)
    score_k_total = random_score_kinship_product(total_amplitude_basis, random_scores)
    forwarded = sparse_forward_null(
        row, col, kij, source_weights, source_den,
        amplitude_subject, amplitude_basis_subject, residual_variance_p2,
        total_basis_subject, total_amplitude_basis,
        ka=score_ka, kb=score_kb, k_total_amplitude_basis=score_k_total,
        chunk_size=edge_chunk,
    )
    null_surface = 0.5 * (forwarded["null_surface"] + forwarded["null_surface"].T)
    raw_diagonal = np.diag(source_raw)
    null_diagonal = np.diag(null_surface)
    raw_tol = 1e-10 * max(float(np.median(np.abs(raw_diagonal))), TINY)
    null_tol = 1e-10 * max(float(np.median(np.abs(null_diagonal))), TINY)
    identifiable = (raw_diagonal > raw_tol) & (null_diagonal > null_tol)
    if int(np.sum(identifiable)) < 2:
        raise ValueError("correlation test has fewer than two identifiable grid points")
    active_raw = source_raw[np.ix_(identifiable, identifiable)]
    active_null = null_surface[np.ix_(identifiable, identifiable)]
    active_den = source_den[np.ix_(identifiable, identifiable)]
    active_weights = source_weights[:, identifiable]
    active_eta_tangents = forwarded["tangent_surfaces"][:, identifiable][:, :, identifiable]
    active_total_tangents = forwarded["total_variance_tangent_surfaces"][:, identifiable][:, :, identifiable]

    # --- residual, gradients, covariance, directional test ---
    krw = sparse_offdiag_product(row, col, kij, (y - np.mean(y))[:, None] * source_weights, chunk_size=edge_chunk)
    active_krw = np.asarray(krw[:, identifiable], float)
    p2_residual, observed_correlation, null_correlation = correlation_residual(active_raw, active_null)
    eta_correlation_tangent = correlation_tangent(active_null, active_eta_tangents)
    total_correlation_tangent = correlation_tangent(active_null, active_total_tangents)
    nuisance_fit_if = local_diagonal_genetic_fraction_influence(
        raw_genetic_gradient, source_total_gradient, raw_genetic, source_total,
        amplitude_fit["genetic_fraction_grid"], smooth_basis.vectors,
    )
    pair_gradient = raw_pair_gradient_block(active_weights, active_krw, active_den, active_null)
    pair_gradient -= nuisance_fit_if @ eta_correlation_tangent.T
    pair_gradient -= source_total_gradient @ total_correlation_tangent.T
    pair_gradient -= np.mean(pair_gradient, axis=0, keepdims=True)
    p2_covariance = random_score_factor_covariance(pair_gradient, amplitude_subject[:, None], residual_variance_p2, random_scores)
    p2_directional = directional_correlation_test(p2_residual, p2_covariance)
    return {
        "p_corr": p2_directional["pvalue_lower"],
        "z": p2_directional["z"],
        "mean_departure": p2_directional["departure"],
        "pair_count": int(len(p2_residual)),
        "identifiable_grid_count": int(np.sum(identifiable)),
        "basis_rank": int(smooth_basis.rank),
        "observed_min_correlation": float(np.min(observed_correlation)),
        "forward_null_min_correlation": float(np.min(null_correlation)),
    }
