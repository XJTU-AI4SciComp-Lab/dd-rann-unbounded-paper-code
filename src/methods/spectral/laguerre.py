"""Boundary-adapted Laguerre-function Galerkin solver for -u''+u=f."""
import time
import numpy as np
from scipy.linalg import solve_banded
from src.errors import halfline_errors, halfline_split_quadrature, halfline_error_metadata
from src.sampling import laguerre_halfline_quadrature


def laguerre_functions(y, degree):
    """Evaluate exp(-y/2)L_k(y) using a rescaled polynomial recurrence."""
    y = np.asarray(y, dtype=float)
    basis = np.empty((len(y), degree+1))
    previous = np.ones_like(y)
    log_scale = np.zeros_like(y)
    basis[:, 0] = np.exp(-y/2)
    if degree == 0:
        return basis
    current = 1-y
    for k in range(1, degree+1):
        large = np.maximum(abs(previous), abs(current)) > 1e100
        previous[large] /= 1e100
        current[large] /= 1e100
        log_scale[large] += np.log(1e100)
        with np.errstate(divide="ignore"):
            basis[:, k] = np.sign(current)*np.exp(np.log(abs(current))+log_scale-y/2)
        if k < degree:
            following = ((2*k+1-y)*current-k*previous)/(k+1)
            previous, current = current, following
    return basis


def assemble_system(dof, beta, source, boundary_value, order=None, scale=None):
    """Correct tridiagonal form; source uses one global half-line quadrature."""
    order = max(180, 2*dof+40) if order is None else int(order)
    scale = 1/beta if scale is None and beta > 0 else scale
    if dof < 1 or beta <= 0 or order < 1 or scale is None or scale <= 0:
        raise ValueError("Require positive spectral DoF, scale and assembly parameters")
    diagonal, off_diagonal = beta/2+2/beta, beta/4-1/beta
    band = np.zeros((3, dof))
    band[1] = diagonal
    band[0, 1:] = off_diagonal
    band[2, :-1] = off_diagonal
    lift = np.zeros(dof)
    lift[0] = .75*beta+1/beta
    if dof > 1:
        lift[1] = .25*beta-1/beta
    x, weights = laguerre_halfline_quadrature(order, scale)
    rhs = np.zeros(dof)
    for start in range(0, len(x), 2048):
        xx, ww = x[start:start+2048], weights[start:start+2048]
        basis = laguerre_functions(beta*xx, dof)
        rhs += (ww*source(xx)) @ (basis[:, 1:]-basis[:, :-1])
    return band, rhs-boundary_value*lift


def evaluate(coefficients, beta, x):
    basis = laguerre_functions(beta*np.asarray(x), len(coefficients)-1)
    derivative = np.polynomial.laguerre.lagder(coefficients)
    derivative = np.pad(derivative, (0, len(coefficients)-len(derivative)))-.5*coefficients
    return basis@coefficients, beta*(basis@derivative)


def solve_halfline(config, exact, exact_gradient, source):
    """Assembly quadrature is independent of the shared error quadrature."""
    if config.get("spectral_source_rule", "laguerre") != "laguerre":
        raise ValueError("Spectral source integration uses only Gauss-Laguerre quadrature")
    dof = sum(config["neurons"])
    beta = float(config.get("spectral_scale", 1.))
    started = time.perf_counter()
    boundary_value = float(config['dirichlet_value'] if 'dirichlet_value' in config else exact(0.))
    band, rhs = assemble_system(dof, beta, source, boundary_value,
                               config.get("spectral_source_order"),
                               config.get("spectral_source_scale"))
    reduced = solve_banded((1, 1), band, rhs)
    coefficients = np.zeros(dof+1)
    coefficients[:-1] -= reduced
    coefficients[1:] += reduced
    coefficients[1] += boundary_value
    seconds = time.perf_counter()-started
    x, weights = halfline_split_quadrature(config, dof)
    values, gradients = evaluate(coefficients, beta, x)
    result = dict(example=61, method="Spectral", neurons=config["neurons"], r_init=[beta],
                  dof=dof, solve_seconds=seconds,
                  error_quadrature=halfline_error_metadata(config, dof),
                  source_quadrature_rule="laguerre",
                  source_quadrature_order=(max(180, 2*dof+40) if config.get("spectral_source_order") is None
                                           else int(config["spectral_source_order"])),
                  quadrature_errors=halfline_errors(x, weights, values, gradients, exact, exact_gradient))
    if 'far_field_points' in config:
        far_points = np.asarray(config['far_field_points'], dtype=float)
        if far_points.ndim != 1 or np.any(~np.isfinite(far_points)) or np.any(far_points < 0):
            raise ValueError('far_field_points must be a one-dimensional nonnegative finite array')
        far_values, _ = evaluate(coefficients, beta, far_points)
        result['far_field_values'] = far_values.tolist()
    return result
