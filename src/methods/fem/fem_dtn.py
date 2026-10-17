"""P1 FEM with a source-only nonhomogeneous DtN exterior on the half-line."""
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import roots_laguerre
from src.errors import (halfline_errors, halfline_cut, halfline_split_quadrature,
                        halfline_error_metadata)


def _element_quadrature_order(config):
    order = int(config.get('fem_quadrature_order', 8))
    if order < 1:
        raise ValueError('fem_quadrature_order must be positive')
    return order


def _assemble_interval(left, right, count, source, quadrature_order=8):
    nodes = np.linspace(left, right, count + 1)
    matrix = np.zeros((count + 1, count + 1))
    rhs = np.zeros(count + 1)
    q, w = np.polynomial.legendre.leggauss(quadrature_order)
    for i in range(count):
        h = nodes[i + 1] - nodes[i]
        x = (nodes[i] + nodes[i + 1]) / 2 + h * q / 2
        weights = h * w / 2
        n = np.vstack(((nodes[i + 1] - x) / h, (x - nodes[i]) / h)).T
        b = np.array([-np.ones_like(x) / h, np.ones_like(x) / h]).T
        matrix[i:i + 2, i:i + 2] += b.T @ (weights[:, None] * b)
        matrix[i:i + 2, i:i + 2] += n.T @ (weights[:, None] * n)
        rhs[i:i + 2] += n.T @ (weights * source(x))
    return nodes, matrix, rhs


def _dtn_load(R, source, quadrature_order=80):
    q, w = roots_laguerre(quadrature_order)
    return float(w @ source(R + q))


def exterior_values(x, R, boundary_value, source, quadrature_order=80):
    """Dirichlet Green reconstruction using only f and the numerical v(R).

    A=int_0^t exp(-y)f(x-y)dy, B=int_0^infinity exp(-y)f(x+y)dy;
    v=boundary_value*exp(-t)+(A+B-exp(-t)*g_R)/2, t=x-R.
    No exact solution or exact derivative enters this reconstruction.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < R):
        raise ValueError('Exterior points must satisfy x >= R')
    q, w = roots_laguerre(int(quadrature_order))
    load = float(w @ source(R+q))
    t = x-R
    a = np.empty_like(x)
    for index, (xx, tt) in enumerate(zip(x, t)):
        # Resolve exp(-y) for large x while retaining the complete [0,t] integral.
        points = [p for p in (1., 4., 16., 64.) if p < tt]
        a[index] = quad(lambda y: float(np.exp(-y)*source(xx-y)), 0., float(tt),
                        points=points, epsabs=1e-13, epsrel=1e-11)[0]
    b = source(x[:, None]+q) @ w
    decay = np.exp(-t)
    c = decay*load
    return (boundary_value*decay+.5*(a+b-c),
            -boundary_value*decay+.5*(-a+b+c))


def solve_fem_dtn(config, exact, exact_gradient, source):
    """True solutions are used only by halfline_errors, after prediction."""
    outer = halfline_cut(config)
    elements = int(config.get('fem_elements', sum(config.get('neurons', [100]))))
    if elements < 1:
        raise ValueError('Require fem_elements >= 1')
    started = time.perf_counter()
    nodes, matrix, rhs = _assemble_interval(0., outer, elements, source,
                                           _element_quadrature_order(config))
    matrix[-1, -1] += 1.
    order = int(config.get('dtn_quadrature_order', 80))
    rhs[-1] += _dtn_load(outer, source, order)
    dirichlet = float(config.get('dirichlet_value', 1.))
    coefficients = np.r_[dirichlet, np.linalg.solve(
        matrix[1:, 1:], rhs[1:]-matrix[1:, 0]*dirichlet)]
    seconds = time.perf_counter()-started
    x, weights = halfline_split_quadrature(config, elements)
    inside = x <= outer
    values, gradients = np.empty_like(x), np.empty_like(x)
    values[inside] = np.interp(x[inside], nodes, coefficients)
    element = np.clip(np.searchsorted(nodes, x[inside], side='right')-1, 0, elements-1)
    gradients[inside] = (np.diff(coefficients)/np.diff(nodes))[element]
    values[~inside], gradients[~inside] = exterior_values(
        x[~inside], outer, coefficients[-1], source, order)
    result = dict(example=61, method='FEM-P1-DtN', neurons=[elements], r_init=[outer],
                  dof=elements, solve_seconds=seconds, fem_outer=outer,
                  exterior_reconstruction='source_only_dirichlet_green',
                  error_quadrature=halfline_error_metadata(config, elements),
                  quadrature_errors=halfline_errors(x, weights, values, gradients, exact, exact_gradient))
    if 'far_field_points' in config:
        far_points = np.asarray(config['far_field_points'], dtype=float)
        if far_points.ndim != 1 or np.any(~np.isfinite(far_points)) or np.any(far_points < outer):
            raise ValueError('far_field_points must be finite points in the FEM exterior region')
        far_values, _ = exterior_values(far_points, outer, coefficients[-1], source, order)
        result['far_field_values'] = far_values.tolist()
    return result
