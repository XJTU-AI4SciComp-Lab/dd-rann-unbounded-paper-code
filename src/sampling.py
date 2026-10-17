import numpy as np


def radial_points(count, dimension, mean, std, rng, angular_range=None):
    if angular_range is not None and dimension == 2:
        radii = np.abs(rng.normal(mean, std, count))
        angles = rng.uniform(*angular_range, count)
        return radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    directions = rng.normal(size=(count, dimension))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions * np.abs(rng.normal(mean, std, count))[:, None]


def sphere_points(count, dimension, radius, rng, angular_range=None):
    if angular_range is not None and dimension == 2:
        angles = rng.uniform(*angular_range, count)
        return radius * np.column_stack((np.cos(angles), np.sin(angles)))
    directions = rng.normal(size=(count, dimension))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return radius * directions


def filtered_points(count, sampler, admissible):
    blocks = []
    remaining = count
    while remaining:
        candidates = sampler(max(remaining * 2, 32))
        accepted = candidates[admissible(candidates)][:remaining]
        blocks.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(blocks)


def tensor_quadrature(dimension, order, rule='mapped_legendre', scale=1.):
    """Product quadrature on R^d or the positive orthant, with Lebesgue weights."""
    if rule == 'mapped_legendre':
        nodes, weights = np.polynomial.legendre.leggauss(order)
        transformed = scale * nodes / np.sqrt(1 - nodes*nodes)
        adjusted = weights * scale / (1 - nodes*nodes)**1.5
    elif rule == 'laguerre':
        if order <= 180:
            transformed, weights = np.polynomial.laguerre.laggauss(order)
            adjusted = scale * weights * np.exp(transformed)
        else:
            transformed, adjusted = _stable_laguerre_lebesgue(order)
            adjusted = scale * adjusted
        transformed = scale * transformed
    elif rule == 'hermite':
        transformed, weights = np.polynomial.hermite.hermgauss(order)
        adjusted = scale * weights * np.exp(transformed*transformed)
        transformed = scale * transformed
    else:
        raise ValueError(rule)
    grids = np.meshgrid(*([transformed]*dimension), indexing='ij')
    weight_grids = np.meshgrid(*([adjusted]*dimension), indexing='ij')
    return (np.stack([grid.ravel() for grid in grids], axis=1),
            np.prod(np.stack(weight_grids, axis=0), axis=0).ravel())


def _stable_laguerre_lebesgue(order):
    """Same Gauss-Laguerre rule, with compensated weights evaluated in log scale."""
    from scipy.linalg import eigh_tridiagonal
    k = np.arange(order, dtype=float)
    nodes = eigh_tridiagonal(2*k+1, np.arange(1, order, dtype=float), eigvals_only=True)
    previous, current = np.ones(order), 1-nodes
    log_scale = np.zeros(order)
    for j in range(1, order-1):
        large = np.maximum(abs(previous), abs(current)) > 1e100
        previous[large] /= 1e100
        current[large] /= 1e100
        log_scale[large] += np.log(1e100)
        following = ((2*j+1-nodes)*current-j*previous)/(j+1)
        previous, current = current, following
    log_function = np.log(abs(current))+log_scale-nodes/2
    # w_i exp(x_i) = x_i / [order^2 * (exp(-x_i/2)L_{order-1}(x_i))^2].
    weights = np.exp(np.log(nodes)-2*np.log(order)-2*log_function)
    return nodes, weights


def laguerre_halfline_quadrature(order, scale=1.):
    """Global Gauss-Laguerre nodes, without segmentation or finite truncation."""
    if order < 1 or scale <= 0:
        raise ValueError("Require positive half-line quadrature order and scale")
    nodes, weights = tensor_quadrature(1, order, 'laguerre', scale)
    return nodes[:, 0], weights
