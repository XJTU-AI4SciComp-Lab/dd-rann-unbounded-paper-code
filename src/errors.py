import numpy as np


def halfline_cut(config):
    """One interface for Example 6.1 training, FEM and error evaluation."""
    cut = float(config.get('x_cut', config.get('interface', 45.)))
    if not np.isfinite(cut) or cut <= 0:
        raise ValueError('Require a positive finite x_cut/interface')
    return cut


def halfline_split_quadrature(config, dof):
    """Shared N-cell near rule and shifted, untruncated far Laguerre rule."""
    from src.sampling import laguerre_halfline_quadrature
    cut = halfline_cut(config)
    order = int(config.get('error_near_order', 16))
    if dof < 1 or order < 1:
        raise ValueError('Require positive DoF and error_near_order')
    if config.get('quadrature_rule', 'laguerre') != 'laguerre':
        raise ValueError('Example 6.1 exterior error integration uses Gauss-Laguerre')
    edges = np.linspace(0., cut, dof+1)
    q, w = np.polynomial.legendre.leggauss(order)
    h = np.diff(edges)
    near = (edges[:-1, None]+h[:, None]*(q+1)/2).ravel()
    near_weights = (h[:, None]*w/2).ravel()
    far, far_weights = laguerre_halfline_quadrature(
        int(config.get('error_order', 180)), float(config.get('quadrature_scale', 1.)))
    return np.r_[near, cut+far], np.r_[near_weights, far_weights]


def halfline_error_metadata(config, dof):
    return dict(rule='elementwise_legendre_shifted_laguerre', cut=halfline_cut(config),
                near_elements=dof, near_order=int(config.get('error_near_order', 16)),
                far_order=int(config.get('error_order', 180)),
                far_scale=float(config.get('quadrature_scale', 1.)))


def halfline_errors(x, weights, values, gradients, exact, exact_gradient):
    """Common unweighted L2/H1 estimates at shared half-line quadrature nodes."""
    target_values, target_gradients = exact(x), exact_gradient(x)
    value_error = np.sum(weights * (values-target_values)**2)
    gradient_error = np.sum(weights * (gradients-target_gradients)**2)
    target = np.sum(weights * target_values**2)
    target_h1 = target + np.sum(weights * target_gradients**2)
    return dict(l2=float(np.sqrt(value_error)), relative_l2=float(np.sqrt(value_error/target)),
                h1=float(np.sqrt(value_error+gradient_error)),
                relative_h1=float(np.sqrt((value_error+gradient_error)/target_h1)))


def relative_errors(points, weights, regions, features, coefficients, exact, exact_gradient):
    from src.methods.ddrann.elliptic_collocation import predict
    value = np.empty(len(points))
    gradient = np.empty_like(points)
    for region in np.unique(regions):
        mask = regions == region
        value[mask], gradient[mask] = predict(features, coefficients, points[mask], int(region), True)
    u, grad = exact(points), exact_gradient(points)
    e2 = np.sum(weights * (value - u) ** 2)
    g2 = np.sum(weights * np.sum((gradient - grad) ** 2, axis=1))
    u2 = np.sum(weights * u ** 2)
    h2 = u2 + np.sum(weights * np.sum(grad ** 2, axis=1))
    return dict(l2=float(np.sqrt(e2)), relative_l2=float(np.sqrt(e2/u2)),
                h1=float(np.sqrt(e2+g2)), relative_h1=float(np.sqrt((e2+g2)/h2)))
