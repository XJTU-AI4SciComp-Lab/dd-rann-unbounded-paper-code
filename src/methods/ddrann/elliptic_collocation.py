import numpy as np
from src.linear_solver import least_squares


def solve_elliptic(features, interiors, source, boundaries, interfaces, beta=1., gamma=1., penalty=10., rank_tolerance=None, regularization=1e-6, solver='scipy', device='cpu', interface_derivatives="gradient", return_diagnostics=False, column_normalize_matrix=False, boundary_penalty=1.):
    """Solve -beta*Delta u + gamma*u=f with Dirichlet/Neumann/interface rows.

    boundaries: (region, points, target, normals_or_None) tuples.
    interfaces: (left, right, points, normals) tuples.
    """
    widths = [f.weights.shape[1] for f in features]
    offsets = np.cumsum([0] + widths)
    total = offsets[-1]
    rows, rhs = [], []

    def block(region, matrix):
        out = np.zeros((len(matrix), total))
        out[:, offsets[region]:offsets[region + 1]] = matrix
        return out

    for region, points in enumerate(interiors):
        value, _, lap = features[region].evaluate(points, laplacian=True)
        rows.append(block(region, -beta * lap + gamma * value))
        rhs.append(source(points))
    for region, points, target, normals in boundaries:
        if normals is None:
            matrix = features[region].evaluate(points)
        else:
            _, grad, _ = features[region].evaluate(points, gradient=True)
            matrix = np.einsum('ndm,nd->nm', grad, normals)
        rows.append(boundary_penalty * block(region, matrix))
        rhs.append(boundary_penalty * target)
    for left, right, points, normals in interfaces:
        va, ga, _ = features[left].evaluate(points, gradient=True)
        vb, gb, _ = features[right].evaluate(points, gradient=True)
        rows.append(penalty * (block(left, va) - block(right, vb)))
        rhs.append(np.zeros(len(points)))
        if interface_derivatives == "gradient":
            for axis in range(ga.shape[1]):
                rows.append(penalty * (block(left, ga[:, axis, :]) - block(right, gb[:, axis, :])))
                rhs.append(np.zeros(len(points)))
        elif interface_derivatives == "normal":
            rows.append(penalty * (block(left, np.einsum('ndm,nd->nm', ga, normals)) -
                                   block(right, np.einsum('ndm,nd->nm', gb, normals))))
            rhs.append(np.zeros(len(points)))
        else:
            raise ValueError(f"Unknown interface derivative matching: {interface_derivatives}")
    matrix = np.concatenate(rows)
    vector = np.concatenate(rhs)
    if regularization:
        matrix = np.vstack((matrix, np.sqrt(regularization) * np.eye(total)))
        vector = np.concatenate((vector, np.zeros(total)))
    if column_normalize_matrix:
        column_scales = np.linalg.norm(matrix, axis=0)
        column_scales[column_scales == 0.0] = 1.0
        solve_matrix = matrix / column_scales
    else:
        column_scales = np.ones(matrix.shape[1])
        solve_matrix = matrix
    solved = least_squares(
        solve_matrix, vector, solver, device, rank_tolerance,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        scaled_coefficients, diagnostics = solved
        coefficients = scaled_coefficients / column_scales
        diagnostics.update({"matrix_rows": matrix.shape[0], "matrix_columns": matrix.shape[1]})
        diagnostics.update({
            "column_normalized": bool(column_normalize_matrix),
            "column_scale_min": float(np.min(column_scales)),
            "column_scale_max": float(np.max(column_scales)),
            "rank_tolerance": rank_tolerance,
        })
        if column_normalize_matrix:
            diagnostics["condition_source"] = "singular_values_of_column_normalized_A"
        return np.split(coefficients, offsets[1:-1]), diagnostics
    coefficients = solved / column_scales
    return np.split(coefficients, offsets[1:-1])


def predict(features, coefficients, points, region, gradient=False):
    if gradient:
        value, grad, _ = features[region].evaluate(points, gradient=True)
        return value @ coefficients[region], np.einsum('ndm,m->nd', grad, coefficients[region])
    return features[region].evaluate(points) @ coefficients[region]
