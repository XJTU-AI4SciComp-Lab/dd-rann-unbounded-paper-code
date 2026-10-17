"""Reusable complex space-time random features and real-coefficient least squares."""
import numpy as np
from src.linear_solver import least_squares


def solve_complex_system(features, interiors, source, initials, interfaces, penalty=1., rank_tolerance=None, real_coefficients=False, solver='scipy', device='cpu', initial_penalty=None, interface_penalty=None):
    """Least squares for i*u_t+u_xx=f with complex or real coefficients."""
    initial_penalty = penalty if initial_penalty is None else initial_penalty
    interface_penalty = penalty if interface_penalty is None else interface_penalty
    widths = [len(f.wx) for f in features]
    offsets = np.cumsum([0]+widths)
    total = offsets[-1]
    matrices, vectors = [], []

    def block(region, values):
        result = np.zeros((len(values), total), dtype=complex)
        result[:, offsets[region]:offsets[region+1]] = values
        return result

    def add(matrix, target):
        if real_coefficients:
            matrices.extend((matrix.real, matrix.imag))
            vectors.extend((target.real, target.imag))
        else:
            matrices.append(matrix)
            vectors.append(target)

    for region, points in enumerate(interiors):
        operator = 1j*features[region].evaluate(points, 't') + features[region].evaluate(points, 'xx')
        add(block(region, operator), source(points))
    for region, points, target in initials:
        add(initial_penalty*block(region, features[region].evaluate(points)), initial_penalty*target)
    for left, right, points in interfaces:
        add(interface_penalty*(block(left, features[left].evaluate(points))-
                     block(right, features[right].evaluate(points))), np.zeros(len(points), complex))
        add(interface_penalty*(block(left, features[left].evaluate(points, 'x'))-
                     block(right, features[right].evaluate(points, 'x'))), np.zeros(len(points), complex))
    matrix = np.concatenate(matrices)
    vector = np.concatenate(vectors)
    values = least_squares(matrix, vector, solver, device, rank_tolerance)
    return np.split(values, offsets[1:-1])


def predict_complex(features, coefficients, points, region, derivative=None):
    return features[region].evaluate(points, derivative) @ coefficients[region]
