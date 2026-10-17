"""Generic one-dimensional broken Petrov--Galerkin least-squares assembly."""
import numpy as np
from src.linear_solver import least_squares


def assemble_region(feature, nodes, source, quadrature_order):
    """Rows for interior piecewise-linear test functions on arbitrary nodes."""
    q, w = np.polynomial.legendre.leggauss(quadrature_order)
    matrix = np.empty((len(nodes)-2, len(feature.weight)))
    rhs = np.empty(len(nodes)-2)
    for j in range(1, len(nodes)-1):
        matrix[j-1] = 0
        rhs[j-1] = 0
        for left, right, rising in ((nodes[j-1], nodes[j], True),
                                    (nodes[j], nodes[j+1], False)):
            h = right-left
            x = (left+right)/2+h*q/2
            phi = (x-left)/h if rising else (right-x)/h
            derivative = 1/h if rising else -1/h
            weights = h*w/2
            matrix[j-1] += (weights*derivative) @ feature.evaluate(x, True)
            matrix[j-1] += (weights*phi) @ feature.evaluate(x)
            rhs[j-1] += np.sum(weights*phi*source(x))
    return matrix, rhs


def solve_pg(features, meshes, source, dirichlet, interface, quadrature_order=5,
             solver='scipy', device='cpu', constraint_weights=None,
             region_weights=None):
    assemblies = [assemble_region(f, x, source, quadrature_order)
                  for f, x in zip(features, meshes)]
    widths = [len(f.weight) for f in features]
    total = sum(widths)
    a0, b0 = assemblies[0]; a1, b1 = assemblies[1]
    region_weights = region_weights or (1.0, 1.0)
    a0, b0 = float(region_weights[0]) * a0, float(region_weights[0]) * b0
    a1, b1 = float(region_weights[1]) * a1, float(region_weights[1]) * b1
    matrix = np.block([[a0, np.zeros((len(a0), widths[1]))],
                       [np.zeros((len(a1), widths[0])), a1]])
    boundary = np.r_[features[0].evaluate(np.array([meshes[0][0]]))[0], np.zeros(widths[1])]
    jump = np.r_[features[0].evaluate(np.array([interface]))[0],
                 -features[1].evaluate(np.array([interface]))[0]]
    flux = np.r_[features[0].evaluate(np.array([interface]), True)[0],
                 -features[1].evaluate(np.array([interface]), True)[0]]
    weights = constraint_weights or {}
    wb = float(weights.get("dirichlet", 1.0))
    wv = float(weights.get("interface_value", 1.0))
    wf = float(weights.get("interface_flux", 1.0))
    matrix = np.vstack((matrix, wb * boundary, wv * jump, wf * flux))
    rhs = np.r_[b0, b1, wb * dirichlet, 0., 0.]
    values = least_squares(matrix, rhs, solver, device)
    return np.split(values, [widths[0]])


def solve_pg_single(feature, mesh, source, dirichlet, quadrature_order=5,
                    solver="scipy", device="cpu", boundary_weight=1.0,
                    region_weight=1.0):
    """Solve the half-line PG system with one globally supported RaNN."""
    matrix, rhs = assemble_region(feature, mesh, source, quadrature_order)
    matrix, rhs = float(region_weight) * matrix, float(region_weight) * rhs
    boundary = feature.evaluate(np.asarray([mesh[0]]))[0]
    boundary_weight = float(boundary_weight)
    values = least_squares(np.vstack((matrix, boundary_weight * boundary)),
                           np.r_[rhs, boundary_weight * dirichlet],
                           solver, device)
    return [values]
