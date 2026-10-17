"""PDE, geometry, boundary data, and sampling choices for paper Example 6.3."""
import numpy as np


class Problem:
    number = 63
    dimension = 2
    interface_radius = 9.0
    feature_ranges = (0.5, 10.0)
    radial_distribution = (6.0, 15.0)
    quadrature_rule = "mapped_legendre"
    sampling_view = ((-30.0, 30.0), (-30.0, 30.0))
    solution_view = (-10.0, 10.0)
    sampling_legend_loc = "upper left"
    interface_plot_label = r"SPs on $\Gamma$"

    def fields(self, points):
        points = np.asarray(points)
        envelope = np.exp(-np.sum(points * points, axis=1) / 10)
        gradient_log = -points / 5
        laplacian_log = np.full(len(points), -points.shape[1] / 5)
        phases = 2 * points
        oscillation = np.cos(phases[:, 1]) - np.cos(phases[:, 0])
        gradient_oscillation = np.column_stack((2 * np.sin(phases[:, 0]), -2 * np.sin(phases[:, 1])))
        laplacian_oscillation = 4 * np.cos(phases[:, 0]) - 4 * np.cos(phases[:, 1])
        value = 0.5 * envelope * oscillation
        gradient = 0.5 * envelope[:, None] * (gradient_log * oscillation[:, None] + gradient_oscillation)
        laplacian = 0.5 * envelope * ((laplacian_log + np.sum(gradient_log**2, axis=1)) * oscillation
                                      + 2 * np.sum(gradient_log * gradient_oscillation, axis=1)
                                      + laplacian_oscillation)
        return value, gradient, laplacian

    def inside(self, points):
        x, y = points[:, 0], points[:, 1]
        ellipse = (x + y - 9) ** 2 / 9 + (y - x + 1) ** 2 <= 2
        circle = (x + 2) ** 2 + (y + 2) ** 2 <= 4
        return ~ellipse & ~circle

    def region(self, points):
        return (np.linalg.norm(points, axis=1) >= self.interface_radius).astype(int)

    def source(self, points):
        value, _, laplacian = self.fields(points)
        return -getattr(self, "beta", 1.) * laplacian + getattr(self, "gamma", 1.) * value

    def boundary_conditions(self, count, rng):
        angle = rng.uniform(0, 2 * np.pi, count)
        a = 3 * np.sqrt(2) * np.cos(angle)
        b = np.sqrt(2) * np.sin(angle)
        ellipse = np.column_stack(((10 + a - b) / 2, (8 + a + b) / 2))
        direction = np.column_stack((np.cos(angle), np.sin(angle)))
        circle = np.array([-2.0, -2.0]) + 2 * direction
        return [(0, ellipse, self.fields(ellipse)[0], None),
                (0, circle, np.sum(self.fields(circle)[1] * -direction, axis=1), -direction)]

    def plot_boundary_groups(self, boundaries):
        labels = (r"SPs on $\Gamma_D$ in the near field",
                  r"SPs on $\Gamma_N$ in the near field")
        colors = ("purple", "black")
        return [(item[1], label, color) for item, label, color in zip(boundaries, labels, colors)]

    def plot_holes(self):
        angle = np.linspace(0.0, 2 * np.pi, 300)
        ellipse = np.column_stack((5 + np.cos(angle) + 2 * np.sin(angle),
                                   4 + 2 * np.cos(angle) + np.sin(angle)))
        circle = np.column_stack((-2 + 2 * np.cos(angle), -2 + 2 * np.sin(angle)))
        return ellipse, circle
