"""PDE, geometry, boundary data, and sampling choices for paper Example 6.2."""
import numpy as np

from src.manufactured import exponential_wave


class Problem:
    number = 62
    dimension = 2
    interface_radius = 18.0
    feature_ranges = (0.5, 1.0)
    radial_distribution = (20.0, 10.0)
    quadrature_rule = "laguerre"
    sampling_view = ((0.0, 50.0), (0.0, 50.0))
    solution_view = (0.0, 20.0)
    sampling_legend_loc = "upper right"
    interface_plot_label = r"SPs on $\Gamma$"

    def fields(self, points):
        return exponential_wave(np.asarray(points), lambda p: np.full_like(p, -0.5),
                                lambda p: np.zeros(len(p)), lambda p: -0.5 * p.sum(axis=1),
                                np.array([1.0, 1.0]))

    def inside(self, points):
        x, y = points[:, 0], points[:, 1]
        ellipse = (x + y - 9) ** 2 / 9 + (y - x + 1) ** 2 <= 2
        circle = (x - 1) ** 2 + (y - 1) ** 2 <= 1
        return (x >= 0) & (y >= 0) & ~ellipse & ~circle

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
        circle = np.array([1.0, 1.0]) + direction
        boundaries = [(0, ellipse, self.fields(ellipse)[0], None),
                      (0, circle, np.sum(self.fields(circle)[1] * -direction, axis=1), -direction)]
        for axis in range(2):
            for index, interval in enumerate(((0.0, self.interface_radius),
                                               (self.interface_radius, getattr(self, "axis_boundary_limit", 50.0)))):
                points = np.zeros((count, 2))
                axis_counts = getattr(self, "axis_boundary_points", None) or [count, count]
                points = np.zeros((axis_counts[index], 2))
                points[:, 1 - axis] = np.linspace(*interval, axis_counts[index])
                points = points[self.inside(points)]
                boundaries.append((index, points, self.fields(points)[0], None))
        return boundaries

    def plot_boundary_groups(self, boundaries):
        order = (3, 5, 2, 4, 0, 1)
        labels = (r"SPs on $\Gamma_{D1}$ in the far field", "_nolegend_",
                  r"SPs on $\Gamma_{D1}$ in the near field", "_nolegend_",
                  r"SPs on $\Gamma_{D2}$ in the near field",
                  r"SPs on $\Gamma_N$ in the near field")
        colors = ("yellow", "yellow", "pink", "pink", "black", "orange")
        return [(boundaries[index][1], label, color)
                for index, label, color in zip(order, labels, colors)]

    def plot_holes(self):
        angle = np.linspace(0.0, 2 * np.pi, 300)
        ellipse = np.column_stack((5 + np.cos(angle) + 2 * np.sin(angle),
                                   4 + 2 * np.cos(angle) + np.sin(angle)))
        circle = np.column_stack((1 + np.cos(angle), 1 + np.sin(angle)))
        return ellipse, circle
