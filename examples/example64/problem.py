"""PDE and sampling choices for paper Example 6.4."""
import numpy as np

from src.manufactured import exponential_wave


class Problem:
    number = 64
    dimension = 3
    interface_radius = 5.0
    feature_ranges = (2.0, 10.0)
    radial_distribution = (2.0, 5.0)
    quadrature_rule = "mapped_legendre"
    sampling_view = ((-10.0, 10.0),) * 3
    solution_view = (-10.0, 10.0)
    sampling_legend_loc = "upper right"
    interface_plot_label = r"SPs on $\Gamma$"

    def _paper_spherical_points(self, radii, rng):
        """Use the two U(0, 2*pi) angles specified in the Example 6.4 PDF."""
        radii = np.asarray(radii, dtype=float)
        points = np.empty((len(radii), 3))
        remainder = np.abs(radii)
        for axis in range(2):
            angles = rng.uniform(*getattr(self, "angular_range", (0., 2 * np.pi)), len(radii))
            points[:, axis] = remainder * np.cos(angles)
            remainder = remainder * np.sin(angles)
        points[:, 2] = remainder
        return points

    def sample_interior(self, count, rng):
        radii = rng.normal(*self.radial_distribution, count)
        return self._paper_spherical_points(radii, rng)

    def sample_interface(self, count, rng):
        return self._paper_spherical_points(
            np.full(count, self.interface_radius), rng
        )

    def fields(self, points):
        return exponential_wave(np.asarray(points), lambda p: -2 * p,
                                lambda p: np.full(len(p), -2 * p.shape[1]),
                                lambda p: -np.sum(p * p, axis=1), np.ones(3))

    def inside(self, points):
        return np.ones(len(points), dtype=bool)

    def region(self, points):
        return (np.linalg.norm(points, axis=1) >= self.interface_radius).astype(int)

    def source(self, points):
        value, _, laplacian = self.fields(points)
        return -getattr(self, "beta", 1.) * laplacian + getattr(self, "gamma", 1.) * value

    def boundary_conditions(self, count, rng):
        return []

    def plot_boundary_groups(self, boundaries):
        return []

    def plot_holes(self):
        return ()
