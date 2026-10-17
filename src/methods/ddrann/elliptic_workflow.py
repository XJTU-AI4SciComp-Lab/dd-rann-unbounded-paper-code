"""Generic execution workflow for two-region elliptic collocation examples."""
import ctypes
import os
import threading
import time

import numpy as np

from src.methods.ddrann.elliptic_collocation import predict, solve_elliptic
from src.errors import relative_errors
from src.methods.ddrann.features import GaussianFeatures
from src.runtime import validate_runtime
from src.config import configure_case, validate_neurons
from src.sampling import filtered_points, radial_points, sphere_points, tensor_quadrature


class _PeakRssMonitor:
    """Sample this process's RSS during one explicitly delimited operation."""

    def __init__(self, interval=0.005):
        self.interval = interval
        self._read_rss = self._make_rss_reader()
        self.peak = self._read_rss()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    @staticmethod
    def _make_rss_reader():
        if os.name == "nt":
            from ctypes import wintypes

            class ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCountersEx),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()

            def read_windows_rss():
                if not psapi.GetProcessMemoryInfo(
                    process, ctypes.byref(counters), counters.cb
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                return int(counters.WorkingSetSize)

            return read_windows_rss

        # Linux fallback without third-party dependencies.
        statm = f"/proc/{os.getpid()}/statm"
        if os.path.exists(statm):
            def read_linux_rss():
                with open(statm, encoding="ascii") as stream:
                    resident_pages = int(stream.read().split()[1])
                return resident_pages * os.sysconf("SC_PAGE_SIZE")

            return read_linux_rss

        # Last-resort fallback measures Python allocations rather than total RSS.
        import tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        return lambda: tracemalloc.get_traced_memory()[0]

    def _sample(self):
        while not self.stop_event.wait(self.interval):
            self.peak = max(self.peak, self._read_rss())

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self._read_rss())
        return self.peak / (1024.0 ** 2)


def _boundary_sampling_rng(config, rng):
    """Optional independent stream makes physical boundary points common to baselines."""
    if config.get("shared_boundary_sampling", False):
        return np.random.default_rng(np.random.SeedSequence([config["seed"], 1]))
    return rng


def sampling_data(config, case):
    """Generate the exact features and point sets used by the elliptic solve."""
    case = configure_case(case, config)
    rng = np.random.default_rng(config["seed"])
    validate_neurons(config["neurons"], 2)
    radii = config.get("r_init", case.feature_ranges)
    if len(radii) != 2 or min(radii) <= 0:
        raise ValueError("r_init must contain two positive radii")
    features = [GaussianFeatures.random(case.dimension, n, radius, rng, config.get("feature_center_range", 1.))
                for n, radius in zip(config["neurons"], radii)]
    mean, std = case.radial_distribution
    if hasattr(case, "sample_interior"):
        sampler = lambda count: case.sample_interior(count, rng)
    else:
        sampler = lambda count: radial_points(count, case.dimension, mean, std, rng, case.angular_range)
    interior = filtered_points(config["interior_points"], sampler, case.inside)
    interiors = [interior[case.region(interior) == index] for index in range(2)]
    count = config.get("interface_points", config["boundary_points"])
    if hasattr(case, "sample_interface"):
        interface_sampler = lambda size: case.sample_interface(size, rng)
    else:
        interface_sampler = lambda size: sphere_points(
            size, case.dimension, case.interface_radius, rng, case.angular_range
        )
    interface = filtered_points(count, interface_sampler, case.inside)
    boundaries = case.boundary_conditions(config["boundary_points"], _boundary_sampling_rng(config, rng))
    return features, interiors, interface, boundaries


def run_elliptic_case(config, case):
    """Solve an example-provided PDE and geometry without importing from ``examples``."""
    validate_runtime(config)
    features, interiors, interface, boundaries = sampling_data(config, case)
    radii = config.get("r_init", case.feature_ranges)
    normals = interface / case.interface_radius
    # A configurable interface may cross a physical boundary: use each point's actual region.
    solve_boundaries = []
    for _, points, target, boundary_normals in boundaries:
        boundary_regions = case.region(points)
        for index in range(2):
            mask = boundary_regions == index
            if np.any(mask):
                solve_boundaries.append((index, points[mask], target[mask],
                                         boundary_normals[mask] if boundary_normals is not None else None))
    collect_diagnostics = config.get("collect_solve_diagnostics", False)
    memory_monitor = _PeakRssMonitor() if collect_diagnostics else None
    if memory_monitor:
        memory_monitor.start()
    try:
        started = time.perf_counter()
        solved = solve_elliptic(
            features, interiors, case.source, solve_boundaries,
            [(0, 1, interface, normals)], penalty=config.get("interface_penalty", config.get("penalty", 10.)),
            beta=case.beta, gamma=case.gamma, boundary_penalty=config.get("boundary_penalty", 1.),
            regularization=config.get("regularization", 0.0),
            solver=config.get("solver", "scipy"), device=config.get("device", "cpu"),
            rank_tolerance=config.get("rank_tolerance"),
            interface_derivatives=config.get("interface_derivatives", "gradient"),
            return_diagnostics=collect_diagnostics,
            column_normalize_matrix=config.get("column_normalize_matrix", False))
        solve_seconds = time.perf_counter() - started
    finally:
        peak_memory_mb = memory_monitor.stop() if memory_monitor else None
    if collect_diagnostics:
        coefficients, solve_diagnostics = solved
        solve_diagnostics["peak_memory_mb"] = peak_memory_mb
    else:
        coefficients = solved
        solve_diagnostics = {}
    test, weights = tensor_quadrature(case.dimension, config["error_order"], case.quadrature_rule, case.quadrature_scale)
    valid = case.inside(test)
    test, weights = test[valid], weights[valid]
    errors = relative_errors(test, weights, case.region(test), features, coefficients,
                             lambda points: case.fields(points)[0],
                             lambda points: case.fields(points)[1])
    if config.get("save_figures", False):
        from pathlib import Path
        from src.plotting import plot_sampling_points, plot_sampling_points_3d, plot_solution_triptych
        figure_dir = Path(config["figures_dir"])
        method_tag = str(config.get("method", "DDRaNN-2net")).replace("/", "_")
        neuron_tag = "-".join(str(n) for n in config["neurons"])
        figure_tag = f"example{case.number}_{method_tag}_m{neuron_tag}_seed{config['seed']}"
        boundary_groups = case.plot_boundary_groups(boundaries)
        sampling_view = case.sampling_view
        if case.dimension == 3:
            plot_sampling_points_3d(interiors, interface,
                                    figure_dir / f"{figure_tag}_sampling_points.png",
                                    labels=("near-field SPs", "far-field SPs"),
                                    view_limits=sampling_view,
                                    legend_loc=case.sampling_legend_loc)
        else:
            plot_sampling_points(interiors, interface,
                                    figure_dir / f"{figure_tag}_sampling_points.png",
                                 boundary_groups=boundary_groups,
                                 labels=("near-field SPs", "far-field SPs"),
                                 view_limits=sampling_view,
                                 legend_loc=case.sampling_legend_loc,
                                 interface_label=case.interface_plot_label,
                                 figure_size=config.get("plot", {}).get("sampling_figure_size", (6, 5)),
                                 legend_above=config.get("plot", {}).get("sampling_legend_above", False),
                                 equal_aspect=config.get("plot", {}).get("sampling_equal_aspect", False),
                                 marker_size=config.get("plot", {}).get("sampling_marker_size", 10))
        lower, upper = case.solution_view
        grid = np.linspace(lower, upper, config.get("plot", {}).get("solution_grid_points", 120))
        xx, yy = np.meshgrid(grid, grid)
        if case.dimension == 2:
            plot_points = np.column_stack((xx.ravel(), yy.ravel()))
        else:
            plot_points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        valid_plot = case.inside(plot_points)
        plot_points = plot_points[valid_plot]
        numerical = np.empty(len(plot_points))
        regions = case.region(plot_points)
        for index in range(2):
            mask = regions == index
            numerical[mask] = predict(features, coefficients, plot_points[mask], index)
        exact_values = case.fields(plot_points)[0]
        plot_solution_triptych(plot_points[:, :2], numerical, exact_values,
                               figure_dir / f"{figure_tag}_solution_comparison.png",
                               hole_polygons=case.plot_holes(),
                               error_colorbar_scientific=config.get("plot", {}).get("error_colorbar_scientific", False))
    return dict(example=case.number, method=config.get("method", "DDRaNN-2net"),
                neurons=config["neurons"], r_init=radii, dof=sum(config["neurons"]),
                solve_seconds=solve_seconds, solve_diagnostics=solve_diagnostics,
                quadrature_errors=errors)


def run_single_elliptic_case(config, case):
    """Run a one-network collocation comparison on an example-provided domain."""
    validate_runtime(config)
    case = configure_case(case, config)
    rng = np.random.default_rng(config["seed"])
    neurons = int(config["neurons"][0])
    radius = config["single_r"]
    radii = ([float(value) for value in radius]
             if isinstance(radius, (list, tuple)) else [float(radius)])
    widths = config.get("single_feature_neurons")
    if widths is None:
        if len(radii) != 1:
            raise ValueError("single_feature_neurons is required when single_r has multiple values")
        widths = [neurons]
    widths = [int(value) for value in widths]
    if len(widths) != len(radii) or min(widths, default=0) <= 0 or sum(widths) != neurons:
        raise ValueError("single_feature_neurons must be positive, match single_r, and sum to neurons[0]")
    feature_blocks = [GaussianFeatures.random(case.dimension, width, value, rng, config.get("feature_center_range", 1.))
                      for width, value in zip(widths, radii)]
    feature = GaussianFeatures(
        np.concatenate([block.weights for block in feature_blocks], axis=1),
        np.concatenate([block.bias for block in feature_blocks], axis=1),
    )
    mean, std = case.radial_distribution
    sampler = lambda count: radial_points(count, case.dimension, mean, std, rng, case.angular_range)
    interior = filtered_points(config["interior_points"], sampler, case.inside)
    boundaries = [(0, points, target, normals)
                  for _, points, target, normals in case.boundary_conditions(
                      config["boundary_points"], _boundary_sampling_rng(config, rng))]
    started = time.perf_counter()
    coefficients = solve_elliptic(
        [feature], [interior], case.source, boundaries, [], penalty=config.get("interface_penalty", config.get("penalty", 10.)),
        beta=case.beta, gamma=case.gamma, boundary_penalty=config.get("boundary_penalty", 1.),
        regularization=config.get("regularization", 0.0),
        rank_tolerance=config.get("rank_tolerance"),
        column_normalize_matrix=config.get("column_normalize_matrix", False),
        solver=config.get("solver", "scipy"), device=config.get("device", "cpu"))
    solve_seconds = time.perf_counter() - started
    test, weights = tensor_quadrature(case.dimension, config["error_order"], case.quadrature_rule, case.quadrature_scale)
    valid = case.inside(test)
    test, weights = test[valid], weights[valid]
    errors = relative_errors(test, weights, np.zeros(len(test), dtype=int), [feature], coefficients,
                             lambda points: case.fields(points)[0],
                             lambda points: case.fields(points)[1])
    return dict(example=case.number, method=config.get("method", "RaNN-1net"), neurons=[neurons],
                r_init=radii, feature_neurons=widths, dof=neurons, solve_seconds=solve_seconds,
                quadrature_errors=errors)
