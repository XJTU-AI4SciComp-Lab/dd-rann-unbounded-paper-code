"""Plotting shared by all numerical methods and examples."""
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ERROR_STYLES = {
    "l2": dict(fmt="ro", label=r"Absolute $L^2$ errors"),
    "relative_l2": dict(fmt="-", color="darkorange", label=r"Relative $L^2$ errors"),
    "h1": dict(fmt="bx", label=r"Absolute $H^1$ errors"),
    "relative_h1": dict(fmt="g--", label=r"Relative $H^1$ errors"),
}


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_sweep_errors(csv_file, output_dir="figures"):
    """Save one log10-error figure per method; complex results use real/imaginary panels."""
    source = Path(csv_file)
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    outputs = []
    for row in rows:
        if not row.get("dof"):
            row["dof"] = str(sum(map(int, row["neurons"].split("-"))))
    for method in dict.fromkeys(row["method"] for row in rows):
        method_rows = [row for row in rows if row["method"] == method]
        prefixes = ("real_", "imag_") if "real_l2" in method_rows[0] else ("",)
        fig, axes = plt.subplots(1, len(prefixes), figsize=(6 * len(prefixes), 4), squeeze=False)
        for axis, prefix in zip(axes[0], prefixes):
            dofs = sorted({int(row["dof"]) for row in method_rows})
            for metric, style in ERROR_STYLES.items():
                means = [np.mean([float(row[prefix + metric]) for row in method_rows
                                  if int(row["dof"]) == dof]) for dof in dofs]
                kwargs = dict(style)
                fmt = kwargs.pop("fmt")
                axis.plot(dofs, np.log10(np.maximum(means, np.finfo(float).tiny)), fmt, **kwargs)
            axis.set_xlabel("Dof")
            axis.set_ylabel(r"$\log_{10}(\mathrm{Error})$")
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            axis.legend(loc="upper right")
        fig.tight_layout()
        output = Path(output_dir) / f"{source.stem}_{_safe_name(method)}_log_errors.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    return outputs


def plot_sampling_points(groups, interface, output, boundary_groups=(), labels=None, view_limits=None,
                         legend_loc="upper left", interface_label=r"Sampling points on $\Gamma$",
                         figure_size=(6, 5), legend_above=False, equal_aspect=False,
                         marker_size=10):
    colors = ("r", "b", "g")
    labels = labels or [f"Sampling points for RaNN{i + 1}" for i in range(len(groups))]
    fig, axis = plt.subplots(figsize=figure_size)
    for points, color, label in zip(groups, colors, labels):
        if len(points):
            axis.scatter(points[:, 0], points[:, 1], color=color, label=label, s=marker_size)
    if interface is not None and len(interface):
        axis.scatter(interface[:, 0], interface[:, 1], color="limegreen",
                     label=interface_label, s=marker_size)
    boundary_colors = ("purple", "orange", "black", "darkorange")
    for index, item in enumerate(boundary_groups):
        points, label, *requested_color = item
        if len(points):
            color = requested_color[0] if requested_color else boundary_colors[index % len(boundary_colors)]
            axis.scatter(points[:, 0], points[:, 1], color=color,
                         label=label, s=marker_size)
    if legend_above:
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                    fontsize=8, framealpha=1., columnspacing=1., borderaxespad=0.)
    else:
        axis.legend(loc=legend_loc)
    if view_limits is not None:
        axis.set_xlim(*view_limits[0])
        axis.set_ylim(*view_limits[1])
    if equal_aspect:
        axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_points_3d(groups, interface, output, labels=None, view_limits=None,
                            legend_loc="upper right"):
    """Plot the three-dimensional sampling cloud used in Example 6.4."""
    colors = ("r", "b", "g")
    labels = labels or [f"Sampling points for RaNN{i + 1}" for i in range(len(groups))]
    fig = plt.figure(figsize=(6, 5))
    axis = fig.add_subplot(111, projection="3d")
    for points, color, label in zip(groups, colors, labels):
        if len(points):
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, label=label, s=7)
    if interface is not None and len(interface):
        axis.scatter(interface[:, 0], interface[:, 1], interface[:, 2], color="limegreen",
                     label=r"Sampling points on $\Gamma$", s=7)
    if view_limits is not None:
        axis.set_xlim(*view_limits[0]); axis.set_ylim(*view_limits[1]); axis.set_zlim(*view_limits[2])
    axis.set_xlabel("x"); axis.set_ylabel("y"); axis.set_zlabel("z")
    axis.legend(loc=legend_loc)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_1d(groups, interface, output, labels=None):
    colors = ("r", "b", "g")
    labels = labels or [f"Sampling points for RaNN{i + 1}" for i in range(len(groups))]
    fig, axis = plt.subplots(figsize=(7, 2.5))
    for row, (points, color, label) in enumerate(zip(groups, colors, labels)):
        points = np.asarray(points).reshape(-1)
        axis.scatter(points, np.full(len(points), row), color=color, label=label, s=10)
    axis.axvline(interface, color="limegreen", linestyle="--", label=r"Interface $\Gamma$")
    axis.set_yticks([])
    axis.set_xlabel("x")
    axis.legend(loc="upper right")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_solution_triptych(points, numerical, exact, output, hole_polygons=(), hole_color="black", error_colorbar_scientific=False):
    points = np.asarray(points)
    values = (np.asarray(numerical), np.asarray(exact), np.abs(np.asarray(numerical) - np.asarray(exact)))
    labels = ("Numerical solution", "Exact solution", "Error")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if points.ndim == 1 or points.shape[1] == 1:
        x = points.reshape(-1)
        order = np.argsort(x)
        for axis, value, label in zip(axes, values, labels):
            axis.plot(x[order], value.reshape(-1)[order], "b-")
            axis.set_title(label)
            axis.set_xlabel("x")
    else:
        for axis, value, label in zip(axes, values, labels):
            image = axis.tricontourf(points[:, 0], points[:, 1], value.reshape(-1), levels=80, cmap="rainbow")
            formatter = None
            if label == "Error" and error_colorbar_scientific:
                from matplotlib.ticker import ScalarFormatter
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((0, 0))
                formatter.set_useOffset(False)
            fig.colorbar(image, ax=axis, format=formatter)
            for polygon in hole_polygons:
                axis.fill(np.asarray(polygon)[:, 0], np.asarray(polygon)[:, 1],
                          color=hole_color, zorder=10)
            axis.set_title(label)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
