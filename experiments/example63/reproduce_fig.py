"""Generate the two Example 6.3 figures: DD mean errors and sampling points."""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.example63.problem import Problem
from src.result_metadata import select_plot_rows, grouped_plot_rows
from src.methods.ddrann.elliptic_workflow import sampling_data

SUMMARY = ROOT / "results" / "reproduce_table63.csv"
ERROR_OUTPUT = ROOT / "figures" / "reproduce_figure_log_errors63.png"
SAMPLING_OUTPUT = ROOT / "figures" / "reproduce_figure_sampling_points63.png"
METHOD = "DD-RaNN-CM"
COMPONENTS = (
    ("l2_mean", "ro", {}, r"Absolute $L^2$ errors"),
    ("relative_l2_mean", "-", {"color": "darkorange"}, r"Relative $L^2$ errors"),
    ("h1_mean", "bx", {}, r"Absolute $H^1$ errors"),
    ("relative_h1_mean", "g--", {}, r"Relative $H^1$ errors"),
)


def new_figure(config):
    """Use the same canvas and axes margins for both paper panels."""
    plot = config.get("plot", {})
    fig, axis = plt.subplots(figsize=plot.get("figure_size", (7.2, 5.2)))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98)
    return fig, axis


def save_figure(fig, output, config):
    """Keep the full canvas so content-dependent cropping cannot change its size."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=config.get("plot", {}).get("figure_dpi", 300),
                bbox_inches=None)
    plt.close(fig)


def draw_mean_errors(rows, config):
    selected = sorted(
        (row for row in rows if row["method"] == METHOD),
        key=lambda row: int(row["dof"]),
    )
    dofs = np.asarray([int(row["dof"]) for row in selected])
    if not selected:
        raise ValueError(f"No {METHOD} results found")
    fig, axis = new_figure(config)
    for family, group in grouped_plot_rows(selected, "allocation"):
        x = np.asarray([int(row["dof"]) for row in group])
        for field, fmt, kwargs, label in COMPONENTS:
            values = np.asarray([float(row[field]) for row in group])
            if np.any(values <= 0) or not np.all(np.isfinite(values)):
                raise ValueError(f"{METHOD} has invalid values in {field}")
            extra = f" ({family})" if len(grouped_plot_rows(selected, "allocation")) > 1 else ""
            axis.plot(x, np.log10(values), fmt, label=label + extra, **kwargs)
    axis.set_xticks(dofs)
    axis.set_xlabel("DoF")
    axis.set_ylabel(r"$\log_{10}$(Error)")
    axis.grid(True, linestyle="--", alpha=0.6)
    axis.legend(loc="upper right")
    save_figure(fig, ERROR_OUTPUT, config)


def split_regions(case, points):
    regions = case.region(points)
    return points[regions == 0], points[regions == 1]


def draw_sampling_points(config, seed=1):
    case = Problem()
    _, (near, far), interface, boundaries = sampling_data(dict(config, seed=seed), case)
    dirichlet_near, dirichlet_far = split_regions(case, boundaries[0][1])
    neumann_near, neumann_far = split_regions(case, boundaries[1][1])
    groups = (
        (near, "red", "near-field SPs"),
        (far, "blue", "far-field SPs"),
        (interface, "green", r"SPs on $\Gamma$"),
        (dirichlet_near, "purple", r"SPs on $\Gamma_D$ in the near field"),
        (dirichlet_far, "orange", r"SPs on $\Gamma_D$ in the far field"),
        (neumann_near, "black", r"SPs on $\Gamma_N$ in the near field"),
        (neumann_far, "darkorange", r"SPs on $\Gamma_N$ in the far field"),
    )
    plot = config.get("plot", {})
    fig, axis = new_figure(config)
    for points, color, label in groups:
        if len(points):
            axis.scatter(points[:, 0], points[:, 1], color=color, label=label,
                         s=plot.get("sampling_marker_size", 6))
    axis.set_xlim(*case.sampling_view[0])
    axis.set_ylim(*case.sampling_view[1])
    if plot.get("sampling_equal_aspect", False):
        axis.set_aspect("equal", adjustable="box")
    axis.legend(loc=case.sampling_legend_loc, ncol=2, fontsize=9, framealpha=1.)
    save_figure(fig, SAMPLING_OUTPUT, config)


def main():
    with (ROOT / "configs" / "example63.json").open(encoding="utf-8") as stream:
        config = json.load(stream)["reproduce_fig"]
    source = ROOT / config["source_table"]
    if not source.exists():
        raise FileNotFoundError("Run reproduce_table.py before reproduce_fig.py")
    with source.open(newline="", encoding="utf-8") as stream:
        rows = select_plot_rows(list(csv.DictReader(stream)), config, "allocation")
    draw_mean_errors(rows, config)
    draw_sampling_points(config, config["seed"])
    print(ERROR_OUTPUT)
    print(SAMPLING_OUTPUT)


if __name__ == "__main__":
    main()
