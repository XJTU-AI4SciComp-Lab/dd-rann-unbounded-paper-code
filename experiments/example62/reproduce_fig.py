"""Write separate Example 6.2 sampling and mean-error figures."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import configure_case
from src.result_metadata import select_plot_rows, grouped_plot_rows

from examples.example62.problem import Problem
from src.methods.ddrann.features import GaussianFeatures
from src.plotting import plot_sampling_points
from src.sampling import filtered_points, radial_points, sphere_points

ERRORS = (
    ("l2_mean", r"Absolute $L^2$ errors", "ro", {}),
    ("relative_l2_mean", r"Relative $L^2$ errors", "-", {"color": "darkorange"}),
    ("h1_mean", r"Absolute $H^1$ errors", "bx", {}),
    ("relative_h1_mean", r"Relative $H^1$ errors", "g--", {}),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Example 6.2 scatter and error-order figures.")
    parser.add_argument("--seed", type=int, default=None, help="Override reproduce_fig.seed")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sampling_data(config, seed):
    case = configure_case(Problem(), config)
    rng = np.random.default_rng(seed)
    for neurons, radius in zip(config["neurons"], config["r_init"]):
        GaussianFeatures.random(case.dimension, neurons, radius, rng, config.get("feature_center_range", 1.))
    mean, std = case.radial_distribution
    interior = filtered_points(
        config["interior_points"],
        lambda count: radial_points(count, case.dimension, mean, std, rng, case.angular_range),
        case.inside,
    )
    regions = case.region(interior)
    groups = [interior[regions == index] for index in range(2)]
    interface = filtered_points(
        config.get("interface_points", config["boundary_points"]),
        lambda count: sphere_points(count, case.dimension, case.interface_radius, rng, case.angular_range),
        case.inside,
    )
    boundaries = case.boundary_conditions(config["boundary_points"], rng)
    return case, groups, interface, boundaries


def plot_sampling_reference(case, groups, interface, boundaries, output):
    """Match Fig. 4: 6.4 x 4.8 in, fixed default axes and 10 pt fonts."""
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 10, "figure.autolayout": False,
                         "figure.constrained_layout.use": False}):
        figure, axis = plt.subplots(figsize=(6.4, 4.8))
        figure.subplots_adjust(left=.125, right=.9, bottom=.11, top=.88)
        for points, color, label in zip(groups, ("red", "blue"),
                                        ("near-field SPs", "far-field SPs")):
            axis.scatter(points[:, 0], points[:, 1], color=color, label=label, s=10)
        axis.scatter(interface[:, 0], interface[:, 1], color="limegreen",
                     label=case.interface_plot_label, s=10)
        for points, label, color in case.plot_boundary_groups(boundaries):
            axis.scatter(points[:, 0], points[:, 1], color=color, label=label, s=10)
        axis.set_xlim(*case.sampling_view[0])
        axis.set_ylim(*case.sampling_view[1])
        axis.legend(loc="upper left", fontsize=10)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300)
        plt.close(figure)


def plot_errors(rows, output):
    """Match Fig. 4: 8 x 6 in, fixed default axes and 10 pt fonts."""
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda row: int(row["dof"]))
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans"],
                         "font.size": 10, "axes.unicode_minus": False,
                         "figure.autolayout": False, "figure.constrained_layout.use": False}):
        figure, axis = plt.subplots(figsize=(8, 6))
        figure.subplots_adjust(left=.125, right=.9, bottom=.11, top=.88)
        families = grouped_plot_rows(rows)
        for family, group in families:
            dofs = np.asarray([int(row["dof"]) for row in group])
            for field, label, style, kwargs in ERRORS:
                values = np.asarray([float(row[field]) for row in group])
                if np.any(values <= 0) or not np.all(np.isfinite(values)):
                    raise ValueError(f"invalid values in {field}")
                extra = f" ({family})" if len(families) > 1 else ""
                axis.plot(dofs, np.log10(values), style, label=label + extra, **kwargs)
        axis.set_xlabel("Dof")
        axis.set_ylabel("Log10(error)")
        axis.grid(True, linestyle="--", color="0.69", linewidth=.8)
        axis.legend(loc="upper right", fontsize=10)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300)
        plt.close(figure)


def main():
    args = parse_args()
    config = json.loads((ROOT / "configs" / "example62.json").read_text(encoding="utf-8"))["reproduce_fig"]
    seed = config["seed"] if args.seed is None else args.seed
    table = ROOT / config["source_table"]
    sampling_output = ROOT / "figures" / "reproduce_figure62_sampling.png"
    error_output = ROOT / "figures" / "reproduce_figure62_errors.png"
    print(f"sampling figure: {sampling_output} (seed={seed})")
    print(f"error figure: {error_output} (source={table})")
    if args.dry_run:
        return
    if not table.exists():
        raise FileNotFoundError("Run reproduce_table.py before reproduce_fig.py")
    with table.open(newline="", encoding="utf-8") as stream:
        rows = select_plot_rows(list(csv.DictReader(stream)), config)
    case, groups, interface, boundaries = sampling_data(config, seed)
    plot_sampling_reference(case, groups, interface, boundaries, sampling_output)
    plot_errors(rows, error_output)
    print(f"Wrote {sampling_output}")
    print(f"Wrote {error_output}")


if __name__ == "__main__":
    main()
