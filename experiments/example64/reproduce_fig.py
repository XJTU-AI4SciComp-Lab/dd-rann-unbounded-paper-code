"""Reproduce the two panels of Example 6.4 Figure 6 as separate image files."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.result_metadata import select_plot_rows, grouped_plot_rows

from examples.example64.problem import Problem
from src.methods.ddrann.elliptic_workflow import sampling_data
from src.plotting import ERROR_STYLES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write the Example 6.4 sampling scatter and error-vs-DoF figures."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override reproduce_fig.seed")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sampling_points(config, seed):
    """Reuse the solve's batching and RNG order, including rejection sampling."""
    _, interiors, _, _ = sampling_data(dict(config, seed=seed), Problem())
    return tuple(interiors)


def plot_sampling(near, far, output, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limits = tuple(config["plot"]["sampling_view"][0])
    def visible(points):
        return points[np.all((points >= limits[0]) & (points <= limits[1]), axis=1)]

    near, far = visible(near), visible(far)
    figure = plt.figure(figsize=(6.4, 4.8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(near[:, 0], near[:, 1], near[:, 2], s=10, c="r", alpha=0.5,
                 label="near-field SPs", rasterized=True)
    axis.scatter(far[:, 0], far[:, 1], far[:, 2], s=10, c="b", alpha=0.05,
                 label="far-field SPs", rasterized=True)
    axis.set(xlim=limits, ylim=limits, zlim=limits, xlabel="X", ylabel="Y", zlabel="Z")
    axis.view_init(elev=30, azim=45)
    axis.legend(loc="upper right", fontsize=10)
    # Leave room for 3D axis labels, which tight_layout can clip.
    figure.subplots_adjust(left=.125, right=.9, bottom=.11, top=.88)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def read_error_table(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run reproduce_table_error.py before reproduce_fig.py"
        )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty error table: {path}")
    return sorted(rows, key=lambda row: int(row["dof"]))


def plot_errors(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dofs = np.asarray([int(row["dof"]) for row in rows])
    figure, axis = plt.subplots(figsize=(8, 6))
    figure.subplots_adjust(left=.125, right=.9, bottom=.11, top=.88)
    families = grouped_plot_rows(rows)
    for family, group in families:
        x = np.asarray([int(row["dof"]) for row in group])
        for metric, style in ERROR_STYLES.items():
            field = metric + "_mean"
            values = np.asarray([float(row[field]) for row in group])
            if np.any(values <= 0) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid values in {field}")
            kwargs = dict(style)
            fmt = kwargs.pop("fmt")
            extra = f" ({family})" if len(families) > 1 else ""
            kwargs["label"] += extra
            axis.plot(x, np.log10(values), fmt, **kwargs)
    axis.set_xticks(dofs)
    axis.set_xlabel("DoF")
    axis.set_ylabel(r"$\log_{10}$(Error)")
    axis.grid(True, linestyle="--", alpha=0.6)
    axis.legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main():
    args = parse_args()
    config = json.loads((ROOT / "configs" / "example64.json").read_text(encoding="utf-8"))["reproduce_fig"]
    seed = config["seed"] if args.seed is None else args.seed
    sampling_path = ROOT / "figures" / "reproduce_figure64_sampling.png"
    error_path = ROOT / "figures" / "reproduce_figure64_errors.png"
    table_path = ROOT / config["source_table"]
    print(f"sampling figure: {sampling_path}")
    print(f"error figure: {error_path} (source: {table_path})")
    if args.dry_run:
        return
    rows = select_plot_rows(read_error_table(table_path), config)
    near, far = sampling_points(config, seed)
    plot_sampling(near, far, sampling_path, config)
    plot_errors(rows, error_path)
    print(f"Wrote {sampling_path}")
    print(f"Wrote {error_path}")


if __name__ == "__main__":
    main()
