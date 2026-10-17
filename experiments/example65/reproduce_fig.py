"""Write separate Example 6.5 sampling, real-error, and imaginary-error figures."""
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

from examples.example65.problem import region, COMPATIBLE_H1_NORMS
from src.methods.ddrann.features import TanhHarmonicFeatures
from src.plotting import plot_sampling_points

ERRORS = (
    ("l2_mean", r"Absolute $L^2$ errors", "ro", {}),
    ("relative_l2_mean", r"Relative $L^2$ errors", "-", {"color": "darkorange"}),
    ("h1_mean", r"Absolute $H^1$ errors", "bx", {}),
    ("relative_h1_mean", r"Relative $H^1$ errors", "g--", {}),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Example 6.5 scatter and error-order figures.")
    parser.add_argument("--seed", type=int, default=None, help="Override reproduce_fig.seed")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sampling_data(config, seed):
    from examples.example65.problem import sampling_data as generate
    _, interiors, initials, interfaces = generate(dict(config, seed=seed))
    return interiors, [item[1] for item in initials], [item[2] for item in interfaces]


def plot_errors(rows, part, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: int(row["dof"]))
    dofs = np.asarray([int(row["dof"]) for row in rows])
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for family, group in grouped_plot_rows(rows):
        dofs = np.asarray([int(row["dof"]) for row in group])
        for suffix, label, style, kwargs in ERRORS:
            field = f"{part}_{suffix}"
            values = np.asarray([float(row[field]) for row in group])
            if np.any(values <= 0) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid values in {field}")
            extra = f" ({family})" if len(grouped_plot_rows(rows)) > 1 else ""
            axis.plot(dofs, np.log10(values), style, label=label + extra, **kwargs)
    axis.set_xlabel("DoF")
    axis.set_ylabel(r"$\log_{10}(\mathrm{error})$")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    config = json.loads((ROOT / "configs" / "example65.json").read_text(encoding="utf-8"))["reproduce_fig"]
    seed = config["seed"] if args.seed is None else args.seed
    table = ROOT / config["source_table"]
    sampling_output = ROOT / "figures" / "reproduce_figure65_sampling.png"
    real_output = ROOT / "figures" / "reproduce_figure65_real_errors.png"
    imag_output = ROOT / "figures" / "reproduce_figure65_imag_errors.png"
    print(f"sampling figure: {sampling_output} (seed={seed})")
    print(f"real error figure: {real_output} (source={table})")
    print(f"imaginary error figure: {imag_output} (source={table})")
    if args.dry_run:
        return
    if not table.exists():
        raise FileNotFoundError("Run reproduce_table.py before reproduce_fig.py")
    with table.open(newline="", encoding="utf-8") as stream:
        rows = select_plot_rows(list(csv.DictReader(stream)), config)
    if not rows or any(row.get("h1_norm") not in COMPATIBLE_H1_NORMS for row in rows):
        raise ValueError("Run reproduce_table.py again to compute space-time H1 errors before plotting.")
    interiors, initials, interfaces = sampling_data(config, seed)
    plot_sampling_points(
        interiors, None, sampling_output,
        labels=("left far-field SPs", "near-field SPs", "right far-field SPs"),
        boundary_groups=(
            (interfaces[0], r"SPs on $\Gamma_1$", "yellow"),
            (interfaces[1], r"SPs on $\Gamma_2$", "pink"),
            (initials[0], r"SPs on $\Gamma$ in $\Omega_1$", "black"),
            (initials[1], r"SPs on $\Gamma$ in $\Omega_2$", "purple"),
            (initials[2], r"SPs on $\Gamma$ in $\Omega_3$", "orange"),
        ),
        view_limits=config["plot"]["sampling_view"],
        legend_loc=config["plot"]["legend_loc"],
    )
    plot_errors(rows, "real", real_output)
    plot_errors(rows, "imag", imag_output)
    print(f"Wrote {sampling_output}")
    print(f"Wrote {real_output}")
    print(f"Wrote {imag_output}")


if __name__ == "__main__":
    main()
