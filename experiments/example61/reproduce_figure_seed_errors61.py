"""Draw relative-L2 and relative-H1 seed stem figures for the available DoFs."""
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import allocation_key
from src.result_metadata import select_plot_rows
from examples.example61.problem import normalize_method
DETAIL_CANDIDATES = (
    ROOT / "results" / "reproduce_table_error61_detail.csv",
    # Backward compatibility for results created before the script rename.
    ROOT / "results" / "reproduce_table61_detail.csv",
)
METHODS = (
    ("single", r"RaNN-1net, $r=0.5$", "#2878B5", "o"),
    ("mixed single", r"RaNN-1net, mixed $r=0.7,0.2$", "#F39B2F", "^"),
    ("equal-split double", r"DD-RaNN-2net, $r_1=0.7,r_2=0.2$", "#2A9D67", "s"),
    ("DD-RaNN-decay", r"DD-RaNN-decay", "#7B2CBF", "D"),
)
METRICS = {
    "l2": ("relative_l2", r"$\log_{10}(e_{L^2}^{\mathrm{rel}})$"),
    "h1": ("relative_h1", r"$\log_{10}(e_{H^1}^{\mathrm{rel}})$"),
}
BASELINES = (("FEM-P1-DtN", "#B22222", "--"),
             ("Spectral", "#555555", ":"))
METHOD_NAMES = {"single": "RaNN-1net", "mixed single": "RaNN-1net-mixed",
                "equal-split double": "DDRaNN-2net", "DD-RaNN-decay": "DD-RaNN-decay"}


def canonical_method(method):
    return normalize_method(METHOD_NAMES.get(method, method))


def configured_rows(rows, config):
    """Ignore disabled methods even when old CSV files still contain them."""
    enabled = config.get("methods")
    if enabled is None:
        return rows
    enabled = {canonical_method(method) for method in enabled}
    return [row for row in rows if canonical_method(row["method"]) in enabled]


def configured_baselines(config):
    """Use the selected reference group in the order of the parameter list."""
    styles = {entry[0]: entry for entry in BASELINES}
    selected = config.get("methods", list(styles))
    names = dict.fromkeys(canonical_method(method) for method in selected)
    return [styles[name] for name in names if name in styles]


def interval_edges(lower, upper, reference_values):
    """Partition by reference lines only; integer gridlines do not split counts."""
    return np.unique(np.r_[lower, reference_values, upper])


def draw(rows, metric, config):
    rows = configured_rows(rows, config)
    reference_group = configured_baselines(config)
    available = {canonical_method(row["method"]) for row in rows}
    methods = [entry for entry in METHODS if canonical_method(entry[0]) in available]
    if not methods:
        raise ValueError("Seed figures require at least one enabled RaNN method with source results")
    field, ylabel = METRICS[metric]
    allocations = sorted({allocation_key(row["allocation"]) for row in rows}, key=lambda n: (sum(n), n))
    if not allocations:
        raise ValueError("detail data must contain at least one DoF")
    fig, axes = plt.subplots(len(methods), len(allocations), figsize=(4.3 * len(allocations), 3.4 * len(methods)),
                             sharex=True, squeeze=False, constrained_layout=True)
    max_seed = max(int(row["seed"]) for row in rows)
    min_seed = min(int(row["seed"]) for row in rows)
    span = max(max_seed - min_seed, 1)
    divider, annotation = max_seed + .02 * span, max_seed + .07 * span
    titles = {"single": f"RaNN-1net, r={config['single_r']}",
              "mixed single": f"RaNN-1net, mixed r={config['single_mixed_r']}",
              "equal-split double": f"DD-RaNN-2net, r={config['r_init']}",
              "DD-RaNN-decay": "DD-RaNN-decay"}
    for column, allocation in enumerate(allocations):
        dof = sum(allocation)
        axes[0, column].set_title(f"DoF = {dof}, m={allocation}", fontsize=12, pad=8)
        baselines = []
        for label, color, style in reference_group:
            baseline = next((row for row in rows
                             if allocation_key(row["allocation"]) == allocation
                             and canonical_method(row["method"]) == label), None)
            if baseline is not None:
                baselines.append((label, color, style, np.log10(float(baseline[field]))))
            elif "methods" in config:
                plt.close(fig)
                raise ValueError(f"Missing selected reference {label} at allocation {allocation}; recompute the source table")
        # The first selected reference starts the stems. With no baseline,
        # start stems at the lower integer log-error bound of each panel.
        for row_index, (method, title, color, marker) in enumerate(methods):
            ax = axes[row_index, column]
            group = sorted((row for row in rows
                            if allocation_key(row["allocation"]) == allocation
                            and canonical_method(row["method"]) == canonical_method(method)),
                           key=lambda row: int(row["seed"]))
            seeds = np.asarray([int(row["seed"]) for row in group])
            values = np.log10([float(row[field]) for row in group])
            if not group:
                plt.close(fig)
                raise ValueError(f"Missing source results for {METHOD_NAMES[method]} at allocation {allocation}")
            bounds = np.r_[values, [b[3] for b in baselines]]
            lower = int(np.floor(bounds.min()))
            upper = int(np.ceil(bounds.max()))
            if lower == upper:
                lower, upper = lower - 1, upper + 1
            reference_value = baselines[0][3] if baselines else lower
            edges = interval_edges(lower, upper, [b[3] for b in baselines])

            ax.vlines(seeds, reference_value, values, color=color, alpha=0.28,
                      linewidth=0.65, zorder=1)
            ax.scatter(seeds, values, s=16, marker=marker, color=color,
                       edgecolors="white", linewidths=0.25, alpha=0.88, zorder=3)
            for _, baseline_color, style, baseline_value in baselines:
                ax.axhline(baseline_value, color=baseline_color, linestyle=style,
                           linewidth=1.25, zorder=2)
            for integer in range(lower, upper + 1):
                ax.axhline(integer, color="0.86", linewidth=0.65, zorder=0)

            ax.axvline(divider, color="0.82", linewidth=0.7)
            for index in range(len(edges) - 1):
                left, right = edges[index], edges[index + 1]
                closed = index == len(edges) - 2
                count = int(np.sum((values >= left) &
                                   (values <= right if closed else values < right)))
                ax.text(annotation, (left + right) / 2.0, f"n={count}", color=color,
                        ha="center", va="center", fontsize=7.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                                  edgecolor="none", alpha=0.82))
            ax.set_xlim(min_seed - .02 * span, max_seed + .14 * span)
            ax.set_ylim(upper, lower)
            ax.set_yticks(np.arange(lower, upper + 1, 1))
            ax.set_xticks(np.unique(np.linspace(min_seed, max_seed, min(6, max_seed - min_seed + 1), dtype=int)))
            ax.grid(axis="x", color="0.94", linewidth=0.45)
            ax.tick_params(labelsize=8)
            if column == 0:
                ax.set_ylabel(f"{titles[method]}\n{ylabel}", fontsize=9.5)
            if row_index == len(methods) - 1:
                ax.set_xlabel("Seed", fontsize=10)

    present = [b for b in reference_group if any(
        canonical_method(row["method"]) == b[0]
        for row in rows)]
    if present:
        fig.legend([plt.Line2D([0], [0], color=color, linestyle=style, linewidth=1.25)
                    for _, color, style in present],
                   ["P1-FEM-DtN" if label == "FEM-P1-DtN" else label
                    for label, _, _ in present],
                   loc="outside lower center", ncol=len(present), frameon=False)
    destination = ROOT / "figures" / f"reproduce_figure_seed_errors61_{metric}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(destination)


def main():
    config = json.loads((ROOT / "configs" / "example61.json").read_text(encoding="utf-8"))["reproduce_figure_seed_errors"]
    detail = ROOT / config["source_detail"]
    if not detail.exists():
        raise FileNotFoundError("Run reproduce_table_error61.py before plotting")
    with detail.open(newline="", encoding="utf-8") as stream:
        rows = select_plot_rows(list(csv.DictReader(stream)), config, "allocation")
    for metric in ("l2", "h1"):
        draw(rows, metric, config)


if __name__ == "__main__":
    main()
