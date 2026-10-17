"""Reusable parameter-sweep execution and CSV output."""
import csv
from collections import defaultdict
from pathlib import Path

from src.config import parse_seeds
from src.plotting import plot_sweep_errors


def average_seed_rows(rows, expected_seed_count):
    """Average repeated-seed measurements for each method/DoF pair."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["example"], row["method"], row["neurons"], row["dof"])].append(row)
    averaged = []
    identity = {"example", "method", "neurons", "dof", "seed"}
    for key, group in groups.items():
        required = (expected_seed_count.get(key[1])
                    if isinstance(expected_seed_count, dict) else expected_seed_count)
        seeds = {int(row["seed"]) for row in group}
        if len(group) != required or len(seeds) != required:
            raise ValueError(f"Expected {required} distinct seeds for {key}, got {sorted(seeds)}")
        numeric = [name for name in group[0] if name not in identity]
        summary = dict(zip(("example", "method", "neurons", "dof"), key))
        summary["seed_count"] = required
        summary.update({name: sum(float(row[name]) for row in group) / required
                        for name in numeric})
        averaged.append(summary)
    return averaged


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows):
    """Print the final all-seed arithmetic means as one compact table."""
    identity = ("example", "method", "neurons", "dof", "seed_count")
    metrics = [name for name in rows[0] if name not in identity]
    columns = [*identity, *metrics]
    formatted = []
    for row in rows:
        formatted.append([
            f"{row[name]:.6e}" if isinstance(row[name], float) else str(row[name])
            for name in columns
        ])
    widths = [max(len(name), *(len(row[index]) for row in formatted))
              for index, name in enumerate(columns)]
    print("\nAveraged sweep results (arithmetic mean over all seeds):")
    print("  ".join(name.ljust(width) for name, width in zip(columns, widths)))
    print("  ".join("-" * width for width in widths))
    for row in formatted:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def run_parameter_sweep(single, sweep, run, output, config_adapter=None):
    rows = []
    seeds = parse_seeds(sweep["seeds"])
    deterministic = set(sweep.get("deterministic_methods", ()))
    expected_counts = {method: (1 if method in deterministic else len(seeds))
                       for method in sweep["methods"]}
    for method in sweep["methods"]:
        neuron_sets = sweep.get("method_neurons", {}).get(method, sweep["neuron_pairs"])
        for neurons in neuron_sets:
            method_seeds = seeds[:1] if method in deterministic else seeds
            for seed in method_seeds:
                config = dict(single, method=method, neurons=neurons, seed=seed)
                if config_adapter:
                    config_adapter(config)
                result = run(config)
                errors = result["quadrature_errors"]
                flat_errors = {}
                for name, value in errors.items():
                    if isinstance(value, dict):
                        flat_errors.update({f"{name}_{metric}": number for metric, number in value.items()})
                    else:
                        flat_errors[name] = value
                row = dict(example=result["example"], method=method,
                           neurons="-".join(map(str, neurons)), dof=result["dof"], seed=seed,
                           seconds=result["solve_seconds"], **flat_errors)
                rows.append(row)
                print(row)
    if not rows:
        raise ValueError("The sweep produced no results")
    destination = Path(output)
    raw_destination = destination.with_name(f"{destination.stem}_detail{destination.suffix}")
    averaged = average_seed_rows(rows, expected_counts)
    _write_rows(raw_destination, rows)
    _write_rows(destination, averaged)
    _print_summary(averaged)
    print(f"\nMean results saved to: {destination}")
    print(f"Per-seed results saved to: {raw_destination}")
    plot_sweep_errors(destination, destination.parents[1] / "figures")
    return destination
