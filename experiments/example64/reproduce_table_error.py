"""Reproduce the Example 6.4 error table with seed-level diagnostics."""
import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.example64.run import run_example
from src.config import parse_seeds, validate_allocations


MEAN_METRICS = (
    "l2",
    "relative_l2",
    "h1",
    "relative_h1",
    "solve_seconds",
    "condition_number",
    "effective_condition_number",
    "matrix_rank",
    "peak_memory_mb",
)


def seed_argument(value):
    return int(value) if value.isdigit() else value


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def detail_row(result, seed, **labels):
    errors = result["quadrature_errors"]
    diagnostics = result["solve_diagnostics"]
    row = {
        **labels,
        "example": result["example"],
        "method": result["method"],
        "dof": result["dof"],
        "neurons": json.dumps(result["neurons"]),
        "seed": seed,
        "l2": errors["l2"],
        "relative_l2": errors["relative_l2"],
        "h1": errors["h1"],
        "relative_h1": errors["relative_h1"],
        "solve_seconds": result["solve_seconds"],
        # Both values are for A itself, never A.T @ A.
        "condition_number": diagnostics["condition_number"],
        "effective_condition_number": diagnostics["truncated_condition_number"],
        "singular_value_max": diagnostics["singular_value_max"],
        "singular_value_min": diagnostics["singular_value_min"],
        "retained_singular_value_min": diagnostics["retained_singular_value_min"],
        "discarded_singular_value_max": diagnostics["discarded_singular_value_max"],
        "matrix_rank": diagnostics["matrix_rank"],
        "matrix_rows": diagnostics["matrix_rows"],
        "matrix_columns": diagnostics["matrix_columns"],
        "peak_memory_mb": diagnostics["peak_memory_mb"],
        "condition_source": diagnostics["condition_source"],
        "rank_tolerance": diagnostics["rank_tolerance"],
    }
    return row


def mean_rows(details, group_fields):
    """Return arithmetic seed means, preserving the requested group order."""
    groups = {}
    for row in details:
        key = tuple(row[field] for field in group_fields) + (row["neurons"],)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        row = {field: value for field, value in zip(group_fields, key)}
        row.update({
            "dof": group[0]["dof"],
            "neurons": group[0]["neurons"],
            "seed_count": len(group),
            "seeds": json.dumps([item["seed"] for item in group]),
        })
        for metric in MEAN_METRICS:
            row[f"{metric}_mean"] = float(np.mean([item[metric] for item in group]))
        summary.append(row)
    return summary


def load_setup(section):
    config = json.loads((ROOT / "configs" / "example64.json").read_text(encoding="utf-8"))
    base = dict(
        config[section],
        save_figures=False,
    )
    return base, config[section]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example 6.4 error table: rcond=1e-7 and arithmetic means over seeds 1-10."
    )
    parser.add_argument("--seeds", help="Override the configured inclusive seed range.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base, setup = load_setup("reproduce_table_error")
    allocations = validate_allocations(setup["neuron_pairs"], 2)
    seeds = parse_seeds(seed_argument(args.seeds) if args.seeds else setup["seeds"])
    rcond = float(setup["rcond"])
    print(f"DoFs={[sum(pair) for pair in allocations]}; seeds={seeds}; rcond={rcond:.0e}")
    if args.dry_run:
        print(f"Dry run: {len(allocations) * len(seeds)} solves")
        return

    detail_path = ROOT / "results" / "reproduce_table_error64_detail.csv"
    summary_path = ROOT / "results" / "reproduce_table_error64.csv"
    details = []
    for neurons in allocations:
        for seed in seeds:
            gc.collect()
            result = run_example(dict(base, neurons=neurons, seed=seed, rank_tolerance=rcond))
            row = detail_row(result, seed, rcond=rcond)
            details.append(row)
            write_csv(detail_path, details)  # checkpoint after every expensive solve
            print(row, flush=True)

    # rcond is fixed here, so one mean row is required per allocation rather than one global row.
    summary = []
    for neurons in allocations:
        group = [row for row in details if json.loads(row["neurons"]) == neurons]
        rows = mean_rows(group, ("rcond",))
        if len(group) != len(seeds):
            raise RuntimeError(f"incomplete seed group for DoF={sum(neurons)}")
        summary.extend(rows)
    write_csv(summary_path, summary)
    print(f"Detail data: {detail_path}")
    print(f"Mean table: {summary_path}")


if __name__ == "__main__":
    main()
