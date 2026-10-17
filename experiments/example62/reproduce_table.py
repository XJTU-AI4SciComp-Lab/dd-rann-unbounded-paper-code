"""Reproduce the Example 6.2 four-error table and retain every seed result."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.example62.run import run_example
from src.config import parse_seeds, allocation_key, validate_allocations

ERRORS = ("l2", "relative_l2", "h1", "relative_h1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example 6.2 means over configured seeds, with seed-level detail."
    )
    parser.add_argument("--seeds", help="Override the configured seed or inclusive range.")
    parser.add_argument("--fresh", action="store_true", help="Compatibility option; every run already computes fresh results.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_spec(value):
    return int(value) if value and value.isdigit() else value


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def format_errors(row, suffix=""):
    labels = ("L2", "RelativeL2", "H1", "RelativeH1")
    return " | ".join(
        f"{label}={float(row[name + suffix]):.6e}"
        for label, name in zip(labels, ERRORS)
    )


def detail_row(result, seed, config):
    errors = result["quadrature_errors"]
    return {
        "example": result["example"],
        "method": result["method"],
        "neurons": json.dumps(result["neurons"]),
        "r_init": json.dumps(result["r_init"]),
        "dof": result["dof"],
        "seed": seed,
        "interior_points": config["interior_points"],
        "boundary_points": config["boundary_points"],
        "error_order": config["error_order"],
        "penalty": config.get("interface_penalty", config.get("penalty", 1. if 62 == 65 else 10.)),
        "solver": config.get("solver", "scipy"),
        "device": config.get("device", "cpu"),
        **{name: errors[name] for name in ERRORS},
        "solve_seconds": result["solve_seconds"],
    }


def summarize(details, allocations, seeds):
    summary = []
    for neurons in allocations:
        dof = sum(neurons)
        group = [row for row in details if allocation_key(row["neurons"]) == tuple(neurons)]
        actual = sorted(int(row["seed"]) for row in group)
        if actual != sorted(seeds):
            raise RuntimeError(f"DoF={dof}: expected seeds {seeds}, got {actual}")
        summary.append({
            "dof": dof,
            "neurons": json.dumps(neurons),
            "seed_count": len(group),
            "seeds": json.dumps(seeds),
            **{
                f"{name}_mean": float(np.mean([float(row[name]) for row in group]))
                for name in ERRORS
            },
        })
    return summary


def main():
    args = parse_args()
    config = json.loads((ROOT / "configs" / "example62.json").read_text(encoding="utf-8"))
    setup = config["reproduce_table"]
    base = dict(setup, save_figures=False)
    if setup["methods"] != ["DDRaNN-2net"]:
        raise ValueError("Example 6.2 reproduce_table requires DDRaNN-2net")
    allocations = validate_allocations(setup["neuron_pairs"], 2)
    seeds = parse_seeds(seed_spec(args.seeds) if args.seeds else setup["seeds"])
    print(f"allocations={allocations}; seeds={seeds}")
    if args.dry_run:
        print(f"Dry run: {len(allocations) * len(seeds)} solves")
        return

    detail_path = ROOT / "results" / "reproduce_table_error62_detail.csv"
    summary_path = ROOT / "results" / "reproduce_table_error62.csv"
    details = []
    for neurons in allocations:
        for seed in seeds:
            result = run_example(dict(base, neurons=neurons, seed=seed))
            row = detail_row(result, seed, base)
            status = "computed"
            print(f"{status} | DoF={sum(neurons)} | seed={seed} | {format_errors(row)}", flush=True)
            details.append(row)
            write_csv(detail_path, details)

    details.sort(key=lambda row: (int(row["dof"]), allocation_key(row["neurons"]), int(row["seed"])))
    write_csv(detail_path, details)
    summary = summarize(details, allocations, seeds)
    write_csv(summary_path, summary)
    print("\nArithmetic mean errors over all seeds:")
    for row in summary:
        print(f"DoF={row['dof']} | {format_errors(row, '_mean')}")
    print(f"Detail data: {detail_path}")
    print(f"Four-error mean table: {summary_path}")


if __name__ == "__main__":
    main()
