"""Reproduce the complete three-method statistics table for Example 6.3."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.example63.run import run_example
from src.config import parse_seeds, allocation_key, validate_allocations as parse_allocations


METHODS = (
    ("RaNN-1net", "single"),
    ("RaNN-1net-mixed", "mixed single"),
    ("DDRaNN-2net", "DD-RaNN-CM"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the three-method mean/std/best table for Example 6.3."
    )
    parser.add_argument(
        "--seeds",
        help="Override the configured seeds with an integer or inclusive range, e.g. 1-2.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print the experiment matrix without solving it.",
    )
    return parser.parse_args()


def command_line_seeds(value):
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def validate_allocations(allocations):
    return parse_allocations(allocations, 2)


def detail_row(result, label, allocation, seed, setup):
    errors = result["quadrature_errors"]
    radii = result.get("r_init", [])
    feature_neurons = result.get("feature_neurons", allocation)
    return {
        "allocation": json.dumps(allocation),
        "dof": result["dof"],
        "method": label,
        "seed": seed,
        "feature_neurons": json.dumps(feature_neurons),
        "radii": json.dumps(radii),
        "l2": errors["l2"],
        "relative_l2": errors["relative_l2"],
        "h1": errors["h1"],
        "relative_h1": errors["relative_h1"],
        "solve_seconds": result["solve_seconds"],
    }


def result_key(row):
    return allocation_key(row.get("allocation", row["feature_neurons"])), row["method"], int(row["seed"])


def summarize(details, allocations, seeds, methods=METHODS):
    rows = []
    for allocation in allocations:
        dof = sum(allocation)
        for _, label in methods:
            group = [row for row in details
                     if allocation_key(row["allocation"]) == tuple(allocation) and row["method"] == label]
            actual_seeds = sorted({int(row["seed"]) for row in group})
            if len(group) != len(seeds) or actual_seeds != sorted(seeds):
                raise ValueError(
                    f"Expected seeds {sorted(seeds)} for DoF={dof}, method={label}; "
                    f"got {actual_seeds}"
                )
            metrics = {
                name: np.asarray([float(row[name]) for row in group], dtype=float)
                for name in ("l2", "relative_l2", "h1", "relative_h1", "solve_seconds")
            }
            rows.append({
                "allocation": json.dumps(allocation),
                "seeds": json.dumps(seeds),
                "dof": dof,
                "method": label,
                "seed_count": len(group),
                "l2_mean": float(np.mean(metrics["l2"])),
                "l2_std": float(np.std(metrics["l2"], ddof=1 if len(group) > 1 else 0)),
                "l2_best": float(np.min(metrics["l2"])),
                "relative_l2_mean": float(np.mean(metrics["relative_l2"])),
                "relative_l2_std": float(np.std(metrics["relative_l2"], ddof=1 if len(group) > 1 else 0)),
                "relative_l2_best": float(np.min(metrics["relative_l2"])),
                "h1_mean": float(np.mean(metrics["h1"])),
                "h1_std": float(np.std(metrics["h1"], ddof=1 if len(group) > 1 else 0)),
                "h1_best": float(np.min(metrics["h1"])),
                "relative_h1_mean": float(np.mean(metrics["relative_h1"])),
                "relative_h1_std": float(np.std(metrics["relative_h1"], ddof=1 if len(group) > 1 else 0)),
                "relative_h1_best": float(np.min(metrics["relative_h1"])),
                "solve_seconds_mean": float(np.mean(metrics["solve_seconds"])),
                "solve_seconds_std": float(np.std(metrics["solve_seconds"], ddof=1 if len(group) > 1 else 0)),
            })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = json.loads((ROOT / "configs" / "example63.json").read_text(encoding="utf-8"))
    setup = config["reproduce_table"]
    base = dict(setup, save_figures=False)
    expected_methods = [method for method, _ in METHODS]
    if not setup["methods"] or any(method not in expected_methods for method in setup["methods"]):
        raise ValueError(f"Supported methods are {expected_methods}")
    active_methods = [(method, label) for method, label in METHODS if method in setup["methods"]]
    allocations = validate_allocations(setup["double_neurons"])
    seed_spec = command_line_seeds(args.seeds)
    seeds = parse_seeds(setup["seeds"] if seed_spec is None else seed_spec)
    if not seeds:
        raise ValueError("at least one seed is required")


    print(f"Seeds: {seeds}")
    print(f"Allocations: {allocations}")
    if args.dry_run:
        print(f"Dry run: {len(allocations) * len(active_methods) * len(seeds)} runs")
        return

    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "reproduce_table63_detail.csv"
    details = []
    for allocation in allocations:
        dof = sum(allocation)
        print(f"DoF={dof}, paper allocation={allocation}")
        for method, label in active_methods:
            for seed in seeds:
                if method == "DDRaNN-2net":
                    method_config = dict(base, method=method, neurons=allocation, seed=seed)
                elif method == "RaNN-1net-mixed":
                    method_config = dict(
                        base, method=method, neurons=[dof], seed=seed,
                        single_r=setup["single_mixed_r"],
                        single_feature_neurons=allocation,
                    )
                else:
                    method_config = dict(
                        base, method=method, neurons=[dof], seed=seed,
                        single_r=setup["single_r"],
                    )
                    method_config.pop("single_feature_neurons", None)
                result = run_example(method_config)
                details.append(detail_row(result, label, allocation, seed, setup))
                print(details[-1])
            print(f"DoF={dof}, method={label}: computed={len(seeds)}")
            write_csv(detail_path, details)

    expected = len(allocations) * len(active_methods) * len(seeds)
    if len(details) != expected or len({result_key(row) for row in details}) != expected:
        raise ValueError(f"Expected {expected} distinct results, got {len(details)}")
    details.sort(key=lambda row: (int(row["dof"]), allocation_key(row["allocation"]), row["method"], int(row["seed"])))
    summary = summarize(details, allocations, seeds, active_methods)
    summary_path = output / "reproduce_table63.csv"
    write_csv(detail_path, details)
    write_csv(summary_path, summary)
    print(f"Detail data: {detail_path}")
    print(f"Mean +- std and best table: {summary_path}")


if __name__ == "__main__":
    main()
