"""Write selected-seed far-field values for all Example 6.1 methods."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.example61.problem import exact, normalize_method, run
from src.config import parse_seeds
from experiments.example61.reproduce_table_error61 import split_test_elements


RANDOM_METHODS = (
    ("DDRaNN-2net", "equal-split double"),
    ("DD-RaNN-decay", "DD-RaNN-decay"),
)
DETERMINISTIC_METHODS = ("FEM-P1-DtN", "Spectral")
ALL_METHODS = tuple(method for method, _ in RANDOM_METHODS) + DETERMINISTIC_METHODS
DISPLAY_NAMES = {"DDRaNN-2net": "DD-RaNN-PG", "FEM-P1-DtN": "P1-FEM-DtN"}
CANONICAL_NAMES = {DISPLAY_NAMES.get(method, method): method
                   for method, _ in RANDOM_METHODS}


def parse_selected_seeds(value):
    """Accept the shared seed syntax plus a convenient single integer string."""
    if isinstance(value, str) and re.fullmatch(r"\s*[+-]?\d+\s*", value):
        return [int(value)]
    if isinstance(value, str) and re.fullmatch(r"\s*[+-]?\d+(\s*,\s*[+-]?\d+)+\s*", value):
        return parse_seeds([int(item.strip()) for item in value.split(',')])
    return parse_seeds(value)


def theoretical_far_field_limit(method, result):
    """Return the basis-implied limit; FEM is deliberately left unreported."""
    if method == "FEM-P1-DtN":
        return "-"
    if method == "Spectral":
        return 0.0
    return float(result["corrected_far_field_limits"][-1])


def selected_setup(base, method, allocation, seed, target_rows, test_elements, points):
    setup = dict(base, method=method, seed=seed, far_field_points=list(points),
                 save_figures=False)
    dof = sum(allocation)
    if method in ("DDRaNN-2net", "DD-RaNN-decay"):
        setup.update(neurons=list(allocation), mesh_points=test_elements)
    elif method in ("RaNN-1net", "RaNN-1net-mixed"):
        setup.update(neurons=[dof], single_mesh_points=target_rows)
        if method == "RaNN-1net-mixed":
            setup["single_feature_neurons"] = list(allocation)
    elif method == "FEM-P1-DtN":
        setup.update(neurons=[dof], fem_elements=dof)
    elif method == "Spectral":
        setup.update(neurons=[dof])
    else:
        raise ValueError(f"Unsupported Example 6.1 method: {method}")
    return setup


def build_rows(config, seeds, allocations, points):
    if len(points) != 2 or not np.allclose(points, [100.0, 1000.0]):
        raise ValueError("far_field_points must be exactly [100, 1000]")
    base = dict(config)
    rows = []
    for allocation in allocations:
        dof = sum(allocation)
        target_rows = int(round(float(base["rann_row_col_ratio"]) * dof))
        test_elements = split_test_elements(target_rows, float(base["test_region_ratio"]))
        for method in ALL_METHODS:
            if method not in base["methods"]:
                continue
            method_seeds = seeds if method in dict(RANDOM_METHODS) else seeds[:1]
            for seed in method_seeds:
                setup = selected_setup(base, method, allocation, seed, target_rows,
                                       test_elements, points)
                result = run(setup)
                values = result.get("far_field_values")
                if values is None or len(values) != 2:
                    raise RuntimeError(f"Method {method} did not return two far-field values")
                rows.append({
                    "allocation": json.dumps(allocation),
                    "dof": dof,
                    "method": DISPLAY_NAMES.get(method, method),
                    "seed": seed,
                    "value_at_100": float(values[0]),
                    "value_at_1000": float(values[1]),
                    "theoretical_far_field_limit": theoretical_far_field_limit(method, result),
                })
        exact_values = exact(np.asarray(points, dtype=float))
        rows.append({
            "allocation": json.dumps(allocation),
            "dof": dof,
            "method": "Exact solution",
            "seed": "-",
            "value_at_100": float(exact_values[0]),
            "value_at_1000": float(exact_values[1]),
            "theoretical_far_field_limit": 0.0,
        })
    return rows


def select_representative_rows(rows, selection_metrics=None):
    """Select one coherent row per method and DoF from the seed sweep.

    The selected row is the one whose configured metric is closest to its
    seed-wise median. Ties are resolved by the smallest seed. Deterministic
    methods and the exact solution are retained directly.
    """
    selection_metrics = selection_metrics or {
        "DDRaNN-2net": "theoretical_far_field_limit",
        "DD-RaNN-decay": "value_at_1000",
    }
    grouped = {}
    for row in rows:
        key = (row["allocation"], row["dof"], row["method"])
        grouped.setdefault(key, []).append(row)

    selected = []
    for key, group in grouped.items():
        method = CANONICAL_NAMES.get(group[0]["method"])
        metric = selection_metrics.get(method) if method is not None else None
        if metric is None:
            chosen = dict(group[0])
            chosen["selection_metric"] = "deterministic"
            chosen["selection_median"] = ""
        else:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            target = float(np.median(values))
            distance = np.abs(values - target)
            nearest = np.flatnonzero(np.isclose(distance, distance.min(),
                                                 rtol=0., atol=1.e-15))
            index = min(nearest, key=lambda item: int(group[int(item)]["seed"]))
            chosen = dict(group[int(index)])
            chosen["selection_metric"] = metric
            chosen["selection_median"] = target
        selected.append(chosen)

    def order(row):
        method_order = {name: index for index, name in enumerate(
            ["DD-RaNN-PG", "DD-RaNN-decay", "P1-FEM-DtN", "Spectral", "Exact solution"])}
        return (int(row["dof"]), method_order.get(row["method"], 99))

    return sorted(selected, key=order)


def main():
    parser = argparse.ArgumentParser(description="Example 6.1 selected-seed far-field table")
    parser.add_argument("--seeds", default=None,
                        help="integer list/range such as 1,3 or 1-5 (default: configured seeds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the plan without solving or writing results")
    args = parser.parse_args()

    data = json.loads((ROOT / "configs" / "example61.json").read_text(encoding="utf-8"))
    config = dict(data["reproduce_table_error"])
    farfield_config = dict(data.get("reproduce_table_farfield", {}))
    config["methods"] = [normalize_method(method) for method in config["methods"]]
    seed_spec = farfield_config.get("seeds", "1") if args.seeds is None else args.seeds
    seeds = parse_selected_seeds(seed_spec)
    allocations = [list(pair) for pair in farfield_config.get("double_neurons", config["double_neurons"])]
    points = [float(value) for value in farfield_config.get("points", [100.0, 1000.0])]
    enabled_random = sum(method in config["methods"] for method, _ in RANDOM_METHODS)
    enabled_deterministic = sum(method in config["methods"] for method in DETERMINISTIC_METHODS)
    runs = len(allocations) * (enabled_random * len(seeds) + enabled_deterministic)
    if args.dry_run:
        print(f"Far-field points: x=100, 1000; allocations={allocations}; seeds={seeds}")
        print(f"Dry run: {runs} solves")
        return

    rows = build_rows(config, seeds, allocations, points)
    selected = select_representative_rows(rows, farfield_config.get("selection_metrics"))
    output = ROOT / "results" / "reproduce_table_farfield61.csv"
    detail_output = ROOT / "results" / "reproduce_table_farfield61_detail.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with detail_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)
    print(f"Far-field table: {output}")
    print(f"Seed detail table: {detail_output}")
    print("allocation | dof | method | seed | u(100) | u(1000) | theoretical limit")
    for row in selected:
        print(f"{row['allocation']} | {row['dof']} | {row['method']} | {row['seed']} | "
              f"{row['value_at_100']:.8e} | {row['value_at_1000']:.8e} | "
              f"{row['theoretical_far_field_limit']}")


if __name__ == "__main__":
    main()
