"""Reproduce all Example 6.1 seed details and the final error table."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.example61.problem import run, normalize_method
from src.config import parse_seeds, allocation_key, validate_allocations as parse_allocations

METHODS = (
    ("RaNN-1net", "single"),
    ("RaNN-1net-mixed", "mixed single"),
    ("DDRaNN-2net", "equal-split double"),
    ("DD-RaNN-decay", "DD-RaNN-decay"),
)
DETERMINISTIC_METHODS = ("FEM-P1-DtN", "Spectral")
METHOD_ORDER = ("single", "mixed single", "equal-split double", "DD-RaNN-decay",
                *DETERMINISTIC_METHODS)


def split_test_elements(target_rows, near_far_ratio):
    """Split tests while retaining the requested full RaNN matrix row count."""
    if target_rows < 3 or near_far_ratio <= 0:
        raise ValueError("target rows must be >= 3 and test_region_ratio must be positive")
    total_elements = target_rows - 1
    near = int(round(total_elements * near_far_ratio / (1.0 + near_far_ratio)))
    near = min(max(near, 1), total_elements - 1)
    return [near, total_elements - near]


def detail_row(result, label, dof, system_rows, neurons, test_elements, seed, setup, allocation):
    error = result["quadrature_errors"]
    return {
        "allocation": json.dumps(allocation),
        "dof": dof,
        "system_rows": system_rows,
        "method": label,
        "seed": seed,
        "neurons": json.dumps(neurons),
        "test_elements": json.dumps(test_elements),
        "solver": setup["solver"],
        "device": setup["device"],
        "l2": error["l2"],
        "relative_l2": error["relative_l2"],
        "h1": error["h1"],
        "relative_h1": error["relative_h1"],
        "solve_seconds": result["solve_seconds"],
        **{"error_"+key: value for key, value in result['error_quadrature'].items()},
    }


def stats(values):
    values = np.asarray(values, dtype=float)
    return (float(np.min(values)), float(np.median(values)),
            float(np.mean(values)),
            float(np.std(values, ddof=1 if len(values) > 1 else 0)))


def summarize(details, allocations):
    rows = []
    for allocation in allocations:
        dof = sum(allocation)
        for method in METHOD_ORDER:
            group = [row for row in details
                     if allocation_key(row["allocation"]) == tuple(allocation)
                     and ("FEM-P1-DtN" if row["method"] == "FEM" else normalize_method(row["method"])) == method]
            if not group:
                continue
            l2 = stats([row["relative_l2"] for row in group])
            h1 = stats([row["relative_h1"] for row in group])
            rows.append({
                "allocation": json.dumps(allocation),
                "seeds": json.dumps([row["seed"] for row in group]),
                "dof": dof,
                "method": method,
                "samples": len(group),
                "relative_l2_best": l2[0],
                "relative_l2_median": l2[1],
                "relative_l2_mean": l2[2],
                "relative_l2_std": l2[3],
                "relative_h1_best": h1[0],
                "relative_h1_median": h1[1],
                "relative_h1_mean": h1[2],
                "relative_h1_std": h1[3],
                **{key: value for key, value in group[0].items() if key.startswith('error_')},
            })
    return rows


def validate_allocations(allocations):
    return parse_allocations(allocations, 2)


def main():
    parser = argparse.ArgumentParser(description="Example 6.1 error table with independent configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the run plan without solving or writing results.")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs" / "example61.json").read_text(encoding="utf-8"))
    setup = dict(config["reproduce_table_error"])
    # Canonical label; retain compatibility with existing configuration files.
    setup["methods"] = [normalize_method(method) for method in setup["methods"]]
    base = dict(setup, save_figures=False)
    expected = [method for method, _ in METHODS] + list(DETERMINISTIC_METHODS)
    if not setup["methods"] or any(method not in expected for method in setup["methods"]):
        raise ValueError(f"Supported methods are {expected}")

    allocations = validate_allocations(setup["double_neurons"])
    dofs = [sum(pair) for pair in allocations]
    seeds = parse_seeds(setup["seeds"])
    if args.dry_run:
        for allocation, dof in zip(allocations, dofs):
            target_rows = int(round(float(setup["rann_row_col_ratio"]) * dof))
            tests = split_test_elements(target_rows, float(setup["test_region_ratio"]))
            print(f"DoF={dof}: allocation={allocation}, rows={target_rows}, tests={tests}; "
                  f"error: {dof} near cells x {setup.get('error_near_order',16)} points, "
                  f"{setup.get('error_order',180)} shifted exterior nodes")
            if "Spectral" in setup["methods"]:
                source_order = setup.get("spectral_source_order")
                source_order = max(180, 2*dof+40) if source_order is None else int(source_order)
                print(f"  Spectral source: laguerre, {source_order} nodes")
        random_methods = sum(method in setup["methods"] for method, _ in METHODS)
        deterministic = sum(method in setup["methods"] for method in DETERMINISTIC_METHODS)
        runs = len(allocations) * (random_methods * len(seeds) + deterministic)
        print(f"Dry run: {runs} solves; seeds={seeds}")
        return

    details = []
    for allocation, dof in zip(allocations, dofs):
        target_rows = int(round(float(setup["rann_row_col_ratio"]) * dof))
        test_elements = split_test_elements(target_rows, float(setup["test_region_ratio"]))
        print(f"DoF={dof}: RaNN={target_rows}x{dof}, double={allocation}, tests={test_elements}")
        for method, label in METHODS:
            if method not in setup["methods"]:
                continue
            for seed in seeds:
                if method in ("DDRaNN-2net", "DD-RaNN-decay"):
                    neurons = allocation
                    method_config = dict(base, method=method, neurons=neurons,
                                         mesh_points=test_elements, seed=seed)
                else:
                    neurons = [dof]
                    method_config = dict(base, method=method, neurons=neurons,
                                         single_mesh_points=target_rows, seed=seed)
                    if method == "RaNN-1net-mixed":
                        method_config["single_feature_neurons"] = allocation
                result = run(method_config)
                stored_tests = (test_elements
                                if method in ("DDRaNN-2net", "DD-RaNN-decay")
                                else [target_rows])
                details.append(detail_row(result, label, dof, target_rows, neurons,
                                          stored_tests, seed, setup, allocation))

        for method in DETERMINISTIC_METHODS:
            if method not in setup["methods"]:
                continue
            elements = dof
            method_config = dict(base, method=method, neurons=[dof],
                                 fem_elements=elements, seed=seeds[0])
            result = run(method_config)
            details.append(detail_row(result, method, dof, dof, [dof],
                                      [elements] if method != "Spectral" else [],
                                      seeds[0], setup, allocation))

    summary = summarize(details, allocations)
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "reproduce_table_error61_detail.csv"
    summary_path = output / "reproduce_table_error61.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"Detail data: {detail_path}")
    print(f"Summary table: {summary_path}")


if __name__ == "__main__":
    main()
