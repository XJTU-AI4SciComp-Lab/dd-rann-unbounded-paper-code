"""Run the Example 6.1 two-network sensitivity study from configuration."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.example61.problem import run
from experiments.example61.reproduce_table_error61 import split_test_elements
from src.config import parse_seeds


BASELINE_PARAMETERS = ("x_cut", "radii", "scaling", "eta", "xi")
SWEEPS = (
    "x_cut",
    "r1_fixed_r2",
    "r2_fixed_r1",
    "radii_joint",
    "scaling",
    "eta",
    "xi",
)
SETTING_KEYS = {
    "x_cut": "x_cut",
    "r1_fixed_r2": "radii",
    "r2_fixed_r1": "radii",
    "radii_joint": "radii",
    "scaling": "scaling",
    "eta": "eta",
    "xi": "xi",
}
DISPLAY_NAMES = {
    "x_cut": "x_cut",
    "r1_fixed_r2": "r1 (r2 fixed)",
    "r2_fixed_r1": "r2 (r1 fixed)",
    "radii_joint": "(r1, r2) joint",
    "scaling": "S",
    "eta": "eta (interface)",
    "xi": "xi (boundary)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-factor-at-a-time sensitivity test for Example 6.1."
    )
    parser.add_argument(
        "--seeds",
        help="Override configured seeds with one integer or an inclusive range, e.g. 1-5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the experiment matrix without solving it.",
    )
    return parser.parse_args()


def command_line_seeds(value):
    """Accept the integer spelling that argparse necessarily returns as text."""
    if value is not None and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return value


def normalized_value(parameter, value):
    if parameter == "radii":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("each radii value must be [r1, r2]")
        value = [float(item) for item in value]
        if min(value) <= 0:
            raise ValueError("r1 and r2 must be positive")
        return value
    value = float(value)
    if value <= 0:
        raise ValueError(f"{parameter} values must be positive")
    return value


def validate_setup(setup):
    if setup.get("method") != "DDRaNN-2net":
        raise ValueError("sensitivity study only supports DDRaNN-2net")
    neurons = [int(value) for value in setup["neurons"]]
    if len(neurons) != 2 or min(neurons) <= 0:
        raise ValueError("sensitivity neurons must contain two positive entries")

    baseline = setup["baseline"]
    sequences = setup["sequences"]
    if set(baseline) != set(BASELINE_PARAMETERS):
        raise ValueError(f"baseline must contain exactly {BASELINE_PARAMETERS}")
    if set(sequences) != set(SWEEPS):
        raise ValueError(f"sequences must contain exactly {SWEEPS}")

    normalized_baseline = {
        parameter: normalized_value(parameter, baseline[parameter])
        for parameter in BASELINE_PARAMETERS
    }
    normalized_sequences = {}
    for sweep in SWEEPS:
        setting_key = SETTING_KEYS[sweep]
        values = sequences[sweep]
        if not values:
            raise ValueError(f"{sweep} sensitivity sequence must not be empty")
        normalized_sequences[sweep] = [
            normalized_value(setting_key, value) for value in values
        ]
        if normalized_baseline[setting_key] not in normalized_sequences[sweep]:
            raise ValueError(f"{sweep} sequence must include its baseline value")
    return neurons, normalized_baseline, normalized_sequences


def experiment_cases(baseline, sequences):
    cases = []
    for sweep in SWEEPS:
        setting_key = SETTING_KEYS[sweep]
        for level, value in enumerate(sequences[sweep], start=1):
            settings = dict(baseline)
            settings[setting_key] = value
            cases.append((sweep, level, value, settings))
    return cases


def penalty_weights(settings, defaults=None):
    """Map the paper's [A; xi B; eta C] notation to solver row weights."""
    weights = dict(defaults or {})
    weights.update(
        dirichlet=settings["xi"],
        interface_value=settings["eta"],
        interface_flux=settings["eta"],
    )
    return weights


def result_row(result, parameter, level, varied_value, settings, baseline, seed,
               neurons, test_elements):
    errors = result["quadrature_errors"]
    return {
        "dof": sum(neurons),
        "method": "DDRaNN-2net",
        "sweep_parameter": parameter,
        "level": level,
        "varied_value": json.dumps(varied_value, separators=(",", ":")),
        "is_baseline": settings == baseline,
        "seed": seed,
        "neurons": json.dumps(neurons),
        "test_elements": json.dumps(test_elements),
        "x_cut": settings["x_cut"],
        "r1": settings["radii"][0],
        "r2": settings["radii"][1],
        "scaling": settings["scaling"],
        "eta": settings["eta"],
        "xi": settings["xi"],
        "l2": errors["l2"],
        "relative_l2": errors["relative_l2"],
        "h1": errors["h1"],
        "relative_h1": errors["relative_h1"],
        "solve_seconds": result["solve_seconds"],
    }


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {
        "best": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1 if len(values) > 1 else 0)),
    }


def summarize(details):
    summary = []
    for parameter in SWEEPS:
        levels = sorted({row["level"] for row in details
                         if row["sweep_parameter"] == parameter})
        for level in levels:
            group = [row for row in details
                     if row["sweep_parameter"] == parameter and row["level"] == level]
            row = {
                "dof": group[0]["dof"],
                "method": "DDRaNN-2net",
                "sweep_parameter": parameter,
                "level": level,
                "varied_value": group[0]["varied_value"],
                "samples": len(group),
            }
            for metric in ("relative_l2", "relative_h1"):
                for name, value in statistics([item[metric] for item in group]).items():
                    row[f"{metric}_{name}"] = value
            summary.append(row)
    return summary


def main():
    args = parse_args()
    config = json.loads((ROOT / "configs" / "example61.json").read_text(encoding="utf-8"))
    setup = config["reproduce_table_sensitivity"]
    neurons, baseline, sequences = validate_setup(setup)
    seed_spec = command_line_seeds(args.seeds) if args.seeds is not None else setup["seeds"]
    seeds = parse_seeds(seed_spec)
    if not seeds:
        raise ValueError("at least one seed is required")

    target_rows = int(round(float(setup["rann_row_col_ratio"]) * sum(neurons)))
    test_elements = split_test_elements(target_rows, float(setup["test_region_ratio"]))
    cases = experiment_cases(baseline, sequences)
    print(f"DDRaNN-2net: DoF={sum(neurons)}, neurons={neurons}, rows={target_rows}, tests={test_elements}")
    print(f"Seeds: {seeds}")
    for parameter in SWEEPS:
        print(f"{DISPLAY_NAMES[parameter]}: {sequences[parameter]}")
    if args.dry_run:
        print(f"Dry run: {len(cases)} table entries, {len(cases) * len(seeds)} reported runs")
        return

    common = dict(setup, method="DDRaNN-2net", neurons=neurons,
                  mesh_points=test_elements, save_figures=False)
    cache = {}
    details = []
    for parameter, level, varied_value, settings in cases:
        print(f"{DISPLAY_NAMES[parameter]} level {level}/{len(sequences[parameter])}: {varied_value}")
        for seed in seeds:
            cache_key = (json.dumps(settings, sort_keys=True), seed)
            if cache_key not in cache:
                weights = penalty_weights(settings, setup.get("constraint_weights"))
                run_config = dict(
                    common,
                    seed=seed,
                    interface=settings["x_cut"],
                    r_init=settings["radii"],
                    scaling=settings["scaling"],
                    constraint_weights=weights,
                )
                cache[cache_key] = run(run_config)
            details.append(result_row(cache[cache_key], parameter, level, varied_value,
                                      settings, baseline, seed, neurons, test_elements))

    summary = summarize(details)
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "reproduce_table_sensitivity61_detail.csv"
    summary_path = output / "reproduce_table_sensitivity61.csv"
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
    print(f"Unique solves: {len(cache)} (identical baseline cases were reused)")


if __name__ == "__main__":
    main()
