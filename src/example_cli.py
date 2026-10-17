import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_example_config

def write_result_csv(path, result, seed):
    """Keep a single-run result in one CSV row, including nested diagnostics."""
    row = {"seed": seed}

    def flatten(values, prefix=""):
        for name, value in values.items():
            key = prefix + name
            if isinstance(value, dict):
                flatten(value, "" if key == "quadrature_errors" else key + "_")
            else:
                row[key] = json.dumps(value) if isinstance(value, (list, tuple)) else value

    flatten(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

def main(run):
    parser = argparse.ArgumentParser(description="Run one paper example")
    default_config = ROOT / "configs" / f"example{Path(sys.argv[0]).resolve().parent.name.replace('example', '')}.json"
    parser.add_argument("--config", default=str(default_config),
                        help="configuration file (defaults to the matching example config)")
    args = parser.parse_args()
    config, _ = load_example_config(args.config, "single")
    if isinstance(config.get("seed"), list):
        raise ValueError("single.seed must be an integer")
    config["save_figures"] = bool(config.get("save_figures", True))
    config["save_sampling_figure"] = bool(
        config.get("save_sampling_figure", config["example"] != 61))
    config["save_solution_figure"] = bool(config.get("save_solution_figure", True))
    config["figures_dir"] = str(ROOT / "figures")
    result = run(config)
    output = ROOT / "results" / f"example{config['example']}_seed{config['seed']}.csv"
    write_result_csv(output, result, config['seed'])
    q = result.get("quadrature_errors", {})
    if "relative_l2" in q:
        errors = (
            f"L2={q['l2']:.3e} | RelativeL2={q['relative_l2']:.3e} | "
            f"H1={q['h1']:.3e} | RelativeH1={q['relative_h1']:.3e}"
        )
    else:
        parts = []
        for name in ("real", "imag"):
            values = q.get(name, {})
            parts.append(
                f"{name.capitalize()}[L2={values.get('l2', float('nan')):.3e}, "
                f"RelativeL2={values.get('relative_l2', float('nan')):.3e}, "
                f"H1={values.get('h1', float('nan')):.3e}, "
                f"RelativeH1={values.get('relative_h1', float('nan')):.3e}]"
            )
        errors = " | ".join(parts)
    method = result.get('method', config.get('method'))
    fields = [
        str(method),
        f"m={result.get('neurons')}",
        f"DOF={result.get('dof')}",
        f"r={result.get('r_init')}",
        errors,
        f"SolveTime={result.get('solve_seconds', 0):.6f}s",
    ]
    diagnostics = result.get("solve_diagnostics", {})
    if diagnostics:
        fields.extend([
            f"ConditionSource={diagnostics.get('condition_source')}",
            f"ConditionNumber(A)={diagnostics.get('condition_number', float('nan')):.6e}",
            f"TruncatedConditionNumber(A)={diagnostics.get('truncated_condition_number', float('nan')):.6e}",
            f"SigmaMax(A)={diagnostics.get('singular_value_max', float('nan')):.6e}",
            f"SigmaMin(A)={diagnostics.get('singular_value_min', float('nan')):.6e}",
            f"SigmaMinRetained(A)={diagnostics.get('retained_singular_value_min', float('nan')):.6e}",
            f"SigmaMaxDiscarded(A)={diagnostics.get('discarded_singular_value_max', float('nan')):.6e}",
            f"Rank={diagnostics.get('matrix_rank')}",
            f"MatrixShape={diagnostics.get('matrix_rows')}x{diagnostics.get('matrix_columns')}",
            f"PeakMemory={diagnostics.get('peak_memory_mb', float('nan')):.3f}MiB",
        ])
    print(" | ".join(fields))
    if config["save_figures"]:
        print(f"Figures saved to: {ROOT / 'figures'}")
