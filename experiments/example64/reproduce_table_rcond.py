"""Example 6.4 sensitivity to the relative SVD cutoff used by GELSD."""
import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.example64.run import run_example
from experiments.example64.reproduce_table_error import (
    detail_row,
    load_setup,
    mean_rows,
    seed_argument,
    write_csv,
)
from src.config import parse_seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example 6.4 rcond sensitivity using configured neurons and seeds."
    )
    parser.add_argument("--seeds", help="Override the configured inclusive seed range.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base, setup = load_setup("reproduce_table_rcond")
    neurons = [int(value) for value in setup["neurons"]]
    if len(neurons) != 2 or min(neurons) <= 0:
        raise ValueError("neurons must contain two positive entries")
    seeds = parse_seeds(seed_argument(args.seeds) if args.seeds else setup["seeds"])
    rconds = [float(value) for value in setup["rconds"]]
    print(f"DoF={sum(neurons)}; seeds={seeds}; rconds={rconds}")
    if args.dry_run:
        print(f"Dry run: {len(rconds) * len(seeds)} solves")
        return

    detail_path = ROOT / "results" / "reproduce_table_rcond64_detail.csv"
    summary_path = ROOT / "results" / "reproduce_table_rcond64.csv"
    details = []
    for rcond in rconds:
        for seed in seeds:
            gc.collect()
            result = run_example(dict(
                base, neurons=neurons, seed=seed, rank_tolerance=rcond
            ))
            row = detail_row(result, seed, rcond=rcond)
            details.append(row)
            write_csv(detail_path, details)
            print(row, flush=True)

    summary = mean_rows(details, ("rcond",))
    write_csv(summary_path, summary)
    print(f"Detail data: {detail_path}")
    print(f"Mean table: {summary_path}")


if __name__ == "__main__":
    main()
