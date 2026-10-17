"""Single-run entry point for Example 65; PDE logic is kept in reusable modules."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.example65.problem import run as solve_example
from src.example_cli import main

METHOD = "DDRaNN-3net"
def run_example(config):
    if config.get("method") != METHOD:
        raise ValueError(f"Example 65 requires method={METHOD} for this entry point")
    return solve_example(config)

if __name__ == "__main__":
    main(run_example)
