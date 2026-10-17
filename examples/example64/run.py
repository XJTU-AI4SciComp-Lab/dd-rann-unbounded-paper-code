"""Single-run entry point for Example 64; PDE logic is kept in reusable modules."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.example64.problem import Problem
from src.methods.ddrann.elliptic_workflow import run_elliptic_case
from src.example_cli import main

METHOD = "DDRaNN-2net"
def run_example(config):
    if config.get("method") != METHOD:
        raise ValueError(f"Example 64 requires method={METHOD} for this entry point")
    return run_elliptic_case(config, Problem())

if __name__ == "__main__":
    main(run_example)
