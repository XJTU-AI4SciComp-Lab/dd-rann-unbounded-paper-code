"""Single-run entry point for Example 61; PDE logic is kept in reusable modules."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.example61.problem import run as solve_example, METHODS, normalize_method
from src.example_cli import main

METHOD = "DDRaNN-2net"
def run_example(config):
    if normalize_method(config.get("method", METHOD)) not in METHODS:
        raise ValueError("Unsupported Example 6.1 method")
    return solve_example(config)

if __name__ == "__main__":
    main(run_example)
