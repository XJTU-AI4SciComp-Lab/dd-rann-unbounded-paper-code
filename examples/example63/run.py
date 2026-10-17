"""Single-run entry point for Example 63; PDE logic is kept in reusable modules."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.example63.problem import Problem
from src.methods.ddrann.elliptic_workflow import run_elliptic_case, run_single_elliptic_case
from src.example_cli import main

METHOD = "DDRaNN-2net"
def run_example(config):
    if config.get("method") == METHOD:
        return run_elliptic_case(config, Problem())
    if config.get("method") in ("RaNN-1net", "RaNN-1net-mixed"):
        config = dict(config)
        allocation = list(config["neurons"])
        if config["method"] == "RaNN-1net-mixed":
            config["single_r"] = config.get("single_mixed_r", config["single_r"])
            if len(allocation) > 1:
                config.setdefault("single_feature_neurons", allocation)
        config["neurons"] = [sum(allocation)]
        return run_single_elliptic_case(config, Problem())
    raise ValueError("Example 63 requires DDRaNN-2net, RaNN-1net, or RaNN-1net-mixed")

if __name__ == "__main__":
    main(run_example)
