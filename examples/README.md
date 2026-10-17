# Examples

Each example has a standalone `run.py` entry point. Run commands from the repository root and pass the matching JSON configuration from `configs/`:

```bash
python -m examples.example61.run --config configs/example61.json
python -m examples.example62.run --config configs/example62.json
python -m examples.example63.run --config configs/example63.json
python -m examples.example64.run --config configs/example64.json
python -m examples.example65.run --config configs/example65.json
```

The JSON files keep the numerical parameters, seeds, solver choices, and output settings in one place. Single-run outputs are written to `results/`; figures are written to `figures/` when enabled by the selected configuration.
