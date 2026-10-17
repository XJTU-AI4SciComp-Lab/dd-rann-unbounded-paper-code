# Domain-Decomposed Randomized Neural Networks Paper Code

This repository contains the paper code for **Domain-Decomposed Randomized Neural Networks for Partial Differential Equations in Unbounded Domains**.

It provides the numerical experiments for semi-unbounded, fully unbounded, perforated-domain, and time-dependent partial differential equations using domain-decomposed randomized neural networks.

## Paper Information

- Title: Domain-Decomposed Randomized Neural Networks for Partial Differential Equations in Unbounded Domains
- Authors: Haixin Wang, Haoning Dang, Fei Wang, and Shimin Guo
- Journal / preprint: arXiv preprint
- DOI / arXiv: [arXiv:2606.31342](https://arxiv.org/abs/2606.31342)
- Corresponding author: Fei Wang, Xi'an Jiaotong University
- Code maintainer: Haixin Wang
- Repository: - Repository: https://github.com/XJTU-AI4SciComp-Lab/sfrbe-rann-adaptive-paper-code

## Method Summary

This repository implements domain-decomposed randomized neural network methods for solving partial differential equations in unbounded domains. Hidden-layer parameters are randomly generated and fixed, while output-layer coefficients are computed from linear least-squares systems. The implementation includes a semi-unbounded Petrov--Galerkin formulation and collocation formulations for perforated, fully unbounded, three-dimensional, and time-dependent problems. Near-field and far-field subnetworks are coupled through physical boundary and interface equations.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── configs/
├── src/
│   └── methods/
│       ├── ddrann/
│       ├── fem/
│       └── spectral/
├── examples/
│   ├── example61/
│   ├── example62/
│   ├── example63/
│   ├── example64/
│   └── example65/
├── experiments/
│   ├── example61/
│   ├── example62/
│   ├── example63/
│   ├── example64/
│   └── example65/
├── data/
├── results/
├── figures/
├── tests/
└── docs/
```
## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ddrann-paper-code
```

Or install dependencies with pip:

```bash
python -m pip install -r requirements.txt
```

The optional PyTorch backend can be installed separately:

```bash
python -m pip install torch
```

## Quick Start

Run a configured example from the repository root:

```bash
python -m examples.example61.run --config configs/example61.json
```

The corresponding entry points for Examples 6.2--6.5 are:

```bash
python -m examples.example62.run --config configs/example62.json
python -m examples.example63.run --config configs/example63.json
python -m examples.example64.run --config configs/example64.json
python -m examples.example65.run --config configs/example65.json
```

Run the complete regression suite:

```bash
python -m unittest discover -s tests -v
```

## Reproduce Results

Run commands from the repository root. The paper-scale table scripts are organized under `experiments/example61/` through `experiments/example65/`.

### Tables

```bash
python -m experiments.example61.reproduce_table_error61
python -m experiments.example61.reproduce_table_farfield61
python -m experiments.example61.reproduce_table_sensitivity61
python -m experiments.example62.reproduce_table
python -m experiments.example63.reproduce_table
python -m experiments.example64.reproduce_table_error
python -m experiments.example64.reproduce_table_rcond
python -m experiments.example64.reproduce_table_sample
python -m experiments.example65.reproduce_table
```

### Figures

Run the matching table script first when the figure script requires its CSV output:

```bash
python -m experiments.example61.reproduce_figure_seed_errors61
python -m experiments.example62.reproduce_fig
python -m experiments.example63.reproduce_fig
python -m experiments.example64.reproduce_fig
python -m experiments.example65.reproduce_fig
```

Generated CSV summaries and seed-level results are written to `results/`. Generated plots and sampling figures are written to `figures/`. The mapping between paper tables, figures, configurations, and scripts is documented in [`docs/reproduction.md`](docs/reproduction.md).

Examples 6.4 and 6.5 can require substantial CPU time and memory. The JSON files in `configs/` record the seeds, sampling counts, quadrature rules, solver settings, and output paths used by the experiments. The figure scripts do not silently launch missing expensive table computations.

The tanh trial spaces in Examples 6.1 and 6.5 do not enforce zero limits at spatial infinity by default, so their whole-domain errors are finite-quadrature estimates. Example 6.1 supports the `asymptotic_correction` option for the decay-constrained far-field variant described in [`docs/method.md`](docs/method.md).

## Folder Description

- `src/`: reusable configuration, solver, sampling, error, plotting, and method implementations.
- `src/methods/ddrann/`: domain-decomposed randomized neural network formulations and features.
- `src/methods/fem/`: FEM-P1-DtN comparison implementation.
- `src/methods/spectral/`: Laguerre spectral comparison implementation.
- `examples/`: problem definitions and single-run entry points for Examples 6.1--6.5.
- `experiments/`: scripts for reproducing paper tables, sensitivity studies, and figures.
- `configs/`: numerical parameters, seeds, solver choices, and output settings.
- `data/`: data documentation; the current examples do not require external datasets.
- `results/`: generated numerical summaries and seed-level CSV files.
- `figures/`: generated paper figures and sampling plots.
- `tests/`: regression tests for numerical and pipeline behavior.
- `docs/`: method notes and reproduction instructions.

## Data

No external dataset is required for the included experiments. Manufactured solutions, source terms, geometries, and sampling rules are defined in `examples/`, `src/`, and `configs/`.


## Results

Generated CSV summaries and seed-level results should be saved in `results/`. Generated figures should be saved in `figures/`.


## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@misc{wang2026ddrann,
  title         = {Domain-Decomposed Randomized Neural Networks for Partial Differential Equations in Unbounded Domains},
  author        = {Wang, Haixin and Dang, Haoning and Wang, Fei and Guo, Shimin},
  year          = {2026},
  eprint        = {2606.31342},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2606.31342}
}
```

The machine-readable citation metadata is also provided in `CITATION.cff`.

## License

This project is released under the BSD 3-Clause License. See `LICENSE`.

## Contact

Fei Wang  
School of Mathematics and Statistics, Xi'an Jiaotong University.

