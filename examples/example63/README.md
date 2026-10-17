# Example 6.3

This directory implements the strong collocation formulation for the
two-dimensional full-space exterior problem in Example 6.3.

The paper-table comparison is generated with:

```powershell
python -m experiments.example63.reproduce_table
```

The `reproduce_table` block in `configs/example63.json` records the paper
settings: 4,000 interior points, 500 points on each physical boundary and the
near--far interface, penalty 10, mapped-Legendre error order 50, radii
`(0.5, 10)`, and seeds 1--30. It compares:

- one full-domain network whose features all use `r=3` and no ridge regularization;
- one full-domain network mixing `DoF-50` features with `r=0.5` and 50
  features with `r=10`;
- the paper's near--far two-network method with the same mixed feature counts.

All three methods use the same strong PDE and physical-boundary collocation
equations. Only the two-network method adds interface matching equations. The
final table reports the arithmetic mean, sample standard deviation (`ddof=1`),
and best (minimum) value for absolute/relative L2 and H1 errors. Seed-level values are retained in
`results/reproduce_table63_detail.csv`.

After generating the table data, draw the DD-RaNN-CM mean logarithmic error
curve with:

```powershell
python -m experiments.example63.reproduce_fig
```

Only the two-network method is plotted. Component styles follow the paper
exactly: red circles for absolute L2, a dark-orange solid line for relative L2,
blue crosses for absolute H1, and a green dashed line for relative H1.

The same command also reproduces the sampling-point scatter plot from the
2026-06-30 paper. It uses seed 1 and the same 4,000 interior points, 500 interface
points, and 500 points on each physical boundary as the table experiment. Its
colors, component labels, and viewing window follow the paper figure.
