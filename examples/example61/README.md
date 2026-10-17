# Example 6.1

This directory implements the half-line problem

`-u''+u=f` on `[0,infinity)`, `u(0)=1`, with exact solution `u=1/(x+1)`.

The draft-paper settings are stored in `configs/example61.json`: `xcut=45`,
`S=10`, two tanh feature ranges `(0.7,0.2)`, five Gauss-Legendre points per
test-function element, and 180 Laguerre-Gauss points for error evaluation.

Run one configured case with:

```powershell
python examples/example61/run.py
```

For a budget `M`, the comparison uses `M` features for one RaNN, `M/2+M/2`
features for the two-network methods, and `M` free FEM nodal values. The
`DD-RaNN-decay` variant uses the same two-network allocation as `DDRaNN-2net`
but subtracts the positive-infinity limit from every far-field `tanh` feature.
Randomized methods are reported over all configured seeds; FEM is deterministic.

The final six-method table is reproduced with:

```powershell
python experiments/example61/reproduce_table_error61.py
```

Its complete numerical setup is the `reproduce_table_error` block in
`configs/example61.json`. Set `double_neurons` to the two-network allocation;
the script uses their sum as the common DoF of both single networks and FEM.
`rann_row_col_ratio` determines the RaNN test-row count by
`round(ratio * DoF)`, while `test_region_ratio` splits those tests between the
near and far regions. The latter does not change FEM. Feature radii (`r_init`,
`single_r`, and `single_mixed_r`), constraint/test-region weights, quadrature,
solver, device, and seeds are also declared in that block.

The script writes the presentation-ready statistics to
`results/reproduce_table_error61.csv`, and all seed-level values to
`results/reproduce_table_error61_detail.csv`.

After generating the table data, draw the relative-L2 and relative-H1 stem
figures with:

```powershell
python experiments/example61/reproduce_figure_seed_errors61.py
```

To inspect the far-field values without computing errors, use the selected-seed
table (default seed 1):

```powershell
python experiments/example61/reproduce_table_farfield61.py
```

Use `--seeds 1-5` or `--seeds 1,3` as needed. The table reports the numerical
values at `x=100` and `x=1000`, together with the basis-implied far-field limit;
the FEM limit is shown as `-`.

The two-network, DoF-400 one-factor-at-a-time sensitivity table is generated
with:

```powershell
python experiments/example61/reproduce_table_sensitivity61.py
```

Its parameter sequences and baseline are declared in the
`reproduce_table_sensitivity` block in `configs/example61.json`. Statistics
use seeds 1--100 by default. The default
three-level sequences are `x_cut = [30, 45, 60]`,
three radius groups (vary `r1` with `r2=0.2`, vary `r2` with `r1=0.7`, and
vary both from small to large), `S = [5, 10, 20]`, and `[0.1, 1, 10]` for
both penalty parameters. Following equation (3.11),
`xi` weights the Dirichlet boundary equation, while `eta` weights both the
solution-value and flux-continuity equations at the interface. All other
settings remain at the declared baseline while one parameter is varied.
