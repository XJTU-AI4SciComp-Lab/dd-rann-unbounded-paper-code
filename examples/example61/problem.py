"""Paper Example 6.1 on [0, infinity)."""
import time
import numpy as np
from scipy.special import roots_hermite

from src.methods.ddrann.features import TanhFeatures
from src.methods.ddrann.petrov_galerkin import solve_pg, solve_pg_single
from src.methods.spectral.laguerre import solve_halfline
from src.methods.fem.fem_dtn import solve_fem_dtn
from src.errors import (halfline_errors, halfline_cut, halfline_split_quadrature,
                        halfline_error_metadata)
from src.runtime import validate_runtime


def exact(x):
    return 1/(x+1)


def exact_gradient(x):
    return -1/(x+1)**2


def source(x):
    return 1/(x+1)-2/(x+1)**3


METHOD_ALIASES = {"FEM": "FEM-P1-DtN", "P1-FEM": "FEM-P1-DtN",
                  "FEM-DtN": "FEM-P1-DtN", "P1-FEM-DtN": "FEM-P1-DtN",
                  "DDRaNN-decay": "DD-RaNN-decay",
                  }
METHODS = ("DDRaNN-2net", "DD-RaNN-decay", "RaNN-1net", "RaNN-1net-mixed",
           "Spectral", "FEM-P1-DtN")


def normalize_method(method):
    return METHOD_ALIASES.get(method, method)


def run(config):
    config = dict(config, method=normalize_method(config.get("method", "DDRaNN-2net")))
    # DD-RaNN-decay is the hard-constrained positive-half-line variant.  Keep
    # the flag available for controlled comparisons with the original method.
    asymptotic_correction = (config['method'] == 'DD-RaNN-decay'
                             or bool(config.get("asymptotic_correction", False)))
    config['interface'] = halfline_cut(config)
    validate_runtime(config)
    if config['method'] not in METHODS:
        raise ValueError("unsupported Example 6.1 method")
    if config['method'] == 'FEM-P1-DtN':
        return solve_fem_dtn(config, exact, exact_gradient, source)
    if config.get('method', 'DDRaNN-2net') == 'Spectral':
        return solve_halfline(config, exact, exact_gradient, source)
    rng = np.random.default_rng(config['seed'])
    if config.get('method') in ('RaNN-1net', 'RaNN-1net-mixed'):
        neurons = int(sum(config['neurons']))
        radius = (config.get('single_mixed_r', [0.7, 0.2])
                  if config.get('method') == 'RaNN-1net-mixed'
                  else config['single_r'])
        if np.isscalar(radius):
            feature = TanhFeatures.random(
                neurons, float(radius), rng,
                asymptotic_correction=asymptotic_correction)
            single_radii = [float(radius)]
        else:
            single_radii = [float(value) for value in radius]
            widths = config.get("single_feature_neurons")
            if widths is None and len(config['neurons']) == len(single_radii):
                widths = list(config['neurons'])
            if widths is None:
                widths = [neurons // len(single_radii)] * len(single_radii)
                for index in range(neurons % len(single_radii)):
                    widths[index] += 1
            if len(widths) != len(single_radii) or min(widths) <= 0 or sum(widths) != neurons:
                raise ValueError("single_feature_neurons must match radii and sum to the total neurons")
            parts = [TanhFeatures.random(
                         width, value, rng,
                         asymptotic_correction=asymptotic_correction)
                     for width, value in zip(widths, single_radii)]
            feature = TanhFeatures(
                np.concatenate([part.weight for part in parts]),
                np.concatenate([part.bias for part in parts]),
                asymptotic_correction=asymptotic_correction)
        single_mesh_points = int(config.get('single_mesh_points', neurons))
        hermite, _ = roots_hermite(2 * single_mesh_points)
        mesh = np.r_[0.0, config['scaling'] * hermite[hermite > 0]]
        start = time.perf_counter()
        coefficients = solve_pg_single(feature, mesh, source, float(config.get('dirichlet_value', 1.)),
                                       config['quadrature_order'], config.get('solver', 'scipy'),
                                       config.get('device', 'cpu'),
                                       config.get('constraint_weights', {}).get('dirichlet', 1.0),
                                       config.get('test_region_weights', [1.0])[0])
        seconds = time.perf_counter() - start
        features = [feature]
        active_neurons = [neurons]
        cut = np.inf
        radii = single_radii
    else:
        radii = config.get('r_init', config.get('ranges', [0.7, 0.2]))
        # The near-field network is unchanged.  On the positive half-line the
        # far-field network is the one whose basis must decay at infinity.
        features = [TanhFeatures.random(
                        n, r, rng,
                        asymptotic_correction=asymptotic_correction and index == 1)
                    for index, (n, r) in enumerate(zip(config['neurons'], radii))]
        active_neurons = list(config['neurons'])
        cut = config['interface']
        near = np.linspace(0, cut, config['mesh_points'][0]+1)
        hermite, _ = roots_hermite(2*config['mesh_points'][1])
        far = np.r_[cut, cut+config['scaling']*hermite[hermite>0]]
        start = time.perf_counter()
        coefficients = solve_pg(features, (near, far), source, float(config.get('dirichlet_value', 1.)), cut, config['quadrature_order'],
                                 config.get('solver', 'scipy'), config.get('device', 'cpu'),
                                 config.get('constraint_weights'),
                                 config.get('test_region_weights'))
        seconds = time.perf_counter()-start
    x, w = halfline_split_quadrature(config, sum(active_neurons))
    values = np.empty_like(x); gradients = np.empty_like(x)
    regions = ((0, np.ones(len(x), dtype=bool)),) if len(features) == 1 else ((0, x<cut), (1, x>=cut))
    for index, mask in regions:
        values[mask] = features[index].evaluate(x[mask])@coefficients[index]
        gradients[mask] = features[index].evaluate(x[mask], True)@coefficients[index]
    errors = halfline_errors(x, w, values, gradients, exact, exact_gradient)
    far_field_values = None
    if 'far_field_points' in config:
        far_points = np.asarray(config['far_field_points'], dtype=float)
        if far_points.ndim != 1 or np.any(~np.isfinite(far_points)) or np.any(far_points < 0):
            raise ValueError('far_field_points must be a one-dimensional nonnegative finite array')
        far_regions = (((0, np.ones(len(far_points), dtype=bool)),) if len(features) == 1 else
                       ((0, far_points < cut), (1, far_points >= cut)))
        far_field_values = np.empty(len(far_points))
        for index, mask in far_regions:
            far_field_values[mask] = features[index].evaluate(far_points[mask]) @ coefficients[index]
    if config.get('save_figures', False):
        from pathlib import Path
        from src.plotting import plot_sampling_1d, plot_solution_triptych
        figure_dir = Path(config['figures_dir'])
        method_tag = str(config.get('method', 'DDRaNN-2net')).replace('/', '_')
        neuron_tag = '-'.join(str(n) for n in config['neurons'])
        figure_tag = f"example61_{method_tag}_m{neuron_tag}_seed{config['seed']}"
        if config.get('save_sampling_figure', False) and len(features) == 2:
            plot_sampling_1d((near, far), cut, figure_dir / f'{figure_tag}_sampling_points.png',
                             labels=('near-field SPs', 'far-field SPs'))
        if config.get('save_solution_figure', True):
            plot = config.get("plot", {})
            plot_x = np.linspace(*plot.get("solution_view", [0, 100]), plot.get("solution_grid_points", 1000))
            plot_value = np.empty_like(plot_x)
            plot_regions = ((0, np.ones(len(plot_x), dtype=bool)),) if len(features) == 1 else (
                (0, plot_x < cut), (1, plot_x >= cut))
            for index, mask in plot_regions:
                plot_value[mask] = features[index].evaluate(plot_x[mask]) @ coefficients[index]
            plot_solution_triptych(plot_x, plot_value, exact(plot_x),
                                   figure_dir / f'{figure_tag}_solution_comparison.png')
    far_field_limits = [float(feature.far_field_limit @ coefficient)
                        for feature, coefficient in zip(features, coefficients)]
    corrected_far_field_limits = [float((feature.far_field_limit - feature.asymptotic_shift)
                                        @ coefficient)
                                 for feature, coefficient in zip(features, coefficients)]
    result = dict(example=61, method=config.get('method', 'DDRaNN-2net'), neurons=active_neurons,
                  r_init=radii, dof=sum(active_neurons), solve_seconds=seconds,
                  asymptotic_correction=asymptotic_correction,
                  far_field_limits=far_field_limits,
                  corrected_far_field_limits=corrected_far_field_limits,
                  error_quadrature=halfline_error_metadata(config, sum(active_neurons)),
                  quadrature_errors=errors)
    if far_field_values is not None:
        result['far_field_values'] = far_field_values.tolist()
    return result
