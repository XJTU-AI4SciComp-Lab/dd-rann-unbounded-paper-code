"""Paper Example 6.5."""
import numpy as np

from src.sampling import tensor_quadrature
from src.methods.ddrann.features import TanhHarmonicFeatures
from src.methods.ddrann.spacetime_collocation import solve_complex_system, predict_complex
from src.runtime import validate_runtime
from src.config import validate_neurons

H1_NORM = "broken_spacetime_h1"
# Accept the earlier name for the same norm when reading existing results.
COMPATIBLE_H1_NORMS = {H1_NORM, "broken_spacetime_h1_v1"}


def region(points, cut=2.5):
    return np.where(points[:, 0] < -cut, 0, np.where(points[:, 0] > cut, 2, 1))


def exact(points):
    x, t = points[:, 0], points[:, 1]
    return np.exp(-x*x+1j*t)


def source(points):
    x = points[:, 0]
    return (4*x*x-3)*exact(points)


def sampling_data(config):
    validate_neurons(config['neurons'], 3)
    cut = float(config.get('interface_cut', 2.5))
    t0, t1 = config.get('time_interval', [0., 2.])
    if cut <= 0 or t1 <= t0 or config['time_range'] <= 0:
        raise ValueError('interface_cut and time_range must be positive and time_interval increasing')
    rng = np.random.default_rng(config['seed'])
    radii = config.get('r_init', config.get('ranges', [3, 0.5, 3]))
    if len(radii) != 3 or min(radii) <= 0:
        raise ValueError('r_init must contain three positive radii')
    features = [TanhHarmonicFeatures.random(n, r, config['time_range'], rng) for n, r in zip(config['neurons'], radii)]
    points = np.column_stack((rng.normal(*config.get('interior_distribution', [2., 5.]), config['interior_points']),
                              rng.uniform(t0, t1, config['interior_points'])))
    interiors = [points[region(points, cut)==index] for index in range(3)]
    initial_x = rng.normal(*config.get('initial_distribution', [0., 10.]), config['initial_points'])
    initial = np.column_stack((initial_x, np.full(len(initial_x), t0)))
    initials = [(i, initial[region(initial, cut)==i], exact(initial[region(initial, cut)==i])) for i in range(3)]
    interfaces = []
    for index, x in ((0, -cut), (1, cut)):
        points = np.column_stack((np.full(config['interface_points'], x),
                                  rng.uniform(t0, t1, config['interface_points'])))
        interfaces.append((index, index+1, points))
    return features, interiors, initials, interfaces


def run(config):
    validate_runtime(config)
    if config.get('method', 'DDRaNN-3net') != 'DDRaNN-3net':
        raise ValueError("Example 6.5 supports method='DDRaNN-3net' only")
    cut = float(config.get('interface_cut', 2.5))
    t0, t1 = config.get('time_interval', [0., 2.])
    radii = config.get('r_init', config.get('ranges', [3, .5, 3]))
    features, interiors, initials, interfaces = sampling_data(config)
    import time
    start = time.perf_counter()
    coefficients = solve_complex_system(features, interiors, source, initials, interfaces,
                                        penalty=config.get('penalty', 1.),
                                        real_coefficients=config.get('real_coefficients', False),
                                        solver=config.get('solver', 'scipy'), device=config.get('device', 'cpu'),
                                        rank_tolerance=config.get('rank_tolerance'),
                                        initial_penalty=config.get('initial_penalty', config.get('penalty', 1.)),
                                        interface_penalty=config.get('interface_penalty', config.get('penalty', 1.)))
    seconds = time.perf_counter()-start
    xnodes, xweights = tensor_quadrature(1, config['error_order'], config.get('quadrature_rule', 'mapped_legendre'), config.get('quadrature_scale', 1.))
    tnodes, tweights = np.polynomial.legendre.leggauss(config['error_order'])
    tnodes = (t0 + t1) / 2 + (t1 - t0) * tnodes / 2
    tweights = (t1 - t0) * tweights / 2
    xx, tt = np.meshgrid(xnodes[:, 0], tnodes, indexing='ij')
    points = np.column_stack((xx.ravel(), tt.ravel()))
    weights = (xweights[:, None]*tweights[None, :]).ravel()
    output = np.empty(len(points), complex)
    dx = np.empty(len(points), complex)
    dt = np.empty(len(points), complex)
    for index in range(3):
        mask = region(points, cut)==index
        output[mask] = predict_complex(features, coefficients, points[mask], index)
        dx[mask] = predict_complex(features, coefficients, points[mask], index, 'x')
        dt[mask] = predict_complex(features, coefficients, points[mask], index, 't')
    truth = exact(points)
    truth_dx = -2*points[:, 0]*truth
    truth_dt = 1j*truth
    values = {}
    for name, part in (('real', np.real), ('imag', np.imag)):
        err = part(output-truth); grad_err = part(dx-truth_dx)
        target = part(truth); target_grad = part(truth_dx)
        time_err = part(dt-truth_dt)
        target_time = part(truth_dt)
        l2 = np.sqrt(np.sum(weights*err**2))
        # Broken space-time H1: integrate value, x derivative, and t derivative.
        h1 = np.sqrt(np.sum(weights*(err**2+grad_err**2+time_err**2)))
        values[name] = dict(l2=float(l2), relative_l2=float(l2/np.sqrt(np.sum(weights*target**2))),
                            h1=float(h1), relative_h1=float(h1/np.sqrt(np.sum(weights*(target**2+target_grad**2+target_time**2)))))
    if config.get('save_figures', False):
        from pathlib import Path
        from src.plotting import plot_sampling_points, plot_solution_triptych
        figure_dir = Path(config['figures_dir'])
        method_tag = str(config.get('method', 'DDRaNN-2net')).replace('/', '_')
        neuron_tag = '-'.join(str(n) for n in config['neurons'])
        figure_tag = f"example65_{method_tag}_m{neuron_tag}_seed{config['seed']}"
        interface_points = np.vstack([item[2] for item in interfaces])
        initial_groups = []
        initial_colors = ("black", "purple", "orange")
        for (index, initial_points, _), color in zip(initials, initial_colors):
            initial_groups.append((initial_points, rf"SPs on $\Gamma$ in $\Omega_{index + 1}$", color))
        plot_sampling_points(interiors, None,
                             figure_dir / f'{figure_tag}_sampling_points.png',
                             labels=('left far-field SPs', 'near-field SPs', 'right far-field SPs'),
                             boundary_groups=[(interfaces[0][2], r"SPs on $\Gamma_1$", "yellow"),
                                              (interfaces[1][2], r"SPs on $\Gamma_2$", "pink"),
                                              *initial_groups],
                             view_limits=config.get("plot", {}).get("sampling_view", ((-30., 30.), (t0, t1))),
                             legend_loc=config.get("plot", {}).get("legend_loc", "upper left"))
        plot_limits = config.get("plot", {}).get("sampling_view", [[-30., 30.], [t0, t1]])[0]
        plot_mask = (points[:, 0] >= plot_limits[0]) & (points[:, 0] <= plot_limits[1])
        plot_solution_triptych(points[plot_mask], output.real[plot_mask], truth.real[plot_mask],
                               figure_dir / f'{figure_tag}_solution_comparison.png')
    return dict(example=65, method=config.get('method', 'DDRaNN-3net'), neurons=config['neurons'], r_init=radii,
                dof=sum(config['neurons']), solve_seconds=seconds, h1_norm=H1_NORM,
                quadrature_errors=values)
