import json
import re
from pathlib import Path


def load_example_config(path, section='single'):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if section in data:
        config = dict(data[section])
        config.setdefault('example', data.get('example'))
        return config, data
    if 'single' in data:
        raise KeyError(f'Configuration section {section!r} is missing')
    # Backward-compatible flat configuration.
    return data, data



def allocation_key(neurons):
    return tuple(int(value) for value in (json.loads(neurons) if isinstance(neurons, str) else neurons))


def validate_neurons(neurons, count):
    values = list(neurons)
    if len(values) != count or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in values):
        raise ValueError(f'neurons must contain {count} positive integers')
    return values


def validate_allocations(allocations, count):
    values = [validate_neurons(item, count) for item in allocations]
    if not values or len({tuple(item) for item in values}) != len(values):
        raise ValueError('allocations must be nonempty and contain no duplicate allocations')
    return values


def configure_case(case, config):
    """Apply the current entry point's geometry, sampling, and quadrature options."""
    case.interface_radius = float(config.get('interface_radius', case.interface_radius))
    case.radial_distribution = tuple(config.get('radial_distribution', case.radial_distribution))
    case.quadrature_rule = config.get('quadrature_rule', case.quadrature_rule)
    case.quadrature_scale = float(config.get('quadrature_scale', 1.0))
    case.angular_range = tuple(config.get('angular_range', [0.0, 2 * 3.141592653589793]))
    case.axis_boundary_limit = float(config.get('axis_boundary_limit', 50.0))
    case.axis_boundary_points = config.get('axis_boundary_points')
    case.beta = float(config.get('beta', 1.0))
    case.gamma = float(config.get('gamma', 1.0))
    if case.interface_radius <= 0 or case.radial_distribution[1] <= 0 or case.quadrature_scale <= 0:
        raise ValueError('interface radius, radial standard deviation, and quadrature scale must be positive')
    plot = config.get('plot', {})
    case.sampling_view = plot.get('sampling_view', case.sampling_view)
    case.solution_view = plot.get('solution_view', case.solution_view)
    case.sampling_legend_loc = plot.get('legend_loc', case.sampling_legend_loc)
    return case


def parse_seeds(value):
    """Return seed integers from an integer, a list, or an inclusive ``"a-b"`` range."""
    if isinstance(value, bool):
        raise ValueError("seeds must be an integer, a list of integers, or an inclusive range such as '1-5'")
    if isinstance(value, int):
        return [value]
    if isinstance(value, list) and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in value):
        if not value or len(set(value)) != len(value):
            raise ValueError("seeds must be nonempty and contain no duplicates")
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([+-]?\d+)\s*-\s*([+-]?\d+)\s*", value)
        if match:
            start, stop = map(int, match.groups())
            step = 1 if stop >= start else -1
            return list(range(start, stop + step, step))
    raise ValueError("seeds must be an integer, a list of integers, or an inclusive range such as '1-5'")
