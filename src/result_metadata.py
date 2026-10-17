"""Allocation and seed selection for independent plotting configs."""
import json

from src.config import allocation_key, parse_seeds


def select_plot_rows(rows, config, allocation_field='neurons'):
    """Select results for configured allocations and seeds."""
    wanted = config.get('neuron_pairs', config.get('double_neurons'))
    wanted = [allocation_key(item) for item in wanted] if wanted is not None else None
    seeds = parse_seeds(config['seeds']) if 'seeds' in config else None
    selected = []
    for row in rows:
        allocation = allocation_key(row[allocation_field])
        if wanted is not None and allocation not in wanted:
            continue
        if seeds is not None:
            if 'seed' in row and int(row['seed']) not in seeds:
                continue
            if 'seeds' in row and sorted(json.loads(row['seeds'])) != sorted(seeds):
                raise ValueError('Result seeds differ from the plotting config; recompute the source table.')
        selected.append(row)
    if not selected:
        raise ValueError('No results match the plotting config')
    if wanted is not None:
        missing = set(wanted) - {allocation_key(row[allocation_field]) for row in selected}
        if missing:
            raise ValueError(f'Missing configured allocations in source results: {sorted(missing)}')
    return selected


def grouped_plot_rows(rows, allocation_field='neurons'):
    """Separate allocation families when multiple allocations have the same DoF."""
    groups = {}
    for row in rows:
        allocation = allocation_key(row[allocation_field])
        # Follow a curve for each fixed far-field allocation; arbitrary allocations remain distinct.
        family = allocation[1:] if len(allocation) == 2 else (allocation[0], allocation[-1])
        groups.setdefault(family, []).append(row)
    return [(family, sorted(group, key=lambda row: (int(row['dof']), allocation_key(row[allocation_field]))))
            for family, group in groups.items()]
