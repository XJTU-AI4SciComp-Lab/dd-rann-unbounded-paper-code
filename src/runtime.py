"""Runtime options shared by all examples."""
def validate_runtime(config):
    solver = config.get('solver', 'scipy').lower()
    device = config.get('device', 'cpu').lower()
    if solver not in ('scipy', 'torch'):
        raise ValueError("solver must be 'scipy' or 'torch'")
    if device != 'cpu' and device != 'cuda' and not device.startswith('cuda:'):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if solver == 'scipy' and device != 'cpu':
        raise ValueError('solver=scipy runs on CPU; choose solver=torch to use a CUDA device')
    if solver == 'torch':
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('solver=torch requires PyTorch') from exc
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(f'{device} requested but CUDA is unavailable')
    return solver, device
