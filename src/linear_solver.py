"""Shared dense least-squares backend."""
import numpy as np
from scipy.linalg import lstsq

from src.runtime import validate_runtime


def _spectral_diagnostics(singular_values, rank):
    """Describe the singular spectrum of A, including its rank-truncated condition."""
    values = np.sort(np.abs(np.asarray(singular_values, dtype=float)))[::-1]
    rank = int(rank)
    if values.size == 0:
        return {
            "condition_number": float("nan"),
            "truncated_condition_number": float("nan"),
            "singular_value_max": float("nan"),
            "singular_value_min": float("nan"),
            "retained_singular_value_min": float("nan"),
            "discarded_singular_value_max": float("nan"),
        }
    largest = float(values[0])
    smallest = float(values[-1])
    retained_min = float(values[rank - 1]) if rank > 0 else 0.0
    discarded_max = float(values[rank]) if rank < values.size else float("nan")
    return {
        # These are singular values of A itself; A.T @ A is never formed here.
        "condition_number": float("inf") if smallest == 0.0 else largest / smallest,
        "truncated_condition_number": (
            float("inf") if retained_min == 0.0 else largest / retained_min
        ),
        "singular_value_max": largest,
        "singular_value_min": smallest,
        "retained_singular_value_min": retained_min,
        "discarded_singular_value_max": discarded_max,
    }


def least_squares(matrix, vector, solver="scipy", device="cpu", rank_tolerance=None,
                  return_diagnostics=False):
    solver, device = validate_runtime({"solver": solver, "device": device})
    if solver == "scipy":
        solution, _, rank, singular_values = lstsq(
            matrix, vector, cond=rank_tolerance, lapack_driver="gelsd"
        )
        if return_diagnostics:
            diagnostics = _spectral_diagnostics(singular_values, rank)
            diagnostics["matrix_rank"] = int(rank)
            diagnostics["condition_source"] = "singular_values_of_A"
            return solution, diagnostics
        return solution
    import torch
    is_complex = np.iscomplexobj(matrix) or np.iscomplexobj(vector)
    dtype = torch.complex128 if is_complex else torch.float64
    result = torch.linalg.lstsq(
        torch.as_tensor(matrix, dtype=dtype, device=device),
        torch.as_tensor(vector, dtype=dtype, device=device),
        rcond=rank_tolerance,
    )
    solution = result.solution.detach().cpu().numpy()
    if return_diagnostics:
        singular_values = result.singular_values.detach().cpu().numpy()
        rank = int(result.rank.detach().cpu().item())
        diagnostics = _spectral_diagnostics(singular_values, rank)
        diagnostics["matrix_rank"] = rank
        diagnostics["condition_source"] = "singular_values_of_A"
        return solution, diagnostics
    return solution
