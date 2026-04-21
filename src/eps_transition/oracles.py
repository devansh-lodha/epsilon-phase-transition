import cvxpy as cp
import numpy as np
import torch


def _solve_margin(x: torch.Tensor, y: torch.Tensor, norm_type: int | str) -> torch.Tensor:
    """Internal helper to solve the hard-margin SVM for a given norm."""
    x_np = x.numpy()
    y_np = y.numpy()
    d = x_np.shape[1]

    w = cp.Variable(d)

    if norm_type == 2:
        objective = cp.Minimize(0.5 * cp.sum_squares(w))
    elif norm_type == "inf":
        objective = cp.Minimize(cp.norm(w, "inf"))
    else:
        raise ValueError("Unsupported norm_type. Use 2 or 'inf'.")

    constraints = [cp.multiply(y_np, x_np @ w) >= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    w_opt = w.value
    if w_opt is None:
        raise RuntimeError(f"Optimization failed for L_{norm_type} margin.")

    # Return the normalized theoretical direction
    return torch.tensor(w_opt / np.linalg.norm(w_opt), dtype=torch.float64)


def compute_l2_max_margin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _solve_margin(x, y, 2)


def compute_linf_max_margin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _solve_margin(x, y, "inf")
