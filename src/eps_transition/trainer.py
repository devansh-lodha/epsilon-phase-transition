import torch


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer | object,  # Handles custom optimizers
    epochs: int,
    w_l2_oracle: torch.Tensor | None = None,
    w_linf_oracle: torch.Tensor | None = None,
    log_interval: int = 5000,
) -> dict[str, list[float]]:
    """
    Centralized training loop that tracks optimizer kinematics and geometries.
    """
    history: dict[str, list[float]] = {
        "loss": [],
        "rho_t": [],
        "sim_l2": [],
        "sim_linf": [],
    }

    for epoch in range(epochs):
        logits = x @ w
        # Logistic loss to allow gradients to decay smoothly (crucial for Phase 2)
        loss = torch.nn.functional.softplus(-y * logits).mean()

        if w.grad is not None:
            w.grad.zero_()
        loss.backward()

        kinematics = optimizer.step()  # type: ignore

        # Track trajectory metrics
        with torch.no_grad():
            w_norm = w / torch.linalg.norm(w)

            sim_l2 = torch.dot(w_norm, w_l2_oracle).item() if w_l2_oracle is not None else 0.0
            sim_linf = torch.dot(w_norm, w_linf_oracle).item() if w_linf_oracle is not None else 0.0

            history["sim_l2"].append(sim_l2)
            history["sim_linf"].append(sim_linf)
            history["rho_t"].append(kinematics.get("rho_t", 0.0))
            history["loss"].append(loss.item())

        if epoch % log_interval == 0:
            print(
                f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
                f"Rho: {kinematics.get('rho_t', 0.0):.2e} | "
                f"L2 Sim: {sim_l2:.4f} | Linf Sim: {sim_linf:.4f}"
            )

    return history
