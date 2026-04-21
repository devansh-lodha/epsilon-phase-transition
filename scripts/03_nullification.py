from collections.abc import Callable
from pathlib import Path

import torch

from eps_transition import setup_environment
from eps_transition.data import generate_separable_data
from eps_transition.optimizers import DeterministicAdamW, DeterministicSGD, DeterministicSignGD
from eps_transition.oracles import compute_l2_max_margin
from eps_transition.trainer import train_model
from eps_transition.vis import COLORS, get_fig_ax


def run() -> None:
    device = setup_environment(42)
    x, y = generate_separable_data()
    x, y = x.to(device), y.to(device)

    w_l2 = compute_l2_max_margin(x, y).to(device)

    # Define optimization strategies
    configs: dict[str, tuple[Callable[[torch.nn.Parameter], object], str]] = {
        r"SGD (Pure $\mathcal{L}_2$)": (
            lambda w: DeterministicSGD(params=[w], lr=0.1),
            COLORS["sgd_l2"],
        ),
        r"SignGD (Pure $\mathcal{L}_\infty$)": (
            lambda w: DeterministicSignGD(params=[w], lr=0.001),
            COLORS["signgd"],
        ),
        r"AdamW (The $\epsilon$-Bridge)": (
            lambda w: DeterministicAdamW(params=[w], lr=0.01, eps=1e-4),
            COLORS["rho_tracker"],
        ),
    }

    results = {}

    print("\n--- Running Probe 3: The Nullification ---")
    for name, (opt_builder, color) in configs.items():
        w_init = torch.randn(2, dtype=torch.float64, device=device) * 1e-4
        w = torch.nn.Parameter(w_init)

        opt = opt_builder(w)
        history = train_model(x, y, w, opt, epochs=25000, w_l2_oracle=w_l2, log_interval=10000)
        results[name] = (history["sim_l2"], color)

    # Visualization
    result = get_fig_ax(layout="column", twin_x=False)
    fig, ax = result[0], result[1]

    for name, (history_l2, color) in results.items():
        ax.plot(history_l2, color=color, lw=2.5, label=name)

    ax.set_xlabel("Epochs", fontweight="bold")
    ax.set_ylabel(r"Similarity to $\mathcal{L}_2$ Max-Margin", fontweight="bold")

    # Place legend in the empty void on the right
    ax.legend(loc="center right")

    fig.suptitle("AdamW Bridges the Geometric Extremes", fontweight="bold")

    out_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "03_nullification.pdf", bbox_inches="tight")
    print(f"Saved: {out_dir / '03_nullification.pdf'}")


if __name__ == "__main__":
    run()
