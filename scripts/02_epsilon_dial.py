from pathlib import Path

import torch

from eps_transition import setup_environment
from eps_transition.data import generate_separable_data
from eps_transition.optimizers import DeterministicAdamW
from eps_transition.oracles import compute_l2_max_margin
from eps_transition.trainer import train_model
from eps_transition.vis import get_fig_ax


def run() -> None:
    device = setup_environment(42)
    x, y = generate_separable_data()
    x, y = x.to(device), y.to(device)

    w_l2 = compute_l2_max_margin(x, y).to(device)

    epsilons = [1e-2, 1e-4, 1e-6, 1e-8]
    colors = ["#E74C3C", "#F39C12", "#4A90E2", "#9B59B6"]  # Red, Orange, Blue, Purple
    results = {}

    print("\n--- Running Probe 2: The Epsilon Dial ---")
    for eps in epsilons:
        w_init = torch.randn(2, dtype=torch.float64, device=device) * 1e-4
        w = torch.nn.Parameter(w_init)

        opt = DeterministicAdamW(params=[w], lr=0.01, eps=eps)
        history = train_model(x, y, w, opt, epochs=25000, w_l2_oracle=w_l2, log_interval=10000)
        results[eps] = history["sim_l2"]

    # Visualization
    result = get_fig_ax(layout="column", twin_x=False)
    fig, ax = result[0], result[1]

    for (eps, history_l2), color in zip(results.items(), colors, strict=True):
        ax.plot(history_l2, color=color, lw=2.5, label=rf"$\epsilon = {eps:.0e}$")

    ax.set_xlabel("Epochs", fontweight="bold")
    ax.set_ylabel(r"Similarity to $\mathcal{L}_2$ Max-Margin", fontweight="bold")

    # Place legend in the empty bottom right corner
    ax.legend(loc="lower right")

    fig.suptitle("Controlling the Phase Transition", fontweight="bold")

    out_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "02_epsilon_dial.pdf", bbox_inches="tight")
    print(f"Saved: {out_dir / '02_epsilon_dial.pdf'}")


if __name__ == "__main__":
    run()
