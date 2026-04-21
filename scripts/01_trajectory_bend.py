from pathlib import Path

import torch

from eps_transition import setup_environment
from eps_transition.data import generate_separable_data
from eps_transition.optimizers import DeterministicAdamW
from eps_transition.oracles import compute_l2_max_margin, compute_linf_max_margin
from eps_transition.trainer import train_model
from eps_transition.vis import COLORS, get_fig_ax


def run() -> None:
    device = setup_environment(42)
    x, y = generate_separable_data()
    x, y = x.to(device), y.to(device)

    w_l2 = compute_l2_max_margin(x, y).to(device)
    w_linf = compute_linf_max_margin(x, y).to(device)

    w_init = torch.randn(2, dtype=torch.float64, device=device) * 1e-4
    w = torch.nn.Parameter(w_init)

    opt = DeterministicAdamW(params=[w], lr=0.01, eps=1e-4)

    print("\n--- Running Probe 1: The Trajectory Bend ---")
    history = train_model(x, y, w, opt, epochs=25000, w_l2_oracle=w_l2, w_linf_oracle=w_linf)

    # Visualization
    result = get_fig_ax(layout="column", twin_x=True)
    # Type hinting workaround for unpack
    fig, ax1, ax2 = result[0], result[1], result[2]  # type: ignore

    ax1.plot(
        history["sim_linf"], color=COLORS["adam_linf"], lw=2.5, label=r"Sim to $\mathcal{L}_\infty$"
    )
    ax1.plot(history["sim_l2"], color=COLORS["sgd_l2"], lw=2.5, label=r"Sim to $\mathcal{L}_2$")
    ax1.set_xlabel("Epochs", fontweight="bold")
    ax1.set_ylabel("Cosine Similarity", fontweight="bold")

    ax2.plot(
        history["rho_t"], color=COLORS["rho_tracker"], lw=2, ls="--", label=r"$\rho_t$ (log scale)"
    )
    ax2.axhline(1.0, color=COLORS["accent"], ls=":", lw=2, label=r"Phase Transition ($\rho_t=1$)")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"Epsilon-Variance Ratio ($\rho_t$)", color=COLORS["rho_tracker"])

    # Combine legends cleanly and place them in the empty middle-left pocket
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # Tweak the (x, y) floats until it fits perfectly in the gap.
    # loc="center" means the center of the legend box is placed at these coordinates.
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="lower left",             # Anchors the bottom-left corner of the legend box
        bbox_to_anchor=(0.05, -0.02),  # Places it slightly off the bottom-left corner axes
    )
     
    fig.suptitle(
        r"$\mathcal{L}_\infty \to \mathcal{L}_2$ Transition in AdamW",
        fontweight="bold",
    )

    # Save output
    out_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "01_trajectory_bend.pdf", bbox_inches="tight")
    print(f"Saved: {out_dir / '01_trajectory_bend.pdf'}")


if __name__ == "__main__":
    run()
