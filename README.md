# Epsilon-Driven $\mathcal{L}_{\infty}$ to $\mathcal{L}_2$ Geometric Phase Transitions in Adaptive Optimizers

## Core Thesis
Current theoretical analyses of adaptive optimizers treat the stability parameter $\epsilon$ as a passive numerical safeguard. This repository provides mathematical and empirical proof that $\epsilon$ acts as the engine of a deterministic geometric phase transition in the loss landscape.

Gradient Descent (GD) is known to maximize the $\mathcal{L}_2$ margin, learning simple, generalizable features. SignGD and Adam (as typically analyzed in the $\epsilon \to 0$ limit) maximize the $\mathcal{L}_\infty$ margin, favoring rich, coordinate-aligned features. We demonstrate that Adam does not possess a static implicit bias. By tracking the Epsilon-Variance Ratio ($\rho_t = \sqrt{v_t} / \epsilon$), we show Adam dynamically bridges these two extremes.

## Optimization Kinematics
Isolating the preconditioner, the Adam update can be factored as:

$$ \Delta w_t = - \frac{\eta}{\epsilon} \left( \frac{1}{\rho_t + 1} \right) \hat{m}_t $$

This formulation yields three distinct optimization phases:
1. Phase 1 ($\mathcal{L}&#95;{\infty}$ Regime, $\rho&#95;t \gg 1$): The variance buffer dominates. The update mimics SignGD, maximizing the $\mathcal{L}&#95;\infty$ margin.
2. Phase 2 (The Collapse, $v_t \to 0$): As the model separates the data, gradients decay exponentially.
3. Phase 3 ($\mathcal{L}_2$ Regime, $\rho_t \ll 1$): $\epsilon$ dominates the preconditioner. The update collapses into heavy-ball SGD, aggressively curving toward the $\mathcal{L}_2$ max-margin.

## Codebase Architecture
To ensure absolute bit-for-bit reproducibility of the geometric fracture point, this codebase enforces strict hardware determinism. GPU execution (CUDA/MPS) is explicitly disabled via the environment setup. Cross-hardware floating-point accumulation discrepancies violate sub-epsilon parity; CPU execution guarantees standard IEEE 754 compliance across all environments.

### Installation
The project relies on `uv` for strict dependency resolution and virtual environment management.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
make setup
```

### Empirical Probes
The repository contains three formal optimization probes tested on an anisotropic 2D dataset. The dataset is specifically constructed to geometrically decouple the $\mathcal{L}_2$ and $\mathcal{L}_\infty$ theoretical vectors, which are computed natively via `cvxpy`.

Probe 1: The Trajectory Bend
Proves the optimizer abandons the $\mathcal{L}_\infty$ geometry and tracks $\mathcal{L}_2$ exactly when $\rho_t$ drops below 1.
```bash
uv run python scripts/01_trajectory_bend.py
```

Probe 2: The Epsilon Dial
Proves strict causality. Sweeping $\epsilon \in \{10^{-2}, 10^{-4}, 10^{-6}, 10^{-8}\}$ directly dictates the precise epoch of the geometric transition.
```bash
uv run python scripts/02_epsilon_dial.py
```

Probe 3: The Nullification
The control experiment. Demonstrates that SGD and SignGD remain permanently locked into their respective geometries, establishing Adam as the mathematical bridge.
```bash
uv run python scripts/03_nullification.py
```

All outputs are generated as IEEE-compliant vector graphics in `results/figures/`.

### Code Quality
Type checking and linting are strictly enforced via `ruff` and `ty` against Python 3.14 specifications.
```bash
make check
```

## References
1. Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., & Srebro, N. (2018). The Implicit Bias of Gradient Descent on Separable Data. Journal of Machine Learning Research, 19(70), 1-57.
2. Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., & Anandkumar, A. (2018). signSGD: Compressed Optimisation for Non-Convex Problems. Proceedings of the 35th International Conference on Machine Learning (ICML), 80, 560-569.
3. Cattaneo, M. D., Klusowski, J. M., & Shigida, B. (2024). On the Implicit Bias of Adam. Proceedings of the 41st International Conference on Machine Learning (ICML), 235, 5862-5906.
4. Zhang, C., Zou, D., & Cao, Y. (2024). The Implicit Bias of Adam on Separable Data. arXiv preprint arXiv:2406.10650.
5. Fan, C., Schmidt, M., & Thrampoulidis, C. (2025). Implicit Bias of SignGD and Adam on Multiclass Separable Data. arXiv preprint arXiv:2502.04664.
