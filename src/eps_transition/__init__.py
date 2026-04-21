import os
import random

import numpy as np
import torch


def setup_environment(seed: int = 42) -> torch.device:
    """
    Locks all stochasticity and forces CPU execution to guarantee
    bit-for-bit identical epsilon fracture points across all machines.
    """
    # 1. Lock Python's hash seed for deterministic dict/set iteration
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Force CPU execution.
    # For a 2D mathematical probe, GPU dispatch overhead is slower,
    # and CPU guarantees strict cross-platform IEEE 754 determinism.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # 3. Standard seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 4. Strict deterministic algorithms
    # If a PyTorch op lacks a deterministic implementation, crash immediately.
    torch.use_deterministic_algorithms(True, warn_only=False)

    return torch.device("cpu")
