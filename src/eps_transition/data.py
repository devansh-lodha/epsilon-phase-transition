import torch


def generate_separable_data() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a linearly separable 2D anisotropic dataset.
    The elongated margin channel splits the L2 and Linf geometries.
    """
    x = torch.tensor(
        [
            [1.0, 0.1],
            [1.2, 0.2],
            [2.0, 1.0],
            [-1.0, -0.1],
            [-1.2, -0.2],
            [-2.0, -1.0],
        ],
        dtype=torch.float64,
    )
    y = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=torch.float64)

    return x, y
