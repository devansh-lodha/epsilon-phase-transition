import matplotlib.pyplot as plt

# A professional, flat color palette
COLORS = {
    "adam_linf": "#4A90E2",  # Crisp Blue
    "sgd_l2": "#F39C12",  # Vibrant Orange
    "rho_tracker": "#2ECC71",  # Emerald Green
    "signgd": "#9B59B6",  # Amethyst Purple
    "text": "#2C3E50",  # Slate Gray (softer than black)
    "grid": "#E5E8E8",  # Subtle Gray
    "accent": "#E74C3C",  # Alizarin Red (for the fracture line)
}


def set_publication_theme() -> None:
    """Applies a clean, minimalistic theme using the Lato font."""
    plt.rcParams["font.family"] = "sans-serif"
    # Fallbacks in case Lato isn't installed on the system
    plt.rcParams["font.sans-serif"] = ["Lato", "Helvetica Neue", "Arial", "sans-serif"]

    # Text colors
    plt.rcParams["text.color"] = COLORS["text"]
    plt.rcParams["axes.labelcolor"] = COLORS["text"]
    plt.rcParams["xtick.color"] = COLORS["text"]
    plt.rcParams["ytick.color"] = COLORS["text"]

    # Axes and grids
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.edgecolor"] = COLORS["text"]
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = COLORS["grid"]
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7

    # Legend formatting
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["legend.loc"] = "best"

    # Remove top and right spines for a modern, open look
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def get_fig_ax(
    layout: str = "wide", twin_x: bool = False
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Generates a figure sized perfectly for IEEE double-column format.
    'wide' spans both columns (7.1 inches).
    'column' spans a single column (3.5 inches).
    """
    set_publication_theme()

    if layout == "wide":
        figsize = (7.1, 3.5)
    elif layout == "column":
        figsize = (3.5, 3.0)
    else:
        raise ValueError("layout must be 'wide' or 'column'")

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    if twin_x:
        ax2 = ax.twinx()
        # Keep top open, but RESTORE the right spine for the secondary y-axis
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(COLORS["text"])
        ax2.spines["right"].set_linewidth(1.0)
        return fig, ax, ax2

    return fig, ax
