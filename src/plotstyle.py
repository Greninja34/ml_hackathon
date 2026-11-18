"""One consistent look for every report figure (matplotlib only, no extra dependencies)."""
import matplotlib
import matplotlib.pyplot as plt

# colour-blind-friendly palette, reused everywhere (figures AND maps)
COLORS = {
    "train": "#2a6fdb",
    "student_test": "#f28c28",
    "gps": "#1f4fd8",
    "reference": "#1b9e55",
    "predicted": "#e63946",
    "network": "#9aa0a6",
    "accent": "#6a4c93",
    "neutral": "#5f6368",
}
MODEL_COLORS = ["#2a6fdb", "#f28c28", "#1b9e55", "#e63946", "#6a4c93", "#17a2b8", "#b8860b"]


def apply():
    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "#fafafa",
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 11, "axes.labelcolor": "#333333",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#888888", "axes.grid": True, "grid.color": "#e3e3e3",
        "grid.linewidth": .8, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 10,
        "xtick.color": "#444444", "ytick.color": "#444444",
    })


def finish(fig, path, subtitle=None):
    """Save with an optional grey subtitle (e.g. which partition the data comes from)."""
    if subtitle:
        fig.text(0.0, -0.02, subtitle, fontsize=9, color=COLORS["neutral"], ha="left", va="top")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
