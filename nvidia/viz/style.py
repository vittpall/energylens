"""Shared look of the paper figures: fonts, axis styling, and per-family line styles.

A model's colour identifies the model; its marker and line style identify its
architecture family, so curves of different families stay distinguishable in
greyscale and when colours are close.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 22, 21, 18
FACECOLOR = "#f9f9f9"

# family -> (marker, linestyle, legend name)
FAMILY = {
    "dense":  ("s", "-", "Dense transformer"),
    "moe":    ("^", "--", "Mixture-of-experts"),
    "ssm":    ("D", ":", "State-space (Mamba)"),
    "hybrid": ("p", "-.", "Hybrid Mamba-transformer"),
}
FAMILY_ORDER = list(FAMILY)
# one hue per family for figures that colour by family (bars)
FAMILY_COLOR = {"dense": "#4c72b0", "moe": "#dd8452", "ssm": "#55a868", "hybrid": "#c44e52"}


def line_kw(family, color):
    marker, ls, _ = FAMILY[family]
    return dict(marker=marker, linestyle=ls, color=color, markersize=8, linewidth=2,
                markeredgecolor="white", markeredgewidth=0.8)


def style(ax, xlabel="", ylabel=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(FACECOLOR)


def family_handles():
    """Legend entries for the families alone (black marker + line style)."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], **line_kw(f, "#333333"), label=FAMILY[f][2]) for f in FAMILY_ORDER]


def save(fig, out, name, formats=("png", "pdf")):
    os.makedirs(out, exist_ok=True)
    if not getattr(fig, "_laid_out", False):
        fig.tight_layout()
    for ext in formats:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.join(out, name))
