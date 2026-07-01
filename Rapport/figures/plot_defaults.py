# figures/plot_defaults.py
# Shared matplotlib defaults for Rapport figures.
# Edit this file to change styling globally.

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Colour palette (aligned with main.tex accent colours) ---
COLOURS = {
    "primary": "#2E5A88",
    "secondary": "#E0A85B",
    "tertiary": "#1F6F54",
    "quaternary": "#d62728",
    "grey": "#555555",
    "light_grey": "#c7c7c7",
    "train": "#2E5A88",
    "test": "#E0A85B",
}
COLOUR_CYCLE = [
    COLOURS["primary"],
    COLOURS["secondary"],
    COLOURS["tertiary"],
    COLOURS["quaternary"],
    COLOURS["grey"],
]

# --- Figure sizes (inches) ---
SINGLE_COL = (6.5, 4.0)
DOUBLE_COL = (6.5, 3.0)
HALF_COL = (3.15, 3.0)

# --- Font and text ---
FONT_FAMILY = "serif"
FONT_SIZE = 10
LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 9

# --- Line and marker ---
LINE_WIDTH = 1.5
MARKER_SIZE = 4

# --- Export ---
DPI = 300
FORMATS = ["pdf", "png"]


def apply() -> None:
    """Apply report defaults to matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.figsize": SINGLE_COL,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "axes.prop_cycle": mpl.cycler(color=COLOUR_CYCLE),
        }
    )


def savefig(fig, path_stem: str) -> None:
    """Save figure in all configured formats."""
    for fmt in FORMATS:
        fig.savefig(f"{path_stem}.{fmt}", dpi=DPI, bbox_inches="tight")
