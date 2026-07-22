"""Shared plotting style so every figure in the project looks like one system.

Import this before plotting and call `apply_style()` once:

    import viz
    viz.apply_style()
    fig, ax = viz.figure()
    ...
    viz.save(fig, "my_chart.png")

COLOR CHOICES ARE VALIDATED, NOT EYEBALLED. Every pair below was checked for
colourblind separation (OKLab dE, deuteran/protan/tritan simulated):

  side identity  blue #2a78d6 vs red #e34948   dE 21.6  PASS
  outcome        aqua #1baf7a vs orange #eb6834 dE  9.2  PASS (+ W/L text labels)
  REJECTED       green #0ca30c vs red #d03b3b   dE  4.1  FAIL <- the classic
                 green/loss-red trap: invisible to red-green colourblind readers
                 (~8% of men). Never use it for win/loss.

Outcome colour is always paired with a text label or legend, so meaning never
rests on hue alone.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Palette -----------------------------------------------------------------
BLUE_SIDE = "#2a78d6"   # team 100
RED_SIDE = "#e34948"    # team 200
WON = "#1baf7a"         # outcome: win  (aqua-green)
LOST = "#eb6834"        # outcome: loss (orange)
NEUTRAL_MID = "#f0efec"  # diverging midpoint ("no effect")

# Chrome & ink (recessive by design -- data should be the only loud thing)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style() -> None:
    """Hairline grid, no top/right spines, system sans, generous padding."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "text.color": INK,
        # Solid hairline grid on one axis only -- never dashed (dashes read as
        # "threshold" and add noise).
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10,
        "axes.titlesize": 13,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# Log-odds -> percentage points of win chance.
# A logistic model's coefficient is in log-odds, which means nothing to a reader.
# Near an even game (50/50) the sigmoid's slope is 1/4, so a coefficient of b
# shifts the win chance by about b/4 -- i.e. b * 25 percentage points. Every one
# of our predictions sits near 50%, so this conversion is accurate here, and it
# turns "0.067 log-odds" into the far more honest "+1.7 points of win chance".
PP_PER_LOGODD = 25.0


def to_points(log_odds):
    """Coefficient in log-odds -> percentage points of win chance."""
    return log_odds * PP_PER_LOGODD


def figure(width: float = 8, height: float = 4.5):
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def caption(ax, text: str, width: int = 92) -> None:
    """A plain-language takeaway printed under the chart -- the sentence you'd
    say out loud if someone asked 'so what?'. Wraps itself, so callers just pass
    one plain sentence."""
    import textwrap
    ax.annotate(textwrap.fill(text, width), xy=(0, 0), xycoords="axes fraction",
                xytext=(0, -48), textcoords="offset points",
                fontsize=9.5, color=INK_SECONDARY, va="top", ha="left",
                annotation_clip=False)


def title_block(ax, title: str, sub: str | None = None) -> None:
    """Left-aligned title + optional one-line subtitle, both placed in axes
    fraction coords so they never collide with each other or with in-plot text
    (matplotlib's own title padding can't see the subtitle)."""
    ax.set_title("")  # ensure no competing built-in title
    # Offsets in POINTS, not axes fractions, so the title/subtitle gap is
    # identical on a 4in and a 7in tall chart.
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 26 if sub else 10), textcoords="offset points",
                fontsize=13, fontweight="semibold", color=INK, va="bottom", ha="left")
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=9.5, color=INK_SECONDARY, va="bottom", ha="left")


def reference_line(ax, value: float = 0.5, label: str = "coin flip",
                   axis: str = "y") -> None:
    """A labelled threshold rule (50% = no information). The label sits INSIDE
    the plot so it can't collide with the title block."""
    if axis == "y":
        ax.axhline(value, color=INK_MUTED, lw=1, zorder=1)
        ax.text(0.995, value, label, transform=ax.get_yaxis_transform(),
                fontsize=8.5, color=INK_MUTED, va="bottom", ha="right")
    else:
        ax.axvline(value, color=INK_MUTED, lw=1, zorder=1)
        ax.text(value, 0.97, f" {label}", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color=INK_MUTED, ha="left", va="top")


def despine_x(ax) -> None:
    """For horizontal-bar charts: drop the y spine and its ticks."""
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def save(fig, filename: str, figures_dir=None):
    """Save into reports/figures/ and return the path."""
    if figures_dir is None:
        import config
        figures_dir = config.FIGURES
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path
