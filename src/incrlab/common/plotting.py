"""One consistent, professional matplotlib theme for every chart in the repo."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHANNEL_LABELS, FIG_DIR  # noqa: E402

PALETTE = {
    "navy": "#1F3A5F", "teal": "#2A9D8F", "orange": "#E76F51", "gold": "#E9C46A", "slate": "#5C677D",
    "red": "#C0392B", "green": "#2E8B57", "purple": "#6C5B7B", "sky": "#7FB3D5", "grey": "#B8BCC6",
    "light": "#EEF1F5",
}
SERIES = [PALETTE[k] for k in ("navy", "orange", "teal", "gold", "purple", "slate", "sky", "red")]
ROLE = {"treatment": PALETTE["orange"], "control": PALETTE["navy"], "synthetic": PALETTE["teal"],
        "buffer": PALETTE["gold"], "excluded": PALETTE["grey"], "truth": PALETTE["green"],
        "estimate": PALETTE["navy"], "attribution": PALETTE["red"], "donor": PALETTE["sky"]}
CHANNEL_COLORS = {"tv": PALETTE["slate"], "social": PALETTE["purple"], "search_brand": PALETTE["navy"],
                  "search_nonbrand": PALETTE["sky"], "dsp_a": PALETTE["orange"], "dsp_b": PALETTE["gold"],
                  "organic": PALETTE["green"], "base": PALETTE["grey"]}


def set_theme() -> None:
    plt.rcParams.update({
        "figure.figsize": (10, 5.2), "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10.5, "axes.labelcolor": "#333333",
        "axes.edgecolor": "#888888", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E3E6EB", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "legend.frameon": False, "legend.fontsize": 9.5,
        "xtick.color": "#555555", "ytick.color": "#555555", "lines.linewidth": 2.0,
    })


set_theme()


def shade_window(ax, start, end, label: Optional[str] = "Test window", color: str = PALETTE["gold"],
                 alpha: float = 0.18, text: bool = True):
    """Shade an x-range (inclusive start, end) and optionally annotate it at the top."""
    ax.axvspan(start, end, color=color, alpha=alpha, lw=0, label=None, zorder=0)
    if text and label:
        ax.text(start, 1.0, f" {label}", transform=ax.get_xaxis_transform(), va="top", ha="left",
                fontsize=9, color="#6b5a1e")
    return ax


def mark_line(ax, x, label: Optional[str] = None, color: str = PALETTE["slate"], ls: str = "--"):
    ax.axvline(x, color=color, ls=ls, lw=1.2, zorder=1)
    if label:
        ax.text(x, 0.02, f" {label}", transform=ax.get_xaxis_transform(), rotation=90, va="bottom",
                fontsize=8.5, color=color)


def subtitle(ax, text: str):
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9.5, color="#555555", va="bottom")


def pct_axis(ax, axis: str = "y", decimals: int = 0):
    from matplotlib.ticker import PercentFormatter
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=decimals))


def money_axis(ax, axis: str = "y"):
    from matplotlib.ticker import FuncFormatter

    def f(v, _):
        a = abs(v)
        return f"${v / 1e6:.1f}M" if a >= 1e6 else (f"${v / 1e3:.0f}k" if a >= 1e3 else f"${v:.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(f))


def channel_label(ch: str) -> str:
    return CHANNEL_LABELS.get(ch, ch)


def source_note(fig, text: str = "Synthetic data - illustrative magnitudes only."):
    fig.text(0.01, -0.02, text, fontsize=8, color="#777777", ha="left", va="top")


def save_fig(fig, module: str, name: str, fig_dir: Optional[Path] = None, close: bool = True) -> Path:
    """Save to reports/figures/{module}/{name}.png at 150 dpi; returns the path."""
    d = Path(fig_dir or FIG_DIR) / module
    d.mkdir(parents=True, exist_ok=True)
    p = d / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return p


def plot_actual_vs_counterfactual(index, actual, counterfactual, window: Tuple, title: str,
                                  ylabel: str, band: Optional[Tuple[Sequence, Sequence]] = None,
                                  actual_label: str = "Treated (actual)", cf_label: str = "Counterfactual"):
    fig, ax = plt.subplots()
    ax.plot(index, actual, color=ROLE["treatment"], label=actual_label)
    ax.plot(index, counterfactual, color=ROLE["synthetic"], ls="--", label=cf_label)
    if band is not None:
        ax.fill_between(band[0], band[1][0], band[1][1], color=ROLE["synthetic"], alpha=0.15, lw=0,
                        label="Interval")
    shade_window(ax, *window)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
    return fig, ax


def bar_compare(labels: Iterable[str], series: dict, title: str, ylabel: str, pct: bool = False):
    """Grouped bar chart: series = {name: values}."""
    labels = list(labels)
    k = len(series)
    x = np.arange(len(labels))
    w = 0.8 / max(k, 1)
    fig, ax = plt.subplots()
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (k - 1) / 2) * w, vals, w, label=name, color=SERIES[i % len(SERIES)])
    ax.set_xticks(x, labels, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if pct:
        pct_axis(ax)
    ax.legend(ncol=min(k, 4))
    return fig, ax
