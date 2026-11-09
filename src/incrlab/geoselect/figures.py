"""Figures for the geoselect module (real-geo clusters, power ladder, monitoring, final readout).
Saved to reports/figures/geoselect/*.png using the repo's shared matplotlib theme.
"""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.plotting import PALETTE, ROLE, SERIES, mark_line, pct_axis, save_fig, shade_window
from ..common.plotting import source_note as _base_source_note
from ..common.plotting import subtitle as _base_subtitle

MODULE = "geoselect"


def subtitle(ax, text: str):  # noqa: F811 - local override: keep subtitle clear of the title
    ax._set_title_offset_trans(24)
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9.5, color="#555555", va="bottom")


def source_note(fig, text: str = "Synthetic data - illustrative magnitudes only."):  # noqa: F811
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    y = min(-0.02, bb.y0 / fig.get_figheight() - 0.015)
    fig.text(0.01, y, text, fontsize=8, color="#777777", ha="left", va="top")

SRC_NOTE = ("Clustering/matching/power on REAL US Census ACS 2018-2022 5yr demographics for MA towns "
           "(pop >= 40k); baseline KPI, test outcomes and monitoring are SIMULATED - see readout.")


def fig_clusters(d: pd.DataFrame, pair: dict) -> plt.Figure:
    """Scatter of MA towns by median income x pct bachelor's+, colored by cluster, treatment/control starred."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in sorted(d["cluster"].unique()):
        sub = d[d["cluster"] == c]
        ax.scatter(sub["median_income"], sub["pct_bachelors_plus"], s=np.sqrt(sub["pop"]) * 1.1,
                  color=SERIES[c % len(SERIES)], alpha=0.65, label=f"Cluster {c} (n={len(sub)})",
                  edgecolor="white", linewidth=0.6)
    t_name, c_name = pair["treatment"]["town"], pair["control"]["town"]
    for role, name, color in (("treatment", t_name, ROLE["treatment"]), ("control", c_name, ROLE["control"])):
        r = d[d["town"] == name].iloc[0]
        ax.scatter([r["median_income"]], [r["pct_bachelors_plus"]], marker="*", s=550, color=color,
                  edgecolor="black", linewidth=1.0, zorder=5, label=f"{role.title()}: {name}")
    for _, r in d.iterrows():
        if r["town"] in (t_name, c_name) or r["pop"] > d["pop"].quantile(0.75):
            ax.annotate(r["town"], (r["median_income"], r["pct_bachelors_plus"]), fontsize=7.5,
                       xytext=(4, 4), textcoords="offset points", color="#333333")
    ax.set_xlabel("Median household income ($, ACS 2018-2022 5yr)")
    ax.set_ylabel("Share with bachelor's degree or higher (ACS)")
    ax.set_title("Massachusetts towns clustered on real Census demographics")
    subtitle(ax, f"{len(d)} MA county subdivisions, pop >= 40k; K-means k={d['cluster'].nunique()}; "
                 f"marker size ~ population")
    pct_axis(ax, "y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    source_note(fig, SRC_NOTE)
    return fig


def fig_power_ladder(power_df: pd.DataFrame, mde: float, st) -> plt.Figure:
    """Power-sizing ladder: power vs weeks, one curve per candidate lift (like dsp's sizing ladder)."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for i, L in enumerate(sorted(power_df["lift"].unique())):
        sub = power_df[power_df["lift"] == L].sort_values("weeks")
        ax.plot(sub["weeks"], sub["power"], marker="o", ms=4, color=SERIES[i % len(SERIES)],
               label=f"lift={L:.0%}")
    ax.axhline(st.power_target, color=PALETTE["slate"], ls="--", lw=1.1)
    ax.text(power_df["weeks"].max(), st.power_target, f" target power {st.power_target:.0%}",
           va="bottom", ha="right", fontsize=8.5, color=PALETTE["slate"])
    mark_line(ax, st.test_weeks, label=f"planned test ({st.test_weeks} wk)", color=PALETTE["orange"])
    ax.set_xlabel("Test length (weeks)")
    ax.set_ylabel("Power (closed-form, one-sided)")
    ax.set_title("Power-sizing ladder: treatment vs matched control")
    subtitle(ax, f"MDE at {st.test_weeks} wk / {st.power_target:.0%} power = {mde:.1%}; "
                 f"true injected lift = {st.true_lift:.0%}")
    pct_axis(ax, "y")
    ax.legend(ncol=2, fontsize=8.5)
    source_note(fig, "Closed-form AR(1)-inflated DiD power (incrlab.common.power); sigma from simulated "
                     "pre-period daily noise on real-population-scaled baseline volume.")
    return fig


def fig_monitoring(looks: pd.DataFrame, st) -> plt.Figure:
    """Sequential monitoring: cumulative lift estimate + narrowing CI across interim test-week looks."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(looks["weeks_elapsed"], looks["lift"], color=ROLE["estimate"], marker="o", label="DiD lift estimate")
    ax.fill_between(looks["weeks_elapsed"], looks["ci_low"], looks["ci_high"], color=ROLE["estimate"], alpha=0.18,
                   label=f"{st.did_alpha * 0 + (1 - st.did_alpha):.0%} CI")
    ax.axhline(st.true_lift, color=ROLE["truth"], ls="--", lw=1.4, label=f"true lift ({st.true_lift:.0%})")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("Test weeks elapsed (interim look)")
    ax.set_ylabel("Cumulative DiD lift estimate")
    ax.set_title("Sequential monitoring: interim DiD reads across the test window")
    subtitle(ax, "One interim look per completed test week; CI narrows as more test data accrues")
    pct_axis(ax, "y")
    ax.legend(loc="lower right", fontsize=9)
    source_note(fig, SRC_NOTE)
    return fig


def fig_final_readout(final: dict, looks: pd.DataFrame, pair: dict, st) -> plt.Figure:
    """Final readout: bar of estimate vs truth with CI, small inset table."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = [0, 1]
    vals = [final["lift"], st.true_lift]
    colors = [ROLE["estimate"], ROLE["truth"]]
    bars = ax.bar(x, vals, color=colors, width=0.5)
    if final.get("ci_low") is not None:
        ax.errorbar([0], [final["lift"]], yerr=[[final["lift"] - final["ci_low"]], [final["ci_high"] - final["lift"]]],
                   fmt="none", ecolor="black", capsize=6, lw=1.6)
    ax.set_xticks(x, [f"DiD estimate\n(90% CI)", "True injected lift\n(ground truth)"])
    ax.axhline(0, color="#999999", lw=0.8)
    pct_axis(ax, "y")
    ax.set_title(f"Final readout: {pair['treatment']['town']} vs {pair['control']['town']}")
    cov = "covers" if final.get("ci_covers_truth") else "misses"
    subtitle(ax, f"Estimate {final['lift']:+.1%} [{final.get('ci_low', float('nan')):.1%}, "
                 f"{final.get('ci_high', float('nan')):.1%}] - CI {cov} the true lift")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004 * np.sign(v or 1), f"{v:+.1%}", ha="center",
               va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    source_note(fig, SRC_NOTE)
    return fig


def make_all(S) -> List[str]:
    paths = []
    paths.append(str(save_fig(fig_clusters(S.clustered, S.pair), MODULE, "01_clusters")))
    paths.append(str(save_fig(fig_power_ladder(S.power_table, S.mde, S.st), MODULE, "02_power_ladder")))
    paths.append(str(save_fig(fig_monitoring(S.monitoring, S.st), MODULE, "03_monitoring")))
    paths.append(str(save_fig(fig_final_readout(S.final, S.monitoring, S.pair, S.st), MODULE, "04_final_readout")))
    return paths
