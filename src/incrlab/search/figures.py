"""Figures for the Paid Search module (shared theme from incrlab.common.plotting)."""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Polygon

from ..common import plotting as pl
from ..common.plotting import PALETTE, ROLE, save_fig, shade_window, source_note, subtitle
from ..geo import voronoi_polygons
from . import panel as P
from .settings import ARM_LABELS, ARMS, MODULE, SOURCE_LABELS, SOURCES, wk


# --- layout helpers: keep subtitles clear of titles and source notes clear of tick labels -----------
def subtitle(ax, text: str):  # noqa: F811 - local override of common.plotting.subtitle
    """Grey subtitle between the axes and a title lifted out of its way."""
    ax._set_title_offset_trans(24)
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9.5, color="#555555", va="bottom")


def source_note(fig, text: str = "Synthetic data - illustrative magnitudes only."):  # noqa: F811
    """Footnote placed below everything already drawn (rotated tick labels, x labels)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    y = min(-0.02, bb.y0 / fig.get_figheight() - 0.015)
    fig.text(0.01, y, text, fontsize=8, color="#777777", ha="left", va="top")

METHOD_LABELS = {"ratio_did": "Ratio DiD", "ols_fe": "OLS DiD (unit+week FE)", "bsts": "BSTS / CausalImpact",
                 "synth_control": "Synthetic control", "triangulated": "Triangulated"}
METRIC_LABELS = {"qs": "Quote starts", "rated": "Rated quotes", "sales": "Sales", "visits": "Site visits"}
ARM_COLORS = {"all_search": PALETTE["orange"], "branded_only": PALETTE["purple"], "control": PALETTE["navy"],
              "bau": PALETTE["sky"]}


def _window(ax, s, label="Dark-out"):
    shade_window(ax, s.post[0] - 0.5, s.post[1] + 0.5, label=label)


def fig_map(o) -> plt.Figure:
    s, d = o["settings"], o["design"]
    from ..simulate import get_world
    z = get_world(s.cfg).geo.zips
    z = z[z["region"] == s.region].reset_index(drop=True)
    role = {c: "bau" for c in z["county_id"].unique()}
    for a in ARMS:
        for c in d[a]:
            role[c] = a
    role[d["control"][0]] = "control"
    polys = voronoi_polygons(z["lon"], z["lat"])
    fig, ax = plt.subplots(figsize=(10, 7))
    for poly, cid in zip(polys, z["county_id"]):
        if len(poly) >= 3:
            ax.add_patch(Polygon(poly, closed=True, fc=ARM_COLORS[role[cid]], ec="white", lw=0.25,
                                 alpha=0.9 if role[cid] != "bau" else 0.45))
    cs = o["design_stage"]["mde_table"]
    names = o["panel"].groupby("county_id")["county"].first()
    cent = z.groupby("county_id").apply(lambda g: pd.Series({"lon": np.average(g["lon"], weights=g["pop"]),
                                                              "lat": np.average(g["lat"], weights=g["pop"])}),
                                        include_groups=False)
    for cid, r in cent.iterrows():
        ax.text(r["lon"], r["lat"], names[cid].replace(" County", ""), ha="center", va="center", fontsize=7.5,
                color="white" if role[cid] in ("control", "branded_only") else "#222222",
                fontweight="bold" if role[cid] != "bau" else "normal")
    ax.set_xlim(z["lon"].min() - 0.05, z["lon"].max() + 0.05)
    ax.set_ylim(z["lat"].min() - 0.05, z["lat"].max() + 0.05)
    ax.set_aspect(1 / np.cos(np.radians(z["lat"].mean())))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.legend(handles=[Patch(color=ARM_COLORS[k], label=ARM_LABELS[k]) for k in ("all_search", "branded_only", "control", "bau")],
              loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Pipeline-selected design: Northvale counties by arm")
    subtitle(ax, f"Selected from {len(cs)} candidate units (single counties + adjacent pairs); zip Voronoi cells")
    source_note(fig)
    return fig


def fig_mde(o) -> plt.Figure:
    t = o["design_stage"]["mde_table"].sort_values("mde")
    d = o["design"]["units"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(t))
    col = [ARM_COLORS["all_search"] if u == d["all_search"] else ARM_COLORS["branded_only"] if u == d["branded_only"]
           else PALETTE["grey"] for u in t["unit"]]
    ax.barh(y, t["mde"], color=col, label="Rated quotes")
    ax.scatter(t["mde_sales"], y, color=PALETTE["slate"], marker="D", s=22, zorder=3, label="Sales")
    ax.set_yticks(y, t["label"])
    ax.invert_yaxis()
    pl.pct_axis(ax, "x")
    ax.set_xlabel("MDE, 6-week dark-out (two-sided 90%, 80% power)")
    ax.axvline(o["settings"].prior_lift_all, color=PALETTE["red"], ls="--", lw=1.2)
    ax.text(o["settings"].prior_lift_all, len(t) - 0.4, " prior all-search lift", color=PALETTE["red"], fontsize=8.5)
    ax.legend(loc="lower right")
    ax.set_title("County / bundle MDE and injected-lift recovery")
    subtitle(ax, "Orange = all-search arm, purple = branded-only arm (selected by pipeline)")
    ax2.barh(y, t["abs_err_10"], color=col)
    ax2.axvline(o["settings"].recovery_gate, color=PALETTE["red"], ls="--", lw=1.2)
    pl.pct_axis(ax2, "x", 1)
    ax2.set_xlabel("|detected - injected| at +10% (locked weights, pseudo-launch wk 73)")
    source_note(fig)
    return fig


def fig_injected(o) -> plt.Figure:
    t = o["design_stage"]["mde_table"]
    s = o["settings"]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for lift in s.injected_lifts:
        k = f"detected_{int(round(lift * 100))}"
        ax.scatter(np.full(len(t), lift) + np.random.default_rng(1).uniform(-0.004, 0.004, len(t)), t[k],
                   color=PALETTE["grey"], s=18, alpha=0.8)
        for arm in ARMS:
            r = t[t["unit"] == o["design"]["units"][arm]].iloc[0]
            ax.scatter(lift, r[k], color=ARM_COLORS[arm], s=70, zorder=3, edgecolor="white",
                       label=ARM_LABELS[arm] if lift == s.injected_lifts[0] else None)
    lim = [0, max(s.injected_lifts) * 1.25]
    ax.plot(lim, lim, color=ROLE["truth"], ls="--", lw=1.4, label="Perfect recovery")
    ax.set_xlim(lim)
    pl.pct_axis(ax, "x")
    pl.pct_axis(ax)
    ax.set_xlabel("Injected lift (rated quotes per capita)")
    ax.set_ylabel("Detected lift (bias-corrected)")
    ax.legend(loc="upper left")
    ax.set_title("Injected-lift validation - same metric fit and injected")
    subtitle(ax, f"{len(t)} candidates (grey); weights fit wk {s.validation_train[0]}-{s.validation_train[1]}, locked for "
                 f"wk {s.validation_post[0]}-{s.validation_post[1]}")
    source_note(fig)
    return fig


def fig_balance(o) -> plt.Figure:
    b = o["design_stage"]["balance"]
    cov = list(o["settings"].balance_covariates)
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(cov))
    for i, arm in enumerate(ARMS):
        ax.scatter(b.loc[arm, [f"smd_{c}" for c in cov]].astype(float), y + (i - 0.5) * 0.25, color=ARM_COLORS[arm],
                   s=40, label=f"{ARM_LABELS[arm]} vs BAU pool")
        ax.scatter(b.loc[arm, [f"smd_synth_{c}" for c in cov]].astype(float), y + (i - 0.5) * 0.25, facecolor="none",
                   edgecolor=ARM_COLORS[arm], s=40, label=f"{ARM_LABELS[arm]} vs its synthetic control")
    ax.axvspan(-0.25, 0.25, color=PALETTE["green"], alpha=0.08)
    ax.axvline(0, color="#777777", lw=1)
    ax.set_yticks(y, cov)
    ax.invert_yaxis()
    ax.set_xlabel("Standardised mean difference (pool SD units)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Covariate balance (design period)")
    subtitle(ax, "Shaded |SMD| < 0.25; static ACS-like + pre-period behaviour")
    source_note(fig, "Synthetic data. With 1-2 treated counties per arm perfect balance is impossible; "
                     "synthetic control matches on the outcome path instead.")
    return fig


def fig_trends(o) -> plt.Figure:
    s, up, d = o["settings"], o["units"], o["design"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, arm in zip(axes, ARMS):
        r = o["rich"][(arm, "rated", "synth_control")]
        f = r["fit"]
        x = f.actual.index
        ax.plot(x, f.actual, color=ARM_COLORS[arm], label=f"{ARM_LABELS[arm]} (actual)")
        ax.plot(x, f.synthetic, color=ROLE["synthetic"], ls="--", label="Synthetic control (locked weights)")
        c = up[up["county_id"] == d["control"][0]].set_index("week_idx")["rated_pc"].loc[x]
        ax.plot(x, c * f.actual.loc[wk(s.pre)].mean() / c.loc[wk(s.pre)].mean(), color=ROLE["control"], lw=1.2,
                alpha=0.7, label="Control county (scaled to arm pre-mean)")
        _window(ax, s)
        ax.axvline(s.pre[0] - 0.5, color=PALETTE["slate"], ls=":", lw=1)
        ax.set_ylabel("Rated quotes per 10k residents")
        ax.set_title(f"{ARM_LABELS[arm]}: {o['panel'].set_index('county_id').loc[d[arm], 'county'].unique().tolist()}")
        ax.legend(loc="lower left", ncol=3, fontsize=8.5)
    axes[-1].set_xlabel("Week index")
    source_note(fig, "Synthetic data. Dotted line = start of the 10-week readout pre-period; synthetic control is fit on "
                     f"weeks {s.sc_fit_start}-{s.pre[1]}.")
    return fig


def fig_placebo(o) -> plt.Figure:
    s = o["settings"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, arm in zip(axes, ARMS):
        r = o["rich"][(arm, "rated", "synth_control")]
        plc = r["placebo_space"]
        ax.hist(plc["stat"], bins=10, color=PALETTE["grey"], label="Placebo donors")
        ax.axvline(r["treated_stat"], color=ARM_COLORS[arm], lw=2.5, label=f"{ARM_LABELS[arm]}")
        ax.set_xlabel("Post / pre RMSPE ratio")
        ax.set_ylabel("Units")
        ax.set_title(f"{ARM_LABELS[arm]}: in-space placebo")
        subtitle(ax, f"p = {r['p_value']:.3f} (min attainable {r['min_attainable_p']:.3f}, {len(plc)} placebos)")
        ax.legend()
    source_note(fig, "Synthetic data. p = rank of the treated unit's RMSPE ratio among all units - never 0, and "
                     "bounded below by 1/(J+1).")
    return fig


def fig_bsts(o, arm="all_search", metric="rated") -> plt.Figure:
    s = o["settings"]
    r = o["rich"][(arm, metric, "bsts")]
    pw = r["pointwise"]
    up = o["units"]
    y = up[up["county_id"] == arm].set_index("week_idx")[f"{metric}_pc"].loc[wk((s.bsts_pre_start, s.post[1]))]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(y.index, y, color="#222222", label="Actual")
    axes[0].plot(pw.index, pw["predicted"], color=PALETTE["red"], ls="--", label="Counterfactual (BSTS)")
    axes[0].fill_between(pw.index, pw["pred_low"], pw["pred_high"], color=PALETTE["grey"], alpha=0.5, label="90% interval")
    axes[0].set_ylabel("Per 10k residents")
    axes[0].legend(loc="upper left", ncol=3)
    eff = pw["predicted"] - pw["actual"]
    axes[1].bar(pw.index, eff, color=ARM_COLORS[arm])
    axes[1].axhline(0, color="#555555", lw=1)
    axes[1].set_ylabel("Weekly loss (cf - actual)")
    cum = eff.cumsum()
    axes[2].plot(pw.index, cum / pw["predicted"].cumsum(), color=ARM_COLORS[arm], marker="o")
    axes[2].axhspan(r["ci_low"], r["ci_high"], color=PALETTE["grey"], alpha=0.35,
                    label=f"90% interval on 6-week lift [{r['ci_low']:.1%}, {r['ci_high']:.1%}]")
    axes[2].axhline(o["truth"][arm][metric]["true_lift_pct"], color=ROLE["truth"], ls="--", label="True lift")
    axes[2].axhline(0, color="#555555", lw=1)
    pl.pct_axis(axes[2])
    axes[2].set_ylabel("Cumulative relative loss")
    axes[2].legend(loc="lower right")
    for ax in axes:
        _window(ax, s, label=None)
    axes[2].set_xlabel("Week index")
    axes[0].set_title(f"BSTS counterfactual - {ARM_LABELS[arm]}, {METRIC_LABELS[metric]}")
    subtitle(axes[0], f"{r['model']}; covariates = control + all BAU donors; significance = interval excludes 0 "
                      f"-> {'significant' if r['significant'] else 'not significant'}")
    source_note(fig)
    return fig


def fig_forest(o) -> plt.Figure:
    est = pd.concat([o["estimates"], o["triangulated"]], ignore_index=True)
    s = o["settings"]
    meths = ["ratio_did", "ols_fe", "bsts", "synth_control", "triangulated"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
    for i, arm in enumerate(ARMS):
        for j, m in enumerate(s.readout_metrics):
            ax = axes[i, j]
            g = est[(est["arm"] == arm) & (est["metric"] == m)].set_index("method").loc[meths]
            y = np.arange(len(meths))
            for k, (meth, r) in enumerate(g.iterrows()):
                c = PALETTE["navy"] if meth != "triangulated" else PALETTE["orange"]
                ax.errorbar(r["estimate"], k, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                            fmt="o", color=c, ecolor=c, capsize=4, ms=6, mfc=c if r["significant"] else "white")
            ax.axvline(g["truth"].iloc[0], color=ROLE["truth"], ls="--", lw=1.6, label="True lift")
            ax.axvline(0, color="#777777", lw=1)
            ax.set_yticks(y, [METHOD_LABELS[mm] for mm in meths])
            ax.invert_yaxis() if j == 0 and i == 0 else None
            pl.pct_axis(ax, "x")
            ax.set_title(f"{ARM_LABELS[arm]} | {METRIC_LABELS[m]}", fontsize=11)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Estimate vs truth: four estimators + triangulation (90% intervals; filled = significant)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    source_note(fig, "Synthetic data. Lift = share of volume lost when the arm's paid search was dark (positive = "
                     "paid search contribution). Truth = scenario expected lift.")
    return fig


def fig_power(o) -> plt.Figure:
    c = o["power"]["curves"]
    s = o["settings"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    lifts = list(s.lift_grid)
    for ax, arm in zip(axes, ARMS):
        g = c[(c["arm"] == arm) & (c["metric"] == "rated")]
        for k, lift in enumerate(lifts):
            h = g[g["lift"] == lift]
            ax.plot(h["weeks"], h["power"], color=plt.cm.viridis(k / max(1, len(lifts) - 1)), label=f"{lift:.0%}")
            hit = h[h["power"] >= s.power]
            if len(hit):
                w = int(hit["weeks"].iloc[0])
                ax.scatter(w, h[h["weeks"] == w]["power"], color="black", s=18, zorder=3)
        ax.axhline(s.power, color="black", ls="--", lw=1)
        ax.axvline(s.n_post, color=PALETTE["red"], ls=":", lw=1.4)
        ax.text(s.n_post, 0.03, " planned 6 wk", color=PALETTE["red"], fontsize=8.5)
        sim = o["power"]["simulation"]
        sm = sim[sim["arm"] == arm]
        ax.scatter(np.full(len(sm), s.n_post), sm["power"], marker="x", color=PALETTE["slate"], s=40,
                   label="Simulated (Welch, 6 wk)")
        ax.set_xlabel("Test weeks (pre-period = max(10, weeks))")
        ax.set_title(f"{ARM_LABELS[arm]} - rated quotes")
        pl.pct_axis(ax)
    axes[0].set_ylabel("Power (two-sided alpha = 0.10)")
    axes[1].legend(title="True lift", loc="upper left", bbox_to_anchor=(1.01, 1))
    source_note(fig, "Synthetic data. Closed-form DiD power on the synthetic-control gap (AR(1)-inflated); "
                     "x = simulation power at 3/6/10/15% injected loss over random historical windows.")
    return fig


def fig_sources(o) -> plt.Figure:
    s = o["settings"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, arm in zip(axes, ARMS):
        f = o["source_effects"][arm]
        x = np.arange(len(SOURCES) + 1)
        labs = [SOURCE_LABELS[b] for b in SOURCES] + ["Total"]
        ax.bar(x - 0.2, f["estimate"], 0.4, color=ROLE["estimate"], label="Estimated (synthetic control per source)")
        ax.bar(x + 0.2, f["truth"], 0.4, color=ROLE["truth"], label="Truth (vs BAU twin)")
        ax.axhline(0, color="#555555", lw=1)
        ax.set_xticks(x, labs, rotation=25, ha="right")
        ax.set_title(f"{ARM_LABELS[arm]}: rated quotes by last-touch source")
        subtitle(ax, "Change per week during dark-out vs counterfactual")
    axes[0].set_ylabel("Rated quotes / week (actual - counterfactual)")
    axes[0].legend(loc="lower left")
    source_note(fig, "Synthetic data. Paid search conversions vanish; organic picks part of them back up "
                     "(cannibalisation), the remainder is the true loss.")
    return fig


def fig_recapture(o) -> plt.Figure:
    r = o["recapture"]
    r = r[r["ref"] == "synthetic"]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, k = [], 0
    for arm in ARMS:
        for st in ("qs", "rated", "sales"):
            row = r[(r["arm"] == arm) & (r["stage"] == st)].iloc[0]
            lo, hi = max(row["ci_low"], -0.5), min(row["ci_high"], 1.5)
            ax.errorbar(k, row["recapture"], yerr=[[row["recapture"] - lo], [hi - row["recapture"]]], fmt="o",
                        color=ARM_COLORS[arm], capsize=5, ms=8)
            ax.scatter(k, row["truth"], marker="_", s=600, color=ROLE["truth"], zorder=4,
                       label="True recapture" if k == 0 else None)
            labels.append(f"{ARM_LABELS[arm]}\n{METRIC_LABELS[st]}")
            k += 1
    ax.set_xticks(range(k), labels, fontsize=8.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axhline(0, color="#777777", lw=1)
    pl.pct_axis(ax)
    ax.set_ylabel("Organic recapture (% of lost paid conversions)")
    ax.set_title("Cannibalisation: organic recapture when paid search goes dark")
    subtitle(ax, "Recapture = Δ organic / -Δ paid (per-source synthetic control); 90% week-bootstrap CI (clipped)")
    ax.legend(loc="upper left")
    source_note(fig, "Synthetic data. True brand recapture = 40%, non-brand = 15%; the all-search blend depends "
                     "on the arm's brand/non-brand mix.")
    return fig


def fig_costs(o) -> plt.Figure:
    e = o["economics"]["table"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m in zip(axes, ("rated", "sales")):
        g = e[e["metric"] == m].set_index("arm").loc[list(ARMS)]
        x = np.arange(len(ARMS))
        est = g["incr_cost_per_est"].where(g["incr_cost_per_est"] > 0)
        cap = np.nanmax(np.r_[g["incr_cost_per_true"].astype(float), g["last_touch_cost_per"], est.fillna(0)]) * 1.6
        ax.bar(x - 0.27, g["last_touch_cost_per"], 0.27, color=ROLE["attribution"], label="Last-touch")
        ax.bar(x, est.fillna(0).clip(upper=cap), 0.27, color=ROLE["estimate"], label="Incremental (estimated)")
        lo = g["incr_cost_per_ci_low"].fillna(0)
        hi = g["incr_cost_per_ci_high"].replace(np.inf, cap).fillna(cap).clip(upper=cap)
        ax.errorbar(x, est.fillna(0).clip(upper=cap), yerr=[np.maximum(est.fillna(0) - lo, 0), np.maximum(hi - est.fillna(0), 0)],
                    fmt="none", ecolor="#222222", capsize=4)
        ax.bar(x + 0.27, g["incr_cost_per_true"].astype(float), 0.27, color=ROLE["truth"], label="Incremental (truth)")
        for xi, v in zip(x, est):
            if not np.isfinite(v):
                ax.text(xi, cap * 0.05, "n/a\n(est. <= 0)", ha="center", fontsize=8)
        ax.set_ylim(0, cap)
        ax.set_xticks(x, [ARM_LABELS[a] for a in ARMS])
        pl.money_axis(ax)
        ax.set_title(f"Cost per {METRIC_LABELS[m].lower()[:-1]}")
        ax.legend(loc="upper left", fontsize=8.5)
    subtitle(axes[0], "Spend removed / incremental outcomes; 90% CI (upper end capped)")
    source_note(fig)
    return fig


def fig_guardrail(o) -> plt.Figure:
    s = o["settings"]
    g = o["guardrail"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(g))
    ax.bar(x - 0.18, g["estimate"], 0.36, color=ROLE["estimate"], label="Estimated visit loss (SC)")
    ax.errorbar(x - 0.18, g["estimate"], yerr=[g["estimate"] - g["ci_low"], g["ci_high"] - g["estimate"]], fmt="none",
                ecolor="#222222", capsize=4)
    ax.bar(x + 0.18, g["truth"], 0.36, color=ROLE["truth"], label="True visit loss")
    ax.axhline(s.traffic_guardrail_max_loss, color=PALETTE["red"], ls="--", label="Guardrail")
    ax.set_xticks(x, [ARM_LABELS[a] for a in g["arm"]])
    pl.pct_axis(ax)
    ax.set_ylabel("Site visits lost during dark-out")
    ax.set_title("Traffic guardrail: total site visits")
    ax.legend(loc="upper right")
    source_note(fig)
    return fig


def fig_truth_table(o) -> plt.Figure:
    t = o["truth_table"]
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(t) + 1.2))
    ax.axis("off")
    cols = ["quantity", "estimate", "ci", "truth", "error"]
    cell = [[r["quantity"], r["estimate_fmt"], r["ci_fmt"], r["truth_fmt"], r["error_fmt"]] for _, r in t.iterrows()]
    tb = ax.table(cellText=cell, colLabels=["Quantity", "Estimate", "90% CI", "Truth", "Est - truth"], loc="center",
                  cellLoc="left", colLoc="left", colWidths=[0.46, 0.12, 0.2, 0.11, 0.11])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#DDDDDD")
        if i == 0:
            c.set_facecolor(PALETTE["navy"])
            c.set_text_props(color="white", fontweight="bold")
        elif i % 2 == 0:
            c.set_facecolor(PALETTE["light"])
    ax.set_title("Truth vs estimate - every headline quantity", loc="left")
    return fig


def make_all(o) -> List[str]:
    specs = [("01_design_map", fig_map), ("02_county_mde", fig_mde), ("03_injected_lift", fig_injected),
             ("04_balance", fig_balance), ("05_trends_sc", fig_trends), ("06_placebo", fig_placebo),
             ("07_bsts_all_search_rated", fig_bsts), ("08_forest_truth", fig_forest), ("09_power", fig_power),
             ("10_source_decomposition", fig_sources), ("11_recapture", fig_recapture),
             ("12_cost_per", fig_costs), ("13_guardrail", fig_guardrail), ("14_truth_table", fig_truth_table)]
    paths = []
    for name, fn in specs:
        fig = fn(o)
        p = save_fig(fig, MODULE, name)
        paths.append(str(p.relative_to(p.parents[3])))
    return paths
