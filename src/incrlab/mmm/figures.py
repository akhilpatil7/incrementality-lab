"""Figures for the calibrated MMM module (reports/figures/mmm/*.png). Every function returns a Figure."""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..common.plotting import (CHANNEL_COLORS, PALETTE, ROLE, channel_label, money_axis, save_fig,
                               shade_window, source_note, subtitle)
from ..config import get_config
from . import analysis as A
from .pipeline import CURVE_MULTS, HOLDOUT_WEEKS, MMMState, calibration_fit_table, roi_table, triangulation_table

MOD = "mmm"
FIT_STYLE = {"uncalibrated": (PALETTE["grey"], "Uncalibrated MMM"),
             "calibrated": (PALETTE["navy"], "Calibrated MMM (lift tests)"),
             "calibrated_powered": (PALETTE["teal"], "Calibrated, well-powered tests (sensitivity)")}
TEST_COLORS = {"meta": PALETTE["purple"], "search": PALETTE["sky"], "dsp_a": PALETTE["orange"],
               "dsp_b": PALETTE["gold"]}
SHORT = {"tv": "TV / Brand", "social": "Social", "search_brand": "Search - Brand",
         "search_nonbrand": "Search - Non-brand", "dsp_a": "DSP - Vendor A", "dsp_b": "DSP - Vendor B"}


def _title(ax, title: str, sub: str = None):
    ax.set_title(title, pad=22 if sub else 8)
    if sub:
        subtitle(ax, sub)


def _note(fig, S: MMMState, extra: str = "", y: float = -0.02):
    stub = f" Calibration inputs for {', '.join(S.stub_tests)} are STUB (fallback) geo tests." if S.stub_tests else ""
    fig.text(0.01, y, "Synthetic data - illustrative magnitudes only." + stub + (" " + extra if extra else ""),
             fontsize=8, color="#777777", ha="left", va="top")


def _shade_tests(ax, S, label=True):
    cfg = get_config()
    for name, td in cfg.tests.items():
        ax.axvspan(td.test_start, td.test_end, color=TEST_COLORS[name], alpha=0.15, lw=0, zorder=0)
        if label:
            ax.text((td.test_start + td.test_end) / 2, 1.0, name.replace("_", " ").upper(), ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(), fontsize=7.5, color="#555555")


# 01 ---------------------------------------------------------------------------------------
def fig_data_overview(S: MMMState):
    d = S.data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax1.plot(d.week_idx, d.y, color=PALETTE["navy"], lw=1.6)
    ax1.set_ylabel("National sales / week")
    _shade_tests(ax1, S)
    ax1.axvspan(d.week_idx[-HOLDOUT_WEEKS], d.week_idx[-1], color=PALETTE["slate"], alpha=0.10, lw=0)
    ax1.text(d.week_idx[-HOLDOUT_WEEKS] + 0.5, 0.04, "holdout", transform=ax1.get_xaxis_transform(), fontsize=8,
             color=PALETTE["slate"])
    ax1.set_title("MMM input: weekly national sales and paid media spend", pad=16)
    ax2.stackplot(d.week_idx, d.spend, colors=[CHANNEL_COLORS[c] for c in d.channels],
                  labels=[SHORT[c] for c in d.channels], alpha=0.9)
    _shade_tests(ax2, S, label=False)
    money_axis(ax2)
    ax2.set_ylabel("Spend / week")
    ax2.set_xlabel("Week index (0 = first week of year 1)")
    ax2.legend(ncol=6, loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8.5)
    _note(fig, S, "Coloured bands = geo-test windows used for calibration.", y=-0.10)
    fig.tight_layout()
    return fig


# 02 ---------------------------------------------------------------------------------------
def fig_calibration_inputs(S: MMMState):
    tab = calibration_fit_table(S)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y = np.arange(len(tab))[::-1]
    k = 1000
    ax.errorbar(tab["observed_ipd"] * k, y + 0.22, xerr=1.645 * tab["se"] * k, fmt="o", color=ROLE["estimate"],
                capsize=3, label="Experiment readout (90% CI)")
    for off, name in ((0.0, "uncalibrated"), (-0.22, "calibrated")):
        col, lab = FIT_STYLE[name]
        ax.errorbar(tab[f"{name}_ipd"] * k, y + off,
                    xerr=[(tab[f"{name}_ipd"] - tab[f"{name}_ipd_lo"]) * k, (tab[f"{name}_ipd_hi"] - tab[f"{name}_ipd"]) * k],
                    fmt="s", color=col, capsize=3, label=f"{lab}: implied (90% HDI)")
    tr = tab["truth_ipd"].astype(float)
    ax.scatter(tr * k, y, marker="D", s=50, color=ROLE["truth"], zorder=5, label="Truth (test geos)")
    labels = [f"{r.test} / {r.arm}\n({r.channels}){'  [STUB]' if str(r.source).startswith('stub') else ''}"
              for r in tab.itertuples()]
    ax.set_yticks(y, labels, fontsize=9)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_xlabel("Incremental sales per $1,000 removed (test window)")
    _title(ax, "Lift tests vs what each MMM implies for the same test", "The calibrated MMM must explain these readouts; noisy tests (wide CIs) pull it only weakly")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    _note(fig, S)
    return fig


# 03 ---------------------------------------------------------------------------------------
def fig_fit_ppc(S: MMMState):
    d = S.data
    f = S.fits["calibrated"]
    p = f.diag["_ppc"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(d.week_idx, p["lo"], p["hi"], color=PALETTE["sky"], alpha=0.35, lw=0, label="90% posterior predictive")
    ax.plot(d.week_idx, p["mean"], color=PALETTE["navy"], lw=1.5, label="Posterior mean")
    ax.plot(d.week_idx, d.y, "o", ms=2.8, color=PALETTE["orange"], label="Actual sales")
    shade_window(ax, d.week_idx[-HOLDOUT_WEEKS], d.week_idx[-1], label="Holdout (not fit)", color=PALETTE["slate"])
    dg = f.diag
    _title(ax, "Calibrated MMM: posterior predictive check", f"Train MAPE {dg['train_mape']:.1%}, R² {dg['train_r2']:.2f} | holdout MAPE {dg['holdout_mape']:.1%}, "
                 f"R² {dg['holdout_r2']:.2f} | max r-hat {dg['max_r_hat']:.3f}, min ESS {dg['min_ess_bulk']:.0f}, "
                 f"divergences {dg['divergences']}")
    ax.set_xlabel("Week index")
    ax.set_ylabel("Sales / week")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.13))
    _note(fig, S, y=-0.10)
    return fig


def _forest(ax, S, key: str, truth_key: str, fits: List[str]):
    chans = S.data.channels
    y = np.arange(len(chans))[::-1]
    n = len(fits)
    for i, name in enumerate(fits):
        f = S.fits[name]
        x = getattr(f, key)
        lo, hi = A.hdi(x)
        m = x.mean(0)
        off = (i - (n - 1) / 2) * 0.22
        col, lab = FIT_STYLE[name]
        ax.errorbar(m, y + off, xerr=[m - lo, hi - m], fmt="o", color=col, capsize=3, label=f"{lab} (90% HDI)")
    ax.scatter(S.truth[truth_key], y, marker="D", s=55, color=ROLE["truth"], zorder=5, label="Truth")
    ax.set_yticks(y, [SHORT[c] for c in chans])
    ax.axvline(0, color="#999999", lw=1)


# 04 ---------------------------------------------------------------------------------------
def fig_roi_forest(S: MMMState):
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _forest(ax, S, "roi", "roi", [f for f in ("uncalibrated", "calibrated", "calibrated_powered") if f in S.fits])
    ax.set_xlabel("ROI = incremental sales value / spend (whole 3-yr period)")
    _title(ax, "Channel ROI: uncalibrated vs experiment-calibrated MMM vs truth", "Same data, same priors; the calibrated fit adds one likelihood term per lift test")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    _note(fig, S)
    return fig


# 05 ---------------------------------------------------------------------------------------
def fig_roi_error(S: MMMState):
    t = roi_table(S)
    fits = [f for f in ("uncalibrated", "calibrated", "calibrated_powered") if f in S.fits]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(t))
    w = 0.8 / len(fits)
    for i, name in enumerate(fits):
        col, lab = FIT_STYLE[name]
        v = t[f"{name}_roi_abs_err"].to_numpy()
        ax.bar(x + (i - (len(fits) - 1) / 2) * w, v, w, color=col, label=lab)
    ax.set_xticks(x, [SHORT[c] for c in t["channel"]], fontsize=9)
    ax.set_ylabel("|posterior mean ROI - true ROI|")
    mae = {n: t[f"{n}_roi_abs_err"].mean() for n in fits}
    _title(ax, "ROI error vs ground truth, by channel", "Mean abs. error: " + " | ".join(f"{FIT_STYLE[n][1].split(' (')[0].split(',')[0]} {v:.2f}"
                                                for n, v in mae.items()))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
    _note(fig, S, y=-0.10)
    return fig


# 06 ---------------------------------------------------------------------------------------
def fig_mroi_forest(S: MMMState):
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _forest(ax, S, "mroi", "mroi", ["uncalibrated", "calibrated"])
    ax.axvline(1.0, color=PALETTE["red"], ls="--", lw=1.2)
    ax.text(1.0, 1.0, " break-even", color=PALETTE["red"], fontsize=8.5, transform=ax.get_xaxis_transform(), va="bottom")
    ax.set_xlabel("Marginal ROI of the next +1% spend")
    _title(ax, "Marginal ROI: where the next dollar works", "Budget moves are driven by marginal, not average, ROI")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    _note(fig, S)
    return fig


# 07 ---------------------------------------------------------------------------------------
def fig_response_curves(S: MMMState):
    d = S.data
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for j, (c, ax) in enumerate(zip(d.channels, axes.ravel())):
        sp = CURVE_MULTS * d.spend[j].sum()
        for name in ("uncalibrated", "calibrated"):
            col, lab = FIT_STYLE[name]
            cur = S.fits[name].curves[:, j, :]
            lo, hi = np.quantile(cur, [0.05, 0.95], axis=0)
            ax.fill_between(sp, lo, hi, color=col, alpha=0.22, lw=0)
            ax.plot(sp, cur.mean(0), color=col, lw=2, label=lab)
        if "curves" in S.truth:
            ax.plot(sp, S.truth["curves"][j], color=ROLE["truth"], ls="--", lw=2, label="Truth")
        ax.axvline(d.spend[j].sum(), color="#999999", ls=":", lw=1.2)
        ax.set_title(SHORT[c], fontsize=11)
        money_axis(ax, "x")
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=8.5)
        if j % 3 == 0:
            ax.set_ylabel("Incremental sales (3 yrs)")
    axes[1, 1].set_xlabel("Total 3-yr spend (BAU flighting scaled; dotted = actual)")
    h = [Line2D([], [], color=FIT_STYLE[n][0], lw=2) for n in ("uncalibrated", "calibrated")] + \
        [Line2D([], [], color=ROLE["truth"], ls="--", lw=2)]
    fig.legend(h, ["Uncalibrated (90% band)", "Calibrated (90% band)", "Truth"], ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Response curves with posterior uncertainty", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _note(fig, S)
    return fig


# 08 ---------------------------------------------------------------------------------------
def fig_adstock(S: MMMState):
    d = S.data
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharex=True)
    bins = np.linspace(0, 1, 41)
    for j, (c, ax) in enumerate(zip(d.channels, axes.ravel())):
        for name in ("uncalibrated", "calibrated"):
            col, lab = FIT_STYLE[name]
            ax.hist(S.fits[name].draws.decay[:, j], bins=bins, density=True, color=col, alpha=0.55, label=lab)
        ax.axvline(S.truth["decay"][j], color=ROLE["truth"], lw=2.2, ls="--", label="Truth")
        ax.set_title(SHORT[c], fontsize=11)
        ax.set_yticks([])
        if j >= 3:
            ax.set_xlabel("Geometric adstock decay")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Adstock (carry-over) posteriors - prior Beta(2, 2.5)", x=0.01, ha="left", fontweight="bold",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _note(fig, S, "Weekly national data identify carry-over only loosely; tests on window totals add little here.")
    return fig


# 09 ---------------------------------------------------------------------------------------
def fig_contribution_area(S: MMMState):
    d = S.data
    dr = S.fits["calibrated"].draws
    base = A.baseline(dr, d).mean(0)
    con = A.contributions(dr, d).mean(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(d.week_idx, np.vstack([base, con]), colors=[CHANNEL_COLORS["base"]] + [CHANNEL_COLORS[c] for c in d.channels],
                 labels=["Baseline (no paid media)"] + [SHORT[c] for c in d.channels], alpha=0.92)
    ax.plot(d.week_idx, d.y, color="black", lw=1, label="Actual sales")
    ax.set_xlabel("Week index")
    ax.set_ylabel("Sales / week")
    ax.set_title("Sales decomposition (calibrated MMM, posterior mean)")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8.5)
    _note(fig, S, y=-0.10)
    return fig


# 10 ---------------------------------------------------------------------------------------
def fig_waterfall(S: MMMState):
    d = S.data
    dr = S.fits["calibrated"].draws
    base = A.baseline(dr, d).sum(1)
    con = A.contributions(dr, d).sum(-1)
    order = np.argsort(-con.mean(0))
    names = ["Baseline"] + [SHORT[d.channels[j]] for j in order] + ["Total (model)"]
    vals = [base.mean()] + [con[:, j].mean() for j in order]
    errs = [None] + [np.quantile(con[:, j], [0.05, 0.95]) for j in order]
    fig, ax = plt.subplots(figsize=(11, 5))
    cum = 0.0
    for i, (n, v) in enumerate(zip(names[:-1], vals)):
        col = CHANNEL_COLORS["base"] if i == 0 else CHANNEL_COLORS[d.channels[order[i - 1]]]
        ax.bar(i, v, bottom=cum, color=col, width=0.65)
        if errs[i] is not None:
            ax.errorbar(i, cum + v, yerr=[[v - errs[i][0]], [errs[i][1] - v]], color="#333333", capsize=3, lw=1)
            ax.text(i + 0.36, cum + v / 2, f"+{v:,.0f}\n({v / d.y.sum():.1%})", ha="left", va="center", fontsize=8)
        else:
            ax.text(i, v - 0.02 * v, f"{v:,.0f}\n({v / d.y.sum():.0%})", ha="center", va="top", fontsize=8.5,
                    color="#333333")
        cum += v
    ax.bar(len(names) - 1, cum, color=PALETTE["navy"], width=0.65)
    ax.text(len(names) - 1, cum * 0.985, f"{cum:,.0f}\n(actual\n{d.y.sum():,.0f})", ha="center", va="top",
            fontsize=8.5, color="white")
    ax.set_xticks(range(len(names)), names, fontsize=8.5, rotation=15)
    ax.set_xlim(-0.5, len(names) - 0.2)
    lo_y = 0.85 * base.mean()
    ax.set_ylim(lo_y, cum * 1.10)
    ax.set_ylabel("Sales, 3 years (axis truncated)")
    _title(ax, "Contribution waterfall: where 3 years of sales came from", "Calibrated MMM posterior mean, 90% intervals on channel bars")
    _note(fig, S)
    return fig


# 11 ---------------------------------------------------------------------------------------
def fig_triangulation(S: MMMState):
    t = triangulation_table(S)
    series = [("last_touch_roi", "Last-touch attribution", ROLE["attribution"]),
              ("experiment_roi", "Geo experiment (test window)", PALETTE["purple"]),
              ("uncalibrated_roi", "Uncalibrated MMM", FIT_STYLE["uncalibrated"][0]),
              ("calibrated_roi", "Calibrated MMM", FIT_STYLE["calibrated"][0]),
              ("true_roi", "Truth", ROLE["truth"])]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(t))
    w = 0.16
    for i, (k, lab, col) in enumerate(series):
        v = t[k].astype(float).to_numpy()
        xs = x + (i - 2) * w
        ax.bar(xs, np.nan_to_num(v), w, color=col, label=lab)
        if k == "experiment_roi":
            se = t["experiment_roi_se"].astype(float).to_numpy()
            ok = np.isfinite(v)
            ax.errorbar(xs[ok], v[ok], yerr=1.645 * se[ok], fmt="none", color="#333333", capsize=2, lw=0.9)
            for xi, vi in zip(xs, v):
                if not np.isfinite(vi):
                    ax.text(xi, 0.05, "no test", rotation=90, fontsize=7.5, ha="center", va="bottom", color="#777777")
    ax.axhline(0, color="#999999", lw=1)
    ax.set_xticks(x, [SHORT[c] for c in t["channel"]], fontsize=9)
    ax.set_ylabel("ROI (sales value / spend)")
    ymax = np.nanmax(t[[s[0] for s in series]].astype(float).to_numpy()) * 1.25
    ax.set_ylim(min(-1.0, np.nanmin(t["experiment_roi"].astype(float)) - 0.5), ymax)
    _title(ax, "Triangulation: five answers to 'what does this channel return?'", "Experiment bars show 90% CI (clipped); experiments measure one window in a few geos")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=8.5)
    _note(fig, S, "DSP last-touch splits the pooled site last-touch by vendor-claimed conversions.", y=-0.10)
    return fig


# 12 ---------------------------------------------------------------------------------------
def fig_optimizer_mix(S: MMMState):
    o = S.optimizer["calibrated"]
    d = S.data
    sq = np.array([o["status_quo"]["spend_by_channel"][c] for c in d.channels])
    op = np.array([o["optimal"]["spend_by_channel"][c] for c in d.channels])
    ou = np.array([S.optimizer["uncalibrated"]["optimal"]["spend_by_channel"][c] for c in d.channels])
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(d.C)
    ax.bar(x - 0.27, sq, 0.27, color=PALETTE["grey"], label="Status quo (last 52 wks)")
    ax.bar(x, ou, 0.27, color=PALETTE["sky"], label="Optimal - uncalibrated MMM")
    ax.bar(x + 0.27, op, 0.27, color=PALETTE["navy"], label="Optimal - calibrated MMM")
    for i in range(d.C):
        ch = op[i] / sq[i] - 1
        ax.text(x[i] + 0.27, op[i] * 1.02, f"{ch:+.0%}", ha="center", fontsize=8.5,
                color=PALETTE["green"] if ch > 0 else PALETTE["red"])
    money_axis(ax)
    ax.set_xticks(x, [SHORT[c] for c in d.channels], fontsize=9)
    ax.set_ylabel("Annual spend")
    dl = o["optimal"]["delta_vs_ref"]
    _title(ax, "Recommended reallocation (same total budget, ±30% per channel)", f"Calibrated plan: +{dl['mean']:,.0f} incremental sales/yr (90% HDI {dl['hdi_low']:,.0f} to "
                 f"{dl['hdi_high']:,.0f}), P(better) = {o['optimal']['prob_better']:.0%}")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    _note(fig, S, y=-0.10)
    return fig


# 13 ---------------------------------------------------------------------------------------
def fig_optimizer_outcome(S: MMMState):
    o = S.optimizer["calibrated"]
    bp = A.BudgetProblem(S.fits["calibrated"].draws, S.data, S.value_per_sale)
    m_opt = np.array([o["optimal"]["multipliers"][c] for c in S.data.channels])
    sq = bp.incremental(np.ones(S.data.C))
    op = bp.incremental(m_opt)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.linspace(min(sq.min(), op.min()), max(sq.max(), op.max()), 40)
    ax1.hist(sq, bins, color=PALETTE["grey"], alpha=0.7, density=True, label="Status quo")
    ax1.hist(op, bins, color=PALETTE["navy"], alpha=0.6, density=True, label="Optimized")
    tc = o.get("truth_check")
    if tc:
        ax1.axvline(tc["status_quo_true_incremental"], color=PALETTE["slate"], ls="--", lw=2, label="True: status quo")
        ax1.axvline(tc["optimal_true_incremental"], color=ROLE["truth"], ls="--", lw=2, label="True: optimized")
    ax1.set_yticks([])
    ax1.set_xlabel("Media-driven incremental sales, plan year")
    ax1.set_title("Expected incremental sales", fontsize=11.5)
    ax1.legend(fontsize=8)
    dd = op - sq
    ax2.hist(dd, 40, color=PALETTE["teal"], alpha=0.8, density=True)
    ax2.axvline(0, color="#555555", lw=1)
    if tc:
        ax2.axvline(tc["true_gain"], color=ROLE["truth"], ls="--", lw=2, label=f"True gain {tc['true_gain']:+,.0f}")
        ax2.legend(fontsize=8.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Gain vs status quo (sales)")
    cpa_sq, cpa_op = o["status_quo"]["cpa"], o["optimal"]["cpa"]
    ax2.set_title(f"Gain: P(>0) = {np.mean(dd > 0):.0%}; CPA \\${cpa_sq['mean']:,.0f} -> \\${cpa_op['mean']:,.0f}",
                  fontsize=11.5)
    fig.suptitle("Posterior-aware budget plan: what we expect, and how sure we are", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _note(fig, S)
    return fig


# 14 ---------------------------------------------------------------------------------------
def fig_scenarios(S: MMMState):
    o = S.optimizer["calibrated"]
    rows = [("-20% budget\nproportional", o["scenarios"]["0.8x"]["proportional"]),
            ("-20% budget\noptimized", o["scenarios"]["0.8x"]["optimized"]),
            ("Status quo", o["status_quo"]),
            ("Same budget\noptimized", o["optimal"]),
            ("+20% budget\nproportional", o["scenarios"]["1.2x"]["proportional"]),
            ("+20% budget\noptimized", o["scenarios"]["1.2x"]["optimized"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(rows))
    cols = [PALETTE["grey"], PALETTE["navy"], PALETTE["slate"], PALETTE["navy"], PALETTE["grey"], PALETTE["navy"]]
    for ax, key, lab in ((ax1, "incremental_sales", "Incremental sales (plan year)"), (ax2, "cpa", "Cost per incremental sale")):
        m = np.array([r[1][key]["mean"] for r in rows])
        lo = np.array([r[1][key]["hdi_low"] for r in rows])
        hi = np.array([r[1][key]["hdi_high"] for r in rows])
        ax.bar(x, m, 0.6, color=cols, yerr=[m - lo, hi - m], capsize=3, error_kw={"lw": 1})
        if key == "incremental_sales":
            tv = [r[1].get("true_incremental", None) for r in rows]
            tv[2] = (o.get("truth_check") or {}).get("status_quo_true_incremental")
            tv[3] = (o.get("truth_check") or {}).get("optimal_true_incremental")
            ok = [i for i, v in enumerate(tv) if v is not None]
            if ok:
                ax.scatter(x[ok], [tv[i] for i in ok], marker="D", color=ROLE["truth"], zorder=5, label="Truth")
                ax.legend(fontsize=8.5, loc="upper left")
        else:
            money_axis(ax)
        ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
        ax.set_title(lab, fontsize=11.5)
    fig.suptitle("Budget scenarios: -20% / status quo / +20%, proportional vs optimized mix (90% HDI)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    _note(fig, S)
    return fig


# 15 ---------------------------------------------------------------------------------------
def fig_voi(S: MMMState):
    v = S.voi
    chans = sorted(v, key=lambda c: -v[c]["evppi_value"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    vals = [v[c]["evppi_value"] for c in chans]
    ax1.barh(range(len(chans))[::-1], vals, color=[CHANNEL_COLORS[c] for c in chans])
    ax1.set_yticks(range(len(chans))[::-1], [SHORT[c] for c in chans])
    money_axis(ax1, "x")
    ax1.set_xlabel("Expected value of perfect information on ROI ($ / plan year)")
    ax1.set_title(f"Next test: {SHORT[chans[0]]}", fontsize=11.5)
    for i, c in enumerate(chans):
        ax1.text(vals[i], len(chans) - 1 - i, f"  {v[c]['evppi_sales']:,.0f} sales", va="center", fontsize=8.5)
    ax1.set_xlim(0, max(vals) * 1.35 + 1)
    for c in S.data.channels:
        j = S.data.channels.index(c)
        ax2.scatter(v[c]["roi_cv"], S.fits["calibrated"].mroi[:, j].mean(), s=80 + 4e-3 * v[c]["evppi_value"],
                    color=CHANNEL_COLORS[c])
        ax2.annotate(SHORT[c], (v[c]["roi_cv"], S.fits["calibrated"].mroi[:, j].mean()), fontsize=8,
                     xytext=(5, 4), textcoords="offset points")
    ax2.axhline(1.0, color=PALETTE["red"], ls="--", lw=1)
    ax2.set_xlabel("Posterior ROI uncertainty (coef. of variation)")
    ax2.set_ylabel("Marginal ROI (posterior mean)")
    ax2.set_title("Uncertainty vs decision relevance", fontsize=11.5)
    fig.suptitle("Value of information: which test would change the budget most?", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _note(fig, S, "EVPPI: re-optimise within posterior ROI quartiles; gain vs current optimum.")
    return fig


# 16 ---------------------------------------------------------------------------------------
def fig_uncertainty_shrink(S: MMMState):
    t = roi_table(S)
    fits = [f for f in ("uncalibrated", "calibrated", "calibrated_powered") if f in S.fits]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(t))
    w = 0.8 / len(fits)
    for i, name in enumerate(fits):
        col, lab = FIT_STYLE[name]
        wd = (t[f"{name}_roi_hi"] - t[f"{name}_roi_lo"]).to_numpy()
        cov = t[f"{name}_roi_covers_truth"].to_numpy()
        b = ax.bar(x + (i - (len(fits) - 1) / 2) * w, wd, w, color=col, label=lab)
        for bb, c in zip(b, cov):
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height(), "✓" if c else "✗", ha="center", va="bottom",
                    fontsize=9, color=PALETTE["green"] if c else PALETTE["red"])
    ax.set_xticks(x, [SHORT[c] for c in t["channel"]], fontsize=9)
    ax.set_ylabel("Width of 90% ROI HDI")
    _title(ax, "Uncertainty: how much do the tests narrow the ROI posterior?", "✓ / ✗ = 90% HDI contains the true ROI")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
    _note(fig, S, y=-0.10)
    return fig


ALL = [("01_data_overview", fig_data_overview), ("02_calibration_inputs", fig_calibration_inputs),
       ("03_fit_ppc", fig_fit_ppc), ("04_roi_forest", fig_roi_forest), ("05_roi_error", fig_roi_error),
       ("06_marginal_roi", fig_mroi_forest), ("07_response_curves", fig_response_curves),
       ("08_adstock_posteriors", fig_adstock), ("09_contribution_area", fig_contribution_area),
       ("10_contribution_waterfall", fig_waterfall), ("11_triangulation", fig_triangulation),
       ("12_optimizer_mix", fig_optimizer_mix), ("13_optimizer_outcome", fig_optimizer_outcome),
       ("14_budget_scenarios", fig_scenarios), ("15_value_of_information", fig_voi),
       ("16_uncertainty_width", fig_uncertainty_shrink)]


def make_all(S: MMMState, fig_dir=None) -> list:
    paths = []
    for name, fn in ALL:
        fig = fn(S)
        paths.append(save_fig(fig, MOD, name, fig_dir=fig_dir))
    return paths
