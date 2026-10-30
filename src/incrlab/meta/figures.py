"""Figures for the Social geo-holdout module (common/plotting theme; saved to reports/figures/meta/)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Polygon

from .. import geo
from ..common.plotting import (PALETTE, ROLE, SERIES, mark_line, money_axis, pct_axis, save_fig, shade_window,
                               source_note, subtitle)
from ..simulate import load_table
from .readout import STAGE_LABEL, STAGES


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

MODULE = "meta"
METHOD_LABEL = {"did_week_fe_hc1": "DiD (week FE, HC1)", "ratio_did": "Ratio DiD",
                "synthetic_control_gap": "Synthetic-control gap"}


def _county_names(S) -> dict:
    return S.county_static.set_index("county_id")["county"].str.replace(" County", "", regex=False).to_dict()


def _unit_label(S, unit: str) -> str:
    n = _county_names(S)
    return " + ".join(n.get(m, m) for m in unit.split("+"))


def _windows(ax, S, design=True, test=True):
    st = S.st
    if design:
        shade_window(ax, st.train_end + 0.5, st.design_end + 0.5, label="Pseudo-post (validation)",
                     color=PALETTE["sky"], alpha=0.15)
    if test:
        shade_window(ax, st.test_start - 0.5, st.test_end + 0.5, label="Holdout test", alpha=0.25)


# ------------------------------------------------------------------------------------------
def fig_map(S):
    zs = load_table("zip_static", S.cfg.data_dir)
    zs = zs[zs["region"] == S.st.region].reset_index(drop=True)
    polys = geo.voronoi_polygons(zs["lon"], zs["lat"])
    treated = set(S.unit.split("+"))
    w = S.dfit.weights
    cand = set(m for ms in S.candidates["members"] for m in ms)
    fig, ax = plt.subplots(figsize=(8.5, 9))
    wmax = float(w.max())
    for i, poly in enumerate(polys):
        if len(poly) < 3:
            continue
        c = zs.loc[i, "county_id"]
        if c in treated:
            fc, a = ROLE["treatment"], 0.9
        elif c in w.index:
            fc, a = ROLE["control"], 0.15 + 0.75 * float(w[c]) / wmax
        elif c in cand:
            fc, a = PALETTE["gold"], 0.35
        else:
            fc, a = ROLE["excluded"], 0.35
        ax.add_patch(Polygon(poly, closed=True, facecolor=fc, alpha=a, edgecolor="white", lw=0.3))
    cs = S.county_static.set_index("county_id")
    for c in sorted(zs["county_id"].unique()):
        r = cs.loc[c]
        lab = r["county"].replace(" County", "")
        if c in w.index:
            lab += f"\nw={w[c]:.2f}"
        dark = c in treated or (c in w.index and float(w[c]) / wmax > 0.6)
        ax.text(r["lon"], r["lat"], lab, fontsize=7, ha="center", va="center",
                color="white" if dark else "#222222", fontweight="bold" if c in treated else "normal")
    ax.set_xlim(zs["lon"].min() - 0.05, zs["lon"].max() + 0.05)
    ax.set_ylim(zs["lat"].min() - 0.05, zs["lat"].max() + 0.05)
    ax.set_aspect(1 / np.cos(np.radians(zs["lat"].mean())))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.set_title(f"Eastmoor: selected holdout market and synthetic-control donors")
    subtitle(ax, f"Treated: {_unit_label(S, S.unit)} (pop {S.truth['pop']/1e3:,.0f}k). "
                 f"Donor shading proportional to locked weight.")
    ax.legend(handles=[Patch(color=ROLE["treatment"], label="Holdout (Social paused)"),
                       Patch(color=ROLE["control"], label="Synthetic-control donor"),
                       Patch(color=PALETTE["gold"], alpha=0.35, label="Other candidate member"),
                       Patch(color=ROLE["excluded"], alpha=0.35, label="Unused county")],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5, frameon=False)
    source_note(fig)
    return save_fig(fig, MODULE, "01_market_map")


def fig_kmeans(S):
    km = S.kmeans
    f = km["features"].copy()
    f["cluster"] = km["labels"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    g = km["grid"]
    a1.plot(g["k"], g["silhouette"], marker="o", color=PALETTE["navy"])
    a1.axvline(km["k"], color=PALETTE["slate"], ls="--", lw=1)
    a1.set_xlabel("Number of clusters k")
    a1.set_ylabel("Silhouette score")
    a1.set_title("K-means sanity lens: cluster quality")
    a1.annotate(f"k={km['k']} used", (km["k"], g.set_index("k").loc[km["k"], "silhouette"]),
                xytext=(8, 8), textcoords="offset points", fontsize=9)
    treated = set(S.unit.split("+"))
    names = _county_names(S)
    for k, sub in f.groupby("cluster"):
        a2.scatter(sub["median_hh_income"] / 1e3, sub["qs_per_capita"] * 1e4, s=np.exp(sub["log_pop"]) / 4e3,
                   color=SERIES[int(k) % len(SERIES)], alpha=0.75, label=f"Cluster {k}", edgecolor="white")
    for c, r in f.iterrows():
        a2.annotate(names.get(c, c), (r["median_hh_income"] / 1e3, r["qs_per_capita"] * 1e4), fontsize=7,
                    xytext=(3, 3), textcoords="offset points",
                    fontweight="bold" if c in treated else "normal",
                    color=ROLE["treatment"] if c in treated else "#333333")
    a2.set_xlabel("Median household income ($k)")
    a2.set_ylabel("Quote starts per 10k residents / week")
    a2.set_title("Counties by income and quote propensity (bubble = population)")
    a2.legend(ncol=3, fontsize=8)
    fig.suptitle("")
    source_note(fig, "Clusters are a plausibility check only; they are not used by the synthetic control.")
    return save_fig(fig, MODULE, "02_kmeans_sanity")


def fig_ranking(S):
    r = S.ranked.copy()
    r["label"] = [_unit_label(S, u) for u in r["unit"]]
    r = r.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(r))))
    cols = [ROLE["treatment"] if u == S.unit else (PALETTE["teal"] if p else ROLE["excluded"])
            for u, p in zip(r["unit"], r["passes"])]
    ax.barh(r["label"], r["score"], color=cols)
    for y, (s, rr) in enumerate(zip(r["score"], r["rejected_because"])):
        if rr:
            ax.text(s + 0.01, y, rr.split(";")[0], va="center", fontsize=7, color="#777777")
    ax.set_xlabel("Composite design score (fit, diversification, recovery, readout noise)")
    ax.set_title("Candidate markets ranked by design score")
    subtitle(ax, f"{int(S.ranked['passes'].sum())} of {len(S.ranked)} candidates pass all hard filters; "
                 "first failed filter shown for rejected ones")
    ax.set_xlim(0, 1.25)
    ax.legend(handles=[Patch(color=ROLE["treatment"], label="Selected"), Patch(color=PALETTE["teal"], label="Passes filters"),
                       Patch(color=ROLE["excluded"], label="Rejected")], loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    return save_fig(fig, MODULE, "03_candidate_ranking")


def fig_design_fit(S):
    st = S.st
    r = S.dfit.fit
    idx = st.design_weeks
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(idx, r.actual.loc[idx], color=ROLE["treatment"], label=f"{_unit_label(S, S.unit)} (actual)")
    ax.plot(idx, r.synthetic.loc[idx], color=ROLE["synthetic"], ls="--", label="Synthetic control (locked weights)")
    _windows(ax, S, test=False)
    pct_axis(ax)
    ax.set_xlabel("Week index")
    ax.set_ylabel("Social share of quote starts (last touch)")
    ax.set_title("Design fit: Social share of quote starts, treated vs synthetic")
    subtitle(ax, f"Weights fit on weeks {st.design_start}-{st.train_end} and frozen; pre-RMSPE "
                 f"{r.pre_rmspe:.3f} ({r.rel_rmse:.1%} of mean), {int((S.dfit.weights >= .01).sum())} donors, "
                 f"effective N {1/float((S.dfit.weights**2).sum()):.1f}")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    return save_fig(fig, MODULE, "04_design_fit_share")


def fig_injected(S):
    v = S.validation
    fig, ax = plt.subplots(figsize=(8, 6))
    lim = [0, max(0.2, float(v["detected"].max()) + 0.02)]
    ax.plot(lim, lim, color=PALETTE["slate"], ls=":", lw=1, label="Perfect recovery")
    ax.fill_between(lim, np.array(lim) - S.st.max_recovery_error, np.array(lim) + S.st.max_recovery_error,
                    color=PALETTE["light"], label=f"±{S.st.max_recovery_error:.0%} tolerance")
    units = sorted(v["unit"].unique())
    for j, (u, sub) in enumerate(v.groupby("unit")):
        sel = u == S.unit
        ax.scatter(sub["injected"] + (0.0 if sel else -0.004 + 0.008 * units.index(u) / max(len(units) - 1, 1)),
                   sub["detected"], s=70 if sel else 25, zorder=3,
                   color=ROLE["treatment"] if sel else PALETTE["sky"], edgecolor="white",
                   label=f"Selected: {_unit_label(S, u)}" if sel else None)
    ax.scatter([], [], color=PALETTE["sky"], s=25, label="Other shortlisted candidates")
    sel = v[v["unit"] == S.unit]
    for _, r in sel.iterrows():
        ax.annotate(f"{r['detected']:.1%}", (r["injected"], r["detected"]), xytext=(8, -4),
                    textcoords="offset points", fontsize=9, color=ROLE["treatment"])
    ax.set_xlim(0.02, 0.18)
    ax.set_ylim(lim)
    pct_axis(ax)
    pct_axis(ax, "x")
    ax.set_xlabel("Injected holdout loss (pseudo-post weeks)")
    ax.set_ylabel("Recovered loss (locked weights, out of sample)")
    ax.set_title("Injected-lift validation on quote starts")
    subtitle(ax, f"Loss injected into weeks {S.st.train_end+1}-{S.st.design_end}; no refit, no bias subtraction")
    ax.legend(loc="upper left", fontsize=8.5)
    return save_fig(fig, MODULE, "05_injected_lift_recovery")


def fig_weights(S):
    p = S.pysyncon
    names = _county_names(S)
    if not p.get("available") or "weights" not in p:
        wv = pd.DataFrame({"locked_design": S.dfit.weights})
    else:
        wv = p["weights"]
    wv = wv[(wv > 0.005).any(axis=1)].sort_values(wv.columns[-1] if "locked_design" not in wv else "locked_design",
                                                  ascending=False)
    labels = [names.get(c, c) for c in wv.index]
    cols = {"locked_design": "Locked design fit (path + predictors)", "in_house_nested_v": "In-house nested-V (predictors)",
            "pysyncon": "pysyncon Synth (predictors, nested V)"}
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(wv))
    k = len(wv.columns)
    for i, c in enumerate(["locked_design", "in_house_nested_v", "pysyncon"]):
        if c in wv:
            ax.bar(x + (i - (k - 1) / 2) * 0.8 / k, wv[c], 0.8 / k, label=cols[c], color=SERIES[i])
    ax.set_xticks(x, labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("Donor weight")
    ax.set_title("Synthetic-control donor weights: design fit vs pysyncon cross-check")
    if p.get("available") and "weights" in p:
        d = p["detected_10"]
        subtitle(ax, f"Synthetic path corr (pysyncon vs design) {p['synthetic_corr_pysyncon_vs_design']:.3f}; pre-RMSPE "
                     f"{p['pre_rmspe']['locked_design']:.4f} vs {p['pre_rmspe']['pysyncon']:.4f}; 10% injection recovered "
                     f"{d['locked_design']:.1%} / {d['in_house_nested_v']:.1%} / {d['pysyncon']:.1%}")
    ax.legend()
    return save_fig(fig, MODULE, "06_donor_weights_pysyncon")


def fig_power(S):
    st = S.st
    pw = S.power
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(0.5, 3, color=PALETTE["grey"], alpha=0.15, lw=0)
    ax.text(1.75, 0.5, "too short:\nnoise-dominated", ha="center", va="center", rotation=90, color="#777777", fontsize=9)
    for i, (L, sub) in enumerate(pw.groupby("lift")):
        c = SERIES[i % len(SERIES)]
        ax.plot(sub["weeks"], sub["power"], color=c, label=f"{L:.1%} true loss")
        w80 = S.weeks_to_power.set_index("lift").loc[L, "weeks_to_80"]
        if w80 is not None and not pd.isna(w80) and w80 <= st.power_max_weeks:
            ax.scatter([w80], [st.power], color=c, edgecolor="black", zorder=4)
            ax.annotate(f"{int(w80)}w", (w80, st.power), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8.5, color=c, fontweight="bold")
    ax.axhline(st.power, color="black", ls="--", lw=1)
    mark_line(ax, st.n_test_weeks, f"Planned test: {st.n_test_weeks} weeks", color=ROLE["treatment"])
    pct_axis(ax)
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0.5, st.power_max_weeks)
    ax.set_xticks(range(1, st.power_max_weeks + 1))
    ax.set_xlabel("Test length (weeks)")
    ax.set_ylabel("Power (one-sided)")
    ax.set_title("Test-duration analysis: power vs weeks by true Social contribution")
    subtitle(ax, f"Closed form, alpha={st.alpha:.2f} one-sided, weekly gap sd {S.noise['sigma']:.1%}, AR(1) "
                 f"rho={S.noise['rho']:.2f}, {len(st.readout_pre_weeks)} pre weeks. "
                 f"Simulation check @10%: {S.power_sim[0.10]['power']:.0%}")
    ax.legend(title="True loss", loc="center left", bbox_to_anchor=(1.0, 0.5))
    return save_fig(fig, MODULE, "07_power_curve")


def fig_jackknife(S):
    jk = S.jackknife.iloc[1:].copy()
    names = _county_names(S)
    jk["label"] = [f"{names.get(d, d)}\n(w={w:.2f})" for d, w in zip(jk["dropped"], jk["dropped_weight"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, max(4.5, 0.42 * len(jk))), sharey=True)
    y = np.arange(len(jk))
    c1 = [PALETTE["purple"] if s else PALETTE["red"] for s in jk["stable"]]
    a1.barh(y, jk["change_pp"] * 100, color=c1)
    a1.axvline(0, color="black", lw=0.8)
    for x in (-S.st.jk_max_change_pp, S.st.jk_max_change_pp):
        a1.axvline(100 * x, color=PALETTE["slate"], ls=":", lw=1)
    a1.set_yticks(y, jk["label"], fontsize=8)
    a1.set_xlabel("Change in recovered 10% injection (pp)")
    a1.set_title("Leave-one-donor-out: recovered lift change")
    a2.barh(y, jk["rmspe_ratio"], color=[PALETTE["teal"] if s else PALETTE["red"] for s in jk["stable"]])
    a2.axvline(1.0, color="black", lw=0.8)
    a2.axvline(S.st.jk_max_rmspe_ratio, color=PALETTE["slate"], ls=":", lw=1)
    a2.set_xlabel("Pre-period RMSPE ratio (refit / baseline)")
    a2.set_title("Leave-one-donor-out: fit degradation")
    base = S.jackknife.iloc[0]
    fig.suptitle(f"Jackknife stability for {_unit_label(S, S.unit)}: baseline recovery {base['detected_10']:.1%}; "
                 f"{int(jk['stable'].sum())}/{len(jk)} drops within ±{S.st.jk_max_change_pp*100:.0f}pp and "
                 f"RMSPE ratio ≤ {S.st.jk_max_rmspe_ratio:.0f}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, MODULE, "08_jackknife_stability")


def fig_readout_series(S):
    st, ts = S.st, S.ts
    idx = list(range(st.design_start, st.test_end + 1))
    g = S.estimates["synthetic_control_gap"]["qs"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    a1.plot(idx, ts.loc[idx, "qs_actual"], color=ROLE["treatment"], label="Treated (actual)")
    a1.plot(idx, g["cf_series"].loc[idx], color=ROLE["synthetic"], ls="--", label="Synthetic (locked weights, level-adj.)")
    _windows(a1, S)
    a1.set_ylabel("Quote starts / week")
    a1.set_title(f"Readout: quote starts in {_unit_label(S, S.unit)} vs synthetic control")
    subtitle(a1, f"SC-gap estimate of Social contribution: {g['estimate']:.1%} of quote starts over the test "
                 f"(truth {S.truth['qs']['true_lift_pct']:.1%})")
    a1.legend(loc="upper left", ncol=2)
    a2.plot(idx, ts.loc[idx, "share_actual"], color=ROLE["treatment"], label="Treated")
    a2.plot(idx, ts.loc[idx, "share_synth"], color=ROLE["synthetic"], ls="--", label="Synthetic")
    _windows(a2, S, design=False)
    pct_axis(a2)
    a2.set_ylabel("Social last-touch share of QS")
    a2.set_xlabel("Week index")
    post = st.test_weeks
    drop = 1 - ts.loc[post, "share_actual"].mean() / ts.loc[post, "share_synth"].mean()
    a2.set_title("Holdout compliance: Social-attributed share fell but not to zero")
    subtitle(a2, f"Treated Social LT share {drop:.0%} below synthetic in test weeks; "
                 f"{S.spend['delivered_share']:.0%} of BAU spend still delivered (leakage)")
    a2.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    return save_fig(fig, MODULE, "09_readout_treated_vs_synthetic")


def fig_did_weekly(S):
    st, ts = S.st, S.ts
    e = S.estimates["did_week_fe_hc1"]["qs"]
    weeks = st.readout_pre_weeks + st.test_weeks
    pre = st.readout_pre_weeks
    adj = ts.loc[pre, "qs_actual"].mean() - ts.loc[pre, "qs_synth"].mean()
    contrib = (ts.loc[weeks, "qs_synth"] + adj) - ts.loc[weeks, "qs_actual"]  # positive = lost volume
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
    cols = [PALETTE["grey"] if w < st.test_start else (PALETTE["green"] if v >= 0 else PALETTE["red"])
            for w, v in contrib.items()]
    a1.bar(weeks, contrib, color=cols)
    a1.axhline(0, color="black", lw=0.8)
    a1.axhline(e["abs_per_week"], color=PALETTE["navy"], ls="--", lw=1.2)
    a1.text(st.test_end + 0.4, e["abs_per_week"], f"DiD: {e['abs_per_week']:.1f}/wk", va="bottom", ha="right", fontsize=9,
            color=PALETTE["navy"])
    mark_line(a1, st.test_start - 0.5, "Social paused")
    a1.set_ylabel("Quote starts lost / week")
    a1.set_title(f"Week-FE DiD: weekly and cumulative quote starts attributable to Social")
    subtitle(a1, f"Contribution {e['estimate']:.1%} (90% CI {e['ci_low']:.1%} to {e['ci_high']:.1%}), "
                 f"HC1 p={e['p_value']:.2f}, HAC p={e['hac_p_value']:.2f}; grey = readout pre-period residuals")
    cum = contrib.loc[st.test_weeks].cumsum()
    a2.plot(st.test_weeks, cum, color=PALETTE["purple"], marker="o")
    a2.fill_between(st.test_weeks, cum - 1.645 * e["incremental_se"] * np.sqrt(np.arange(1, len(cum) + 1) / len(cum)),
                    cum + 1.645 * e["incremental_se"] * np.sqrt(np.arange(1, len(cum) + 1) / len(cum)),
                    color=PALETTE["purple"], alpha=0.15, lw=0, label="approx. 90% band")
    a2.plot(st.test_weeks, np.cumsum([S.truth["qs"]["true_incremental"] / st.n_test_weeks] * st.n_test_weeks),
            color=ROLE["truth"], ls=":", label="Ground truth (expected)")
    a2.axhline(0, color="black", lw=0.8)
    a2.set_ylabel("Cumulative QS lost")
    a2.set_xlabel("Week index")
    a2.legend(loc="upper left")
    fig.tight_layout()
    return save_fig(fig, MODULE, "10_did_weekly_cumulative")


def fig_placebo(S):
    pl = S.placebo["qs"]
    d = pl["placebos"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(d["lift"], bins=12, color=PALETTE["sky"], edgecolor="white", label=f"Placebo counties (n={len(d)})")
    ax.axvline(pl["treated"]["lift"], color=ROLE["treatment"], lw=2.5, label=f"Treated: {pl['treated']['lift']:.1%}")
    ax.axvline(S.truth["qs"]["true_lift_pct"], color=ROLE["truth"], ls=":", lw=2,
               label=f"Ground truth: {S.truth['qs']['true_lift_pct']:.1%}")
    pct_axis(ax, "x")
    ax.set_xlabel("Estimated QS 'contribution' over the test window (same design procedure)")
    ax.set_ylabel("Number of placebo units")
    ax.set_title("In-space placebo: is the treated gap unusual among untreated counties?")
    subtitle(ax, f"One-sided permutation p = {pl['p_value_lift_one_sided']:.2f} (min attainable {pl['min_attainable_p']:.2f}); "
                 f"RMSPE-ratio p = {pl['p_value_rmspe_ratio']:.2f}; placebo sd {pl['placebo_lift_sd']:.1%}")
    ax.legend()
    return save_fig(fig, MODULE, "11_placebo_in_space")


def fig_truth_vs_estimate(S, table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    methods = list(METHOD_LABEL)
    x = np.arange(len(STAGES))
    k = len(methods) + 1
    wd = 0.8 / k
    for i, m in enumerate(methods):
        sub = table[table["method"] == m].set_index("metric").loc[list(STAGES)]
        err = None
        if sub["ci_low"].notna().all():
            err = np.vstack([sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]])
        elif sub["se_rel"].notna().all():
            err = 1.645 * sub["se_rel"].to_numpy()
        ax.bar(x + (i - (k - 1) / 2) * wd, sub["estimate"], wd, yerr=err, capsize=3, color=SERIES[i],
               label=METHOD_LABEL[m], error_kw={"elinewidth": 1, "ecolor": "#444444"})
    tr = table[table["method"] == methods[0]].set_index("metric").loc[list(STAGES), "truth"]
    ax.bar(x + (k - 1 - (k - 1) / 2) * wd, tr, wd, color=ROLE["truth"], label="Ground truth (expected)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [STAGE_LABEL[s] for s in STAGES])
    pct_axis(ax)
    ax.set_ylabel("Social contribution (% of counterfactual volume)")
    ax.set_title("Funnel lifts: every estimator vs ground truth")
    subtitle(ax, "Error bars = 90% intervals (DiD: HC1; ratio DiD: bootstrap; SC gap: ±1.645 × placebo sd)")
    ax.legend(ncol=2, loc="upper left")
    return save_fig(fig, MODULE, "12_truth_vs_estimate")


def fig_attribution(S):
    a = S.attribution.set_index("stage").loc[list(STAGES)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(STAGES))
    tr = [1000 * S.truth[s]["true_incremental"] / S.truth["spend_removed"] for s in STAGES]
    ax.bar(x - 0.27, a["lt_per_1k"], 0.27, color=ROLE["attribution"], label="Last-touch claimed (platform view)")
    ax.bar(x, a["incremental_per_1k"], 0.27, color=ROLE["estimate"], label="Measured incremental (holdout)")
    ax.bar(x + 0.27, tr, 0.27, color=ROLE["truth"], label="Ground truth incremental")
    for i, s in enumerate(STAGES):
        r = a.loc[s, "overclaim_ratio"]
        top = max(a.loc[s, "lt_per_1k"], a.loc[s, "incremental_per_1k"], tr[i])
        word = "over-claims" if r > 1 else "under-claims"
        ax.text(i, top * 1.05, f"LT {word}\n{r:.1f}×" if r > 1 else f"LT {word}\n{1/r:.1f}×", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x, [STAGE_LABEL[s] for s in STAGES])
    ax.set_ylabel("Outcomes per $1,000 Social spend (log scale)")
    ax.set_title("Attribution vs incrementality: what last-touch credits Social with")
    subtitle(ax, "Ratio = last-touch claimed per $ / measured incremental per $")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    return save_fig(fig, MODULE, "13_attribution_vs_incrementality")


def fig_cost_per(S):
    a = S.attribution.set_index("stage").loc[list(STAGES)]
    e = S.efficiency.set_index("stage").loc[list(STAGES)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, s in zip(axes, STAGES):
        vals = [a.loc[s, "lt_cost_per"], e.loc[s, "blended_cost_per"], a.loc[s, "incremental_cost_per"],
                e.loc[s, "true_incremental_cost_per"] or np.nan]
        labs = ["Last-touch\n(platform)", "Blended\nall-channel", "Incremental\n(measured)", "Incremental\n(truth)"]
        cols = [ROLE["attribution"], PALETTE["slate"], ROLE["estimate"], ROLE["truth"]]
        ax.bar(labs, vals, color=cols)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=8.5)
        money_axis(ax)
        ax.set_title(f"Cost per {STAGE_LABEL[s].lower().rstrip('s')}", fontsize=11)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Cost per outcome")
    fig.suptitle("Efficiency: incremental cost-per vs platform-reported and blended benchmarks", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    source_note(fig, "Measured incremental cost-per = spend removed / DiD incremental volume; sales are thin "
                     "(~1-2 per week) so the sales figure is very noisy.")
    return save_fig(fig, MODULE, "14_cost_per_comparison")


def make_all(S) -> List[Path]:
    from .pipeline import estimates_table
    figs = [fig_map(S), fig_kmeans(S), fig_ranking(S), fig_design_fit(S), fig_injected(S), fig_weights(S),
            fig_power(S), fig_jackknife(S), fig_readout_series(S), fig_did_weekly(S), fig_placebo(S),
            fig_truth_vs_estimate(S, estimates_table(S)), fig_attribution(S), fig_cost_per(S)]
    return figs
