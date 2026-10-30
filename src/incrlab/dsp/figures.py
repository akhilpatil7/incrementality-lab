"""All DSP figures (design A, design B, readout). Every label that names a design parameter
(buffer miles, gates, alpha, K, weights) is generated from the parameter objects, so a chart can
never disagree with the code that produced it."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import geo as G
from ..common.plotting import (PALETTE, ROLE, SERIES, mark_line, money_axis, pct_axis, save_fig, shade_window,
                               source_note, subtitle)
from .params import FEATURE_LABELS, STAGE_LABELS
from .readout import METHOD_LABELS, truth_table

MOD = "dsp"
CLUSTER_COLORS = ["#1a5fa8", "#2ca076", "#e05c38", "#8b5cf6", "#f59e0b", "#ec4899", "#14b8a6", "#f43f5e",
                  "#84cc16", "#0ea5e9", "#a16207", "#64748b", "#be185d", "#4d7c0f", "#7c3aed", "#0f766e"]
GROUP_COLORS = {"treatment": ROLE["treatment"], "control": ROLE["control"], "buffer": PALETTE["gold"],
                "eligible (unused)": "#dfe6ee", "not eligible": "#f4f5f7", "excluded": "#9aa1ab"}
VENDOR_LABEL = {"A": "Vendor A", "B": "Vendor B"}


# ======================================================================================
# helpers
# ======================================================================================
def _aspect(ax, lat):
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(lat)))))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)


def _sub(ax, text: str):
    """Subtitle under a left-aligned title without colliding with it."""
    ax._autotitlepos = False  # otherwise matplotlib re-positions the title on draw
    for t in (ax.title, ax._left_title, ax._right_title):  # theme uses left-located titles
        t.set_y(1.075)
    subtitle(ax, text)


def _cells(zs: pd.DataFrame):
    return G.voronoi_polygons(zs["lon"].to_numpy(), zs["lat"].to_numpy())


def _draw_cells(ax, polys, colors, edge="white", lw=0.3, idx=None, zorder=1, alpha=1.0):
    idx = range(len(polys)) if idx is None else idx
    pp = [polys[i] for i in idx if len(polys[i])]
    cc = [colors[i] for i in idx if len(polys[i])]
    ax.add_collection(PolyCollection(pp, facecolors=cc, edgecolors=edge, linewidths=lw, zorder=zorder, alpha=alpha))


def _buffer_contour(ax, zs, treated_ids, miles, color="#7a5c00"):
    """Iso-distance line at exactly ``miles`` from the nearest treated centroid (same parameter
    as the BallTree buffer in the design code)."""
    t = zs[zs["zip"].isin(set(treated_ids))]
    x0, x1 = zs["lon"].min() - 0.05, zs["lon"].max() + 0.05
    y0, y1 = zs["lat"].min() - 0.05, zs["lat"].max() + 0.05
    gx, gy = np.meshgrid(np.linspace(x0, x1, 220), np.linspace(y0, y1, 220))
    d = G.haversine_miles(gy.ravel()[:, None], gx.ravel()[:, None], t["lat"].to_numpy()[None], t["lon"].to_numpy()[None])
    d = d.min(1).reshape(gx.shape)
    ax.contour(gx, gy, d, levels=[miles], colors=[color], linestyles="--", linewidths=1.3, zorder=5)


def _scalebar(ax, zs, miles=20):
    lat = zs["lat"].mean()
    dlon = miles / (69.17 * np.cos(np.radians(lat)))
    x0, y0 = zs["lon"].min(), zs["lat"].min() - 0.08
    ax.plot([x0, x0 + dlon], [y0, y0], color="#333", lw=2.5, solid_capstyle="butt", zorder=6)
    ax.text(x0 + dlon / 2, y0 + 0.03, f"{miles} mi", ha="center", va="bottom", fontsize=8, color="#333")


def _set_extent(ax, zs, pad=0.12):
    ax.set_xlim(zs["lon"].min() - pad, zs["lon"].max() + pad)
    ax.set_ylim(zs["lat"].min() - pad - 0.08, zs["lat"].max() + pad)


def _fmt_pop(p):
    return f"{p / 1e6:.2f}M" if p >= 1e6 else f"{p / 1e3:.0f}k"


def _power_panel(ax, curve: pd.DataFrame, title: str, alpha_label: str, max_weeks: int, test_weeks: int):
    ramp = ["#ffcccc", "#ff8a8a", PALETTE["green"], "#cc0000", "#8b0000", "#3d0000"]
    lifts = sorted(curve["lift"].unique())
    for i, l in enumerate(lifts):
        c = curve[curve["lift"] == l].sort_values("weeks")
        col = ramp[i % len(ramp)]
        lw = 3 if abs(l - 0.15) < 1e-9 else 1.8
        ax.plot(c["weeks"], c["power"], color=col, lw=lw, label=f"{l:.0%} lift")
        hit = c[c["power"] >= 0.8]
        if len(hit):
            w = hit["weeks"].iloc[0]
            ax.scatter([w], [hit["power"].iloc[0]], color=col, edgecolor="black", zorder=5, s=36)
            ax.annotate(f"{w}w", (w, 0.8), xytext=(0, -24 - 9 * (i % 2)), textcoords="offset points",
                        ha="center", fontsize=8, color=col if i != 0 else "#c76", fontweight="bold")
    ax.axhline(0.8, color="#888", ls="--", lw=1)
    ax.text(max_weeks, 0.81, "80% power", ha="right", va="bottom", fontsize=8.5, color="#666")
    ax.axvspan(0.5, 2.5, color="#d9dce1", alpha=0.5, lw=0)
    ax.text(1.5, 0.5, "noise zone\n(too early)", ha="center", va="center", fontsize=8, color="#555", rotation=90)
    ax.axvline(test_weeks, color=PALETTE["navy"], ls=":", lw=1.4)
    ax.text(test_weeks, 0.03, f" planned test = {test_weeks} wk", fontsize=8.5, color=PALETTE["navy"])
    ax.set_xlim(0.5, max_weeks)
    ax.set_ylim(-0.02, 1.04)
    pct_axis(ax)
    ax.set_xlabel("Weeks of holdout")
    ax.set_ylabel("Power")
    ax.set_title(title)
    _sub(ax, alpha_label)
    ax.legend(loc="lower right", ncol=2, fontsize=8.5, title="True lift", title_fontsize=8.5)


# ======================================================================================
# 01 pipeline flowchart (graphviz dot; matplotlib fallback when dot is unavailable)
# ======================================================================================
def fig_flowchart(A, B, P) -> Path:
    a, b = P.a, P.b
    sg, rg = a.strict_gates, a.relaxed_gates
    dot = f"""digraph G {{
  graph [rankdir=TB, newrank=true, fontname="Helvetica", fontsize=11, bgcolor="white", nodesep=0.35, ranksep=0.28, pad=0.2,
         label="Programmatic DSP geo-experiment pipeline (python -m incrlab.dsp) - every box is code that ran", labelloc=t];
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9.5, color="#5C677D", penwidth=1.1, margin="0.12,0.06"];
  edge [color="#5C677D", arrowsize=0.7];
  data [label="Synthetic world\\nweekly zip panel + ACS-like zip features\\n(design data: week <= launch-1 only)", fillcolor="#EEF1F5"];
  agg [label="Granularity-safe aggregation\\nzip-static dedupe | monthly media dedupe\\nrates from summed num/den", fillcolor="#EEF1F5"];
  subgraph cluster_a {{ label="Vendor A - statistical twin"; fontsize=10; color="#E76F51"; style="rounded";
    a1 [label="Eligible zips\\nserved, pop >= {a.min_zip_pop:,}\\nno concurrent test", fillcolor="#FBE3DB"];
    a2 [label="{a.adjacency_miles:g}-mi adjacency graph\\nBFS patches N/S/E/W\\ntarget {a.pop_target/1e3:.0f}k +-{a.pop_tolerance:.0%} (closest)", fillcolor="#FBE3DB"];
    a3 [label="{a.buffer_miles:g}-mi BallTree buffer\\n(haversine) -> donor pool", fillcolor="#FBE3DB"];
    a4 [label="Control patches (every seed)\\n6-feature Mahalanobis + ridge\\n+ {a.w_compact:.2f} x compactness", fillcolor="#FBE3DB"];
    a5 [label="Pre-trend gates\\nstrict corr>={sg[0]}, slope {sg[1]}-{sg[2]}, CV<={sg[3]}\\nrelaxed tier logged (never force-pass)", fillcolor="#FBE3DB"];
    a6 [label="Rank + SMD balance\\npower (1-sided a={a.alpha}) + injected lift", fillcolor="#FBE3DB"];
    a1 -> a2 -> a3 -> a4 -> a5 -> a6; }}
  subgraph cluster_b {{ label="Vendor B - synthetic twin"; fontsize=10; color="#2A9D8F"; style="rounded";
    b1 [label="Inclusion-list zips\\nminus concurrent-test zips", fillcolor="#D8F0EC"];
    b2 [label="Vendor-switch Welch test\\n(level shift in vendor share)", fillcolor="#D8F0EC"];
    b3 [label="Geo-weighted K-means (x{b.geo_weight:g})\\nsilhouette among K with\\nmean pop <= {b.cluster_pop_target/1e3:.0f}k -> K={B.kmeans['k']}", fillcolor="#D8F0EC"];
    b4 [label="SC tournament: every cluster\\n+{b.injected_lift:.0%} injected, locked weights\\nscore {b.score_weights[0]:.2f}/{b.score_weights[1]:.2f}/{b.score_weights[2]:.2f} acc/fit/speed", fillcolor="#D8F0EC"];
    b5 [label="Final SC (buffer-trimmed donors)\\nplacebos in-space / in-time / random geo\\nridge-augmented SC", fillcolor="#D8F0EC"];
    b1 -> b2 -> b3 -> b4 -> b5; }}
  run [label="Holdout realized by the\\nscenario runner on the\\nSELECTED zips (spend -> {P.a.test} / {P.b.test} paused)", fillcolor="#FFF4D6"];
  tri [label="Triangulated readout (90% intervals)\\nchange-ratio DiD + bootstrap | Welch gaps | 2-way FE\\nstructural TS | SC + placebo | AugSynth", fillcolor="#FFF4D6"];
  biz [label="Business translation\\nincremental cost-per vs vendor-reported\\nspillover check | ROI | go / no-go", fillcolor="#FFF4D6"];
  mmm [label="extra.incremental_per_dollar\\n-> calibrated MMM", fillcolor="#E4E9F7"];
  {{rank=same; run; tri; biz; mmm;}}
  data -> agg; agg -> a1; agg -> b1; a6 -> run; b5 -> run;
  run -> tri -> biz -> mmm;
}}"""
    dot_bin = shutil.which("dot")
    if dot_bin:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "f.dot"
            src.write_text(dot)
            out = Path(td) / "f.png"
            subprocess.run([dot_bin, "-Tpng", "-Gdpi=150", str(src), "-o", str(out)], check=True,
                           capture_output=True)
            img = plt.imread(out)
        h, w = img.shape[:2]
        fig = plt.figure(figsize=(w / 150, h / 150))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis("off")
        return save_fig(fig, MOD, "01_pipeline_flowchart")
    # fallback: plain matplotlib boxes (CI without graphviz)
    steps = ["Synthetic panel", "Granularity-safe aggs", "A: BFS patches + buffer", "A: Mahalanobis + gates",
             "B: switch test + K-means", "B: SC tournament", "Scenario holdout", "Triangulated readout", "Go / no-go"]
    fig, ax = plt.subplots(figsize=(14, 2.6))
    for i, s in enumerate(steps):
        ax.text(i, 0, s, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="#EEF1F5", ec="#5C677D"))
        if i:
            ax.annotate("", (i - 0.42, 0), (i - 0.58, 0), arrowprops=dict(arrowstyle="<-", color="#5C677D"))
    ax.set_xlim(-0.6, len(steps) - 0.4)
    ax.axis("off")
    ax.set_title("DSP geo-experiment pipeline (graphviz not installed - simplified)")
    return save_fig(fig, MOD, "01_pipeline_flowchart")


# ======================================================================================
# Vendor A design
# ======================================================================================
def fig_a_patches(A) -> Path:
    zs = A.region_zips.reset_index(drop=True)
    polys = _cells(zs)
    patches = A.patches
    n = len(patches)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    dir_col = {"North": "#2563eb", "South": "#dc2626", "East": "#16a34a", "West": "#d97706"}
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    elig = set(A.eligible["zip"])
    for ax, (_, pt) in zip(axes, patches.iterrows()):
        in_cty = (zs["county_id"] == pt["county_id"]).to_numpy()
        base = np.where(in_cty, "#cfd8e3", np.where(zs["zip"].isin(elig), "#eceff3", "#f7f7f8"))
        mem = zs["zip"].isin(set(pt["zips"])).to_numpy()
        cols = np.where(mem, dir_col.get(pt["direction"], ROLE["treatment"]), base)
        _draw_cells(ax, polys, list(cols))
        _draw_cells(ax, polys, list(cols), edge="black", lw=0.9, idx=np.where(mem)[0], zorder=2)
        s = zs[zs["zip"] == pt["seed"]]
        ax.scatter(s["lon"], s["lat"], marker="*", s=90, color="white", edgecolor="black", zorder=4)
        _set_extent(ax, zs)
        _aspect(ax, zs["lat"])
        ok = "in window" if pt["in_window"] else "OUT of window"
        ax.set_title(f"{pt['county']} - {pt['direction']}", fontsize=10.5)
        ax.text(0.02, 0.02, f"{pt['n_zips']} zips | pop {_fmt_pop(pt['pop'])} ({pt['pop_gap_pct']:+.1%} vs target)\n"
                f"stop-at-lower-bound rule would give {_fmt_pop(pt['pop_if_stop_at_lower'])} | {ok}",
                transform=ax.transAxes, fontsize=7.5, color="#333", va="bottom")
    for ax in axes[n:]:
        ax.axis("off")
    p = A.params
    fig.suptitle(f"Vendor A - candidate treatment patches (BFS on {p.adjacency_miles:g}-mi adjacency graph, "
                 f"target {_fmt_pop(p.pop_target)} +-{p.pop_tolerance:.0%}, growth stops closest to target)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Synthetic geography (Voronoi cells around synthetic zip centroids). Star = BFS seed "
                     "(extreme zip of the county in that direction).")
    return save_fig(fig, MOD, "02_a_treatment_patches")


def fig_a_donor_buffer(A) -> Path:
    zs = A.region_zips.reset_index(drop=True)
    polys = _cells(zs)
    w = A.winner
    pool = A.donor_pools.set_index("patch_id").loc[w["patch_id"]]
    valid, buf = set(pool["valid_zips"]), set(pool["buffer_zips"])
    treated = set(A.treated)
    t_cty = zs["county_id"] == w["t_county"]
    cols = np.full(len(zs), "#f4f5f7", dtype=object)
    cols[zs["zip"].isin(set(A.eligible["zip"])).to_numpy()] = "#e6e9ee"
    cols[zs["zip"].isin(buf).to_numpy()] = "#fca5a5"
    cols[zs["zip"].isin(valid).to_numpy()] = "#86efac"
    cols[(t_cty & zs["zip"].isin(set(A.eligible["zip"]))).to_numpy()] = "#bfdbfe"
    cols[zs["zip"].isin(treated).to_numpy()] = "#2563eb"
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.3), gridspec_kw={"width_ratios": [1.35, 1]})
    _draw_cells(ax, polys, list(cols))
    _draw_cells(ax, polys, list(cols), edge="black", lw=0.9, idx=np.where(zs["zip"].isin(treated))[0], zorder=2)
    _buffer_contour(ax, zs, A.treated, A.params.buffer_miles)
    _set_extent(ax, zs)
    _aspect(ax, zs["lat"])
    _scalebar(ax, zs)
    bl = A.params.buffer_miles
    handles = [Patch(color="#2563eb", label=f"Treatment patch ({len(treated)} zips)"),
               Patch(color="#bfdbfe", label="Other eligible zips in treatment county (never donors)"),
               Patch(color="#fca5a5", label=f"Excluded: centroid within {bl:g} mi of a treated zip ({len(buf)})"),
               Patch(color="#86efac", label=f"Valid donors ({len(valid)}, {pool['n_counties']} counties)"),
               Patch(color="#e6e9ee", label="Eligible elsewhere"), Patch(color="#f4f5f7", label="Not eligible"),
               Line2D([], [], color="#7a5c00", ls="--", label=f"{bl:g}-mi line (same parameter as the BallTree)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2, fontsize=8)
    ax.set_title(f"Donor pool for the selected patch - {bl:g}-mi spillover buffer")
    dp = A.donor_pools.copy()
    dp = dp.sort_values("n_valid")
    y = np.arange(len(dp))
    ax2.barh(y, dp["n_valid"], color="#4ade80", label="valid donors")
    ax2.barh(y, dp["n_excluded_buffer"], left=dp["n_valid"], color="#fca5a5", label=f"inside {bl:g}-mi buffer")
    ax2.set_yticks(y, [p.replace(":", " - ") for p in dp["patch_id"]], fontsize=8.5)
    for yi, r in zip(y, dp.itertuples()):
        ax2.text(r.n_valid + r.n_excluded_buffer + 2, yi, f"{r.n_counties} counties", va="center", fontsize=8)
    ax2.set_xlabel("Eligible zips outside the treatment county")
    ax2.set_title("Donor pool size per candidate patch")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)
    ax2.set_xlim(0, (dp["n_valid"] + dp["n_excluded_buffer"]).max() * 1.18)
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    source_note(fig)
    return save_fig(fig, MOD, "03_a_donor_pool_buffer")


def fig_a_corr(A) -> Path:
    c = A.corr
    lab = [FEATURE_LABELS.get(f, f) for f in c.columns]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.6 else "#222")
    ax.set_xticks(range(len(c)), lab, rotation=35, ha="right")
    ax.set_yticks(range(len(c)), lab)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r (eligible zips)")
    ax.set_title("Vendor A - Mahalanobis feature correlations")
    _sub(ax, f"Correlated features are why Mahalanobis (inverse covariance, ridge {A.params.maha_ridge:g}) "
                 "beats Euclidean distance")
    source_note(fig)
    return save_fig(fig, MOD, "04_a_feature_correlation")


def fig_a_candidates(A) -> Path:
    c = A.candidates
    p = A.params
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    tier_c = {"strict": PALETTE["green"], "relaxed": PALETTE["gold"], "fail": "#b9bec7"}
    for t, s in c.groupby("tier"):
        ax.scatter(s["maha"], s["wow_corr"], s=40 + 4 * s["c_n"], color=tier_c[t], edgecolor="#333", lw=0.5,
                   label=f"{t} ({len(s)})", alpha=0.9)
    w = c.iloc[0]
    ax.scatter([w["maha"]], [w["wow_corr"]], s=260, facecolor="none", edgecolor=ROLE["treatment"], lw=2.2,
               label="selected (rank 1)")
    ax.axhline(p.strict_gates[0], color=PALETTE["green"], ls="--", lw=1)
    ax.axhline(p.relaxed_gates[0], color=PALETTE["gold"], ls="--", lw=1)
    ax.text(ax.get_xlim()[0], p.strict_gates[0], f" strict corr gate {p.strict_gates[0]}", ha="left", va="bottom",
            fontsize=8, color=PALETTE["green"])
    ax.text(ax.get_xlim()[0], p.relaxed_gates[0], f" relaxed {p.relaxed_gates[0]}", ha="left", va="top",
            fontsize=8, color="#9a7b16")
    ax.set_xlabel("Mahalanobis distance (static similarity, lower = better)")
    ax.set_ylabel(f"Week-over-week correlation ({p.validation_metric})")
    ax.set_title(f"Top {len(c)} candidate pairs: static vs dynamic similarity")
    ax.legend(loc="lower left", fontsize=8.5)
    top = c.head(12)
    cols = ["rank", "patch_id", "c_county", "maha", "composite", "wow_corr", "slope_ratio", "treat_cv", "ctrl_cv",
            f"weeks_{p.rank_lift:.2f}", "placebo_drift", "tier"]
    tab = top[cols].copy()
    tab["patch_id"] = (tab["patch_id"].str.replace(r"^.*:", "", regex=True) + " "
                       + top["t_county"].str.replace(r"^.*-", "", regex=True))
    tab["c_county"] = tab["c_county"].str.replace(" County", "")
    for k in ("maha", "composite", "wow_corr", "slope_ratio", "treat_cv", "ctrl_cv"):
        tab[k] = tab[k].map(lambda v: f"{v:.2f}")
    tab["placebo_drift"] = tab["placebo_drift"].map(lambda v: f"{v:+.1%}")
    tab[f"weeks_{p.rank_lift:.2f}"] = tab[f"weeks_{p.rank_lift:.2f}"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    ax2.axis("off")
    hdr = ["#", "Treatment patch", "Control", "Maha", "Score", "WoW r", "Slope", "CV(T)", "CV(C)",
           f"Wks@{p.rank_lift:.0%}", "Drift", "Tier"]
    tb = ax2.table(cellText=tab.to_numpy(), colLabels=hdr, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.45)
    tb.auto_set_column_width(list(range(len(hdr))))
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r_ == 0:
            cell.set_facecolor("#1F3A5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif c_ == len(hdr) - 1:
            cell.set_facecolor({"strict": "#c9ecd7", "relaxed": "#fbefc4", "fail": "#eceef1"}[top["tier"].iloc[r_ - 1]])
    ax2.set_title(f"Ranked candidate table (strict gates {p.strict_gates}; relaxed {p.relaxed_gates})", fontsize=11)
    fig.tight_layout()
    source_note(fig, "Rank = gate tier first, then rank(composite) + rank(-WoW corr) + rank(weeks to detect "
                     f"{p.rank_lift:.0%}) + rank(|in-time placebo drift|). No pair is force-passed.")
    return save_fig(fig, MOD, "05_a_candidate_ranking")


def fig_a_overlays(A) -> Path:
    k = list(A.weekly)
    cand = A.candidates.set_index("rank")
    fig, axes = plt.subplots(len(k), 1, figsize=(11, 2.1 * len(k)), sharex=True)
    for ax, key in zip(np.atleast_1d(axes), k):
        d = A.weekly[key]
        r = cand.loc[int(key.replace("rank", ""))]
        ax.plot(d.index, d["t_qs"], color="#2563eb", marker="o", ms=3.5, label="Treatment patch")
        ax.plot(d.index, d["c_qs"], color="#dc2626", ls="--", marker="s", ms=3.5, label="Control patch")
        ax.set_title(f"#{key[4:]}  {r['patch_id'].split(':')[1]} ({r['t_county']}) vs {r['c_county']} | "
                     f"WoW corr {r['wow_corr']:.2f} | Maha {r['maha']:.2f} | tier: {r['tier']}", fontsize=10)
        ax.set_ylabel("Weekly QS")
    np.atleast_1d(axes)[0].legend(loc="upper right", ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel("Week index (design window)")
    fig.suptitle("Vendor A - pre-period quote starts, top-5 candidate pairs", fontsize=13, fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig)
    return save_fig(fig, MOD, "06_a_top5_pair_overlays")


def fig_a_profile(A) -> Path:
    feats = ["per_capita_income", "pct_bachelors", "lfpr", "qs_per_capita", "rated_rate", "close_rate"]
    from . import features as F
    cand = A.candidates.head(A.params.top_n_review)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, f in zip(axes.ravel(), feats):
        tv = [F.aggregate_profile(A.eligible, r.t_zips, [f])[f] for r in cand.itertuples()]
        cv = [F.aggregate_profile(A.eligible, r.c_zips, [f])[f] for r in cand.itertuples()]
        x = np.arange(len(cand))
        ax.bar(x - 0.2, tv, 0.4, color="#2563eb", alpha=0.85, label="Treatment")
        ax.bar(x + 0.2, cv, 0.4, color="#dc2626", alpha=0.85, label="Control")
        for xi, a_, b_ in zip(x, tv, cv):
            ax.text(xi, max(a_, b_) * 1.02, f"{a_ / b_ - 1:+.1%}", ha="center", fontsize=8,
                    color=PALETTE["red"] if abs(a_ / b_ - 1) > 0.10 else "#333")
        ax.set_xticks(x, [f"#{r.rank}\n{r.patch_id.split(':')[1]} vs\n{r.c_county.replace(' County', '')}"
                          for r in cand.itertuples()], fontsize=7.5)
        ax.set_title(FEATURE_LABELS[f], fontsize=11)
        ax.set_ylim(0, max(max(tv), max(cv)) * 1.15)
        if f == "per_capita_income":
            money_axis(ax)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Vendor A - attribute profile, treatment vs control (top-5 pairs; red = gap > 10%)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Population-weighted ACS-like features; rates recomputed from summed numerators/denominators.")
    return save_fig(fig, MOD, "07_a_attribute_profile")


def fig_power(A, B) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    pa, pb = A.params, B.params
    _power_panel(ax1, A.power["curves"]["qs"], "Vendor A - matched twin (quote starts)",
                 f"one-sided alpha={pa.alpha}, sigma = sd of weekly relative T-C gap = "
                 f"{A.power['by_metric']['qs']['sigma_gap']:.3f}", pa.max_weeks, A.power["test_weeks"])
    _power_panel(ax2, B.power["by_metric"]["qs"]["curve"], f"Vendor B - synthetic control, cluster {B.treated_cluster}",
                 f"one-sided alpha={pb.alpha}, sigma = sd of weekly relative SC gap = "
                 f"{B.power['by_metric']['qs']['sigma_gap']:.3f}", pb.max_weeks, B.power["test_weeks"])
    ax2.set_ylabel("")
    fig.suptitle("Test-duration analysis: power vs weeks by true lift (noise from design-window residuals)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Same convention for both vendors (unified - the original notebooks mixed three). "
                     "AR(1) residual autocorrelation inflates the SE.")
    return save_fig(fig, MOD, "08_power_curves")


def fig_a_injected(A, B) -> Path:
    d = A.injected
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.2, 1]})
    off = {"qs": -0.006, "rated": 0, "sales": 0.006}
    col = {"qs": PALETTE["navy"], "rated": PALETTE["teal"], "sales": PALETTE["orange"]}
    for m, s in d.groupby("metric"):
        x = s["injected"] + off[m]
        ax.errorbar(x, s["detected"], yerr=[s["detected"] - s["ci_low"], s["ci_high"] - s["detected"]], fmt="o",
                    color=col[m], capsize=3, label=STAGE_LABELS[m])
    lim = [0, d["injected"].max() + 0.04]
    ax.plot(lim, lim, color="#999", ls="--", lw=1, label="perfect recovery")
    pct_axis(ax)
    pct_axis(ax, "x")
    ax.set_xlabel("Injected holdout loss (pseudo-post weeks)")
    ax.set_ylabel("Recovered (change-ratio DiD, natural drift removed)")
    ax.set_title("Vendor A - injected-lift recovery on the selected pair")
    _sub(ax, f"Last {A.params.pseudo_post_weeks} design weeks used as a fake test; 90% bootstrap CI")
    ax.legend(loc="upper left")
    t = B.tournament[B.tournament["status"] == "ok"].sort_values("cluster")
    cc = [ROLE["treatment"] if k == B.treated_cluster else PALETTE["slate"] for k in t["cluster"]]
    ax2.bar(t["cluster"].astype(str), t["detected"], color=cc)
    ax2.axhline(B.params.injected_lift, color=PALETTE["green"], ls="--", lw=1.5,
                label=f"injected +{B.params.injected_lift:.0%}")
    pct_axis(ax2)
    ax2.set_xlabel("Cluster (pseudo-treated in turn)")
    ax2.set_ylabel("Recovered lift (locked weights, natural gap removed)")
    ax2.set_title("Vendor B - injected-lift recovery, every cluster")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    source_note(fig)
    return save_fig(fig, MOD, "09_injected_lift_recovery")


def fig_a_map(A) -> Path:
    zs = A.region_zips.reset_index(drop=True)
    polys = _cells(zs)
    asg = A.assignment().set_index("zip")["group"].reindex(zs["zip"]).to_numpy()
    cols = [GROUP_COLORS.get(g, "#f4f5f7") for g in asg]
    fig, ax = plt.subplots(figsize=(11, 8.4))
    _draw_cells(ax, polys, cols)
    for grp, lw in (("control", 0.9), ("treatment", 0.9)):
        _draw_cells(ax, polys, cols, edge="black", lw=lw, idx=np.where(asg == grp)[0], zorder=2)
    _buffer_contour(ax, zs, A.treated, A.params.buffer_miles)
    for grp in ("treatment", "control"):
        s = zs[asg == grp]
        ax.text(np.average(s["lon"], weights=s["pop"]), s["lat"].max() + 0.07,
                f"{grp.upper()}\n{len(s)} zips, {_fmt_pop(s['pop'].sum())}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=GROUP_COLORS[grp],
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    _set_extent(ax, zs)
    _aspect(ax, zs["lat"])
    _scalebar(ax, zs)
    n = pd.Series(asg).value_counts()
    bl = A.params.buffer_miles
    handles = [Patch(color=GROUP_COLORS["treatment"], label=f"Treatment - DSP paused ({n.get('treatment', 0)})"),
               Patch(color=GROUP_COLORS["control"], label=f"Control twin ({n.get('control', 0)})"),
               Patch(color=GROUP_COLORS["buffer"], label=f"Buffer: within {bl:g} mi, not used ({n.get('buffer', 0)})"),
               Patch(color=GROUP_COLORS["eligible (unused)"], label=f"Eligible, unused ({n.get('eligible (unused)', 0)})"),
               Patch(color=GROUP_COLORS["not eligible"], label=f"Not served / below pop floor ({n.get('not eligible', 0)})"),
               Line2D([], [], color="#7a5c00", ls="--", label=f"{bl:g}-mi line")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    w = A.winner
    ax.set_title(f"Vendor A - selected design: {w['patch_id'].split(':')[1]} patch of {w['t_county']} vs twin "
                 f"anchored in {w['c_county']}")
    _sub(ax, f"Mahalanobis {w['maha']:.2f} | WoW corr {w['wow_corr']:.2f} | gate tier: {w['tier']} | "
                 f"pop gap {w['t_pop'] / w['c_pop'] - 1:+.1%}")
    source_note(fig, "Synthetic geography; Voronoi cells around zip centroids stand in for ZCTA polygons. Patches are "
                     f"contiguous on the {A.params.adjacency_miles:g}-mi graph of ELIGIBLE zips, so unserved zips can sit between members.")
    return save_fig(fig, MOD, "10_a_selected_design_map")


def fig_smd(A, B) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, smd, title, cols in ((ax1, A.smd, "Vendor A: treatment vs twin patch", [("smd", "matched twin", ROLE["control"])]),
                                 (ax2, B.smd, f"Vendor B: cluster {B.treated_cluster} vs donors",
                                  [("smd_unweighted_pool", "unweighted donor pool", "#b9bec7"),
                                   ("smd_sc_weighted", "SC-weighted synthetic", ROLE["synthetic"])])):
        y = np.arange(len(smd))
        for col, lab, c in cols:
            ax.scatter(smd[col], y, s=55, color=c, edgecolor="#333", lw=0.5, label=lab, zorder=3)
        if len(cols) == 2:
            for yi, a_, b_ in zip(y, smd[cols[0][0]], smd[cols[1][0]]):
                ax.annotate("", (b_, yi), (a_, yi), arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))
        ax.axvspan(-0.1, 0.1, color=PALETTE["green"], alpha=0.12, lw=0)
        ax.axvspan(-0.25, -0.1, color=PALETTE["gold"], alpha=0.12, lw=0)
        ax.axvspan(0.1, 0.25, color=PALETTE["gold"], alpha=0.12, lw=0)
        ax.axvline(0, color="#666", lw=0.8)
        ax.set_yticks(y, smd["label"])
        ax.set_xlabel("Standardized mean difference (pop-weighted)")
        ax.set_title(title)
        lim = max(0.6, float(np.nanmax(np.abs(smd[[c[0] for c in cols]].to_numpy()))) * 1.1)
        ax.set_xlim(-lim, lim)
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Covariate balance (SMD) - green |SMD| < 0.10, amber < 0.25 (enhancement over the original %-gap bars)",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Imbalance on demographics is reported, not hidden: outcome-rate features drive the counterfactual; "
                     "the DiD / SC designs absorb level differences.")
    return save_fig(fig, MOD, "11_balance_smd")


# ======================================================================================
# Vendor B design
# ======================================================================================
def fig_b_switch(B, cfg) -> Path:
    sw = B.switch
    s = sw["share"]
    td = cfg.tests[B.params.test]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(s.index, s.values, color=PALETTE["navy"], marker="o", ms=4, label="Weekly vendor-attributed QS share")
    if sw.get("n_before", 0) >= 2:
        ax.hlines(sw["mean_before"], s.index.min(), sw["switch_week"] - 0.5, color=PALETTE["orange"], lw=2.5,
                  label=f"mean before {sw['mean_before']:.1%}")
        ax.hlines(sw["mean_after"], sw["switch_week"] - 0.5, s.index.max(), color=PALETTE["teal"], lw=2.5,
                  label=f"mean after {sw['mean_after']:.1%}")
    for w in sw["dip_weeks"]:
        if w in s.index:
            ax.scatter([w], [s.loc[w]], s=110, facecolor="none", edgecolor=PALETTE["red"], lw=1.6, zorder=5)
    ax.scatter([], [], s=80, facecolor="none", edgecolor=PALETTE["red"], label="reporting-dip weeks (excluded)")
    mark_line(ax, sw["switch_week"] - 0.5, "vendor switch", color=PALETTE["orange"])
    lag0 = td.design_end - 2
    ax.axvspan(lag0 - 0.5, td.design_end + 0.5, color="#d9dce1", alpha=0.6, lw=0)
    ax.text(lag0, ax.get_ylim()[1], " attribution\n lag", va="top", fontsize=8, color="#555")
    pct_axis(ax)
    ax.set_xlabel("Week index")
    ax.set_ylabel("Vendor-attributed QS / all QS")
    ax.set_title("Vendor B - structural break at the vendor switch (Welch t-test)")
    _sub(ax, f"shift {sw.get('shift_pct', np.nan):+.1%}, t = {sw.get('t', np.nan):.2f}, p = {sw['p_value']:.3g} "
                 f"(n = {sw['n_before']} vs {sw['n_after']} weeks) -> "
                 + ("vendor-share predictors use post-switch weeks only" if sw["significant"] else "full window usable"))
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    source_note(fig, "The true media effect does not change at the switch - only the vendor's reporting does.")
    return save_fig(fig, MOD, "12_b_vendor_switch")


def fig_b_silhouette(B) -> Path:
    sil = B.kmeans["silhouette"]
    p = B.params
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ok = sil["mean_cluster_pop"] <= p.cluster_pop_target
    ax.plot(sil["k"], sil["silhouette"], color=PALETTE["slate"], lw=1.5, zorder=1)
    ax.scatter(sil.loc[ok, "k"], sil.loc[ok, "silhouette"], color=PALETTE["teal"], s=60, zorder=3,
               label=f"eligible K (mean cluster pop <= {_fmt_pop(p.cluster_pop_target)})")
    ax.scatter(sil.loc[~ok, "k"], sil.loc[~ok, "silhouette"], color="#c5c9d0", s=60, zorder=3,
               label="clusters too large for a holdout")
    kb, k = B.kmeans["k_best_silhouette"], B.kmeans["k"]
    ax.scatter([k], [sil.set_index("k").loc[k, "silhouette"]], s=230, facecolor="none", edgecolor=ROLE["treatment"],
               lw=2.2, zorder=4, label=f"used: K = {k}")
    ax.annotate(f"unconstrained best K = {kb}", (kb, sil.set_index("k").loc[kb, "silhouette"]), xytext=(15, -5),
                textcoords="offset points", fontsize=9)
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.set_xticks(sil["k"])
    ax.set_title("Vendor B - geo-weighted K-means: silhouette curve and the K actually used")
    _sub(ax, f"Rule (documented, not a silent hard-code): {B.kmeans['rule']}; lat/lon weight x{p.geo_weight:g}")
    ax.legend(loc="upper right", fontsize=9)
    source_note(fig)
    return save_fig(fig, MOD, "13_b_silhouette")


def fig_b_cluster_map(B, zips) -> Path:
    zs = zips[zips["region"] == B.params.region].reset_index(drop=True)
    polys = _cells(zs)
    cl = B.feat.set_index("zip")["cluster"]
    k = zs["zip"].map(cl)
    conc = zs["concurrent_test_flag"].to_numpy() == 1
    cols = []
    for kk, cc, inc in zip(k, conc, zs["dsp_b_inclusion"]):
        if cc:
            cols.append("#4b5563")
        elif pd.isna(kk):
            cols.append("#f4f5f7")
        else:
            cols.append(CLUSTER_COLORS[(int(kk) - 1) % len(CLUSTER_COLORS)])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 9.2), gridspec_kw={"width_ratios": [1, 1]})
    _draw_cells(ax, polys, cols, alpha=0.95)
    tmask = zs["zip"].isin(set(B.treated)).to_numpy()
    _draw_cells(ax, polys, cols, edge="black", lw=1.0, idx=np.where(tmask)[0], zorder=2)
    for kk, s in B.feat.groupby("cluster"):
        ax.text(np.average(s["lon"], weights=s["pop"]), np.average(s["lat"], weights=s["pop"]), str(kk),
                fontsize=10, fontweight="bold", ha="center", va="center", color="white",
                bbox=dict(boxstyle="circle", fc="#222" if kk == B.treated_cluster else "#555", ec="none", alpha=0.8))
    _set_extent(ax, zs)
    _aspect(ax, zs["lat"])
    _scalebar(ax, zs)
    ax.set_title(f"Clusters (K={B.kmeans['k']}); treatment = cluster {B.treated_cluster} (outlined)")
    hd = [Patch(color="#4b5563", label=f"Concurrent test zips - excluded ({int(conc.sum())})"),
          Patch(color="#f4f5f7", label="Not on inclusion list")]
    ax.legend(handles=hd, loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=8.5)
    # right: final role map
    asg = B.assignment(zips).set_index("zip")["group"].reindex(zs["zip"]).to_numpy()
    rc = {"treatment": ROLE["treatment"], "synthetic-control donor": ROLE["synthetic"], "donor pool": "#bfe3dd",
          "buffer": PALETTE["gold"], "concurrent test (excluded)": "#4b5563", "not on inclusion list": "#f4f5f7"}
    cols2 = [rc.get(g, "#f4f5f7") for g in asg]
    _draw_cells(ax2, polys, cols2)
    _draw_cells(ax2, polys, cols2, edge="black", lw=1.0, idx=np.where(asg == "treatment")[0], zorder=2)
    _buffer_contour(ax2, zs, B.treated, B.params.buffer_miles)
    _set_extent(ax2, zs)
    _aspect(ax2, zs["lat"])
    n = pd.Series(asg).value_counts()
    w = B.weights
    hd2 = [Patch(color=rc[g], label=f"{g} ({n.get(g, 0)})") for g in rc] + \
          [Line2D([], [], color="#7a5c00", ls="--", label=f"{B.params.buffer_miles:g}-mi buffer line")]
    ax2.legend(handles=hd2, loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=8.5, ncol=2)
    ax2.set_title("Design roles: treatment, weighted donors, buffer")
    _sub(ax2, "SC weights: " + ", ".join(f"{c}={v:.2f}" for c, v in w.sort_values(ascending=False).items()))
    fig.suptitle("Vendor B - geo-weighted K-means clusters and the final holdout design", fontsize=13,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Donor zips inside the buffer are removed from their cluster before the final SC fit.")
    return save_fig(fig, MOD, "14_b_cluster_map")


def fig_b_tournament(B) -> Path:
    t = B.tournament[B.tournament["status"] == "ok"].sort_values("score", ascending=False)
    p = B.params
    lab = [f"C{int(k)}" + ("*" if k == B.treated_cluster else "") for k in t["cluster"]]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))
    sc = [ROLE["treatment"] if k == B.treated_cluster else (PALETTE["navy"] if f else "#b9bec7")
          for k, f in zip(t["cluster"], t["feasible"])]
    axes[0].bar(lab, t["score"], color=sc)
    axes[0].set_title(f"Composite score\n({p.score_weights[0]:.2f} acc + {p.score_weights[1]:.2f} fit + "
                      f"{p.score_weights[2]:.2f} speed)", fontsize=10.5)
    axes[1].bar(lab, t["abs_error"] * 100, color=PALETTE["navy"])
    axes[1].axhline(4, color=PALETTE["red"], ls="--", lw=1)
    axes[1].set_title(f"|error| vs +{p.injected_lift:.0%} injected (pp)", fontsize=10.5)
    axes[2].bar(lab, t["rel_rmse"], color=PALETTE["green"])
    pct_axis(axes[2])
    axes[2].set_title("Pre-period fit: relative RMSE", fontsize=10.5)
    wk = t["weeks_to_80"].astype(float).fillna(p.max_weeks * 3)
    axes[3].bar(lab, np.minimum(wk, p.max_weeks * 3), color=[PALETTE["navy"] if v <= p.max_weeks else "#e05c38" for v in wk])
    axes[3].axhline(p.max_weeks, color=PALETTE["red"], ls="--", lw=1)
    axes[3].text(len(lab) - 0.5, p.max_weeks, f"{p.max_weeks}w limit", ha="right", va="bottom", fontsize=8, color=PALETTE["red"])
    axes[3].set_title(f"Weeks to 80% power at {p.injected_lift:.0%} (capped)", fontsize=10.5)
    for ax in axes:
        ax.tick_params(axis="x", rotation=60, labelsize=8.5)
    hd = [Patch(color=ROLE["treatment"], label="chosen"), Patch(color=PALETTE["navy"], label="feasible"),
          Patch(color="#b9bec7", label="infeasible (pop bounds / largest / >limit)")]
    axes[0].legend(handles=hd, fontsize=8, loc="upper right")
    fig.suptitle(f"Vendor B - synthetic-control tournament (sorted by score). Selected: cluster {B.treated_cluster} - "
                 f"{B.selection_note}", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Feasibility rules are explicit parameters; an analyst override requires a written rationale.")
    return save_fig(fig, MOD, "15_b_tournament_scorecard")


def fig_b_tournament_fits(B) -> Path:
    ks = sorted(B.tournament_series)
    ncol = 4
    nrow = int(np.ceil(len(ks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.9 * nrow), sharex=True)
    t = B.tournament.set_index("cluster")
    for ax, k in zip(axes.ravel(), ks):
        d = B.tournament_series[k]
        ax.plot(d.index, d["actual"], color="black", lw=1.6, label=f"actual (+{B.params.injected_lift:.0%} injected)")
        ax.plot(d.index, d["synthetic"], color="#CC0000", ls="--", lw=1.6, label="fixed-weight synthetic")
        ax.axvspan(B.pseudo_post[0] - 0.5, B.pseudo_post[-1] + 0.5, color="#1a5fa8", alpha=0.08, lw=0)
        r = t.loc[k]
        ax.set_title(f"C{k}{' (chosen)' if k == B.treated_cluster else ''} | detected {r['detected']:+.1%} | "
                     f"rel RMSE {r['rel_rmse']:.1%}", fontsize=9.5,
                     color=ROLE["treatment"] if k == B.treated_cluster else "#222")
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(ks):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Vendor B - every cluster as pseudo-treatment: actual vs locked-weight synthetic (QS per 1k pop; "
                 "shaded = pseudo-post)", fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig)
    return save_fig(fig, MOD, "16_b_tournament_fits")


def fig_b_final_sc(B) -> Path:
    sc = B.sc
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.8, 1]})
    ax.plot(sc.actual.index, sc.actual, color="black", lw=2, label=f"Cluster {B.treated_cluster} (actual)")
    ax.plot(sc.synthetic.index, sc.synthetic, color="#CC0000", ls="--", lw=2, label="Synthetic (weights locked)")
    ax.fill_between(sc.actual.index, sc.actual, sc.synthetic, color="#999", alpha=0.15, lw=0)
    ax.set_xlabel("Week index (design window)")
    ax.set_ylabel("Quote starts per 1k population")
    ax.set_title("Vendor B - final synthetic control (pre-period fit)")
    _sub(ax, f"RMSE {sc.pre_rmspe:.3f} | rel. RMSE {sc.rel_rmse:.1%} | MSPE {sc.pre_mspe:.2e} | "
                 f"{sc.n_donors_used} active donors | injected +{B.params.injected_lift:.0%} recovered as "
                 f"{B.injected_final['detected']:+.1%}")
    ax.legend(loc="lower left")
    w = B.weights.sort_values()
    ax2.barh(w.index, w.values, color="#1a5fa8")
    for i, v in enumerate(w.values):
        ax2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Donor weight (sum = 1, >= 0)")
    ax2.set_title("Donor-cluster weights")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    source_note(fig, "Dependent = QS per capita (ratio of sums), not the vendor-attributed share - the share has a "
                     "reporting level shift and attribution lag.")
    return save_fig(fig, MOD, "17_b_final_synthetic_control")


def fig_b_placebos(B) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ps = B.placebo_space["placebos"]
    real = B.placebo_space["treated_fit"].lift(B.pseudo_post)["lift"]
    axes[0].bar(ps["unit"], ps["lift"], color="#b9bec7")
    axes[0].axhline(real, color=ROLE["treatment"], lw=2, label=f"treated C{B.treated_cluster}: {real:+.1%}")
    pct_axis(axes[0])
    axes[0].set_title(f"In-space (clusters): p = {B.placebo_space['p_value']:.2f}\n"
                      f"min attainable p = {B.placebo_space['min_attainable_p']:.2f} with {B.placebo_space['n_placebos']} placebos",
                      fontsize=10.5)
    axes[0].legend(fontsize=8.5)
    axes[0].tick_params(axis="x", rotation=60, labelsize=8.5)
    pt = B.placebo_time["placebos"]
    axes[1].plot(pt["fake_start"], pt["lift"], marker="o", color=PALETTE["navy"])
    axes[1].axhline(0, color="#666", lw=0.8)
    pct_axis(axes[1])
    axes[1].set_xlabel("Fake launch week")
    axes[1].set_title(f"In-time placebos (no intervention): sd = {B.placebo_time['sd_placebo_lift']:.1%}", fontsize=10.5)
    pg = B.placebo_geo
    axes[2].hist(pg["lift"], bins=25, color="#b9bec7", edgecolor="white")
    axes[2].axvline(0, color="#666", lw=0.8)
    q = np.quantile(pg["lift"], [0.05, 0.95])
    for v in q:
        axes[2].axvline(v, color=PALETTE["red"], ls="--", lw=1)
    pct_axis(axes[2], "x")
    axes[2].set_title(f"Random-geography placebos (n = {len(pg)}):\n90% null band [{q[0]:+.1%}, {q[1]:+.1%}]",
                      fontsize=10.5)
    fig.suptitle("Vendor B - placebo inference on the pseudo-post window (no real effect exists here)",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "With K-1 cluster placebos the smallest p is 1/K; random contiguous geographies give far more "
                     "permutations (fix of the original 9-donor limit).")
    return save_fig(fig, MOD, "18_b_placebo_inference")


# ======================================================================================
# Readout
# ======================================================================================
def fig_r_weekly(R, cfg) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.6))
    for row, v in enumerate("AB"):
        S = R.series[v]
        td = cfg.tests[S.test]
        for col, s in enumerate(("qs", "rated", "sales")):
            ax = axes[row, col]
            idx = [w for w in S.t[s].index if td.design_start <= w <= td.test_end + 2]
            ax.plot(idx, S.t[s].loc[idx], color=ROLE["treatment"], lw=1.8, label="Treatment (holdout)")
            ax.plot(idx, S.c[s].loc[idx], color=ROLE["control"], lw=1.8, ls="--", label=S.control_label)
            shade_window(ax, td.test_start - 0.5, td.test_end + 0.5, label="Holdout" if col == 0 else None)
            ax.axvspan(td.readout_pre_start - 0.5, td.test_start - 0.5, color=PALETTE["sky"], alpha=0.10, lw=0)
            ax.set_title(f"{VENDOR_LABEL[v]} - {STAGE_LABELS[s]}", fontsize=11)
            if col == 0:
                ax.set_ylabel("Weekly count")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    axes[1, 0].legend(loc="lower left", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Week index")
    fig.suptitle("Readout - weekly funnel, treatment vs control (blue band = readout pre-period, gold = holdout)",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Vendor B control = locked SC weights x donor clusters, scaled to the treated population.")
    return save_fig(fig, MOD, "19_readout_weekly_funnel")


def fig_r_dumbbell(R) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for row, v in enumerate("AB"):
        for col, s in enumerate(("qs", "rated", "sales")):
            ax = axes[row, col]
            r = R.results[v][s]["change_ratio_did"]
            ax.plot([0, 1], [r["t_pre"], r["t_post"]], color="#2980b9", marker="o", lw=2, label="Treatment")
            ax.plot([0, 1], [r["c_pre"], r["c_post"]], color="#95a5a6", marker="s", lw=2, ls="--", label="Control")
            ax.scatter([1], [r["cf_per_week"]], s=90, color="#e67e22", edgecolor="black", zorder=5,
                       label="Counterfactual (T_pre x C ratio)")
            ax.annotate("", (1, r["t_post"]), (1, r["cf_per_week"]),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
            for x_, y_ in ((0, r["t_pre"]), (1, r["t_post"]), (0, r["c_pre"]), (1, r["c_post"]), (1, r["cf_per_week"])):
                ax.text(x_ + (0.04 if x_ else -0.04), y_, f"{y_:,.0f}", fontsize=8, va="center",
                        ha="left" if x_ else "right")
            ax.axvline(0.5, color=PALETTE["red"], ls=":", lw=1)
            ax.set_xticks([0, 1], ["Pre (mean/wk)", "Holdout (mean/wk)"])
            ax.set_xlim(-0.4, 1.45)
            ax.set_title(f"{VENDOR_LABEL[v]} - {STAGE_LABELS[s]}", fontsize=11)
            _sub(ax, f"T ratio {r['t_ratio']:.3f} | C ratio {r['c_ratio']:.3f} | contribution {r['lift']:+.1%} "
                         f"[{r['ci_low']:+.1%}, {r['ci_high']:+.1%}]")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9.5)
    fig.suptitle("Change-ratio DiD (headline): counterfactual = treatment pre x control post/pre; contribution = "
                 "(counterfactual - actual)/counterfactual", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(h_pad=2.5)
    source_note(fig, "90% week-bootstrap intervals. Positive = volume the channel contributes (lost when paused).")
    return save_fig(fig, MOD, "20_readout_did_dumbbell")


def fig_r_bsts(R, cfg) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.6))
    for row, v in enumerate("AB"):
        S = R.series[v]
        b = R.results[v]["qs"]["bsts"]
        pw = b["pointwise"]
        td = cfg.tests[S.test]
        ax = axes[row, 0]
        h = [w for w in S.t["qs"].index if S.hist[0] <= w <= td.test_end]
        ax.plot(h, S.t["qs"].loc[h], color="black", lw=1.5, label="Observed (treatment)")
        ax.plot(pw.index, pw["predicted"] if R.params.readout.sign == 1 else pw["predicted"], color="#CC0000", ls="--",
                lw=1.8, label="Counterfactual (structural TS)")
        ax.fill_between(pw.index, pw["pred_low"], pw["pred_high"], color="#CC0000", alpha=0.15, lw=0,
                        label="90% prediction band")
        mark_line(ax, td.test_start - 0.5, "holdout", color=PALETTE["slate"])
        ax.set_title(f"{VENDOR_LABEL[v]} - quote starts: observed vs predicted", fontsize=11)
        ax.legend(loc="lower left", fontsize=8)
        ax2 = axes[row, 1]
        cum = pw["effect"].cumsum()
        ax2.plot(pw.index, cum, color=PALETTE["navy"], marker="o")
        ax2.axhline(0, color=PALETTE["green"], lw=1)
        ax2.set_title(f"Cumulative contribution: {b['lift']:+.1%} [{b['ci_low']:+.1%}, {b['ci_high']:+.1%}] - "
                      f"{'significant' if b['significant'] else 'interval spans 0'}", fontsize=10.5)
        ax2.set_ylabel("Cumulative QS (positive = lost to holdout)")
    for ax in axes[1]:
        ax.set_xlabel("Week index")
    fig.suptitle("CausalImpact-style structural time series (statsmodels UC; significance = interval excludes 0 only)",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, f"Model: {R.results['A']['qs']['bsts']['model']}; long pre-period from week {R.series['A'].hist[0]}.")
    return save_fig(fig, MOD, "21_readout_bsts")


def fig_r_sc(R, cfg) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for row, v in enumerate("AB"):
        r = R.sc[v]["qs"]
        td = cfg.tests[R.series[v].test]
        gap = (r["actual"] - r["synthetic"]) / r["synthetic"]
        ax = axes[row, 0]
        ax.plot(gap.index, gap, color=PALETTE["navy"], lw=1.8)
        ax.axhline(0, color="#666", lw=0.8)
        shade_window(ax, td.test_start - 0.5, td.test_end + 0.5)
        pct_axis(ax)
        ax.set_ylabel("(actual - synthetic) / synthetic")
        ax.set_title(f"{VENDOR_LABEL[v]} - QS gap vs {'county-donor' if v == 'A' else 'locked cluster'} synthetic control",
                     fontsize=11)
        ax2 = axes[row, 1]
        null = np.asarray(r["null"], float)
        null = R.params.readout.sign * (null - null.mean())
        ax2.hist(null, bins=30, color="#b9bec7", edgecolor="white", label=f"placebo null (n={len(null)})")
        ax2.axvline(r["lift"], color=ROLE["treatment"], lw=2.2, label=f"observed {r['lift']:+.1%}")
        pct_axis(ax2, "x")
        ax2.set_title(f"Permutation p = {r['p_value']:.2f} ({'sig.' if r['significant'] else 'n.s.'} at 90%)", fontsize=11)
        ax2.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Synthetic control + placebo inference at readout (in-space + random-geography placebos)",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Late-pre level bias of the locked-weight synthetic is removed, as in the design-time injected check.")
    return save_fig(fig, MOD, "22_readout_sc_placebo")


def fig_r_forest(R) -> Path:
    tt = truth_table(R)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True)
    for ax, v in zip(axes, "AB"):
        d = tt[tt["vendor"] == v]
        y, ticks, labs = 0, [], []
        for s in ("qs", "rated", "sales"):
            ds = d[d["metric"] == s]
            tru = ds["truth"].iloc[0]
            y0 = y
            for r in ds.itertuples():
                c = PALETTE["navy"] if r.significant else "#8a93a3"
                ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt="o",
                            color=c, capsize=3, ms=6, mfc=c if r.significant else "white")
                ticks.append(y)
                labs.append(f"{STAGE_LABELS[s]}: {r.method_label}")
                y += 1
            ax.plot([tru, tru], [y0 - 0.5, y - 0.5], color=PALETTE["green"], lw=2.5)
            ax.axhspan(y0 - 0.5, y - 0.5, color="#f4f6f9" if s == "rated" else "white", zorder=0)
            y += 0.8
        ax.axvline(0, color="#666", lw=0.8)
        ax.set_yticks(ticks, labs, fontsize=8.5)
        ax.invert_yaxis()
        pct_axis(ax, "x")
        ax.set_xlabel("Channel contribution (relative lift, 90% interval)")
        ax.set_title(f"{VENDOR_LABEL[v]} ({R.series[v].test})")
    hd = [Line2D([], [], color=PALETTE["navy"], marker="o", label="interval excludes 0"),
          Line2D([], [], color="#8a93a3", marker="o", mfc="white", label="interval spans 0"),
          Line2D([], [], color=PALETTE["green"], lw=2.5, label="ground truth (scenario expected lift)")]
    fig.legend(handles=hd, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=3, fontsize=9.5)
    fig.suptitle("Triangulation forest plot: every method, every funnel stage, vs the injected ground truth",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig)
    return save_fig(fig, MOD, "23_readout_triangulation_forest")


def fig_r_truth(R) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, v in zip(axes, "AB"):
        test = R.series[v].test
        tr = R.truth[test]["holdout"]
        x = np.arange(3)
        est = [R.results[v][s]["change_ratio_did"] for s in ("qs", "rated", "sales")]
        ax.bar(x - 0.2, [tr[s]["true_lift_pct"] for s in ("qs", "rated", "sales")], 0.38, color=PALETTE["green"],
               label="Truth (expected)")
        ax.bar(x + 0.2, [e["lift"] for e in est], 0.38, color=PALETTE["navy"], label="Estimate (change-ratio DiD)",
               yerr=[[e["lift"] - e["ci_low"] for e in est], [e["ci_high"] - e["lift"] for e in est]], capsize=4,
               error_kw=dict(ecolor="#555", lw=1))
        ax.scatter(x - 0.2, [tr[s]["realized_lift_pct"] for s in ("qs", "rated", "sales")], marker="_", s=300,
                   color="black", zorder=5, label="Truth realized with this noise draw")
        ax.axhline(0, color="#666", lw=0.8)
        ax.set_xticks(x, [STAGE_LABELS[s] for s in ("qs", "rated", "sales")])
        pct_axis(ax)
        ax.set_title(f"{VENDOR_LABEL[v]}: funnel-stage lift, truth vs estimate")
        ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("Channel contribution")
    fig.suptitle("Funnel-stage lifts: Vendor A is full-funnel by construction, Vendor B top-funnel only",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig)
    return save_fig(fig, MOD, "24_readout_funnel_truth_vs_estimate")


def fig_r_costper(R) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, s in zip(axes, ("qs", "rated", "sales")):
        rows = []
        for v in "AB":
            b = R.business[v]
            rows.append((v, b.get(f"cost_per_incremental_{s}"), b.get(f"cost_per_incremental_{s}_true"),
                         b.get(f"last_touch_cost_per_{s}"), b["vendor_reported_cost_per_qs"] if s == "qs" else None))
        x = np.arange(2)
        ser = [("Vendor-reported", 4, PALETTE["red"]), ("Site last-touch", 3, PALETTE["gold"]),
               ("Incremental (estimate)", 1, PALETTE["navy"]), ("Incremental (truth)", 2, PALETTE["green"])]
        wdt = 0.2
        for i, (lab, j, c) in enumerate(ser):
            vals = [r[j] if r[j] is not None and np.isfinite(r[j]) and r[j] > 0 else np.nan for r in rows]
            ax.bar(x + (i - 1.5) * wdt, vals, wdt, color=c, label=lab)
            for xi, val in zip(x, vals):
                if np.isfinite(val):
                    ax.text(xi + (i - 1.5) * wdt, val, f"${val:,.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_yscale("log")
        for xi, r in zip(x, rows):
            if r[1] is None or not np.isfinite(r[1]) or r[1] <= 0:
                ax.text(xi - 0.5 * wdt, ax.get_ylim()[0] * 1.15, "n/a:\nincr. <= 0", ha="center", va="bottom",
                        fontsize=7.5, color=PALETTE["navy"])
        from matplotlib.ticker import FuncFormatter, NullFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(x, ["Vendor A", "Vendor B"])
        ax.set_title(f"Cost per {STAGE_LABELS[s].lower()[:-1]}", fontsize=11)
        ax.set_ylabel("$ (log scale)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=9.5)
    fig.suptitle("Incremental cost-per vs what the vendor and last-touch attribution claim",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Incremental cost-per = spend removed in the treatment geo / incremental volume. Spend-parity caveat: "
                     "small-budget tests are not comparable to scaled spend.")
    return save_fig(fig, MOD, "25_readout_cost_per")


def fig_r_spend(R, cfg) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, v in zip(axes, "AB"):
        S = R.series[v]
        td = cfg.tests[S.test]
        idx = [w for w in S.spend_t.index if td.readout_pre_start - 6 <= w <= td.test_end + 2]
        ax.plot(idx, S.spend_t.loc[idx], color=ROLE["treatment"], marker="o", ms=3.5, label="Treatment geo spend")
        ax.plot(idx, S.spend_c.loc[idx], color=ROLE["control"], ls="--", marker="s", ms=3.5, label="Control spend")
        shade_window(ax, td.test_start - 0.5, td.test_end + 0.5)
        money_axis(ax)
        b = R.business[v]
        ax.set_title(f"{VENDOR_LABEL[v]}: spend compliance - {b['compliance_pct_of_bau']:.1%} of BAU still delivered",
                     fontsize=11)
        ax.set_xlabel("Week index")
        ax.legend(fontsize=8.5, loc="center left")
    fig.suptitle("Did the holdout actually happen? Weekly DSP spend, treatment vs control",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig)
    return save_fig(fig, MOD, "26_readout_spend_compliance")


def fig_r_spillover(R) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, v in zip(axes, "AB"):
        sp = R.spillover[v]
        x = np.arange(3)
        st = ("qs", "rated", "sales")
        est = [sp[s]["buffer_loss_est"] for s in st]
        ci = np.array([sp[s]["ci"] for s in st])
        nv = [sp[s]["naive_adjacent_control_lift"] for s in st]
        nci = np.array([sp[s]["naive_ci"] for s in st])
        good = [R.results[v][s]["change_ratio_did"]["lift"] for s in st]
        tru = [R.truth[R.series[v].test]["holdout"][s]["true_lift_pct"] for s in st]
        ax.bar(x - 0.27, est, 0.25, yerr=[np.array(est) - ci[:, 0], ci[:, 1] - np.array(est)], capsize=3,
               color=PALETTE["gold"], label=f"Loss in served buffer zips (<= {sp['radius_mi']:g} mi)")
        ax.bar(x, nv, 0.25, yerr=[np.array(nv) - nci[:, 0], nci[:, 1] - np.array(nv)], capsize=3,
               color="#b9bec7", label="Lift if buffer zips had been the control")
        ax.bar(x + 0.27, good, 0.25, color=PALETTE["navy"], label="Lift with the buffered design")
        ax.scatter(x + 0.27, tru, marker="_", s=320, color=PALETTE["green"], zorder=5, label="truth")
        ax.axhline(0, color="#666", lw=0.8)
        ax.set_xticks(x, [STAGE_LABELS[s] for s in st])
        pct_axis(ax)
        tl = sp["truth"].get("qs", {}).get("loss_pct", np.nan)
        ax.set_title(f"{VENDOR_LABEL[v]}: {sp['n_buffer_served']} served buffer zips; true spillover loss "
                     f"(all buffer zips) {tl:.1%} of QS", fontsize=10.5)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Spillover check: the buffer zips vs the control, and the bias an adjacent control would carry",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig, "Buffer-zip loss is estimated with the same change-ratio DiD against the design's control; "
                     "buffer zips are geographically different, so their own drift is part of this number.")
    return save_fig(fig, MOD, "27_readout_spillover_check")


def fig_r_roi(R) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, v in zip(axes, "AB"):
        b = R.business[v]
        d = R.decisions[v]
        draws = np.asarray(b["roi_draws"], float)
        lo, hi = np.quantile(draws, [0.01, 0.99])
        ax.hist(np.clip(draws, lo, hi), bins=50, color="#b9bec7", edgecolor="white")
        ax.axvline(1, color=PALETTE["red"], ls="--", lw=1.5, label="break-even (ROI = 1)")
        ax.axvline(b["roi_window_est"], color=PALETTE["navy"], lw=2, label=f"estimate {b['roi_window_est']:.2f}")
        ax.axvline(b["roi_window_true"], color=PALETTE["green"], lw=2.5, label=f"truth {b['roi_window_true']:.2f}")
        ax.set_xlabel("Window ROI = incremental sales x value / spend removed (bootstrap)")
        dc = {"GO": PALETTE["green"], "NO-GO": PALETTE["red"]}.get(d["decision"], "#b7791f")
        if d["decision"].startswith("HOLD"):
            dc = PALETTE["slate"]
        ax.set_title(f"{VENDOR_LABEL[v]}: {d['decision']}  (P(ROI>1) = {d['p_roi_gt_1']:.0%})", color=dc, fontsize=12)
        import textwrap
        ax.text(0.0, -0.2, "\n".join(textwrap.wrap(d["rationale"], 95)), transform=ax.transAxes, fontsize=8.5,
                color="#555", va="top")
        ax.legend(fontsize=8.5)
    fig.suptitle("Go / no-go: probability the test-window ROI clears break-even", fontsize=12.5, fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    source_note(fig)
    return save_fig(fig, MOD, "28_readout_roi_decision")


def fig_a_sizing(A) -> Path:
    """Power-driven holdout sizing: design MDE by stage at each rung of the size ladder."""
    d = A.sizing[A.sizing["status"] == "ok"].copy()
    sd, p = A.sizing_decision, A.params
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    x = np.arange(len(d))
    wdt = 0.26
    cols = {"qs": PALETTE["navy"], "rated": PALETTE["teal"], "sales": PALETTE["orange"]}
    for k, m in enumerate(("qs", "rated", "sales")):
        bars = ax.bar(x + (k - 1) * wdt, d[f"mde_{m}"], wdt, color=cols[m], label=f"{STAGE_LABELS[m]} MDE",
                      edgecolor="#333", lw=0.4)
        for b_, v in zip(bars, d[f"mde_{m}"]):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.axhline(p.target_mde, color=PALETTE["red"], ls="--", lw=1.5,
               label=f"pre-registered target: {p.target_mde:.0%} on {STAGE_LABELS[p.power_metric].lower()}")
    labels = [f"{r.pop_target/1e3:.0f}k target\n({r.t_pop/1e3:.0f}k, {r.tier})" for r in d.itertuples()]
    ax.set_xticks(x, labels)
    sel = int(np.flatnonzero(d["pop_target"].to_numpy() == sd["pop_target"])[0])
    ax.axvspan(sel - 0.45, sel + 0.45, color=PALETTE["green"], alpha=0.10, lw=0, label="selected design")
    pct_axis(ax)
    ax.set_ylabel(f"Design MDE (80% power, one-sided alpha={p.alpha}, {sd['test_weeks']} wk)")
    ax.set_xlabel("Holdout size (population target -> realised patch, best pair's gate tier)")
    ax.set_title(f"Vendor A - holdout sized by its own power analysis: {sd['verdict']}", loc="left")
    _sub(ax, "Smallest strict-gate design whose SALES MDE meets the target; design-window data only")
    ax.set_ylim(0, float(d[[f"mde_{m}" for m in ("qs", "rated", "sales")]].to_numpy().max()) * 1.35)
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    fig.tight_layout()
    source_note(fig, sd["rationale"])
    return save_fig(fig, MOD, "08b_a_power_sizing_ladder")


def make_all(A, B, R, P, cfg, zips) -> List[Path]:
    figs = [fig_flowchart(A, B, P), fig_a_patches(A), fig_a_donor_buffer(A), fig_a_corr(A), fig_a_candidates(A),
            fig_a_overlays(A), fig_a_profile(A), fig_power(A, B), fig_a_sizing(A), fig_a_injected(A, B), fig_a_map(A), fig_smd(A, B),
            fig_b_switch(B, cfg), fig_b_silhouette(B), fig_b_cluster_map(B, zips), fig_b_tournament(B),
            fig_b_tournament_fits(B), fig_b_final_sc(B), fig_b_placebos(B)]
    if R is not None:
        figs += [fig_r_weekly(R, cfg), fig_r_dumbbell(R), fig_r_bsts(R, cfg), fig_r_sc(R, cfg), fig_r_forest(R),
                 fig_r_truth(R), fig_r_costper(R), fig_r_spend(R, cfg), fig_r_spillover(R), fig_r_roi(R)]
    return figs
