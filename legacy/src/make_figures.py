"""Run every analysis and save publication-quality figures to reports/figures/.

Usage:  python -m src.make_figures
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import did, synthetic_control as sc, cuped, mmm, power

FIGDIR = os.path.join("reports", "figures")
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})
BLUE, ORANGE, GREEN, RED = "#2563eb", "#f59e0b", "#16a34a", "#dc2626"


def load():
    df = pd.read_csv("data/geo_panel.csv")
    with open("data/ground_truth.json") as f:
        gt = json.load(f)
    return df, gt


def fig_did(df, gt):
    true_lift = gt["true_lift"]
    # Aggregate treated vs control daily means.
    agg = df.groupby(["day", "treated"])["conversions"].mean().unstack()
    start = df.loc[df["post"] == 1, "day"].min()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(agg.index, agg[1], color=BLUE, label="Treated geos")
    ax.plot(agg.index, agg[0], color=ORANGE, label="Control geos")
    ax.axvline(start, color="grey", ls="--", lw=1)
    ax.text(start + 1, ax.get_ylim()[1] * 0.95, "campaign start", fontsize=9, color="grey")
    ax.set_title("Geo-split DiD: treated vs control conversions")
    ax.set_xlabel("day"); ax.set_ylabel("mean conversions / geo"); ax.legend()
    fig.savefig(os.path.join(FIGDIR, "01_did_trends.png")); plt.close(fig)

    r = did.run_did(df)
    naive = did.naive_pre_post(df)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["Naive\npre/post", "DiD\n(causal)", "TRUE\nlift"]
    vals = [naive, r["lift"], true_lift]
    colors = [RED, BLUE, GREEN]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, alpha=0.85)
    # CI on DiD bar
    lo, hi = r["lift_ci"]
    ax.errorbar(1, r["lift"] * 100, yerr=[[(r["lift"]-lo)*100], [(hi-r["lift"])*100]],
                fmt="none", ecolor="black", capsize=5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*100+0.3, f"{v:+.1%}", ha="center", fontsize=10)
    ax.axhline(true_lift*100, color=GREEN, ls=":", lw=1)
    ax.set_ylabel("estimated lift (%)")
    ax.set_title(f"DiD recovers the injected lift (p={r['pvalue']:.1e})")
    fig.savefig(os.path.join(FIGDIR, "02_did_estimates.png")); plt.close(fig)
    return r, naive


def fig_sc(df, gt):
    r = sc.fit_synthetic_control(df)
    start = r["campaign_start_day"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(r["target"].index, r["target"].values, color=BLUE, label="Treated (observed)")
    ax.plot(r["synthetic"].index, r["synthetic"].values, color=RED, ls="--",
            label="Synthetic control")
    ax.axvline(start, color="grey", ls="--", lw=1)
    ax.fill_between(r["target"].index[r["target"].index >= start],
                    r["synthetic"].values[r["synthetic"].index >= start],
                    r["target"].values[r["target"].index >= start],
                    color=GREEN, alpha=0.2, label="estimated lift")
    ax.set_title(f"Synthetic control: post-period gap = {r['lift']:+.1%} "
                 f"(true {gt['true_lift']:+.0%})")
    ax.set_xlabel("day"); ax.set_ylabel("mean conversions"); ax.legend()
    fig.savefig(os.path.join(FIGDIR, "03_synthetic_control.png")); plt.close(fig)
    return r


def fig_cuped(df):
    frame = cuped.geo_level_frame(df)
    r = cuped.cuped_adjust(frame)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.scatter(frame["x"], frame["y"], c=frame["treated"].map({0: ORANGE, 1: BLUE}), s=40)
    ax.set_xlabel("log pre-period conversions (covariate)")
    ax.set_ylabel("log post-period conversions")
    ax.set_title(f"Pre/post correlation r={r['corr_xy']:.2f} (CUPED lever)")
    ax = axes[1]
    labels = ["Naive", "CUPED"]
    ses = [r["naive"]["se"], r["cuped"]["se"]]
    bars = ax.bar(labels, ses, color=[RED, GREEN], alpha=0.85)
    for b, v in zip(bars, ses):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("standard error of effect")
    ax.set_title(f"CUPED cuts SE by {r['se_reduction']:.0%}")
    fig.savefig(os.path.join(FIGDIR, "04_cuped.png")); plt.close(fig)
    return r


def fig_mmm(df):
    r = mmm.fit_mmm(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    x = np.linspace(0, r["spend_grid"].max()*1.5/(1-r["decay"]), 100)
    ax.plot(x, mmm.hill(x, r["half_sat"], r["shape"]), color=BLUE)
    ax.axvline(r["half_sat"], color=GREEN, ls=":", label=f"half-sat={r['half_sat']:.0f}")
    ax.set_title("Hill saturation curve"); ax.set_xlabel("adstocked spend")
    ax.set_ylabel("saturation (0-1)"); ax.legend()
    ax = axes[1]
    ax.plot(r["spend_grid"], r["response"], color=ORANGE)
    ax.set_title(f"Media response curve (R²={r['r2']:.2f})")
    ax.set_xlabel("daily spend ($)"); ax.set_ylabel("modeled incremental conv/day")
    fig.savefig(os.path.join(FIGDIR, "05_mmm_response.png")); plt.close(fig)
    return r


def fig_power(df):
    var = power.variance_from_data(df)
    n_c = int((df.groupby("geo")["treated"].first() == 0).sum())
    n_t_grid = np.arange(4, 60, 2)
    mdes = power.mde_curve(var, n_c, n_t_grid)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.plot(n_t_grid, [m*100 for m in mdes], color=BLUE)
    ax.axhline(12, color=GREEN, ls=":", label="+12% target lift")
    ax.set_title("MDE vs number of treated geos (@80% power)")
    ax.set_xlabel("# treated geos"); ax.set_ylabel("MDE (%)"); ax.legend()
    ax = axes[1]
    lifts = np.linspace(0.02, 0.5, 60)
    n_t = int(df.groupby("geo")["treated"].first().sum())
    powers = [power.power_for_lift(l, var, n_t, n_c) for l in lifts]
    ax.plot([l*100 for l in lifts], [p*100 for p in powers], color=ORANGE)
    ax.axhline(80, color=GREEN, ls=":", label="80% power")
    ax.set_title(f"Power curve ({n_t} treated / {n_c} control)")
    ax.set_xlabel("true lift (%)"); ax.set_ylabel("power (%)"); ax.legend()
    fig.savefig(os.path.join(FIGDIR, "06_power_mde.png")); plt.close(fig)
    return {"var": var, "mde_80": power.mde(var, n_t, n_c)}


def main():
    df, gt = load()
    print("Generating figures...")
    r_did, naive = fig_did(df, gt)
    r_sc = fig_sc(df, gt)
    r_cuped = fig_cuped(df)
    r_mmm = fig_mmm(df)
    r_pow = fig_power(df)

    summary = {
        "true_lift": gt["true_lift"],
        "did": {"lift": r_did["lift"], "ci": r_did["lift_ci"], "p": r_did["pvalue"],
                "naive_biased": naive},
        "synthetic_control": {"lift": r_sc["lift"], "pre_rmse": r_sc["pre_rmse"]},
        "cuped": {"naive_se": r_cuped["naive"]["se"], "cuped_se": r_cuped["cuped"]["se"],
                  "se_reduction": r_cuped["se_reduction"], "corr": r_cuped["corr_xy"]},
        "mmm": {"r2": r_mmm["r2"], "half_sat": r_mmm["half_sat"], "decay": r_mmm["decay"]},
        "power": {"mde_80": r_pow["mde_80"], "log_var": r_pow["var"]},
    }
    with open(os.path.join("reports", "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    figs = sorted(os.listdir(FIGDIR))
    print("Figures:", figs)
    print("Wrote reports/results_summary.json")


if __name__ == "__main__":
    main()
