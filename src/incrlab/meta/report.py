"""Write reports/results/meta.json and reports/readouts/meta.md from a completed MetaState."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..common.results import write_results
from ..config import REPORTS_DIR, REPO_ROOT
from .pipeline import MetaState, estimates_table, incremental_per_dollar
from .readout import STAGE_LABEL, STAGES

TRUTH_KEY = "scenario.meta.holdout.{s}.true_lift_pct"
UNITS = "relative lift (positive = channel contribution)"


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def build_estimates(S: MetaState) -> list:
    out = []
    for m, by in S.estimates.items():
        for s, e in by.items():
            ci_lvl = 1 - S.st.did_alpha if e.get("ci_low") is not None else None
            p = e.get("p_value")
            note = {"did_week_fe_hc1": f"stacked treated vs locked synthetic, weeks {S.st.readout_pre_start}-"
                                       f"{S.st.test_end}; HC1 SE (HAC p={e.get('hac_p_value', np.nan):.2f})",
                    "ratio_did": "CF = treated_pre x synth_post/synth_pre; week bootstrap CI",
                    "synthetic_control_gap": "locked weights, level-adjusted on 58-week design history; "
                                             "p = in-space placebo (one-sided)"}[m]
            out.append({"test": "meta", "arm": "holdout", "metric": s, "method": m, "estimate": e["estimate"],
                        "ci_low": e.get("ci_low"), "ci_high": e.get("ci_high"), "ci_level": ci_lvl,
                        "se": e.get("se_rel"), "p_value": p,
                        "significant": bool(p is not None and p < S.st.directional_p),
                        "incremental": e["incremental"], "truth": S.truth[s]["true_lift_pct"],
                        "truth_realized": S.truth[s]["realized_lift_pct"], "truth_key": TRUTH_KEY.format(s=s),
                        "units": UNITS, "notes": note})
    return out


def headline(S: MetaState) -> dict:
    e = S.estimates["did_week_fe_hc1"]
    ipd = incremental_per_dollar(S)["social"]
    h = {"treated_unit": S.unit, "treated_unit_name": " + ".join(
        S.county_static.set_index("county_id").loc[S.unit.split("+"), "county"]),
        "treated_pop": S.truth["pop"], "n_candidates": int(len(S.ranked)),
        "n_candidates_passing": int(S.ranked["passes"].sum()), "n_donors": int((S.dfit.weights >= 0.01).sum()),
        "test_weeks": S.st.n_test_weeks, "verdict": S.verdict["label"],
        "design_mde_8wk": S.verdict["design_mde_8wk"], "weeks_to_80pct_power_at_true_lift": None,
        "placebo_p_qs": S.placebo["qs"]["p_value_lift_one_sided"],
        "spend_removed": S.spend["spend_removed"], "delivered_share": S.spend["delivered_share"]}
    for s in STAGES:
        h[f"lift_{s}"] = e[s]["estimate"]
        h[f"lift_{s}_ci_low"] = e[s]["ci_low"]
        h[f"lift_{s}_ci_high"] = e[s]["ci_high"]
        h[f"p_{s}"] = e[s]["p_value"]
        h[f"true_lift_{s}"] = S.truth[s]["true_lift_pct"]
        h[f"incremental_{s}"] = e[s]["incremental"]
        h[f"true_incremental_{s}"] = S.truth[s]["true_incremental"]
        a = S.attribution.set_index("stage").loc[s]
        h[f"incremental_cost_per_{s}"] = a["incremental_cost_per"]
        h[f"true_incremental_cost_per_{s}"] = S.truth.get(f"true_incremental_cost_per_{s}")
        h[f"last_touch_cost_per_{s}"] = a["lt_cost_per"]
        h[f"lt_overclaim_ratio_{s}"] = a["overclaim_ratio"]
    h["incremental_sales_per_1k"] = 1000 * ipd["sales"]
    from . import design as D
    tl = S.truth["qs"]["true_lift_pct"]
    h["weeks_to_80pct_power_at_true_lift"] = D.cpower.weeks_for_power(
        tl, S.noise["sigma"], n_pre=len(S.st.readout_pre_weeks), alpha=S.st.alpha, power=S.st.power,
        rho=S.noise["rho"], max_weeks=104)
    return h


def write_results_json(S: MetaState, results_dir: Optional[Path] = None) -> Path:
    sel = S.ranked[S.ranked["unit"] == S.unit].iloc[0]
    design = {"treated_units": S.unit.split("+"), "treated_unit": S.unit, "control_units": list(S.dfit.weights.index),
              "weights": S.dfit.weights.round(4).to_dict(), "selection_reason": S.selection["reason"],
              "candidate_band_pop": list(S.candidates.attrs["band"]), "target_pop": S.candidates.attrs["target_pop"],
              "train_window": [S.st.design_start, S.st.train_end],
              "pseudo_post_window": [S.st.train_end + 1, S.st.design_end],
              "readout_pre_window": [S.st.readout_pre_start, S.st.test_start - 1],
              "test_window": [S.st.test_start, S.st.test_end], "design_metric": S.st.design_metric,
              "selected_scores": sel.drop(labels=["weights"]).to_dict(),
              "ranked_candidates": S.ranked.drop(columns=["weights"]).to_dict(orient="records"),
              "kmeans": {"k": S.kmeans["k"], "labels": S.kmeans["labels"].to_dict(),
                         "grid": S.kmeans["grid"].to_dict(orient="records")}}
    pys = {k: v for k, v in S.pysyncon.items() if k != "weights"}
    if "weights" in S.pysyncon:
        pys["weights"] = S.pysyncon["weights"].round(4).to_dict(orient="index")
    validation = {
        "injected_lift_recovery": S.validation.to_dict(orient="records"),
        "power": {"sigma_gap": S.noise["sigma"], "rho": S.noise["rho"], "alpha_one_sided": S.st.alpha,
                  "target_power": S.st.power, "weeks_to_80": S.weeks_to_power.to_dict(orient="records"),
                  "mde_8wk": S.verdict["design_mde_8wk"],
                  "simulation_check": {str(k): v for k, v in S.power_sim.items()}},
        "jackknife": S.jackknife.to_dict(orient="records"),
        "placebo": {s: {k: v for k, v in S.placebo[s].items() if k != "placebos"} for s in STAGES},
        "pysyncon_agreement": pys,
        "truth_vs_estimate": estimates_table(S).to_dict(orient="records"),
    }
    truth = {k: v for k, v in S.truth.items() if k != "units"}
    truth["n_zip_units"] = len(S.truth.get("units", []))
    extra = {"incremental_per_dollar": incremental_per_dollar(S), "verdict": S.verdict,
             "attribution": S.attribution.to_dict(orient="records"),
             "efficiency_vs_blended": S.efficiency.to_dict(orient="records"),
             "spend": S.spend}
    return write_results("meta", headline(S), build_estimates(S), design=design, validation=validation,
                         truth=truth, figures=[_rel(f) for f in S.figures], extra=extra, results_dir=results_dir)


def _pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}%}"


def write_readout_md(S: MetaState, path: Optional[Path] = None) -> Path:
    h = headline(S)
    e = S.estimates["did_week_fe_hc1"]
    st = S.st
    a = S.attribution.set_index("stage")
    name = h["treated_unit_name"]
    rows = "\n".join(
        f"| {STAGE_LABEL[s]} | {_pct(e[s]['estimate'])} | {_pct(e[s]['ci_low'])} to {_pct(e[s]['ci_high'])} | "
        f"{e[s]['p_value']:.2f} | {_pct(S.estimates['ratio_did'][s]['estimate'])} | "
        f"{_pct(S.estimates['synthetic_control_gap'][s]['estimate'])} | {_pct(S.truth[s]['true_lift_pct'])} | "
        f"{e[s]['incremental']:,.0f} (truth {S.truth[s]['true_incremental']:,.0f}) |" for s in STAGES)
    cost = "\n".join(
        f"| {STAGE_LABEL[s]} | ${a.loc[s, 'lt_cost_per']:,.0f} | "
        f"{'$' + format(a.loc[s, 'incremental_cost_per'], ',.0f') if np.isfinite(a.loc[s, 'incremental_cost_per']) else 'n/a'} | "
        f"${S.truth.get(f'true_incremental_cost_per_{s}') or float('nan'):,.0f} | "
        f"{a.loc[s, 'lt_share']:.1%} | {a.loc[s, 'overclaim_ratio']:.1f}x |" for s in STAGES)
    v = S.verdict
    under = v["underpowered"]
    wk = h["weeks_to_80pct_power_at_true_lift"]
    md = f"""# Social (Meta) geo holdout — readout

*Synthetic data; illustrative magnitudes. Generated by `python -m incrlab.meta`; every number below is in
`reports/results/meta.json`.*

**Verdict: {v['label'].upper()}.** Pausing Social in {name} cost an estimated **{_pct(e['qs']['estimate'])} of quote
starts, {_pct(e['rated']['estimate'])} of rated quotes and {_pct(e['sales']['estimate'])} of sales** over {st.n_test_weeks}
weeks (ground truth {_pct(S.truth['qs']['true_lift_pct'])} / {_pct(S.truth['rated']['true_lift_pct'])} /
{_pct(S.truth['sales']['true_lift_pct'])}). The direction is right, but no estimate is statistically distinguishable from zero
(QS p={e['qs']['p_value']:.2f}; placebo p={S.placebo['qs']['p_value_lift_one_sided']:.2f}).

## Question
If Social is switched off, how much quote, rated-quote and sales volume is actually lost — and how does that compare
with what last-touch attribution credits Social with?

## Design
- **Market:** {name} (pop {h['treated_pop']/1e3:,.0f}k), chosen by the pipeline from {h['n_candidates']} candidate
  markets (single counties and contiguous 2–3 county bundles near {S.candidates.attrs['target_pop']/1e3:,.0f}k people);
  {h['n_candidates_passing']} passed all hard filters (fit, max donor weight ≤ {st.max_weight:.0%}, effective donors,
  donor count, injected-lift recovery); the top scorer also passed the leave-one-donor-out jackknife.
- **Control:** synthetic control of {h['n_donors']} Eastmoor counties, weights fit on weeks {st.design_start}–{st.train_end}
  of the Social share of quote starts and **locked**; validated out-of-sample by injecting 5/10/15% losses into weeks
  {st.train_end+1}–{st.design_end} (recovered {', '.join(_pct(x) for x in S.validation[S.validation.unit == S.unit]['detected'])}).
- **Test:** Social paused weeks {st.test_start}–{st.test_end} ({st.n_test_weeks} weeks); {S.spend['delivered_share']:.0%} of
  BAU spend kept delivering (incomplete holdout), {_pct(1 - S.spend['delivered_share'], 0)} removed (${S.spend['spend_removed']:,.0f}).
- **Primary estimator:** DiD of treated vs locked synthetic with week fixed effects and HC1 robust SE (HAC as a
  robustness check); cross-checked by a ratio DiD and the synthetic-control gap with an in-space placebo p-value.

## Result (contribution = volume lost without Social)
| Stage | DiD (week FE) | 90% CI | p | Ratio DiD | SC gap | Truth | Incremental, 8 wk |
|---|---|---|---|---|---|---|---|
{rows}

## Efficiency & attribution
| Stage | Last-touch cost-per | Incremental cost-per (measured) | Incremental cost-per (truth) | LT share of volume | LT claim vs incremental |
|---|---|---|---|---|---|
{cost}

Last-touch **over-credits** Social on quote starts and rated quotes and is closer to (or below) incrementality on
sales — Social influences buyers who later close through other channels.

## Confidence
- Design MDE for {st.n_test_weeks} weeks (80% power, one-sided α={st.alpha}) is **{_pct(v['design_mde_8wk'])}**; the true
  effect is ~{_pct(S.truth['qs']['true_lift_pct'], 0)}, so the test was **{'underpowered' if under else 'adequately powered'}**
  {f'(≈{wk} weeks would be needed at this effect size)' if wk else ''}. With weekly sales of ~{S.ts.loc[st.test_weeks, 'sales_actual'].mean():.0f}
  in the market, sales-level conclusions are especially noisy.
- Leakage ({S.spend['delivered_share']:.0%} of spend still delivered) makes the measured loss a **lower bound** of a
  full pause; per-dollar figures use spend actually removed.

## Recommendation
1. Treat the result as **directional evidence that Social is incremental** — keep spend; do not cut on this read.
2. Use the per-dollar estimate (with its SE) only as a weak prior in the MMM; it is in `extra.incremental_per_dollar`.
3. Re-test with ≥{wk or '12+'} weeks or a multi-market design (2–3 matched holdout markets) and tighter delivery controls
   to get a decision-grade read; optimise to incremental cost per sale, not platform-reported cost per quote.
"""
    p = Path(path or REPORTS_DIR / "readouts" / "meta.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md)
    return p
