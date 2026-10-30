"""Truth-vs-estimate table, results JSON (common.results schema) and the exec readout markdown."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common.results import write_results
from ..config import REPORTS_DIR
from .settings import ARM_LABELS, ARMS, MODULE, wk

MLAB = {"qs": "quote starts", "rated": "rated quotes", "sales": "sales"}


def _pct(v, d=1):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}%}"


def _usd(v):
    return "n/a" if v is None or not np.isfinite(v) else f"${v:,.0f}"


def _num(v, d=4):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def truth_table(o) -> pd.DataFrame:
    """One row per headline quantity: estimate, CI, truth, error (formatted + raw)."""
    rows = []

    def add(q, est, lo, hi, truth, kind="pct"):
        f = {"pct": _pct, "usd": _usd, "num": _num}[kind]
        err = est - truth if (est is not None and truth is not None and np.isfinite(est) and np.isfinite(truth)) else np.nan
        rows.append({"quantity": q, "estimate": est, "ci_low": lo, "ci_high": hi, "truth": truth, "error": err,
                     "kind": kind, "estimate_fmt": f(est), "ci_fmt": f"[{f(lo)}, {f(hi)}]" if lo is not None else "",
                     "truth_fmt": f(truth), "error_fmt": (f"{err:+.1%}" if kind == "pct" else
                                                          (f"{err:+,.0f}" if kind == "usd" else f"{err:+.4f}"))
                     if np.isfinite(err) else "n/a"})

    for _, r in o["triangulated"].iterrows():
        add(f"Lift, {ARM_LABELS[r['arm']]}, {MLAB[r['metric']]} (triangulated)", r["estimate"], r["ci_low"],
            r["ci_high"], r["truth"])
    rec = o["recapture"]
    for arm in ARMS:
        for st in ("qs", "rated"):
            r = rec[(rec["arm"] == arm) & (rec["stage"] == st) & (rec["ref"] == "synthetic")].iloc[0]
            add(f"Organic recapture, {ARM_LABELS[arm]}, {MLAB[st]}", r["recapture"], r["ci_low"], r["ci_high"], r["truth"])
    e = o["economics"]["table"]
    for arm in ARMS:
        for m in ("rated", "sales"):
            r = e[(e["arm"] == arm) & (e["metric"] == m)].iloc[0]
            add(f"Incremental cost per {MLAB[m][:-1]}, {ARM_LABELS[arm]}", r["incr_cost_per_est"],
                r["incr_cost_per_ci_low"], r["incr_cost_per_ci_high"], r["incr_cost_per_true"], "usd")
    for ch, d in o["economics"]["incremental_per_dollar"].items():
        for m in ("rated", "sales"):
            z = 1.645
            add(f"Incremental {MLAB[m]} per $1k, {ch}", 1000 * d[m], 1000 * (d[m] - z * d[f"{m}_se"]),
                1000 * (d[m] + z * d[f"{m}_se"]), 1000 * d[f"{m}_truth"], "num")
    for _, r in o["guardrail"].iterrows():
        add(f"Site-visit loss, {ARM_LABELS[r['arm']]}", r["estimate"], r["ci_low"], r["ci_high"], r["truth"])
    t = pd.DataFrame(rows)
    for c in ("truth_fmt",):
        pass
    t.loc[t["kind"] == "num", "truth_fmt"] = t.loc[t["kind"] == "num", "truth"].map(lambda v: f"{v:.2f}")
    t.loc[t["kind"] == "num", "estimate_fmt"] = t.loc[t["kind"] == "num", "estimate"].map(lambda v: f"{v:.2f}")
    t.loc[t["kind"] == "num", "ci_fmt"] = t.loc[t["kind"] == "num"].apply(
        lambda r: f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]", axis=1)
    t.loc[t["kind"] == "num", "error_fmt"] = t.loc[t["kind"] == "num", "error"].map(lambda v: f"{v:+.2f}")
    return t


def _estimates(o) -> list:
    est = pd.concat([o["estimates"], o["triangulated"]], ignore_index=True)
    keep = ["test", "arm", "metric", "method", "estimate", "ci_low", "ci_high", "ci_level", "p_value", "significant",
            "truth", "truth_key", "truth_realized", "units", "notes", "se", "p_cluster", "error_vs_truth", "truth_in_ci"]
    out = [{k: r.get(k) for k in keep if k in r} for r in est.to_dict(orient="records")]
    for _, r in o["guardrail"].iterrows():
        out.append({"test": "search", "arm": r["arm"], "metric": "visits", "method": "synth_control_guardrail",
                    "estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "ci_level": 0.9,
                    "p_value": r["p_value"], "significant": bool(r["p_value"] <= 0.10), "truth": r["truth"],
                    "truth_key": "bau_twin.visits (counterfactual-consistent)",
                    "units": "relative site-visit loss (positive = paid search drives traffic)",
                    "notes": f"guardrail threshold {r['threshold']:.0%}; breach={r['breach']}"})
    return out


def headline(o) -> dict:
    tri = o["triangulated"].set_index(["arm", "metric"])
    rec = o["recapture"]
    e = o["economics"]["table"].set_index(["arm", "metric"])
    ipd = o["economics"]["incremental_per_dollar"]
    d = o["design"]
    h = {"all_search_counties": " + ".join(d["all_search"]), "branded_only_counties": " + ".join(d["branded_only"]),
         "control_county": d["control"][0], "n_bau_donors": len(d["bau_donors"])}
    for arm in ARMS:
        for m in ("qs", "rated", "sales"):
            r = tri.loc[(arm, m)]
            h[f"{arm}_{m}_lift"] = r["estimate"]
            h[f"{arm}_{m}_lift_ci"] = [r["ci_low"], r["ci_high"]]
            h[f"{arm}_{m}_true_lift"] = r["truth"]
        rr = rec[(rec["arm"] == arm) & (rec["stage"] == "rated") & (rec["ref"] == "synthetic")].iloc[0]
        h[f"{arm}_recapture_rated"] = rr["recapture"]
        h[f"{arm}_recapture_rated_ci"] = [rr["ci_low"], rr["ci_high"]]
        h[f"{arm}_true_recapture_rated"] = rr["truth"]
        h[f"{arm}_incr_cost_per_rated"] = e.loc[(arm, "rated"), "incr_cost_per_est"]
        h[f"{arm}_true_incr_cost_per_rated"] = e.loc[(arm, "rated"), "incr_cost_per_true"]
        h[f"{arm}_last_touch_cost_per_rated"] = e.loc[(arm, "rated"), "last_touch_cost_per"]
        h[f"{arm}_incr_cost_per_sale"] = e.loc[(arm, "sales"), "incr_cost_per_est"]
        h[f"{arm}_true_incr_cost_per_sale"] = e.loc[(arm, "sales"), "incr_cost_per_true"]
    for ch in ipd:
        h[f"{ch}_incr_rated_per_1k"] = 1000 * ipd[ch]["rated"]
        h[f"{ch}_true_incr_rated_per_1k"] = 1000 * ipd[ch]["rated_truth"]
    p = o["power"]["table"].set_index(["arm", "metric"])
    for arm in ARMS:
        h[f"{arm}_mde_rated"] = p.loc[(arm, "rated"), "mde_closed_form"]
        h[f"{arm}_mde_sales"] = p.loc[(arm, "sales"), "mde_closed_form"]
    g = o["guardrail"].set_index("arm")
    for arm in ARMS:
        h[f"{arm}_visit_loss"] = g.loc[arm, "estimate"]
        h[f"{arm}_true_visit_loss"] = g.loc[arm, "truth"]
    tt = o["truth_table"]
    h["n_truth_rows"] = len(tt)
    h["share_truth_in_ci"] = float(((tt["truth"] >= tt["ci_low"]) & (tt["truth"] <= tt["ci_high"])).mean())
    h["recommendation"] = o["recommendation"]["lines"][2] if len(o["recommendation"]["lines"]) > 2 else ""
    return h


def write_outputs(o) -> dict:
    s = o["settings"]
    des = o["design_stage"]
    mt = des["mde_table"].drop(columns=["weights"]).copy()
    mt["counties"] = mt["counties"].map(lambda v: list(v))
    design = {"treated_units": {a: o["design"][a] for a in ARMS}, "control_units": o["design"]["control"],
              "bau_donors": o["design"]["bau_donors"], "design_metric": f"{s.design_metric} per 10k residents",
              "windows": {"sc_fit": [s.sc_fit_start, s.pre[1]], "readout_pre": list(s.pre), "test": list(s.post),
                          "design_end": s.design_end},
              "weights": {a: o["rich"][(a, "rated", "synth_control")]["weights"] for a in ARMS},
              "candidate_mde_table": mt, "arm_pair_ranking": des["pair_table"].head(10),
              "control_ranking": des["control_table"], "balance": des["balance"].reset_index(),
              "selection_rule": ("max expected power at planning priors (rated + sales) with injected-lift "
                                 "recovery gate/penalty; control = most stable change ratio vs both arms")}
    validation = {"injected_lift_recovery": mt[["unit", "label"] + [c for c in mt.columns if c.startswith("detected_")
                                                                  or c.startswith("abs_err_")]],
                  "placebo": {a: {"p_value": o["rich"][(a, "rated", "synth_control")]["p_value"],
                                  "min_attainable_p": o["rich"][(a, "rated", "synth_control")]["min_attainable_p"],
                                  "placebos": o["rich"][(a, "rated", "synth_control")]["placebo_space"]} for a in ARMS},
                  "power": {"table": o["power"]["table"], "simulation": o["power"]["simulation"]},
                  "pre_period_unchanged": True}
    extra = {"incremental_per_dollar": o["economics"]["incremental_per_dollar"],
             "truth_table": o["truth_table"][["quantity", "estimate", "ci_low", "ci_high", "truth", "error"]],
             "recapture": o["recapture"], "economics": o["economics"]["table"], "guardrail": o["guardrail"],
             "source_effects_rated": {a: o["source_effects"][a].reset_index() for a in ARMS},
             "recommendation": o["recommendation"]["lines"],
             "fixes_vs_original": [
                 "one design metric (rated quotes per capita) for synth fit, injected-lift validation and MDE",
                 "lift always (cf - actual)/cf, positive = contribution; no inverted ratios",
                 "BSTS significance from the interval only; no text parsing",
                 "OLS DiD with unit + week FE on all donors; cluster SE reported but inference by permutation "
                 "(one treated cluster)",
                 "p-values never 0: permutation p >= 1/(J+1), reported with min attainable p",
                 f"donor pool = {len(o['design']['bau_donors']) + 1} counties (vs 3 in the original)"]}
    truth = {"scenario": o["truth"], "nonbrand_only_reference": o["nonbrand_truth"],
             "incremental_per_dollar_truth": o["ipd_truth"]}
    p = write_results(MODULE, headline(o), _estimates(o), design=design, validation=validation, truth=truth,
                      figures=o.get("figures", []), extra=extra)
    md = write_readout(o)
    return {"results": str(p), "readout": str(md)}


def write_readout(o) -> Path:
    s, d = o["settings"], o["design"]
    names = o["panel"].groupby("county_id")["county"].first()
    nm = lambda ids: " + ".join(names[i] for i in ids)  # noqa: E731
    tri = o["triangulated"].set_index(["arm", "metric"])
    est = o["estimates"]
    e = o["economics"]["table"].set_index(["arm", "metric"])
    rec = o["recapture"]
    pw = o["power"]["table"].set_index(["arm", "metric"])
    g = o["guardrail"].set_index("arm")
    ipd = o["economics"]["incremental_per_dollar"]

    def line(arm, m):
        r = tri.loc[(arm, m)]
        ms = est[(est["arm"] == arm) & (est["metric"] == m)]
        nsig = int(ms["significant"].sum())
        return (f"| {ARM_LABELS[arm]} | {MLAB[m]} | **{_pct(r['estimate'])}** | [{_pct(r['ci_low'])}, {_pct(r['ci_high'])}] "
                f"| {nsig}/4 | {_pct(r['truth'])} | {_pct(pw.loc[(arm, m), 'mde_closed_form'], 0)} |")

    def recl(arm, st):
        r = rec[(rec["arm"] == arm) & (rec["stage"] == st) & (rec["ref"] == "synthetic")].iloc[0]
        return (f"| {ARM_LABELS[arm]} | {MLAB[st]} | {r['delta_paid_per_wk']:+.1f} | {r['delta_organic_per_wk']:+.1f} | "
                f"**{_pct(r['recapture'], 0)}** [{_pct(r['ci_low'], 0)}, {_pct(r['ci_high'], 0)}] | {_pct(r['truth'], 0)} |")

    def costl(arm, m):
        r = e.loc[(arm, m)]
        return (f"| {ARM_LABELS[arm]} | {MLAB[m][:-1]} | {_usd(r['last_touch_cost_per'])} | **{_usd(r['incr_cost_per_est'])}** "
                f"[{_usd(r['incr_cost_per_ci_low'])}, {_usd(r['incr_cost_per_ci_high'])}] | {_usd(r['incr_cost_per_true'])} |")

    tt = o["truth_table"]
    cover = float(((tt["truth"] >= tt["ci_low"]) & (tt["truth"] <= tt["ci_high"])).mean())
    txt = f"""# Paid Search geo dark-out — executive readout (synthetic)

*Module 2 of incrementality-lab. Synthetic data with known ground truth; magnitudes are illustrative.
All intervals are two-sided 90%. Lift = share of volume lost while paid search was dark (positive = paid search contribution).*

## Question
How much of paid search is truly incremental, and how much branded search merely buys conversions that organic
search would have delivered anyway (cannibalisation)?

## Design (selected by the pipeline from {len(o['design_stage']['mde_table'])} candidate units, data through week {s.design_end})
| Arm | Counties | Paused |
|---|---|---|
| All paid search off | {nm(d['all_search'])} | branded + non-branded |
| Branded search off | {nm(d['branded_only'])} | branded only |
| Control | {nm(d['control'])} | nothing (ratio-DiD comparator) |
| BAU donors | {len(d['bau_donors'])} counties | nothing (synthetic control / BSTS / FE-DiD pool) |

Weeks {s.post[0]}–{s.post[1]} dark ({s.n_post} wk); readout pre-period weeks {s.pre[0]}–{s.pre[1]}; synthetic control fit on weeks
{s.sc_fit_start}–{s.pre[1]}. Arms were chosen to maximise expected power (rated quotes + sales) under a recovery gate
on +5/10/15% injected lifts measured on the **same** metric used for fitting.

## Results — triangulated lift vs truth
| Arm | Metric | Lift | 90% CI | Methods significant | Truth | MDE (80% power) |
|---|---|---|---|---|---|---|
{chr(10).join(line(a, m) for a in ARMS for m in s.readout_metrics)}

Methods: ratio DiD vs control, OLS DiD with unit + week fixed effects (inference by donor permutation — cluster-robust
SE with one treated cluster is anti-conservative), BSTS counterfactual on all donors (significance only when its interval
excludes 0), synthetic control with in-space placebo p (never below 1/(J+1)).

## Cannibalisation — organic recapture
| Arm | Stage | Δ paid LT / wk | Δ organic LT / wk | Recapture [90% CI] | Truth |
|---|---|---|---|---|---|
{recl('branded_only', 'qs')}
{recl('branded_only', 'rated')}
{recl('all_search', 'qs')}
{recl('all_search', 'rated')}

## Efficiency — incremental vs last-touch
| Arm | Per | Last-touch | Incremental (est.) [90% CI] | Incremental (truth) |
|---|---|---|---|---|
{costl('all_search', 'rated')}
{costl('branded_only', 'rated')}
{costl('all_search', 'sales')}
{costl('branded_only', 'sales')}

MMM calibration (`extra.incremental_per_dollar`): branded **{1000 * ipd['search_brand']['rated']:.1f}** incremental rated quotes per $1k
(SE {1000 * ipd['search_brand']['rated_se']:.1f}; truth {1000 * ipd['search_brand']['rated_truth']:.1f}); non-branded
**{1000 * ipd['search_nonbrand']['rated']:.1f}** (SE {1000 * ipd['search_nonbrand']['rated_se']:.1f}; truth {1000 * ipd['search_nonbrand']['rated_truth']:.1f}).

Traffic guardrail (≤{s.traffic_guardrail_max_loss:.0%} visit loss): all-search {_pct(g.loc['all_search', 'estimate'])} (truth {_pct(g.loc['all_search', 'truth'])}),
branded-only {_pct(g.loc['branded_only', 'estimate'])} (truth {_pct(g.loc['branded_only', 'truth'])}).

## Recommendation
{chr(10).join('- ' + l for l in o['recommendation']['lines'])}

## Caveats
- Synthetic world; true effects come from the generator's adstock + Hill response, not from this analysis.
- One treated unit per arm: every p-value is rank-based and bounded below by 1/(donors+1).
- Six weeks of binds cannot resolve sales lifts near 10% — sales reads are directional; rated quotes are the decision KPI.
- {cover:.0%} of the {len(tt)} headline quantities have the truth inside their 90% interval (see `reports/figures/search/14_truth_table.png`).
"""
    p = REPORTS_DIR / "readouts" / f"{MODULE}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)
    return p
