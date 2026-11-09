"""End-to-end DSP program: design A + design B -> holdout realized on the selected zips ->
triangulated readout -> figures, ``reports/results/dsp.json`` and ``reports/readouts/dsp.md``."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..config import REPORTS_DIR, get_config
from ..common.results import write_results
from ..simulate import get_world, load_table
from .design_a import DesignA, run_design_a
from .design_b import DesignB, run_design_b
from .params import STAGE_LABELS, DSPParams, default_params
from .readout import METHOD_LABELS, Readout, run_readout, truth_table


@dataclass
class DSPRun:
    cfg: object
    params: DSPParams
    A: DesignA
    B: DesignB
    R: Readout
    figures: List[Path] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    results_path: Optional[Path] = None
    readout_path: Optional[Path] = None


def run(figures: bool = True, cfg=None, params: Optional[DSPParams] = None, write: bool = True) -> DSPRun:
    t0 = time.time()
    cfg = cfg or get_config()
    P = params or default_params(cfg)
    world = get_world(cfg)
    if cfg.small or cfg.seed != get_config().seed or cfg.effect_multiplier != 1.0:
        from ..simulate import simulate
        panels = simulate(cfg, world)
        zw, zs = panels["zip_weekly"], world.geo.zips
    else:
        zw, zs = load_table("zip_weekly"), load_table("zip_static")
    tm = {"load": time.time() - t0}
    A = run_design_a(zw, zs, cfg, P.a)
    tm["design_a"] = time.time() - t0
    B = run_design_b(zw, zs, cfg, P.b)
    tm["design_b"] = time.time() - t0
    R = run_readout(world, A, B, P, cfg)
    tm["readout"] = time.time() - t0
    out = DSPRun(cfg, P, A, B, R, timings=tm)
    if figures:
        from .figures import make_all
        out.figures = make_all(A, B, R, P, cfg, zs)
        tm["figures"] = time.time() - t0
    if write:
        out.results_path = write_dsp_results(out)
        out.readout_path = write_readout_md(out)
    return out


# ======================================================================================
# results JSON
# ======================================================================================
def incremental_per_dollar(run_: DSPRun) -> dict:
    R, cfg = run_.R, run_.cfg
    out = {}
    for v, test in (("A", "dsp_a"), ("B", "dsp_b")):
        b = R.business[v]
        td = cfg.tests[test]
        tr = R.truth[test]["holdout"]
        # Pre-registered SE rule for calibration: SE = max(bootstrap SE, design SE), where the design SE is the
        # pre-period power analysis' standard error of the sales lift (MDE / (z_alpha + z_power)) translated to
        # sales per $. A lucky-looking narrow bootstrap interval from an under-powered design must not
        # dominate the MMM likelihood.
        P_ = run_.params.a if v == "A" else run_.params.b
        mde_s = (run_.A.sizing_decision.get("mde_selected") if v == "A"
                 else run_.B.power["by_metric"]["sales"]["mde"])
        zsum = float(stats.norm.ppf(1 - P_.alpha) + stats.norm.ppf(P_.power))
        cf_sales = R.results[v]["sales"]["change_ratio_did"]["cf_per_week"] * (td.test_end - td.test_start + 1)
        se_design = float(mde_s / zsum * cf_sales / b["spend_saved_est"])
        se_boot = float(b["incremental_per_dollar_sales_se"])
        out[test] = {"test": test, "arm": "holdout", "channels": [test], "window": [td.test_start, td.test_end],
                     "leakage": td.leakage, "method": "change_ratio_did; SE = max(week-bootstrap SE, design SE)",
                     "spend": b["spend_saved_est"], "incremental_sales": b["incremental_sales"],
                     "sales": b["incremental_per_dollar_sales"], "se": max(se_boot, se_design),
                     "se_bootstrap": se_boot, "se_design": se_design, "design_mde_sales": mde_s,
                     "powered_for_sales": R.decisions[v]["powered_for_sales"],
                     "qs": b["incremental_per_dollar_qs"], "qs_se": b["incremental_per_dollar_qs_se"],
                     "rated": b["incremental_per_dollar_rated"], "rated_se": b["incremental_per_dollar_rated_se"],
                     "truth_sales_per_dollar": tr["sales"]["true_incremental"] / tr["spend_removed"],
                     "n_treated_units": len(R.series[v].treated)}
    return out


def headline(run_: DSPRun) -> dict:
    A, B, R = run_.A, run_.B, run_.R
    h = {"a_treatment_patch": A.winner["patch_id"], "a_control_anchor_county": A.winner["c_county"],
         "a_n_treated_zips": len(A.treated), "a_n_control_zips": len(A.control),
         "a_treated_pop": float(A.winner["t_pop"]), "a_control_pop": float(A.winner["c_pop"]),
         "a_mahalanobis": float(A.winner["maha"]), "a_wow_corr": float(A.winner["wow_corr"]),
         "a_gate_tier": A.winner["tier"], "a_n_candidates": int(len(A.candidates)),
         "a_n_pass_strict": int(A.candidates["pass_strict"].sum()),
         "a_n_pass_relaxed_only": int((A.candidates["tier"] == "relaxed").sum()),
         "a_buffer_miles": A.params.buffer_miles, "a_n_buffer_zips": len(A.buffer_zips),
         "a_test_weeks": int(A.power["test_weeks"]), "b_test_weeks": int(B.power["test_weeks"]),
         "a_mde_qs_design": A.power["mde_at_test_length"]["qs"],
         "a_mde_sales_design": A.power["mde_at_test_length"]["sales"],
         "a_sizing_verdict": A.sizing_decision.get("verdict"), "a_sizing_pop_target": A.sizing_decision.get("pop_target"),
         "a_sizing_rationale": A.sizing_decision.get("rationale"), "target_mde_sales": A.params.target_mde,
         "b_k_used": int(B.kmeans["k"]), "b_k_best_silhouette": int(B.kmeans["k_best_silhouette"]),
         "b_treated_cluster": int(B.treated_cluster), "b_n_treated_zips": len(B.treated),
         "b_selection": B.selection_note, "b_sc_rel_rmse": float(B.sc.rel_rmse),
         "b_n_active_donors": int(B.sc.n_donors_used), "b_injected_recovered": float(B.injected_final["detected"]),
         "b_switch_shift_pct": float(B.switch.get("shift_pct", np.nan)), "b_switch_p": float(B.switch["p_value"]),
         "b_mde_qs_design": B.power["by_metric"]["qs"]["mde"],
         "b_mde_sales_design": B.power["by_metric"]["sales"]["mde"]}
    for v in "AB":
        pre = v.lower()
        for s in ("qs", "rated", "sales"):
            r = R.results[v][s]["change_ratio_did"]
            h[f"{pre}_lift_{s}"] = r["lift"]
            h[f"{pre}_lift_{s}_ci"] = [r["ci_low"], r["ci_high"]]
            h[f"{pre}_true_lift_{s}"] = R.truth[R.series[v].test]["holdout"][s]["true_lift_pct"]
        b = R.business[v]
        h[f"{pre}_spend_removed"] = b["spend_saved_est"]
        h[f"{pre}_cost_per_incr_qs"] = b["cost_per_incremental_qs"]
        h[f"{pre}_cost_per_incr_qs_true"] = b["cost_per_incremental_qs_true"]
        h[f"{pre}_cost_per_incr_sale"] = b["cost_per_incremental_sales"]
        h[f"{pre}_cost_per_incr_sale_true"] = b["cost_per_incremental_sales_true"]
        h[f"{pre}_vendor_reported_cost_per_qs"] = b["vendor_reported_cost_per_qs"]
        h[f"{pre}_roi_est"] = b["roi_window_est"]
        h[f"{pre}_roi_true"] = b["roi_window_true"]
        h[f"{pre}_p_roi_gt_1"] = b["p_roi_gt_1"]
        h[f"{pre}_decision"] = R.decisions[v]["decision"]
        tt = truth_table(R)
        d = tt[tt["vendor"] == v]
        h[f"{pre}_interval_coverage"] = f"{int(d['covers_truth'].sum())}/{len(d)}"
    return h


def write_dsp_results(run_: DSPRun) -> Path:
    A, B, R = run_.A, run_.B, run_.R
    design = {
        "dsp_a": {"treated_units": A.treated, "control_units": A.control, "buffer_units": A.buffer_zips,
                  "excluded_counts": {k: len(v) for k, v in A.excluded.items()},
                  "winner": {k: A.winner[k] for k in ("patch_id", "c_county", "maha", "maha_patch_cov", "composite",
                                                      "wow_corr", "slope_ratio", "treat_cv", "ctrl_cv", "tier",
                                                      "reason_strict", "t_pop", "c_pop")},
                  "patches": A.patches.drop(columns=["zips"]),
                  "candidates": A.candidates.drop(columns=["t_zips", "c_zips"]).head(30),
                  "smd": A.smd, "log": A.validation_log, "params": A.params.__dict__,
                  "sizing_ladder": A.sizing, "sizing_decision": A.sizing_decision},
        "dsp_b": {"treated_units": B.treated, "treated_cluster": B.treated_cluster,
                  "weights": B.weights.round(4).to_dict(), "buffer_units": B.buffer_zips,
                  "donor_units": {f"C{k}": v for k, v in B.donor_members.items()},
                  "kmeans": {"k": B.kmeans["k"], "k_best_silhouette": B.kmeans["k_best_silhouette"],
                             "rule": B.kmeans["rule"], "silhouette": B.kmeans["silhouette"]},
                  "switch_test": {k: v for k, v in B.switch.items() if k != "share"},
                  "tournament": B.tournament, "selection": B.selection_note, "v_table": B.v_table.reset_index(),
                  "smd": B.smd, "log": B.log, "params": B.params.__dict__},
    }
    validation = {
        "dsp_a": {"injected_lift_recovery": A.injected,
                  "power": {"convention": A.power["convention"], "mde_at_test_length": A.power["mde_at_test_length"],
                            "weeks_to_power": {m: A.power["by_metric"][m]["weeks_to_power"] for m in A.power["by_metric"]}}},
        "dsp_b": {"injected_lift_recovery": {k: v for k, v in B.injected_final.items()},
                  "placebo_in_space": {"p_value": B.placebo_space["p_value"],
                                       "min_attainable_p": B.placebo_space["min_attainable_p"],
                                       "placebos": B.placebo_space["placebos"]},
                  "placebo_in_time": {"sd": B.placebo_time["sd_placebo_lift"], "placebos": B.placebo_time["placebos"]},
                  "placebo_random_geo": {"n": len(B.placebo_geo), "sd": float(B.placebo_geo["lift"].std()),
                                         "q05": float(B.placebo_geo["lift"].quantile(0.05)),
                                         "q95": float(B.placebo_geo["lift"].quantile(0.95))},
                  "augsynth": {"lambda": B.aug["lambda"], "cv_mse": B.aug["cv_mse"]},
                  "power": {"convention": B.power["convention"],
                            "mde": {m: B.power["by_metric"][m]["mde"] for m in B.power["by_metric"]},
                            "weeks_to_power": {m: B.power["by_metric"][m]["weeks_to_power"] for m in B.power["by_metric"]}}},
        "readout_power": R.power,
        "readout_sc": {v: {s: {k: R.sc[v][s][k] for k in ("lift", "ci_low", "ci_high", "p_value", "n_placebos",
                                                         "n_space_placebos", "n_geo_placebos", "p_in_space_cluster",
                                                         "min_p_in_space", "weights", "placebo_in_time")}
                           for s in R.sc[v]} for v in R.sc},
    }
    biz = {v: {k: x for k, x in R.business[v].items() if k != "roi_draws"} for v in R.business}
    spill = {v: {k: ({kk: vv for kk, vv in x.items() if kk != "buffer_series"} if isinstance(x, dict) else x)
                 for k, x in R.spillover[v].items()} for v in R.spillover}
    truth = {t: {k: v for k, v in R.truth[t]["holdout"].items() if k != "units"} for t in ("dsp_a", "dsp_b")}
    extra = {"incremental_per_dollar": incremental_per_dollar(run_), "business": biz, "spillover": spill,
             "decisions": R.decisions, "timings_s": run_.timings,
             "truth_vs_estimate": truth_table(R).drop(columns=["notes"])}
    figs = [str(Path(f).relative_to(REPORTS_DIR.parent)) for f in run_.figures]
    return write_results("dsp", headline(run_), R.estimates, design=design, validation=validation, truth=truth,
                         figures=figs, extra=extra)


# ======================================================================================
# executive readout markdown
# ======================================================================================
def _p(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}%}"


def _m(x):
    return "n/a (<= 0 incremental)" if x is None or not np.isfinite(x) else f"${x:,.0f}"


def write_readout_md(run_: DSPRun) -> Path:
    A, B, R, cfg = run_.A, run_.B, run_.R, run_.cfg
    tt = truth_table(R)
    L = []
    L.append("# Programmatic DSP geo-incrementality - executive readout (synthetic)\n")
    L.append("> Synthetic data with known ground truth; every number below is generated by `python -m incrlab.dsp` "
             "and stored in `reports/results/dsp.json`. Magnitudes are illustrative and design-dependent.\n")
    sd = A.sizing_decision
    L.append("## Pre-registered design rule\n")
    L.append(f"- **Vendor A was designed to detect a {sd['target_mde']:.0%} contribution to SALES at "
             f"{sd['power']:.0%} power** (one-sided alpha = {sd['alpha']}, {sd['test_weeks']}-week holdout, change-ratio "
             f"DiD). The holdout size was chosen by the module's own pre-period power analysis: the size ladder "
             + ", ".join(f"{r.pop_target/1e3:.0f}k" for r in A.sizing.itertuples())
             + f" was evaluated on design-window data only and the smallest strict-gate design meeting the target "
             f"was kept -> **{sd['verdict']}**, {A.winner['t_pop']:,.0f} pop, design sales MDE {sd['mde_selected']:.1%}.")
    L.append("- Sizing table (design-window data only; no post-launch data or ground truth is used):\n")
    L.append("| Holdout target | Gate tier | Strict pairs | QS MDE | Rated MDE | Sales MDE | Meets rule |\n|---|---|---|---|---|---|---|")
    for r in A.sizing.itertuples():
        if r.status != "ok":
            L.append(f"| {r.pop_target/1e3:.0f}k | {r.status} | | | | | |")
            continue
        L.append(f"| {r.pop_target/1e3:.0f}k | {r.tier} | {r.n_strict} | {r.mde_qs:.1%} | {r.mde_rated:.1%} | "
                 f"{r.mde_sales:.1%} | {'yes' if r.meets_target else 'no'} |")
    L.append(f"\n- The original plan (0.5M pop, 6 weeks) had a sales MDE near 30% and only a relaxed-tier twin; "
             "it would have read a 15% true sales contribution as noise. Hence the longer window and larger holdout.")
    L.append(f"- Vendor B (same 12-week calendar): design sales MDE {B.power['by_metric']['sales']['mde']:.1%}. "
             "Decision rule for both vendors (fixed before launch): **GO** requires a significant SALES lift from a design "
             f"powered for sales (design MDE <= {sd['target_mde']:.0%}) and P(ROI > 1) >= 80%. **NO-GO** when P(ROI > 1) "
             "<= 20%, or when a sales-powered design's interval rules out the break-even lift. A design that is not "
             "powered for sales can at most return **HOLD - RE-TEST POWERED ON SALES**: QS / rated lift alone, or a "
             "'significant' sales read from an under-powered design (exaggerated by construction), never funds a GO.\n")
    L.append("## Bottom line\n")
    for v, test in (("A", "dsp_a"), ("B", "dsp_b")):
        d, b = R.decisions[v], R.business[v]
        cr = {s: R.results[v][s]["change_ratio_did"] for s in ("qs", "rated", "sales")}
        tr = R.truth[test]["holdout"]
        L.append(f"- **Vendor {v} - {d['decision']}.** Contribution (change-ratio DiD, 90% CI): QS {_p(cr['qs']['lift'])} "
                 f"[{_p(cr['qs']['ci_low'])}, {_p(cr['qs']['ci_high'])}], rated {_p(cr['rated']['lift'])}, sales "
                 f"{_p(cr['sales']['lift'])} [{_p(cr['sales']['ci_low'])}, {_p(cr['sales']['ci_high'])}]. Truth: "
                 f"{_p(tr['qs']['true_lift_pct'])} / {_p(tr['rated']['true_lift_pct'])} / {_p(tr['sales']['true_lift_pct'])}. "
                 f"Estimated window ROI {b['roi_window_est']:.2f} (true {b['roi_window_true']:.2f}). {d['rationale']}.")
    L.append("")
    L.append("## What was tested\n")
    td_a, td_b = cfg.tests["dsp_a"], cfg.tests["dsp_b"]
    L.append("| | Vendor A | Vendor B |\n|---|---|---|")
    L.append(f"| Design | Contiguous zip patch vs Mahalanobis-matched twin patch | Geo-weighted K-means cluster vs "
             f"synthetic control of other clusters |")
    L.append(f"| Treatment | {len(A.treated)} zips, {A.winner['t_pop']:,.0f} pop ({A.winner['patch_id']}) | "
             f"{len(B.treated)} zips, cluster {B.treated_cluster} ({B.selection_note}) |")
    L.append(f"| Control | {len(A.control)} zips, {A.winner['c_pop']:,.0f} pop (anchored in {A.winner['c_county']}) | "
             f"weights {', '.join(f'{k}={w:.2f}' for k, w in B.weights.sort_values(ascending=False).items())} |")
    L.append(f"| Spillover buffer | {A.params.buffer_miles:g} mi ({len(A.buffer_zips)} zips not used) | "
             f"{B.params.buffer_miles:g} mi ({len(B.buffer_zips)} donor zips removed) |")
    L.append(f"| Test window | weeks {td_a.test_start}-{td_a.test_end} ({td_a.test_end - td_a.test_start + 1} wk) | "
             f"weeks {td_b.test_start}-{td_b.test_end} ({td_b.test_end - td_b.test_start + 1} wk) |")
    L.append(f"| Spend removed | ${R.business['A']['spend_saved_est']:,.0f} (compliance: "
             f"{R.business['A']['compliance_pct_of_bau']:.1%} of BAU still served) | ${R.business['B']['spend_saved_est']:,.0f} "
             f"({R.business['B']['compliance_pct_of_bau']:.1%} still served) |")
    L.append(f"| Design MDE (QS, 80% power, one-sided a=0.10) | {A.power['mde_at_test_length']['qs']:.1%} | "
             f"{B.power['by_metric']['qs']['mde']:.1%} |")
    L.append("")
    L.append("## Lift table (change-ratio DiD = headline)\n")
    L.append("| Vendor | Stage | Ctrl ratio | Test ratio | Counterfactual / wk | Incremental (test period) | "
             "Contribution [90% CI] | Truth |\n|---|---|---|---|---|---|---|---|")
    for v, test in (("A", "dsp_a"), ("B", "dsp_b")):
        for s in ("qs", "rated", "sales"):
            r = R.results[v][s]["change_ratio_did"]
            L.append(f"| {v} | {STAGE_LABELS[s]} | {r['c_ratio']:.3f} | {r['t_ratio']:.3f} | {r['cf_per_week']:,.0f} | "
                     f"{r['incremental']:+,.0f} [{r['inc_ci_low']:+,.0f}, {r['inc_ci_high']:+,.0f}] | {_p(r['lift'])} "
                     f"[{_p(r['ci_low'])}, {_p(r['ci_high'])}] | {_p(R.truth[test]['holdout'][s]['true_lift_pct'])} |")
    L.append("")
    L.append("## Triangulation (all methods, 90% intervals; significant = interval excludes 0)\n")
    L.append("| Vendor | Stage | Method | Estimate | 90% interval | p | Sig. | Truth | Error (pp) |\n"
             "|---|---|---|---|---|---|---|---|---|")
    for r in tt.itertuples():
        L.append(f"| {r.vendor} | {STAGE_LABELS[r.metric]} | {r.method_label} | {_p(r.estimate)} | "
                 f"[{_p(r.ci_low)}, {_p(r.ci_high)}] | {r.p_value:.2f} | {'yes' if r.significant else 'no'} | "
                 f"{_p(r.truth)} | {r.error_pp:+.1f} |")
    cov = tt.groupby("vendor")["covers_truth"].agg(["sum", "count"])
    L.append(f"\nInterval coverage of the truth: Vendor A {int(cov.loc['A', 'sum'])}/{int(cov.loc['A', 'count'])}, "
             f"Vendor B {int(cov.loc['B', 'sum'])}/{int(cov.loc['B', 'count'])}.\n")
    L.append("## Business translation\n")
    L.append("| | Vendor A est. | Vendor A truth | Vendor B est. | Vendor B truth |\n|---|---|---|---|---|")
    ba, bb = R.business["A"], R.business["B"]
    for s in ("qs", "rated", "sales"):
        L.append(f"| Incremental cost per {STAGE_LABELS[s].lower()[:-1]} | {_m(ba[f'cost_per_incremental_{s}'])} | "
                 f"{_m(ba[f'cost_per_incremental_{s}_true'])} | {_m(bb[f'cost_per_incremental_{s}'])} | "
                 f"{_m(bb[f'cost_per_incremental_{s}_true'])} |")
    L.append(f"| Vendor-reported cost per QS | {_m(ba['vendor_reported_cost_per_qs'])} | | "
             f"{_m(bb['vendor_reported_cost_per_qs'])} | |")
    L.append(f"| Site last-touch cost per QS | {_m(ba['last_touch_cost_per_qs'])} | | {_m(bb['last_touch_cost_per_qs'])} | |")
    L.append(f"| Window ROI (value/sale ${ba['value_per_sale']:,.0f}) | {ba['roi_window_est']:.2f} "
             f"[{ba['roi_window_ci'][0]:.2f}, {ba['roi_window_ci'][1]:.2f}] | {ba['roi_window_true']:.2f} | "
             f"{bb['roi_window_est']:.2f} [{bb['roi_window_ci'][0]:.2f}, {bb['roi_window_ci'][1]:.2f}] | "
             f"{bb['roi_window_true']:.2f} |")
    L.append("")
    L.append("## Spillover\n")
    for v in "AB":
        sp = R.spillover[v]
        tl = sp["truth"].get("qs", {}).get("loss_pct", np.nan)
        L.append(f"- Vendor {v}: {sp['n_buffer_served']} served zips within {sp['radius_mi']:g} mi of the treatment. "
                 f"Estimated QS change there vs the design control {_p(sp['qs']['buffer_loss_est'])} "
                 f"[{_p(sp['qs']['ci'][0])}, {_p(sp['qs']['ci'][1])}]; true spillover loss {tl:.1%}. "
                 f"Using those adjacent zips as the control would have read QS {_p(sp['qs']['naive_adjacent_control_lift'])}.")
    L.append("\nThe buffer-zip comparison mixes real spillover with geographic drift between the buffer and the "
             "control; it is a sanity check, not a spillover estimate. Spillover truth is small here, and the buffer "
             "keeps it out of the control by construction.\n")
    L.append("## Design validation (before launch)\n")
    L.append(f"- Vendor A: {len(A.patches)} treatment patches; {len(A.candidates)} candidate pairs validated; "
             f"{int(A.candidates['pass_strict'].sum())} passed the strict pre-trend gates and "
             f"{int((A.candidates['tier'] == 'relaxed').sum())} passed only the explicitly logged relaxed tier. Selected "
             f"pair tier: **{A.winner['tier']}** ({A.winner['reason_strict']}). Nothing is force-passed.")
    L.append(f"- Vendor A selected pair: in-time placebo drift over the last {A.params.pseudo_post_weeks} design weeks "
             f"{A.winner['placebo_drift']:+.1%} (a pair that drifts with no intervention biases the readout by about "
             "that much); WoW corr {:.2f}.".format(A.winner["wow_corr"]))
    inj = A.injected[A.injected["metric"] == "qs"]
    L.append(f"- Vendor A injected-lift check (QS): " + ", ".join(
        f"{r.injected:.0%} -> {r.detected:.1%}" for r in inj.itertuples()) + ".")
    L.append(f"- Vendor A balance: |SMD| >= 0.25 on {int((A.smd['abs_smd'] >= 0.25).sum())}/{len(A.smd)} features "
             f"(largest: {A.smd.sort_values('abs_smd').iloc[-1]['label']} {A.smd['abs_smd'].max():.2f}). "
             "Change-ratio DiD differences out level gaps; imbalance matters only via differential trends.")
    L.append(f"- Vendor B: vendor switch shifts the vendor-reported share by {B.switch.get('shift_pct', np.nan):+.1%} "
             f"(Welch p = {B.switch['p_value']:.3g}) -> SC dependent is QS per capita. K = {B.kmeans['k']} "
             f"({B.kmeans['rule']}). Tournament: {int(B.tournament['feasible'].sum())}/{len(B.tournament)} clusters feasible.")
    L.append(f"- Vendor B final SC: rel. RMSE {B.sc.rel_rmse:.1%}, {B.sc.n_donors_used} active donors; injected "
             f"+{B.params.injected_lift:.0%} recovered as {B.injected_final['detected']:+.1%}. Cluster placebo p = "
             f"{B.placebo_space['p_value']:.2f} (min attainable {B.placebo_space['min_attainable_p']:.2f}); "
             f"{len(B.placebo_geo)} random-geography placebos give a 90% null band "
             f"[{B.placebo_geo['lift'].quantile(.05):+.1%}, {B.placebo_geo['lift'].quantile(.95):+.1%}].")
    L.append("")
    L.append("## Recommendations\n")
    L.append("1. Power every holdout on the decision metric (sales), not on quote starts: readout-period sales MDE is "
             f"{R.power['A']['sales']['mde']:.0%} (A) / {R.power['B']['sales']['mde']:.0%} (B) at "
             f"{len(R.series['A'].post)} weeks; a 6-week, 0.5M-pop design would have been ~30%.")
    L.append("2. Never use vendor-reported conversions for budget decisions; the vendor over-claims incremental QS "
             f"by ~{(ba['cost_per_incremental_qs_true'] or np.nan) / ba['vendor_reported_cost_per_qs']:.0f}x (A, truth) and "
             f"~{(bb['cost_per_incremental_qs_true'] or np.nan) / bb['vendor_reported_cost_per_qs']:.0f}x (B, truth) in cost-per terms.")
    L.append("3. Feed `extra.incremental_per_dollar` (with SE) into the MMM as calibration likelihood terms, not as "
             "point priors; the wide Vendor A interval should carry little weight.")
    L.append("4. Re-test at scaled spend (spend-parity caveat): cost-pers from small-budget tests do not transfer to "
             "budgets several times larger.")
    L.append("")
    L.append("## Assumptions and limitations\n")
    L.append("- Synthetic world; the true effect follows a Hill + adstock response, so carry-over continues after "
             "the window and is not counted in the window lift.")
    L.append(f"- Change-ratio DiD assumes parallel relative trends; the Vendor A pair passed the "
             f"**{A.winner['tier']}** pre-trend gates, which support but cannot prove that assumption.")
    L.append("- Test windows were lengthened (A: 6 -> 12 weeks, B: 8 -> 12 weeks) after the pre-period power analysis "
             "showed the original plan could not resolve sales; the true effects were not changed.")
    L.append("- Placebo-based SC intervals rely on exchangeability of the placebo geographies.")
    L.append("- Significance in every table is decided only by whether the 90% interval excludes 0.")
    L.append("\nFigures: `reports/figures/dsp/`. Methodology: `docs/dsp_pipeline.md`.")
    p = REPORTS_DIR / "readouts" / "dsp.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n")
    return p
