"""Write reports/results/mmm.json and reports/readouts/mmm.md from an :class:`MMMState`."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common.results import write_results
from ..config import REPORTS_DIR
from . import analysis as A
from .pipeline import (BOUNDS, HORIZON, MMMState, calibration_fit_table, roi_table, settings_summary,
                       triangulation_table)

SHORT = {"tv": "TV / Brand", "social": "Social", "search_brand": "Search - Brand",
         "search_nonbrand": "Search - Non-brand", "dsp_a": "DSP - Vendor A", "dsp_b": "DSP - Vendor B"}


def _relpath(f) -> str:
    from ..config import REPO_ROOT
    try:
        return str(Path(f).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return Path(f).name


def _headline(S: MMMState) -> dict:
    t = roi_table(S)
    o = S.optimizer["calibrated"]
    nxt = max(S.voi, key=lambda c: S.voi[c]["evppi_value"]) if S.voi else None
    h = {"channels": S.data.channels,
         "roi_mae_uncalibrated": float(t["uncalibrated_roi_abs_err"].mean()),
         "roi_mae_calibrated": float(t["calibrated_roi_abs_err"].mean()),
         "roi_hdi_coverage_uncalibrated": float(t["uncalibrated_roi_covers_truth"].mean()),
         "roi_hdi_coverage_calibrated": float(t["calibrated_roi_covers_truth"].mean()),
         "channels_improved_by_calibration": [r.channel for r in t.itertuples()
                                              if r.calibrated_roi_abs_err < r.uncalibrated_roi_abs_err],
         "optimizer_gain_sales_mean": o["optimal"]["delta_vs_ref"]["mean"],
         "optimizer_gain_sales_hdi": [o["optimal"]["delta_vs_ref"]["hdi_low"], o["optimal"]["delta_vs_ref"]["hdi_high"]],
         "optimizer_prob_better": o["optimal"]["prob_better"],
         "optimizer_multipliers": o["optimal"]["multipliers"],
         "cpa_status_quo": o["status_quo"]["cpa"]["mean"], "cpa_optimized": o["optimal"]["cpa"]["mean"],
         "next_test": nxt,
         "calibration_sources": S.sources, "stub_tests": S.stub_tests,
         "calibration_uses_stub": bool(S.stub_tests),
         "runtime_seconds": S.seconds}
    if "calibrated_powered" in S.fits:
        h["roi_mae_calibrated_powered_sensitivity"] = float(t["calibrated_powered_roi_abs_err"].mean())
    if o.get("truth_check"):
        h["optimizer_true_gain_sales"] = o["truth_check"]["true_gain"]
        h["optimizer_true_gain_pct"] = o["truth_check"]["true_gain_pct"]
    for name, f in S.fits.items():
        h[f"diag_{name}"] = {k: v for k, v in f.diag.items() if not k.startswith("_")}
        h[f"diag_{name}"]["fit_seconds"] = f.seconds
    return h


def write_results_json(S: MMMState, results_dir=None) -> Path:
    t = roi_table(S)
    est = []
    for r in t.to_dict("records"):
        for name in S.fits:
            j = S.data.channels.index(r["channel"])
            for metric, arr, truth in (("roi", S.fits[name].roi, r["true_roi"]), ("marginal_roi", S.fits[name].mroi, r["true_mroi"])):
                sm = A.summarize(arr[:, j])
                est.append({"test": "mmm", "arm": r["channel"], "metric": metric, "method": f"pymc_mmm_{name}",
                            "estimate": sm["mean"], "ci_low": sm["hdi_low"], "ci_high": sm["hdi_high"],
                            "ci_level": A.HDI_PROB, "sd": sm["sd"], "truth": truth,
                            "truth_key": f"ground_truth.channels.{r['channel']}.{metric}",
                            "units": "$ sales value per $ spend", "notes": "90% HDI"})
    extra = {"roi_table": t, "triangulation": triangulation_table(S), "calibration_fit": calibration_fit_table(S),
             "calibration_observations": S.obs, "optimizer": S.optimizer, "value_of_information": S.voi,
             "settings": settings_summary(S),
             "calibration_method": ("lift-test likelihood: observed incremental sales per $ removed ~ Normal("
                                    "model-implied value for the same window/leakage/channels, se_model); "
                                    "se_model = hypot(test SE, 10% transport error)")}
    return write_results("mmm", _headline(S), est, design={"horizon_weeks": HORIZON, "bounds": BOUNDS},
                         validation={n: {k: v for k, v in f.diag.items() if not k.startswith("_")} for n, f in S.fits.items()},
                         truth={"roi": dict(zip(S.data.channels, S.truth["roi"])),
                                "marginal_roi": dict(zip(S.data.channels, S.truth["mroi"])),
                                "decay": dict(zip(S.data.channels, S.truth["decay"]))},
                         figures=[_relpath(f) for f in S.figures], extra=extra, results_dir=results_dir)


def _f(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def readout_md(S: MMMState) -> str:
    h = _headline(S)
    t = roi_table(S)
    tri = triangulation_table(S)
    cal = calibration_fit_table(S)
    o = S.optimizer["calibrated"]
    ou = S.optimizer["uncalibrated"]
    du, dc = S.fits["uncalibrated"].diag, S.fits["calibrated"].diag
    stub_warn = ""
    if S.stub_tests:
        stub_warn = (f"\n> **WARNING - STUB CALIBRATION INPUTS:** module results were missing for "
                     f"**{', '.join(S.stub_tests)}**, so those tests use the fallback quick geo tests in "
                     f"`data/calibration_inputs.json`. Re-run `make mmm` after `make dsp` to use the real readouts.\n")
    L = []
    L += ["# Experiment-calibrated MMM - readout", "",
          "*Synthetic data; illustrative magnitudes. Generated by `python -m incrlab.mmm`; every number below is in "
          "`reports/results/mmm.json`.*", stub_warn,
          "## The gap this closes",
          "Three geo lift tests (Social, Paid Search, Programmatic DSP) each answered *one* question in *one* window in "
          "*a few* geos. Always-on budgeting needs a cross-channel, national, saturation-aware answer. A plain MMM "
          "gives that answer but is uncalibrated: it can only separate channels by how their spend happened to move. "
          "This module fits a Bayesian MMM twice - without and with the tests - and turns the calibrated posterior "
          "into a reallocation and a next-test recommendation.", "",
          "## Model",
          "- Weekly national sales (156 weeks; last 13 held out). log y ~ Normal(log mu, sigma); "
          "mu = baseline x (1 + sum_c beta_c Hill(adstock_c)); baseline = exp(intercept + trend + 3-pair Fourier + 5 "
          "holiday dummies).",
          "- Per channel: geometric adstock (normalised, Beta(2,2.5) decay), Hill saturation (half-saturation and slope), "
          "and an ROI-parameterised amplitude with the same LogNormal(log 1, 0.9) ROI prior for every channel.",
          "- **Calibration:** for each test arm, the model computes - for every posterior draw - the incremental sales "
          "per $ it would lose if that arm's channel(s) were cut to the test's leakage level in the test window "
          "(adstock carry-over and saturation included). The test's readout enters as `observed ~ Normal(model-implied, "
          "se_model)` with se_model = sqrt(test SE² + (10% transport error)²). This is the lift-test likelihood used in "
          "PyMC-Marketing / Meridian-style workflows: a test measures a window-average return at the current spend level, "
          "a joint function of amplitude, half-saturation, slope and decay - not a prior on one coefficient.",
          f"- Sampler: NUTS, {S.settings.chains} chains x {S.settings.draws} draws ({S.settings.tune} tune).", "",
          "## Calibration inputs (source per test)", "",
          "| Test / arm | Channels | Observed sales per $1k | SE | Truth (test geos) | Uncal. implied | Cal. implied | Source |",
          "|---|---|---|---|---|---|---|---|"]
    for r in cal.itertuples():
        src = "STUB fallback" if str(r.source).startswith("stub") else Path(str(r.source)).name
        tr = r.truth_ipd
        L.append(f"| {r.test} / {r.arm} | {r.channels} | {r.observed_ipd * 1e3:.2f} | {r.se * 1e3:.2f} | "
                 f"{'–' if tr is None or not np.isfinite(tr) else f'{tr * 1e3:.2f}'} | {r.uncalibrated_ipd * 1e3:.2f} | "
                 f"{r.calibrated_ipd * 1e3:.2f} | {src} |")
    L += ["", "## Diagnostics", "",
          "| Fit | max r-hat | min ESS | divergences | train MAPE | holdout MAPE | holdout R² | PPC 90% coverage (train) |",
          "|---|---|---|---|---|---|---|---|"]
    for n, f in S.fits.items():
        d = f.diag
        L.append(f"| {n} | {d['max_r_hat']:.3f} | {d['min_ess_bulk']:.0f} | {d['divergences']} | {d['train_mape']:.1%} | "
                 f"{d['holdout_mape']:.1%} | {d['holdout_r2']:.2f} | {d['train_ppc_coverage90']:.0%} |")
    L += ["", "## ROI vs truth (posterior mean, 90% HDI)", "",
          "| Channel | Truth | Uncalibrated | Calibrated | abs err uncal. | abs err cal. | Marginal ROI truth / cal. |",
          "|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {SHORT[r.channel]} | {r.true_roi:.2f} | {r.uncalibrated_roi:.2f} [{r.uncalibrated_roi_lo:.2f}, "
                 f"{r.uncalibrated_roi_hi:.2f}] | {r.calibrated_roi:.2f} [{r.calibrated_roi_lo:.2f}, {r.calibrated_roi_hi:.2f}] | "
                 f"{r.uncalibrated_roi_abs_err:.2f} | {r.calibrated_roi_abs_err:.2f} | {r.true_mroi:.2f} / {r.calibrated_mroi:.2f} |")
    L += ["", f"Mean absolute ROI error: **uncalibrated {h['roi_mae_uncalibrated']:.2f} -> calibrated "
              f"{h['roi_mae_calibrated']:.2f}**" +
          (f" (sensitivity - same tests if well powered, ±15% SE: {h['roi_mae_calibrated_powered_sensitivity']:.2f})"
           if "roi_mae_calibrated_powered_sensitivity" in h else "") +
          f". 90% HDI covers truth for {h['roi_hdi_coverage_uncalibrated']:.0%} -> {h['roi_hdi_coverage_calibrated']:.0%} "
          f"of channels. Calibration reduced the error for: {', '.join(SHORT[c] for c in h['channels_improved_by_calibration']) or 'none'}.",
          ""]
    mu, mc = h["roi_mae_uncalibrated"], h["roi_mae_calibrated"]
    mp = h.get("roi_mae_calibrated_powered_sensitivity")
    nz = sum(1 for r in cal.itertuples() if abs(r.observed_ipd) < 1.645 * r.se)
    verdict = ("**Honest verdict.** " + (f"With the tests as actually read out, calibration {'reduced' if mc < mu - 0.02 else 'did not materially reduce'} "
               f"average ROI error ({mu:.2f} -> {mc:.2f}). ") +
               f"{nz} of {len(cal)} test readouts are not distinguishable from zero *at the sales stage*: only "
               + (", ".join(r.test for r in cal.itertuples() if abs(r.observed_ipd) >= 1.645 * r.se) or "none")
               + " was sized for sales; the others were designed on quote starts, and sales are ~6% of quote starts "
               "and far noisier. Their likelihood terms are weak and some point the wrong way (the all-search arm reads "
               "a negative sales effect), which pulls channels sharing the windows off truth. ")
    _ra = {r.channel: r for r in t.itertuples()}
    if "dsp_a" in _ra:
        ra = _ra["dsp_a"]
        verdict += (f"The one test designed for sales (DSP Vendor A) {'moves' if ra.calibrated_roi_abs_err < ra.uncalibrated_roi_abs_err else 'does not move'} "
                    f"its channel toward truth (ROI error {ra.uncalibrated_roi_abs_err:.2f} -> {ra.calibrated_roi_abs_err:.2f}). ")
    if mp is not None:
        tp = (S.optimizer.get("calibrated_powered", {}).get("truth_check") or {})
        if mp < mu - 0.02:
            verdict += (f"Feeding *every* test design with a well-powered readout (simulated: truth ± 15% SE) cuts the error to "
                        f"**{mp:.2f}**" + (f" and its budget plan gains {tp['true_gain']:+,.0f} true sales" if tp else "") +
                        ". The lesson for the program: power the next tests on the decision metric (sales / bound policies).")
        else:
            verdict += (f"Even feeding *every* test with a well-powered, truth-centred readout (simulated: truth ± 15% SE) only "
                        f"reaches **{mp:.2f}**" + (f" (its budget plan gains {tp['true_gain']:+,.0f} true sales)" if tp else "") +
                        ": the remaining gap is model-side, not test-side. A 6-12 week geo-local return maps only partly onto "
                        "a 3-year national ROI (different spend level, carry-over outside the window, channels sharing "
                        "windows), so calibration is necessary but not sufficient. The lesson for the program: power the "
                        "next tests on sales *and* run them at national-like spend levels / longer windows so the "
                        "likelihood term measures the quantity the MMM needs.")
    L += [verdict, "", "### Where calibration did *not* help, and why", ""]
    worse = [r for r in t.itertuples() if r.calibrated_roi_abs_err >= r.uncalibrated_roi_abs_err - 0.02]
    for r in worse:
        o_ = cal[cal["channels"].str.contains(r.channel)]
        why = "no lift test covers it (TV is national; only the MMM speaks to it)" if r.channel == "tv" else ""
        if not why and len(o_):
            q = o_.iloc[0]
            rel = q.se / max(abs(q.observed_ipd), 1e-9)
            tr = q.truth_ipd if q.truth_ipd is not None and np.isfinite(q.truth_ipd) else None
            drift = ""
            if tr is not None:
                drift = (f"; the readout {q.observed_ipd * 1e3:.2f}/$1k {'over' if q.observed_ipd > tr else 'under'}states "
                         f"the truth in its own test geos ({tr * 1e3:.2f}/$1k), so the likelihood pulls the wrong way")
            why = (f"its test is too noisy to move the posterior (SE {q.se * 1e3:.2f} per $1k vs estimate "
                   f"{q.observed_ipd * 1e3:.2f}, i.e. {rel:.1f}x the point estimate)") if rel > 0.8 else \
                  ("the test window measures a short, geo-local return that the national model can only partly map to the "
                   "3-year ROI (different spend level / carry-over)")
            why += drift if (tr is not None and abs(q.observed_ipd - tr) > 0.5 * q.se) else ""
        L.append(f"- **{SHORT[r.channel]}** (error {r.uncalibrated_roi_abs_err:.2f} -> {r.calibrated_roi_abs_err:.2f}): {why}.")
    L += ["", "## Triangulation (ROI)", "",
          "| Channel | Last-touch | Experiment | Uncal. MMM | Cal. MMM | Truth |", "|---|---|---|---|---|---|"]
    for r in tri.itertuples():
        L.append(f"| {SHORT[r.channel]} | {_f(r.last_touch_roi)} | {_f(r.experiment_roi)} | {_f(r.uncalibrated_roi)} | "
                 f"{_f(r.calibrated_roi)} | {_f(r.true_roi)} |")
    dl = o["optimal"]["delta_vs_ref"]
    L += ["", "Last-touch is wrong in both directions (it credits DSP Vendor B and branded search far above their "
              "incremental return and gives TV nothing). Single experiments are unbiased but noisy and local. The calibrated MMM "
              "is the only view that is national, cross-channel and consistent with the tests.", "",
          f"## Budget reallocation (calibrated posterior; next {HORIZON} weeks replayed; total fixed; ±30% per channel)", "",
          "| Channel | Status quo spend | Recommended | Change |", "|---|---|---|---|"]
    for c in S.data.channels:
        a, b = o["status_quo"]["spend_by_channel"][c], o["optimal"]["spend_by_channel"][c]
        L.append(f"| {SHORT[c]} | ${a / 1e6:.2f}M | ${b / 1e6:.2f}M | {b / a - 1:+.0%} |")
    L += ["", f"- Expected gain: **{dl['mean']:+,.0f} incremental sales/yr** (90% HDI {dl['hdi_low']:,.0f} to "
              f"{dl['hdi_high']:,.0f}; P(better) {o['optimal']['prob_better']:.0%}); cost per incremental sale "
              f"${o['status_quo']['cpa']['mean']:,.0f} -> ${o['optimal']['cpa']['mean']:,.0f}."]
    if o.get("truth_check"):
        tc, tu = o["truth_check"], ou.get("truth_check", {})
        L.append(f"- Checked against the simulator: the calibrated plan's **true** gain is {tc['true_gain']:+,.0f} sales "
                 f"({tc['true_gain_pct']:+.1%})" + (f"; the uncalibrated MMM's plan would gain {tu['true_gain']:+,.0f} "
                                                    f"({tu['true_gain_pct']:+.1%})." if tu else "."))
    L += ["", "| Scenario | Incremental sales (mean, 90% HDI) | CPA | True incremental |", "|---|---|---|---|"]
    rows = [("-20%, proportional", o["scenarios"]["0.8x"]["proportional"]), ("-20%, optimized", o["scenarios"]["0.8x"]["optimized"]),
            ("Status quo", o["status_quo"]), ("Same budget, optimized", o["optimal"]),
            ("+20%, proportional", o["scenarios"]["1.2x"]["proportional"]), ("+20%, optimized", o["scenarios"]["1.2x"]["optimized"])]
    tcd = o.get("truth_check") or {}
    tvals = {"Status quo": tcd.get("status_quo_true_incremental"), "Same budget, optimized": tcd.get("optimal_true_incremental")}
    for n, r in rows:
        tv = r.get("true_incremental", tvals.get(n))
        L.append(f"| {n} | {r['incremental_sales']['mean']:,.0f} [{r['incremental_sales']['hdi_low']:,.0f}, "
                 f"{r['incremental_sales']['hdi_high']:,.0f}] | ${r['cpa']['mean']:,.0f} | {'–' if tv is None else f'{tv:,.0f}'} |")
    L += ["", "## Next test (value of information)", "",
          "| Channel | EVPPI (sales/yr) | EVPPI ($/yr) | ROI CV |", "|---|---|---|---|"]
    for c in sorted(S.voi, key=lambda c: -S.voi[c]["evppi_value"]):
        v = S.voi[c]
        L.append(f"| {SHORT[c]} | {v['evppi_sales']:,.0f} | ${v['evppi_value']:,.0f} | {v['roi_cv']:.2f} |")
    if h["next_test"]:
        L += ["", f"**Recommendation: test {SHORT[h['next_test']]} next** - resolving its ROI is worth the most to the "
                  "budget decision (expected gain from re-optimising once its ROI is known). Wide posteriors alone are "
                  "not enough: a channel only scores if learning it could move money across the ±30% bounds."]
    L += ["", "## Caveats",
          "- Synthetic world with a known data-generating process close to the model's functional form; real MMMs face "
          "model misspecification that these numbers do not capture.",
          "- National weekly data: ~150 points for ~35 parameters. Priors matter; they are identical across channels and "
          "documented in `extra.settings.priors`.",
          "- Test geos are a small, local slice; the 10% transport error term and the window-level likelihood are "
          "deliberate hedges, not exact physics.",
          f"- Runtime {S.seconds:.0f}s.", ""]
    return "\n".join(L)


def write_readout_md(S: MMMState, out_dir=None) -> Path:
    d = Path(out_dir or REPORTS_DIR / "readouts")
    d.mkdir(parents=True, exist_ok=True)
    p = d / "mmm.md"
    p.write_text(readout_md(S))
    return p
