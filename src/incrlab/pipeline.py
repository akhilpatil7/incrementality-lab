"""`python -m incrlab.pipeline all` - run the whole measurement program end to end.

    data -> meta (Social holdout) -> search (two-arm dark-out) -> dsp (flagship, two vendors)
         -> mmm (experiment-calibrated MMM + budget optimizer)

Each stage writes ``reports/results/<module>.json``; this runner then condenses the headline
numbers (estimate, interval, truth) into ``reports/results/summary.json``. The README scoreboard
is generated from that file (``python -m incrlab.pipeline readme-table``), so prose and numbers
cannot drift apart.

Stages can be run individually: ``python -m incrlab.pipeline data meta`` etc.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import RESULTS_DIR, get_config

STAGES = ("data", "meta", "search", "dsp", "mmm", "geoselect")
_CMDS = {
    "data": [sys.executable, "-m", "incrlab.simulate"],
    "meta": [sys.executable, "-m", "incrlab.meta"],
    "search": [sys.executable, "-m", "incrlab.search"],
    "dsp": [sys.executable, "-m", "incrlab.dsp"],
    "mmm": [sys.executable, "-m", "incrlab.mmm"],
    "geoselect": [sys.executable, "-m", "incrlab.geoselect"],
}


def run_stage(stage: str) -> float:
    """Run one stage in a fresh interpreter (isolates memory / matplotlib state); returns seconds."""
    t0 = time.time()
    print(f"\n=== [{stage}] {' '.join(_CMDS[stage][1:])} ===", flush=True)
    subprocess.run(_CMDS[stage], check=True)
    dt = time.time() - t0
    print(f"=== [{stage}] done in {dt:.0f}s ===", flush=True)
    return dt


# --------------------------------------------------------------------------------------
# summary.json
# --------------------------------------------------------------------------------------
def _load(module: str, results_dir: Path) -> Optional[dict]:
    p = results_dir / f"{module}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _row(test, quantity, est, lo, hi, truth, units="relative lift", note=""):
    covers = None if lo is None or hi is None or truth is None else bool(lo <= truth <= hi)
    return {"test": test, "quantity": quantity, "estimate": est, "ci_low": lo, "ci_high": hi, "truth": truth,
            "covers_truth": covers, "units": units, "note": note}


def build_summary(results_dir: Path = RESULTS_DIR, timings: Optional[Dict[str, float]] = None) -> dict:
    meta, search, dsp, mmm = (_load(m, results_dir) for m in ("meta", "search", "dsp", "mmm"))
    rows: List[dict] = []
    mods: Dict[str, dict] = {}
    if meta:
        h = meta["headline"]
        for s in ("qs", "sales"):
            rows.append(_row("Social holdout", f"{'Quote starts' if s == 'qs' else 'Sales'} lift", h[f"lift_{s}"],
                             h[f"lift_{s}_ci_low"], h[f"lift_{s}_ci_high"], h[f"true_lift_{s}"]))
        mods["meta"] = {"verdict": h["verdict"], "treated_unit": h["treated_unit"], "test_weeks": h["test_weeks"],
                        "design_mde": h["design_mde_8wk"], "lift_qs": h["lift_qs"], "true_lift_qs": h["true_lift_qs"],
                        "lift_sales": h["lift_sales"], "true_lift_sales": h["true_lift_sales"],
                        "lt_overclaim_ratio_qs": h["lt_overclaim_ratio_qs"],
                        "incremental_cost_per_sale": h["incremental_cost_per_sales"],
                        "true_incremental_cost_per_sale": h["true_incremental_cost_per_sales"]}
    if search:
        h = search["headline"]
        for arm, lab in (("all_search", "All paid search"), ("branded_only", "Branded only")):
            rows.append(_row("Search dark-out", f"{lab}: rated-quote lift", h[f"{arm}_rated_lift"],
                             *h[f"{arm}_rated_lift_ci"], h[f"{arm}_rated_true_lift"]))
        rows.append(_row("Search dark-out", "Branded: organic recapture", h["branded_only_recapture_rated"],
                         *h["branded_only_recapture_rated_ci"], h["branded_only_true_recapture_rated"],
                         units="share of lost rated quotes"))
        mods["search"] = {"recommendation": h["recommendation"], "share_truth_in_ci": h["share_truth_in_ci"],
                          "n_truth_rows": h["n_truth_rows"],
                          "branded_recapture": h["branded_only_recapture_rated"],
                          "branded_true_recapture": h["branded_only_true_recapture_rated"],
                          "all_search_mde_sales": h["all_search_mde_sales"]}
    if dsp:
        h = dsp["headline"]
        for v, lab in (("a", "Vendor A"), ("b", "Vendor B")):
            for s, sl in (("qs", "quote-start"), ("sales", "sales")):
                rows.append(_row(f"DSP {lab}", f"{sl} lift", h[f"{v}_lift_{s}"], *h[f"{v}_lift_{s}_ci"],
                                 h[f"{v}_true_lift_{s}"]))
        mods["dsp"] = {k: h.get(k) for k in (
            "a_sizing_verdict", "a_sizing_pop_target", "a_treated_pop", "a_n_treated_zips", "a_gate_tier",
            "a_test_weeks", "b_test_weeks", "target_mde_sales", "a_mde_sales_design", "b_mde_sales_design",
            "a_decision", "b_decision", "a_roi_est", "a_roi_true", "b_roi_est", "b_roi_true", "a_p_roi_gt_1",
            "b_p_roi_gt_1", "a_cost_per_incr_sale", "a_cost_per_incr_sale_true", "b_cost_per_incr_sale",
            "b_cost_per_incr_sale_true", "a_vendor_reported_cost_per_qs", "a_cost_per_incr_qs_true",
            "a_interval_coverage", "b_interval_coverage")}
    if mmm:
        h = mmm["headline"]
        mods["mmm"] = {k: h.get(k) for k in (
            "roi_mae_uncalibrated", "roi_mae_calibrated", "roi_mae_calibrated_powered_sensitivity",
            "roi_hdi_coverage_uncalibrated", "roi_hdi_coverage_calibrated", "channels_improved_by_calibration",
            "optimizer_gain_sales_mean", "optimizer_prob_better", "optimizer_true_gain_sales",
            "optimizer_true_gain_pct", "optimizer_multipliers", "cpa_status_quo", "cpa_optimized", "next_test")}
        roi = {r["channel"]: r for r in mmm["extra"]["roi_table"]}
        for ch, r in roi.items():
            rows.append(_row("MMM (calibrated)", f"ROI {ch}", r["calibrated_roi"], r["calibrated_roi_lo"],
                             r["calibrated_roi_hi"], r["true_roi"], units="$ value per $"))
        mods["mmm"]["roi_by_channel"] = {ch: {"true": r["true_roi"], "uncalibrated": r["uncalibrated_roi"],
                                              "calibrated": r["calibrated_roi"]} for ch, r in roi.items()}
    cov = [r["covers_truth"] for r in rows if r["covers_truth"] is not None]
    cfg = get_config()
    return {"schema_version": "1.0", "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "seed": cfg.seed, "small": cfg.small, "modules": mods, "scoreboard": rows,
            "scoreboard_coverage": f"{sum(cov)}/{len(cov)}",
            "timings_s": timings or {}}


def write_summary(timings: Optional[Dict[str, float]] = None, results_dir: Path = RESULTS_DIR) -> Path:
    doc = build_summary(results_dir, timings)
    p = results_dir / "summary.json"
    if timings is None and p.exists():  # keep the last full-run timings when only re-summarising
        doc["timings_s"] = json.loads(p.read_text()).get("timings_s", {})
    p.write_text(json.dumps(doc, indent=2))
    return p


# --------------------------------------------------------------------------------------
# README scoreboard (generated, never hand-typed)
# --------------------------------------------------------------------------------------
def _fmt(x, units):
    if x is None:
        return "n/a"
    if units == "$ value per $":
        return f"{x:.2f}"
    return f"{x:+.1%}" if "lift" in units else f"{x:.0%}"


def scoreboard_markdown(summary: Optional[dict] = None) -> str:
    s = summary or json.loads((RESULTS_DIR / "summary.json").read_text())
    L = ["| Test | Quantity | Estimate | 90% interval | Truth | Covered |", "|---|---|---|---|---|---|"]
    for r in s["scoreboard"]:
        u = r["units"]
        iv = "n/a" if r["ci_low"] is None else f"[{_fmt(r['ci_low'], u)}, {_fmt(r['ci_high'], u)}]"
        L.append(f"| {r['test']} | {r['quantity']} | {_fmt(r['estimate'], u)} | {iv} | {_fmt(r['truth'], u)} | "
                 f"{'yes' if r['covers_truth'] else 'no'} |")
    L.append("")
    L.append(f"90% intervals cover the truth for **{s['scoreboard_coverage']}** rows; misses: "
             + ("; ".join(f"{r['test']} - {r['quantity']}" for r in s["scoreboard"] if r["covers_truth"] is False)
                or "none") + ".")
    return "\n".join(L)


def headlines_markdown(summary: Optional[dict] = None) -> str:
    """One generated result line per module (README 'Results at a glance')."""
    s = summary or json.loads((RESULTS_DIR / "summary.json").read_text())
    m = s["modules"]
    L = []
    if "meta" in m:
        x = m["meta"]
        L.append(f"- **Social holdout (first test, {x['test_weeks']} wk):** {x['verdict']} - QS contribution "
                 f"{x['lift_qs']:+.1%} vs true {x['true_lift_qs']:+.1%} (design MDE {x['design_mde']:.1%}); last-touch "
                 f"over-claims incremental quote starts {x['lt_overclaim_ratio_qs']:.1f}x.")
    if "search" in m:
        x = m["search"]
        L.append(f"- **Paid Search two-arm dark-out:** organic recaptures {x['branded_recapture']:.0%} of lost branded "
                 f"rated quotes (true {x['branded_true_recapture']:.0%}); {x['share_truth_in_ci']:.0%} of "
                 f"{x['n_truth_rows']} truth-scored quantities fall inside their 90% intervals.")
    if "dsp" in m:
        x = m["dsp"]
        L.append(f"- **Programmatic DSP (flagship):** Vendor A holdout sized by its own power analysis "
                 f"({x['a_sizing_verdict']}, {x['a_treated_pop'] / 1e6:.2f}M pop, {x['a_test_weeks']} wk, sales MDE "
                 f"{x['a_mde_sales_design']:.1%} vs {x['target_mde_sales']:.0%} target) -> **{x['a_decision']}** "
                 f"(window ROI {x['a_roi_est']:.2f}, true {x['a_roi_true']:.2f}). Vendor B (sales MDE "
                 f"{x['b_mde_sales_design']:.0%}) -> **{x['b_decision']}** (true ROI {x['b_roi_true']:.2f}).")
    if "mmm" in m:
        x = m["mmm"]
        L.append(f"- **Calibrated MMM:** mean abs ROI error uncalibrated {x['roi_mae_uncalibrated']:.2f} -> calibrated "
                 f"{x['roi_mae_calibrated']:.2f} with the tests as read; {x['roi_mae_calibrated_powered_sensitivity']:.2f} "
                 f"if every test were powered on sales. Optimizer: {x['optimizer_true_gain_pct']:+.1%} true sales at "
                 f"equal budget; next test: `{x['next_test']}`.")
    return "\n".join(L)


def _replace_block(txt: str, name: str, body: str) -> str:
    a, b = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    if a not in txt or b not in txt:
        return txt
    return txt[:txt.index(a) + len(a)] + "\n" + body + "\n" + txt[txt.index(b):]


def update_readme(readme: Path = Path(__file__).resolve().parents[2] / "README.md") -> bool:
    """Replace the block between the scoreboard markers in README.md with the generated table."""
    if not readme.exists():
        return False
    txt = readme.read_text()
    new = _replace_block(_replace_block(txt, "SCOREBOARD", scoreboard_markdown()), "HEADLINES", headlines_markdown())
    readme.write_text(new)
    return new != txt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="incrementality-lab: run the measurement program end to end")
    ap.add_argument("stages", nargs="*", default=["all"],
                    help="all | summary | readme-table | any of " + ", ".join(STAGES))
    a = ap.parse_args(argv)
    if a.stages == ["readme-table"]:
        print(scoreboard_markdown())
        print()
        print(headlines_markdown())
        return 0
    stages = list(STAGES) if "all" in a.stages else [s for s in a.stages if s in STAGES]
    t0 = time.time()
    timings = {s: round(run_stage(s), 1) for s in stages}
    if stages:
        timings["total"] = round(time.time() - t0, 1)
    p = write_summary(timings if "all" in a.stages else None)
    changed = update_readme()
    print(f"\n[pipeline] summary -> reports/results/summary.json{' ; README scoreboard refreshed' if changed else ''}")
    if timings:
        print(f"[pipeline] timings (s): {timings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
