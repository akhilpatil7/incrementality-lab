"""End-to-end orchestration of the Social geo-holdout module (design -> scenario -> readout -> outputs)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..simulate import get_world, load_table, run_scenario
from . import design as D
from . import readout as R
from .settings import MetaSettings, get_settings


@dataclass
class MetaState:
    st: MetaSettings
    cfg: Config
    county_static: pd.DataFrame
    candidates: Optional[pd.DataFrame] = None
    panel: Optional[pd.DataFrame] = None              # design-time unit panel
    county_ids: Optional[list] = None
    kmeans: Optional[dict] = None
    ranked: Optional[pd.DataFrame] = None
    selection: Optional[dict] = None
    unit: Optional[str] = None
    dfit: Optional[D.DesignFit] = None
    validation: Optional[pd.DataFrame] = None         # injected lifts for the shortlist
    noise: Optional[dict] = None
    power: Optional[pd.DataFrame] = None
    weeks_to_power: Optional[pd.DataFrame] = None
    power_sim: Optional[dict] = None
    jackknife: Optional[pd.DataFrame] = None
    pysyncon: Optional[dict] = None
    truth: Optional[dict] = None
    readout_panel: Optional[pd.DataFrame] = None
    ts: Optional[pd.DataFrame] = None
    estimates: Dict[str, Dict[str, dict]] = field(default_factory=dict)  # method -> stage -> dict
    placebo: Optional[dict] = None
    spend: Optional[dict] = None
    attribution: Optional[pd.DataFrame] = None
    efficiency: Optional[pd.DataFrame] = None
    verdict: Optional[dict] = None
    figures: list = field(default_factory=list)


# ------------------------------------------------------------------------------------------
def run_design(cfg: Optional[Config] = None, st: Optional[MetaSettings] = None) -> MetaState:
    """Everything that happens BEFORE launch, using only weeks <= design_end of the BAU panel."""
    cfg = cfg or get_config()
    st = st or get_settings(cfg)
    cw_all = load_table("county_weekly", cfg.data_dir)
    cs_all = load_table("county_static", cfg.data_dir)
    cw, cs = D.load_design_panel(st, cw_all, cs_all)
    G = D.county_adjacency_graph(load_table("zip_static", cfg.data_dir), st.region)
    S = MetaState(st=st, cfg=cfg, county_static=cs_all)
    S.candidates = D.find_candidate_units(cs, G, st)
    if S.candidates.empty:
        raise RuntimeError("no candidate units in the population band - widen MetaSettings.pop_band")
    S.panel = D.build_unit_panel(cw, cs, S.candidates)
    S.county_ids = sorted(cs["county_id"])
    S.kmeans = D.kmeans_sanity(S.panel, S.county_ids, st.train_weeks, st)
    evals = pd.DataFrame([D.evaluate_candidate(S.panel, u, S.county_ids, st) for u in S.candidates["unit"]])
    evals = evals.merge(S.candidates[["unit", "size", "pop"]], on="unit")
    S.ranked = D.score_candidates(evals, st)
    S.selection = D.select_treated_unit(S.ranked, S.panel, S.county_ids, st)
    S.unit = S.selection["unit"]
    S.dfit = D.fit_design_sc(S.panel, S.unit, S.county_ids, st)
    short = S.ranked["unit"].head(st.shortlist_n).tolist()
    if S.unit not in short:
        short.append(S.unit)
    S.validation = pd.concat([D.injected_lift_validation(S.panel, D.fit_design_sc(S.panel, u, S.county_ids, st), st)
                              for u in short], ignore_index=True)
    S.noise = D.readout_noise(S.panel, S.dfit, st)
    S.power = D.power_table(S.noise["sigma"], S.noise["rho"], st)
    S.weeks_to_power = D.weeks_to_power(S.noise["sigma"], S.noise["rho"], st)
    S.power_sim = {L: D.simulated_power_check(S.panel, S.dfit, L, st, seed=cfg.seed % 1000)
                   for L in (0.05, 0.10, 0.15)}
    S.jackknife = next(c["jackknife"] for c in S.selection["checked"] if c["unit"] == S.unit)
    S.pysyncon = D.pysyncon_agreement(S.panel, S.dfit, st)
    return S


def run_readout(S: MetaState) -> MetaState:
    """Launch the holdout on the pipeline-selected unit via the scenario runner, then read it out."""
    st, cfg = S.st, S.cfg
    res = run_scenario({st.test: {st.arm: S.unit.split("+")}}, world=get_world(cfg))
    S.truth = res.truth[st.test][st.arm]
    S.readout_panel = R.readout_unit_panel(res.panels["county_weekly"], S.county_static, S.unit, st)
    S.ts = R.treated_vs_synthetic(S.readout_panel, S.unit, S.dfit.weights, st)
    S.estimates = {"did_week_fe_hc1": {}, "ratio_did": {}, "synthetic_control_gap": {}}
    for s in R.STAGES:
        S.estimates["did_week_fe_hc1"][s] = R.did_week_fe(S.ts, s, st)
        S.estimates["ratio_did"][s] = R.ratio_did(S.ts, s, st, seed=cfg.seed % 1000)
        S.estimates["synthetic_control_gap"][s] = R.sc_gap(S.ts, s, st)
    S.placebo = {s: R.placebo_in_space(S.readout_panel, S.unit, S.county_ids, S.dfit, st, stage=s)
                 for s in R.STAGES}
    for s in R.STAGES:  # attach placebo p-value + SE to the SC gap estimate
        g = S.estimates["synthetic_control_gap"][s]
        g["p_value"] = S.placebo[s]["p_value_lift_one_sided"]
        g["se_rel"] = S.placebo[s]["placebo_lift_sd"]
    S.spend = R.spend_removed(S.ts, st)
    S.attribution = R.attribution_comparison(S.ts, S.estimates["did_week_fe_hc1"], S.spend, st)
    S.efficiency = efficiency_vs_blended(S)
    mde = D.cpower.did_mde(S.noise["sigma"], st.n_test_weeks, n_pre=len(st.readout_pre_weeks), alpha=st.alpha,
                           power=st.power, rho=S.noise["rho"])
    S.verdict = R.verdict(S.estimates["did_week_fe_hc1"], S.placebo["qs"], mde, st)
    return S


def efficiency_vs_blended(S: MetaState) -> pd.DataFrame:
    """Incremental Social cost-per vs the blended all-channel cost-per in the region (readout pre).
    Replaces the spreadsheet step of the source workflow."""
    st = S.st
    rw = load_table("region_weekly", S.cfg.data_dir)
    rw = rw[(rw["region"] == st.region) & rw["week_idx"].isin(st.readout_pre_weeks)]
    spend_cols = [c for c in rw.columns if c.startswith("spend_")]
    tot_spend = float(rw[spend_cols].sum().sum())
    social_share = float(rw["spend_social"].sum()) / tot_spend
    rows = []
    est = S.estimates["did_week_fe_hc1"]
    for s in R.STAGES:
        blended = tot_spend / float(rw[s].sum())
        inc = est[s]["incremental"]
        icp = S.spend["spend_removed"] / inc if inc > 0 else np.nan
        tcp = S.truth.get(f"true_incremental_cost_per_{s}")
        rows.append({"stage": s, "blended_cost_per": blended, "incremental_cost_per": icp,
                     "true_incremental_cost_per": tcp, "ratio_to_blended": icp / blended if icp == icp else np.nan,
                     "social_spend_share": social_share})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
def estimates_table(S: MetaState) -> pd.DataFrame:
    """Truth vs estimate table (every estimate compared to the scenario's expected true lift)."""
    rows = []
    for m, by in S.estimates.items():
        for s, e in by.items():
            t = S.truth[s]
            rows.append({"method": m, "metric": s, "estimate": e["estimate"], "ci_low": e.get("ci_low"),
                         "ci_high": e.get("ci_high"), "se_rel": e.get("se_rel"), "p_value": e.get("p_value"),
                         "truth": t["true_lift_pct"], "truth_realized": t["realized_lift_pct"],
                         "error_pp": e["estimate"] - t["true_lift_pct"],
                         "incremental": e["incremental"], "true_incremental": t["true_incremental"]})
    d = pd.DataFrame(rows)
    cover = (d["ci_low"] <= d["truth"]) & (d["truth"] <= d["ci_high"])
    d["ci_covers_truth"] = cover.astype(object).where(d["ci_low"].notna(), None)
    return d


def incremental_per_dollar(S: MetaState) -> dict:
    """MMM calibration input (API.md §7): incremental outcomes per $ of Social spend removed over the
    test window, with SE from the primary (week-FE DiD) estimator."""
    sp = S.spend["spend_removed"]
    e = S.estimates["did_week_fe_hc1"]
    out = {"spend": sp, "window": [S.st.test_start, S.st.test_end], "method": "did_week_fe_hc1",
           "units": S.unit.split("+"), "leakage": S.truth.get("leakage")}
    for s in R.STAGES:
        out[s] = e[s]["incremental"] / sp
        out[f"{s}_se"] = e[s]["incremental_se"] / sp
        out[f"{s}_truth"] = S.truth[s]["true_incremental"] / S.truth["spend_removed"]
    return {"social": out}


def run(cfg: Optional[Config] = None, figures: bool = True, write: bool = True) -> MetaState:
    from . import figures as F
    from . import report

    S = run_design(cfg)
    S = run_readout(S)
    if figures:
        S.figures = F.make_all(S)
    if write:
        report.write_results_json(S)
        report.write_readout_md(S)
    return S
