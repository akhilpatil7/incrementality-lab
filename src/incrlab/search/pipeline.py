"""End-to-end Paid Search pipeline: design -> realise -> readout -> results JSON (+ figures, readout md)."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..simulate import apply_intervention, get_world, realize, run_scenario
from . import design as D
from . import panel as P
from . import readout as R
from .settings import ARMS, MODULE, TEST, SearchSettings, get_settings, wk


def realize_design(design: dict, s: SearchSettings, world=None):
    world = world or get_world(s.cfg)
    res = run_scenario({TEST: {a: design[a] for a in ARMS}}, world=world)
    return res, world


def ipd_truth(world, design: dict, s: SearchSettings) -> dict:
    """True incremental outcomes per $ for each search channel *in the arm where it is measured*.

    Brand: the branded-only arm's own truth. Non-brand: pause ONLY non-brand in the all-search
    arm's counties (a counterfactual-consistent extra scenario, used for scoring only).
    """
    td = s.td
    sc = apply_intervention(world, design["all_search"], (td.test_start, td.test_end), ("search_nonbrand",),
                            {"spend_multiplier": 0.0, "leakage": td.leakage}, test=TEST, arm="nonbrand_only")
    tn = realize(sc).truth[TEST]["nonbrand_only"]
    return tn


def run(s: Optional[SearchSettings] = None, cfg: Optional[Config] = None, make_figures: bool = True,
        write: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    s = s or get_settings(cfg or get_config())
    world = get_world(s.cfg)
    log = (lambda *a: print(f"[search {time.time() - t0:5.1f}s]", *a)) if verbose else (lambda *a: None)

    # ---- design (BAU data, weeks <= design_end only) -----------------------------
    from ..simulate import realize as _realize
    bau = _realize(world, with_truth=False).panels
    zips = world.geo.zips
    cstat = world.geo.counties
    bau_panel = P.build_county_panel(bau["county_weekly"], s.region)
    des = D.run_design(bau_panel, zips, cstat, s)
    design = des["design"]
    log("design:", {k: design[k] for k in ("all_search", "branded_only", "control")},
        f"{len(design['bau_donors'])} BAU donors")

    # ---- realise the test ----------------------------------------------------------
    res, _ = realize_design(design, s, world)
    truth = res.truth[TEST]
    panel = P.build_county_panel(res.panels["county_weekly"], s.region)
    if (panel[panel["week_idx"] <= s.design_end][["qs", "rated", "sales"]].to_numpy()
            != bau_panel[bau_panel["week_idx"] <= s.design_end][["qs", "rated", "sales"]].to_numpy()).any():
        raise AssertionError("pre-period changed by the intervention")
    up = R.readout_units(panel, design)
    bau_up = R.readout_units(bau_panel, design)
    log("realised scenario; truth all_search rated lift", round(truth["all_search"]["rated"]["true_lift_pct"], 4))

    # ---- estimators ----------------------------------------------------------------
    er = R.run_estimators(up, design, truth, bau_up, s)
    tri = R.triangulate(er["table"], s)
    log("estimators done")
    pwr = R.power_analysis(R.readout_units(bau_panel[bau_panel["week_idx"] <= s.design_end], design), design, tri, s)
    log("power done")
    rec_rows = []
    for arm in ARMS:
        for st in ("qs", "rated", "sales"):
            for ref in ("synthetic", "control"):
                r = R.recapture(up, design, arm, st, s, ref=ref)
                r["truth"] = truth[arm].get(f"true_recapture_{st}")
                rec_rows.append(r)
    rec = pd.DataFrame(rec_rows)
    src_truth = {arm: R.source_truth(up, bau_up, arm, "rated", s) for arm in ARMS}
    src_eff = {arm: R.source_effects(up, bau_up, design, arm, "rated", s) for arm in ARMS}
    tn = ipd_truth(world, design, s)
    ipd_t = {"search_brand": {m: truth["branded_only"][m]["true_incremental"] / truth["branded_only"]["spend_removed"]
                              for m in s.readout_metrics},
             "search_nonbrand": {m: tn[m]["true_incremental"] / tn["spend_removed"] for m in s.readout_metrics}}
    eco = R.economics(up, design, tri, truth, ipd_t, s)
    guard = R.guardrail(up, bau_up, design, s)
    reco = R.recommendation(tri, eco["table"], rec, guard, eco["incremental_per_dollar"], pwr["table"], s)
    log("economics / guardrail / recommendation done")

    out = {"settings": s, "design_stage": des, "design": design, "truth": truth, "panel": panel,
           "bau_panel": bau_panel, "units": up, "bau_units": bau_up, "estimates": er["table"], "rich": er["rich"],
           "triangulated": tri, "power": pwr, "recapture": rec, "source_truth": src_truth, "source_effects": src_eff, "economics": eco,
           "guardrail": guard, "recommendation": reco, "nonbrand_truth": tn, "ipd_truth": ipd_t}
    from .report import truth_table
    out["truth_table"] = truth_table(out)
    if make_figures:
        from . import figures as F
        out["figures"] = F.make_all(out)
        log(f"{len(out['figures'])} figures")
    if write:
        from .report import write_outputs
        out["paths"] = write_outputs(out)
        log("wrote", out["paths"])
    return out
