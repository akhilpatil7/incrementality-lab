"""Design stage (pre-launch, uses only weeks <= design_end): candidates, MDE, validation, arm selection.

Steps
-----
1. Candidate treatment units: every Northvale county and every *adjacent* 2-county bundle that
   passes the business-risk cap (population share) and a minimum-volume gate.
2. For each candidate, a synthetic control on per-capita rated quotes (the single design metric),
   donors = every county outside the candidate. Locked weights from a pseudo-launch fit are used
   for injected-lift validation (5/10/15%) on the *same* metric, and the weekly relative gap gives
   the closed-form MDE for a 6-week test.
3. Arm assignment: choose a disjoint pair (all-search unit, branded-only unit) that maximises the
   expected power at the planning priors, with a penalty for poor injected-lift recovery.
4. Control: the remaining county (or bundle) whose per-capita pre-trend best tracks BOTH arms -
   used for the ratio DiD; every other untouched county stays as a BAU donor.
5. Balance table (standardised mean differences) for each arm vs the BAU pool and vs its
   synthetic control.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..common import power as pw
from ..common.synth import fit_synth, injected_lift_recovery, lift_from_series
from ..geo import county_adjacency
from . import panel as P
from .settings import ARMS, SearchSettings, wk


def candidate_units(panel: pd.DataFrame, zips: pd.DataFrame, s: SearchSettings) -> Dict[str, List[str]]:
    """Single counties + adjacent pairs under the population cap with enough weekly volume."""
    design = panel[panel["week_idx"].between(s.sc_fit_start, s.design_end)]
    pop = design.groupby("county_id")["pop"].first()
    vol = design.groupby("county_id")["rated"].mean()
    share = pop / pop.sum()
    counties = sorted(pop.index)
    out: Dict[str, List[str]] = {}
    for c in counties:
        if share[c] <= s.max_treat_pop_share and vol[c] >= s.min_weekly_rated:
            out[c] = [c]
    if s.max_units_per_arm >= 2:
        adj = county_adjacency(zips[zips["region"] == s.region])
        for a, b in adj[["county_a", "county_b"]].itertuples(index=False):
            ids = sorted([a, b])
            if share[ids].sum() <= s.max_treat_pop_share and vol[ids].sum() >= s.min_weekly_rated:
                out["+".join(ids)] = ids
    return out


def _design_wide(panel: pd.DataFrame, cand: Dict[str, List[str]], name: str, s: SearchSettings,
                 metric: Optional[str] = None) -> tuple[pd.DataFrame, List[str]]:
    metric = metric or s.design_metric
    ids = cand[name]
    donors = sorted(set(panel["county_id"]) - set(ids))
    up = P.unit_panel(panel, {name: ids}, donors)
    w = P.wide(up, f"{metric}_pc", wk((s.sc_fit_start, s.design_end)))
    return w, donors


def score_candidate(panel: pd.DataFrame, cand: Dict[str, List[str]], name: str, s: SearchSettings) -> dict:
    """SC fit quality, injected-lift recovery (same metric), and closed-form MDE for one candidate."""
    w, donors = _design_wide(panel, cand, name, s)
    tr, vp = wk(s.validation_train), wk(s.validation_post)
    full_pre = wk((s.sc_fit_start, s.design_end))
    rec = {lift: injected_lift_recovery(w, name, tr, vp, lift, donors=donors, ridge=s.sc_ridge)
           for lift in s.injected_lifts}
    fit_full = fit_synth(w, name, full_pre, donors=donors, ridge=s.sc_ridge)
    fit_val = fit_synth(w, name, tr, donors=donors, ridge=s.sc_ridge)
    gs = pw.gap_sigma(fit_full.actual, fit_full.synthetic, full_pre)
    oos = (fit_val.actual.loc[vp] - fit_val.synthetic.loc[vp]) / fit_val.synthetic.loc[vp]
    sigma = max(gs["sigma"], float(np.sqrt(np.mean(oos ** 2))))  # conservative: in-sample vs out-of-sample
    rho = float(np.clip(gs["rho"], 0.0, 0.8))
    mde = pw.did_mde(sigma, s.n_post, n_pre=len(full_pre), alpha=s.alpha, power=s.power, two_sided=True, rho=rho)
    pop = float(panel[panel["county_id"].isin(cand[name])].groupby("county_id")["pop"].first().sum())
    vol = float(panel[(panel["county_id"].isin(cand[name])) & panel["week_idx"].isin(full_pre)]
                .groupby("week_idx")["rated"].sum().mean())
    out = {"unit": name, "counties": cand[name], "n_counties": len(cand[name]), "pop": pop,
           "weekly_rated": vol, "pre_rel_rmse": fit_full.rel_rmse, "sigma_gap": sigma, "rho": rho,
           "oos_rel_rmse": float(np.sqrt(np.mean(oos ** 2))), "mde": mde,
           "n_donors_used": fit_full.n_donors_used, "max_weight": fit_full.max_weight,
           "natural_gap": rec[s.injected_lifts[0]]["natural_gap"],
           "weights": fit_full.weights[fit_full.weights > 0].round(3).to_dict()}
    for lift, r in rec.items():
        out[f"detected_{int(round(lift * 100))}"] = r["detected"]
        out[f"abs_err_{int(round(lift * 100))}"] = r["abs_error"]
    out["mean_abs_err"] = float(np.mean([r["abs_error"] for r in rec.values()]))
    # sales is a primary KPI but far noisier (few binds per week): score power on BOTH metrics
    ws, _ = _design_wide(panel, cand, name, s, metric="sales")
    fs = fit_synth(ws, name, full_pre, donors=donors, ridge=s.sc_ridge)
    gss = pw.gap_sigma(fs.actual, fs.synthetic, full_pre)
    sig_s, rho_s = gss["sigma"], float(np.clip(gss["rho"], 0.0, 0.8))
    out["sigma_gap_sales"] = sig_s
    out["mde_sales"] = pw.did_mde(sig_s, s.n_post, len(full_pre), s.alpha, s.power, True, rho_s)

    def pw2(lift):
        pr = pw.did_power(lift, sigma, s.n_post, len(full_pre), s.alpha, True, rho)
        ps = pw.did_power(lift * s.sales_lift_ratio, sig_s, s.n_post, len(full_pre), s.alpha, True, rho_s)
        return (1 - s.w_sales_power) * pr + s.w_sales_power * ps
    out["power_all_prior"] = pw2(s.prior_lift_all)
    out["power_brand_prior"] = pw2(s.prior_lift_branded)
    return out


def mde_table(panel: pd.DataFrame, zips: pd.DataFrame, s: SearchSettings) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    cand = candidate_units(panel, zips, s)
    rows = [score_candidate(panel, cand, n, s) for n in cand]
    t = pd.DataFrame(rows).sort_values("mde").reset_index(drop=True)
    names = panel.groupby("county_id")["county"].first()
    t.insert(1, "label", t["counties"].map(lambda ids: " + ".join(names[i].replace(" County", "") for i in ids)))
    return t, cand


def assign_arms(mde: pd.DataFrame, s: SearchSettings) -> dict:
    """Disjoint (all_search, branded_only) pair maximising expected power with a recovery penalty."""
    ok = mde[mde["abs_err_10"] <= s.recovery_gate]
    if len(ok) < 2:
        ok = mde.nsmallest(max(4, len(mde) // 2), "abs_err_10")
    best, rows = None, []
    for a, b in itertools.permutations(ok.index, 2):
        ra, rb = ok.loc[a], ok.loc[b]
        if set(ra["counties"]) & set(rb["counties"]):
            continue
        score = ra["power_all_prior"] + rb["power_brand_prior"] - s.w_recovery * (ra["mean_abs_err"] + rb["mean_abs_err"])
        rows.append({"all_search": ra["unit"], "branded_only": rb["unit"], "score": score,
                     "power_all": ra["power_all_prior"], "power_brand": rb["power_brand_prior"],
                     "mde_all": ra["mde"], "mde_brand": rb["mde"]})
        if best is None or score > best[0]:
            best = (score, ra, rb)
    pairs = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    _, ra, rb = best
    return {"all_search": list(ra["counties"]), "branded_only": list(rb["counties"]),
            "all_search_unit": ra["unit"], "branded_only_unit": rb["unit"], "pair_table": pairs}


def choose_control(panel: pd.DataFrame, arms: Dict[str, List[str]], s: SearchSettings) -> dict:
    """Best single untouched county for the ratio DiD: tracks both arms' pre-period per-capita trend."""
    treated = set(itertools.chain.from_iterable(arms.values()))
    pool = sorted(set(panel["county_id"]) - treated)
    weeks = wk((s.sc_fit_start, s.design_end))
    m = f"{s.design_metric}_pc"
    rows = []
    for c in pool:
        cs = P.pooled_series(panel, [c], s.design_metric).loc[weeks]
        errs, cors = [], []
        for arm, ids in arms.items():
            ts = P.pooled_series(panel, ids, s.design_metric).loc[weeks]
            ratio = ts / cs
            errs.append(float(ratio.std() / ratio.mean()))  # stability of the change ratio
            cors.append(float(np.corrcoef(ts.pct_change().dropna(), cs.pct_change().dropna())[0, 1]))
        vol = float(panel[(panel["county_id"] == c) & panel["week_idx"].isin(weeks)]["rated"].mean())
        rows.append({"county_id": c, "ratio_cv": float(np.mean(errs)), "wow_corr": float(np.mean(cors)),
                     "weekly_rated": vol})
    t = pd.DataFrame(rows)
    t = t[t["weekly_rated"] >= s.min_weekly_rated] if (t["weekly_rated"] >= s.min_weekly_rated).any() else t
    t["score"] = t["ratio_cv"] - 0.05 * t["wow_corr"]
    t = t.sort_values("score").reset_index(drop=True)
    del m
    return {"control": [t.iloc[0]["county_id"]], "table": t,
            "bau_donors": sorted(set(pool) - {t.iloc[0]["county_id"]})}


def smd(a: pd.Series, pool: pd.DataFrame) -> pd.Series:
    """Standardised mean difference of one unit vs a pool (pool sd as the scale)."""
    sd = pool.std(ddof=1).replace(0, np.nan)
    return (a - pool.mean()) / sd


def balance_table(panel: pd.DataFrame, county_static: pd.DataFrame, design: dict, s: SearchSettings,
                  weights: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
    groups = {arm: design[arm] for arm in ARMS}
    groups["control"] = design["control"]
    feats = P.static_features(panel, county_static, wk(s.pre), groups=groups)
    cols = list(s.balance_covariates)
    pool = feats.loc[design["bau_donors"] + design["control"], cols]
    rows = []
    for arm in list(ARMS) + ["control"]:
        a = feats.loc[arm, cols]
        r = {"unit": arm, **{f"{c}": float(a[c]) for c in cols}}
        z = smd(a, pool)
        r.update({f"smd_{c}": float(z[c]) for c in cols})
        if weights and arm in weights:
            w = weights[arm]
            synth = (feats.loc[w.index, cols].T * w.to_numpy()).T.sum()
            zs = (a - synth) / pool.std(ddof=1).replace(0, np.nan)
            r.update({f"smd_synth_{c}": float(zs[c]) for c in cols})
        rows.append(r)
    out = pd.DataFrame(rows).set_index("unit")
    pool_row = pool.mean()
    out.loc["bau_pool_mean", cols] = pool_row.to_numpy()
    return out


def run_design(panel: pd.DataFrame, zips: pd.DataFrame, county_static: pd.DataFrame, s: SearchSettings) -> dict:
    """Full design stage. Uses only weeks <= design_end."""
    dpanel = panel[panel["week_idx"] <= s.design_end]
    mde, cand = mde_table(dpanel, zips, s)
    arms = assign_arms(mde, s)
    ctl = choose_control(dpanel, {a: arms[a] for a in ARMS}, s)
    design = {"all_search": arms["all_search"], "branded_only": arms["branded_only"],
              "control": ctl["control"], "bau_donors": ctl["bau_donors"],
              "units": {"all_search": arms["all_search_unit"], "branded_only": arms["branded_only_unit"]}}
    # SC weights of each arm on BAU donors + control (both arms excluded from each other's pool)
    weights = {}
    donors = design["bau_donors"] + design["control"]
    for arm in ARMS:
        up = P.unit_panel(dpanel, {arm: design[arm]}, donors)
        w = P.wide(up, f"{s.design_metric}_pc", wk((s.sc_fit_start, s.design_end)))
        weights[arm] = fit_synth(w, arm, list(w.index), donors=donors, ridge=s.sc_ridge).weights
    bal = balance_table(dpanel, county_static, design, s, weights={a: weights[a] for a in ARMS})
    return {"design": design, "mde_table": mde, "candidates": cand, "pair_table": arms["pair_table"],
            "control_table": ctl["table"], "balance": bal, "design_weights": weights}
