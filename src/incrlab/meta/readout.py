"""Live readout for the Social geo holdout: estimates, inference, efficiency, attribution comparison.

Lift convention: one definition everywhere - contribution = (counterfactual - actual) / counterfactual,
positive = volume LOST when Social is paused (the source workflow mixed three conventions and
used abs()). Every estimator below differs only in how the counterfactual is built.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..common import did as cdid
from ..common.synth import apply_weights, fit_synth
from .design import DesignFit, build_unit_panel, donors_for, fit_design_sc, per_capita_counterfactual, wide
from .settings import MetaSettings, get_settings

STAGES = ("qs", "rated", "sales")
STAGE_LABEL = {"qs": "Quote starts", "rated": "Rated quotes", "sales": "Sales"}


# ------------------------------------------------------------------------------------------
# Panels
# ------------------------------------------------------------------------------------------
def readout_unit_panel(county_weekly: pd.DataFrame, county_static: pd.DataFrame, unit: str,
                       st: Optional[MetaSettings] = None) -> pd.DataFrame:
    """Unit panel (treated unit + every single county) for weeks design_start..test_end."""
    st = st or get_settings()
    cw = county_weekly[(county_weekly["region"] == st.region)
                       & county_weekly["week_idx"].between(st.design_start, st.test_end)]
    cs = county_static[county_static["region"] == st.region]
    units = pd.DataFrame([{"unit": unit, "members": unit.split("+")}]) if "+" in unit else pd.DataFrame(
        columns=["unit", "members"])
    return build_unit_panel(cw, cs, units)


def treated_vs_synthetic(panel: pd.DataFrame, unit: str, weights: pd.Series, st: Optional[MetaSettings] = None) -> pd.DataFrame:
    """Weekly frame: actual & locked-weight synthetic (count scale) for each stage + spend + social share."""
    st = st or get_settings()
    out = {}
    for s in STAGES:
        cf = per_capita_counterfactual(panel, unit, weights, s)
        out[f"{s}_actual"] = cf["actual"]
        out[f"{s}_synth"] = cf["synthetic"]
    pc = wide(panel, "spend_social") / wide(panel, "pop")
    out["spend_actual"] = wide(panel, "spend_social")[unit]
    out["spend_synth"] = apply_weights(pc, weights) * wide(panel, "pop")[unit]
    for s in STAGES:
        lt = wide(panel, f"lt_social_{s}")
        out[f"lt_{s}_actual"] = lt[unit]
        out[f"lt_{s}_synth"] = apply_weights(lt / wide(panel, "pop"), weights) * wide(panel, "pop")[unit]
    out["share_actual"] = wide(panel, "social_share_qs")[unit]
    out["share_synth"] = apply_weights(wide(panel, "social_share_qs"), weights)
    df = pd.DataFrame(out)
    df["period"] = np.where(df.index >= st.test_start, "test",
                            np.where(df.index >= st.readout_pre_start, "readout_pre", "history"))
    return df


# ------------------------------------------------------------------------------------------
# Estimators
# ------------------------------------------------------------------------------------------
def _hac_did(long: pd.DataFrame, maxlags: int = 2) -> dict:
    """Same week-FE DiD regression with Newey-West (HAC) SE - weekly gaps are autocorrelated,
    which HC1 ignores (source used HC1 only)."""
    X = pd.get_dummies(long[["week"]].astype(str), drop_first=True, dtype=float)
    X.insert(0, "treated", long["treated"].astype(float).to_numpy())
    X.insert(0, "D", (long["treated"] * long["post"]).astype(float).to_numpy())
    X = sm.add_constant(X, has_constant="add")
    # order rows by week so HAC lags run along time within the stacked panel
    o = np.argsort(long["week"].to_numpy(), kind="stable")
    fit = sm.OLS(long["y"].to_numpy(float)[o], X.to_numpy()[o]).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags * 2, "use_correction": True})
    return {"beta": float(fit.params[1]), "se": float(fit.bse[1]), "p_value": float(fit.pvalues[1])}


def did_week_fe(ts: pd.DataFrame, stage: str, st: Optional[MetaSettings] = None, alpha: Optional[float] = None) -> dict:
    """Readout DiD: y ~ treated + C(week) + treated:post on the stacked (treated, locked synthetic)
    series over readout pre + test weeks, HC1 robust SE (+ HAC robustness). Positive = contribution."""
    st = st or get_settings()
    alpha = st.did_alpha if alpha is None else alpha
    weeks = st.readout_pre_weeks + st.test_weeks
    rows = []
    for tr, col in ((1, f"{stage}_actual"), (0, f"{stage}_synth")):
        for wk in weeks:
            rows.append({"unit": "treated" if tr else "synthetic", "week": wk, "treated": tr,
                         "post": int(wk >= st.test_start), "y": float(ts.loc[wk, col])})
    long = pd.DataFrame(rows)
    r = cdid.did_ols_fe(long, "y", "unit", "week", "treated", "post", cluster=None, alpha=alpha)
    hac = _hac_did(long)
    cf_mean = float(ts.loc[st.test_weeks, f"{stage}_actual"].mean()) - r["beta"]
    se = max(r["se"], hac["se"])
    z = stats.t.ppf(1 - alpha / 2, r["n_obs"] - len(weeks) - 2)
    est = -r["beta"] / cf_mean
    return {"method": "did_week_fe_hc1", "estimate": est, "abs_per_week": -r["beta"], "se_abs": r["se"],
            "se_rel": r["se"] / cf_mean, "p_value": r["p_value"],
            "ci_low": (-r["beta"] - z * r["se"]) / cf_mean, "ci_high": (-r["beta"] + z * r["se"]) / cf_mean,
            "hac_se_rel": hac["se"] / cf_mean, "hac_p_value": hac["p_value"],
            "se_rel_conservative": se / cf_mean, "cf_weekly": cf_mean, "n_obs": r["n_obs"],
            "incremental": -r["beta"] * st.n_test_weeks, "incremental_se": se * st.n_test_weeks}


def ratio_did(ts: pd.DataFrame, stage: str, st: Optional[MetaSettings] = None, seed: int = 0) -> dict:
    """Ratio-counterfactual DiD: CF = treated_pre x (synthetic_post / synthetic_pre), bootstrap CI."""
    st = st or get_settings()
    r = cdid.change_ratio_did(ts[f"{stage}_actual"], ts[f"{stage}_synth"], st.readout_pre_weeks, st.test_weeks,
                              sign=-1, n_boot=st.n_boot, seed=seed, alpha=st.did_alpha)
    return {"method": "ratio_did", "estimate": r["lift"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "se_rel": r["boot_se"], "incremental": r["abs_effect"], "cf_total": r["counterfactual_post_total"],
            "control_ratio": r["control_ratio"]}


def sc_gap(ts: pd.DataFrame, stage: str, st: Optional[MetaSettings] = None) -> dict:
    """Locked-weight synthetic-control gap; level-adjusted with the FULL design history (58 weeks,
    far less noisy than a 7-week pre window)."""
    st = st or get_settings()
    pre = st.design_weeks
    a, s = ts[f"{stage}_actual"], ts[f"{stage}_synth"]
    k = a.loc[pre].sum() / s.loc[pre].sum()
    cf = s * k
    post = st.test_weeks
    est = (cf.loc[post].sum() - a.loc[post].sum()) / cf.loc[post].sum()
    g = (a - cf) / cf
    return {"method": "synthetic_control_gap", "estimate": float(est), "level_adj": float(k),
            "incremental": float(cf.loc[post].sum() - a.loc[post].sum()), "cf_total": float(cf.loc[post].sum()),
            "pre_gap_sd": float(g.loc[pre].std(ddof=1)), "cf_series": cf}


def placebo_in_space(panel: pd.DataFrame, unit: str, county_ids: Sequence[str], dfit: DesignFit,
                     st: Optional[MetaSettings] = None, stage: str = "qs") -> dict:
    """In-space placebo using the SAME design procedure (share-metric SC on the training window,
    locked weights, level-adjusted gap on ``stage``). Each donor county takes a turn as the
    'treated' unit (the real treated unit's counties are never donors). Two statistics:
    |lift| over the test window and post/pre RMSPE ratio of the relative gap."""
    st = st or get_settings()
    pre, post = st.design_weeks, st.test_weeks

    def stats_for(u: str, w: pd.Series) -> dict:
        cf = per_capita_counterfactual(panel, u, w, stage)
        k = cf.loc[pre, "actual"].sum() / cf.loc[pre, "synthetic"].sum()
        c = cf["synthetic"] * k
        g = (cf["actual"] - c) / c
        lift = float((c.loc[post].sum() - cf.loc[post, "actual"].sum()) / c.loc[post].sum())
        ratio = float(np.sqrt(np.mean(g.loc[post] ** 2)) / np.sqrt(np.mean(g.loc[pre] ** 2)))
        return {"lift": lift, "rmspe_ratio": ratio, "pre_gap_sd": float(g.loc[pre].std(ddof=1))}

    real = stats_for(unit, dfit.weights)
    rows = []
    pool = dfit.donors
    for d in pool:
        f = fit_design_sc(panel, d, county_ids, st, donors=[c for c in pool if c != d])
        rows.append({"unit": d, **stats_for(d, f.weights)})
    pl = pd.DataFrame(rows)
    n = len(pl)
    p_lift = (1 + int((pl["lift"] >= real["lift"]).sum())) / (n + 1)          # one-sided (holdout)
    p_lift2 = (1 + int((pl["lift"].abs() >= abs(real["lift"])).sum())) / (n + 1)
    p_ratio = (1 + int((pl["rmspe_ratio"] >= real["rmspe_ratio"]).sum())) / (n + 1)
    return {"treated": real, "placebos": pl, "n_placebos": n, "p_value_lift_one_sided": p_lift,
            "p_value_lift_two_sided": p_lift2, "p_value_rmspe_ratio": p_ratio, "min_attainable_p": 1 / (n + 1),
            "placebo_lift_sd": float(pl["lift"].std(ddof=1))}


# ------------------------------------------------------------------------------------------
# Efficiency + attribution
# ------------------------------------------------------------------------------------------
def spend_removed(ts: pd.DataFrame, st: Optional[MetaSettings] = None) -> dict:
    """Spend not delivered in the treated unit during the test = counterfactual spend - actual.
    County-level spend is observed here (the source had to allocate state spend by population);
    the counterfactual spend is the locked-weight synthetic, level-adjusted on the design history."""
    st = st or get_settings()
    pre, post = st.design_weeks, st.test_weeks
    k = ts.loc[pre, "spend_actual"].sum() / ts.loc[pre, "spend_synth"].sum()
    cf = float(ts.loc[post, "spend_synth"].sum() * k)
    act = float(ts.loc[post, "spend_actual"].sum())
    return {"cf_spend": cf, "actual_spend": act, "spend_removed": cf - act,
            "delivered_share": act / cf if cf else np.nan}


def attribution_comparison(ts: pd.DataFrame, est: Dict[str, dict], spend: dict, st: Optional[MetaSettings] = None) -> pd.DataFrame:
    """Last-touch platform view vs measured incrementality, per stage.

    * LT share = Social last-touch volume / total volume in the readout pre-period (what the
      platform would claim for full delivery).
    * LT cost-per = BAU spend / LT volume (readout pre-period run-rate).
    * Incremental cost-per = spend removed / incremental volume (test window).
    * Over-claim ratio = LT-claimed volume per $ / incremental volume per $ (>1 over-claims).
    """
    st = st or get_settings()
    pre = st.readout_pre_weeks
    spend_pre = float(ts.loc[pre, "spend_actual"].sum())
    rows = []
    for s in STAGES:
        lt = float(ts.loc[pre, f"lt_{s}_actual"].sum())
        tot = float(ts.loc[pre, f"{s}_actual"].sum())
        inc = est[s]["incremental"]
        inc_per_k = 1000 * inc / spend["spend_removed"] if spend["spend_removed"] > 0 else np.nan
        lt_per_k = 1000 * lt / spend_pre
        rows.append({"stage": s, "lt_share": lt / tot, "measured_lift": est[s]["estimate"],
                     "lt_cost_per": spend_pre / lt if lt else np.nan,
                     "incremental_cost_per": spend["spend_removed"] / inc if inc > 0 else np.nan,
                     "lt_per_1k": lt_per_k, "incremental_per_1k": inc_per_k,
                     "overclaim_ratio": lt_per_k / inc_per_k if inc_per_k and inc_per_k > 0 else np.nan})
    return pd.DataFrame(rows)


def verdict(primary: Dict[str, dict], placebo: dict, mde: float, st: Optional[MetaSettings] = None) -> dict:
    """Honest verdict: 'significant' only if the primary p-value AND placebo p-value clear alpha and
    the effect exceeds the design MDE; otherwise 'directional' (sign consistent, underpowered) or
    'inconclusive'."""
    st = st or get_settings()
    q = primary["qs"]
    pos = all(primary[s]["estimate"] > 0 for s in STAGES)
    sig = q["p_value"] < st.directional_p and placebo["p_value_lift_one_sided"] < st.directional_p
    if sig and q["estimate"] >= mde:
        label = "significant"
    elif q["estimate"] > 0 and pos:
        label = "directional"
    elif q["estimate"] > 0:
        label = "directional (mixed funnel)"
    else:
        label = "inconclusive"
    return {"label": label, "underpowered": bool(q["estimate"] < mde), "design_mde_8wk": mde,
            "all_stages_positive": bool(pos)}
