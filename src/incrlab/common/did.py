"""Difference-in-differences estimators.

* :func:`did_2x2_ratio` - classic 2x2 on period means, reported in levels and as a relative lift.
* :func:`change_ratio_did` - counterfactual = treated pre-total x (control post/pre ratio); the
  readout form used for geo holdouts. Bootstrap CI over weeks optional.
* :func:`did_ols_fe` - panel OLS with unit + week fixed effects and cluster-robust SE (by unit
  by default, falling back to HC1 when there are too few clusters).

Sign: ``lift`` = (actual - counterfactual)/counterfactual. For a holdout the channel
contribution is ``-lift`` (``sign=-1`` flips it).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def did_2x2_ratio(treat: pd.Series, control: pd.Series, pre_index: Sequence, post_index: Sequence,
                  sign: int = 1) -> dict:
    """2x2 DiD on weekly means of a treated and a control series (time-indexed)."""
    pre, post = list(pre_index), list(post_index)
    t0, t1 = float(treat.loc[pre].mean()), float(treat.loc[post].mean())
    c0, c1 = float(control.loc[pre].mean()), float(control.loc[post].mean())
    did_abs = (t1 - t0) - (c1 - c0)
    cf_level = t0 + (c1 - c0)  # additive counterfactual
    ratio_cf = t0 * (c1 / c0) if c0 else np.nan  # multiplicative counterfactual
    return {"treat_pre": t0, "treat_post": t1, "control_pre": c0, "control_post": c1,
            "did_abs_per_week": sign * did_abs, "lift_additive": sign * did_abs / cf_level if cf_level else np.nan,
            "lift_ratio": sign * (t1 - ratio_cf) / ratio_cf if ratio_cf else np.nan,
            "treat_change_pct": t1 / t0 - 1 if t0 else np.nan, "control_change_pct": c1 / c0 - 1 if c0 else np.nan}


def change_ratio_did(treat: pd.Series, control: pd.Series, pre_index: Sequence, post_index: Sequence,
                     sign: int = 1, n_boot: int = 0, seed: int = 0, alpha: float = 0.10) -> dict:
    """Counterfactual_post = treat_pre_sum x (control_post_sum / control_pre_sum) scaled to post length.

    Uses per-week means so pre/post windows may differ in length. If ``n_boot`` > 0, a
    week-level (block-free) bootstrap within pre and post gives a percentile CI on the lift.
    """
    pre, post = list(pre_index), list(post_index)

    def est(tp, tq, cp, cq):
        cf = tp.mean() * (cq.mean() / cp.mean())
        return (tq.mean() - cf) / cf, cf

    tp, tq = treat.loc[pre].to_numpy(float), treat.loc[post].to_numpy(float)
    cp, cq = control.loc[pre].to_numpy(float), control.loc[post].to_numpy(float)
    lift, cf_wk = est(tp, tq, cp, cq)
    out = {"lift": sign * lift, "counterfactual_post_total": cf_wk * len(post),
           "actual_post_total": float(tq.sum()), "abs_effect": sign * (tq.sum() - cf_wk * len(post)),
           "control_ratio": cq.mean() / cp.mean(), "n_pre": len(pre), "n_post": len(post)}
    if n_boot:
        rng = np.random.default_rng(seed)
        ip = rng.integers(0, len(pre), (n_boot, len(pre)))
        iq = rng.integers(0, len(post), (n_boot, len(post)))
        b = np.array([est(tp[a], tq[c], cp[a], cq[c])[0] for a, c in zip(ip, iq)]) * sign
        lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
        out.update({"ci_low": float(lo), "ci_high": float(hi), "ci_level": 1 - alpha,
                    "boot_se": float(b.std(ddof=1))})
    return out


def did_ols_fe(panel: pd.DataFrame, y: str, unit: str, time: str, treated_col: str, post_col: str,
               cluster: Optional[str] = "unit", log: bool = False, weights: Optional[str] = None,
               alpha: float = 0.05) -> dict:
    """OLS DiD with unit + time fixed effects: y_it = a_i + g_t + beta * (treated_i x post_t) + e.

    cluster: "unit" (default) for cluster-robust SE by unit; "time"; None -> HC1. With < 6
    clusters the cluster-robust SE is unreliable; we report HC1 and flag it.
    ``log=True`` models log(1+y) so beta ~ relative lift. Returns relative lift computed
    from beta / mean counterfactual level of treated post cells when log=False.
    """
    d = panel[[y, unit, time, treated_col, post_col] + ([weights] if weights else [])].copy()
    d["_D"] = (d[treated_col].astype(int) * d[post_col].astype(int)).astype(float)
    yv = np.log1p(d[y].astype(float)) if log else d[y].astype(float)
    X = pd.get_dummies(d[[unit, time]].astype(str), drop_first=True, dtype=float)
    X.insert(0, "_D", d["_D"].to_numpy())
    X = sm.add_constant(X, has_constant="add")
    mod = sm.WLS(yv.to_numpy(), X.to_numpy(), weights=d[weights].to_numpy() if weights else 1.0)
    n_cl = d[unit].nunique() if cluster == "unit" else (d[time].nunique() if cluster == "time" else 0)
    se_type = "HC1"
    if cluster in ("unit", "time") and n_cl >= 6:
        groups = pd.factorize(d[unit if cluster == "unit" else time])[0]
        fit = mod.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
        se_type = f"cluster({cluster}, G={n_cl})"
        dof = n_cl - 1
    else:
        fit = mod.fit(cov_type="HC1")
        dof = fit.df_resid
    beta, se = float(fit.params[1]), float(fit.bse[1])
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    p = float(2 * stats.t.sf(abs(beta / se), dof)) if se > 0 else np.nan
    treated_post = d["_D"] == 1
    if log:
        rel = np.expm1(beta)
        rel_ci = (np.expm1(beta - tcrit * se), np.expm1(beta + tcrit * se))
    else:
        base = float(d.loc[treated_post, y].mean() - beta)
        rel = beta / base if base else np.nan
        rel_ci = ((beta - tcrit * se) / base, (beta + tcrit * se) / base) if base else (np.nan, np.nan)
    return {"beta": beta, "se": se, "t": beta / se if se else np.nan, "p_value": p,
            "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se, "rel_lift": float(rel),
            "rel_ci_low": float(rel_ci[0]), "rel_ci_high": float(rel_ci[1]), "se_type": se_type,
            "n_obs": int(len(d)), "n_units": int(d[unit].nunique()), "n_periods": int(d[time].nunique()),
            "few_clusters_warning": bool(cluster is not None and n_cl < 6), "log": log}


def welch_gap_test(treat: pd.Series, control: pd.Series, pre_index: Sequence, post_index: Sequence,
                   relative: bool = True) -> dict:
    """Welch t-test on weekly (treat - control) gaps, pre vs post (optionally relative gaps)."""
    gap = (treat - control) / control if relative else (treat - control)
    a, b = gap.loc[list(pre_index)].to_numpy(float), gap.loc[list(post_index)].to_numpy(float)
    t, p = stats.ttest_ind(b, a, equal_var=False)
    return {"pre_mean_gap": float(a.mean()), "post_mean_gap": float(b.mean()), "diff": float(b.mean() - a.mean()),
            "t": float(t), "p_value": float(p), "n_pre": len(a), "n_post": len(b)}
