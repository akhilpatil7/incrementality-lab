"""Simulated test execution + sequential monitoring for the geoselect matched pair.

This stage is explicitly SYNTHETIC (per task spec): we generate daily outcome counts for the
real treatment and control markets using their real-population-scaled baseline KPI rate, inject
a known TRUE lift into the treatment market over the test window, and read it out with the same
DiD machinery as the rest of the repo (``incrlab.common.did``). A weekly "monitoring" loop takes
an interim DiD look after each completed week of the test so a sequential-monitoring chart can
be drawn (cumulative lift + narrowing CI), matching the task's "running the test and monitoring"
requirement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..common import did as cdid
from .settings import GeoselectSettings, get_settings


def _daily_series(mean_daily: float, n_days: int, st: GeoselectSettings, rng: np.random.Generator,
                  lift_from_day: Optional[int] = None, true_lift: float = 0.0) -> np.ndarray:
    dow = 1 + st.day_of_week_amp * np.sin(2 * np.pi * np.arange(n_days) / 7.0)
    trend = 1 + 0.0005 * np.arange(n_days)  # mild common drift, same for treat/control
    mult = np.ones(n_days)
    if lift_from_day is not None:
        mult[lift_from_day:] *= (1 + true_lift)
    mu = mean_daily * dow * trend * mult
    y = rng.poisson(mu * np.exp(rng.normal(0, st.daily_noise_sd, n_days) - st.daily_noise_sd ** 2 / 2))
    return y.astype(float)


def simulate_test(pair: dict, st: Optional[GeoselectSettings] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """Daily panel for pre-period + test window, treatment and control, with the true lift
    injected into the treatment market starting at day 0 of the test window."""
    st = st or get_settings()
    rng = np.random.default_rng(seed if seed is not None else st.seed)
    n_pre, n_test = st.pre_period_days, st.test_days
    n = n_pre + n_test

    t_pop, c_pop = pair["treatment"]["pop"], pair["control"]["pop"]
    t_kpi = pair["feature_comparison"]["baseline_kpi_rate"]["treatment"]
    c_kpi = pair["feature_comparison"]["baseline_kpi_rate"]["control"]
    t_daily_mean = t_pop / 1000.0 * t_kpi / 7.0
    c_daily_mean = c_pop / 1000.0 * c_kpi / 7.0

    t = _daily_series(t_daily_mean, n, st, rng, lift_from_day=n_pre, true_lift=st.true_lift)
    c = _daily_series(c_daily_mean, n, st, rng, lift_from_day=None, true_lift=0.0)

    day_idx = np.arange(n)
    phase = np.where(day_idx < n_pre, "pre", "test")
    df = pd.DataFrame({"day_idx": day_idx, "phase": phase, "treat_conversions": t, "control_conversions": c})
    df["week_idx"] = df["day_idx"] // 7
    df.attrs["n_pre_days"] = n_pre
    df.attrs["n_test_days"] = n_test
    df.attrs["true_lift"] = st.true_lift
    return df


def weekly_panel(daily: pd.DataFrame) -> pd.DataFrame:
    w = daily.groupby("week_idx").agg(treat_conversions=("treat_conversions", "sum"),
                                      control_conversions=("control_conversions", "sum"),
                                      phase=("phase", "first")).reset_index()
    return w


def monitoring_looks(weekly: pd.DataFrame, st: Optional[GeoselectSettings] = None) -> pd.DataFrame:
    """Sequential monitoring: after each completed test week, run change-ratio DiD on
    pre-period weeks vs test weeks seen so far. Returns one row per interim look."""
    st = st or get_settings()
    pre = weekly[weekly["phase"] == "pre"].set_index("week_idx")
    test_weeks = weekly[weekly["phase"] == "test"]["week_idx"].tolist()
    rows = []
    for k in range(1, len(test_weeks) + 1):
        seen = test_weeks[:k]
        post = weekly[weekly["week_idx"].isin(seen)].set_index("week_idx")
        treat = pd.concat([pre["treat_conversions"], post["treat_conversions"]])
        control = pd.concat([pre["control_conversions"], post["control_conversions"]])
        res = cdid.change_ratio_did(treat, control, pre.index.tolist(), post.index.tolist(),
                                    sign=1, n_boot=st.n_boot, seed=st.seed, alpha=1 - st.did_alpha)
        rows.append({"look": k, "weeks_elapsed": k, "n_days_elapsed": k * 7,
                    "lift": res["lift"], "ci_low": res.get("ci_low"), "ci_high": res.get("ci_high"),
                    "ci_width": (res["ci_high"] - res["ci_low"]) if res.get("ci_low") is not None else None})
    return pd.DataFrame(rows)


def final_readout(weekly: pd.DataFrame, st: Optional[GeoselectSettings] = None) -> dict:
    """Final DiD readout (full test window) + comparison to ground-truth injected lift."""
    st = st or get_settings()
    pre = weekly[weekly["phase"] == "pre"].set_index("week_idx")
    post = weekly[weekly["phase"] == "test"].set_index("week_idx")
    treat = pd.concat([pre["treat_conversions"], post["treat_conversions"]])
    control = pd.concat([pre["control_conversions"], post["control_conversions"]])
    res = cdid.change_ratio_did(treat, control, pre.index.tolist(), post.index.tolist(),
                                sign=1, n_boot=st.n_boot, seed=st.seed, alpha=1 - st.did_alpha)
    fe = cdid.did_2x2_ratio(treat, control, pre.index.tolist(), post.index.tolist(), sign=1)
    res.update({"true_lift": st.true_lift, "method": "change_ratio_did",
               "ci_covers_truth": bool(res.get("ci_low", -np.inf) <= st.true_lift <= res.get("ci_high", np.inf))
                                 if res.get("ci_low") is not None else None,
               "lift_2x2_ratio": fe["lift_ratio"], "lift_2x2_additive": fe["lift_additive"]})
    return res
