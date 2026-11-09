"""Power analysis for the geoselect treatment/control pair - thin wrapper over
``incrlab.common.power`` (same closed-form MDE / power-curve conventions as meta/search/dsp).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..common import power as cpower
from .settings import GeoselectSettings, get_settings


def estimate_daily_noise(treat_kpi: float, control_kpi: float, treat_pop: float, control_pop: float,
                         st: Optional[GeoselectSettings] = None, seed: int = 0) -> dict:
    """Simulate pre-period daily (treat - control) relative-gap volatility from population-driven
    volume + the configured daily noise sd (no real pre-period panel exists for a brand-new pair,
    so sigma/rho are derived analytically from the same noise model :mod:`run_test` uses)."""
    st = st or get_settings()
    rng = np.random.default_rng(seed)
    n = int(st.pre_period_days)
    t_mean = treat_pop / 1000.0 * treat_kpi / 7.0   # daily conversions
    c_mean = control_pop / 1000.0 * control_kpi / 7.0
    dow = 1 + st.day_of_week_amp * np.sin(2 * np.pi * np.arange(n) / 7.0)
    t = t_mean * dow * np.exp(rng.normal(0, st.daily_noise_sd, n))
    c = c_mean * dow * np.exp(rng.normal(0, st.daily_noise_sd, n))
    gap = (t - c) / c
    rho = float(np.corrcoef(gap[1:], gap[:-1])[0, 1]) if n > 3 else 0.0
    return {"sigma": float(np.std(gap, ddof=1)), "rho": rho, "n": n,
            "treat_daily_mean": float(t_mean), "control_daily_mean": float(c_mean)}


def power_table(sigma: float, rho: float, st: Optional[GeoselectSettings] = None) -> pd.DataFrame:
    """Power x (lift, weeks) grid, closed-form AR(1)-inflated DiD, same convention as common/power."""
    st = st or get_settings()
    rows = []
    for L in st.power_lifts:
        for wk in st.power_weeks_grid:
            n_post = wk  # weekly periods for the closed-form; daily data aggregated to weekly cells
            p = cpower.did_power(L, sigma, n_post, n_pre=int(round(st.pre_period_weeks)), alpha=st.alpha,
                                 two_sided=False, rho=rho)
            rows.append({"lift": L, "weeks": wk, "power": p})
    return pd.DataFrame(rows)


def weeks_for_power(sigma: float, rho: float, lift: float, st: Optional[GeoselectSettings] = None) -> Optional[int]:
    st = st or get_settings()
    return cpower.weeks_for_power(lift, sigma, n_pre=int(round(st.pre_period_weeks)), alpha=st.alpha,
                                  power=st.power_target, two_sided=False, max_weeks=104, rho=rho)


def mde_for_design(sigma: float, rho: float, st: Optional[GeoselectSettings] = None) -> float:
    st = st or get_settings()
    return cpower.did_mde(sigma, st.test_weeks, n_pre=int(round(st.pre_period_weeks)), alpha=st.alpha,
                          power=st.power_target, two_sided=False, rho=rho)
