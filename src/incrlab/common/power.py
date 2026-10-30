"""Power / minimum detectable effect.

Conventions are explicit in every function (one- vs two-sided, which sigma) because mixing
them silently is a classic source of contradictory power tables.

* :func:`did_mde` - closed-form MDE for a geo DiD with ``n_pre`` / ``n_post`` weeks, given the
  weekly sd of the (treat - control) relative gap.
* :func:`weeks_for_power` - smallest post length that reaches the target power for a lift.
* :func:`power_curve` - analytic power for a grid of lifts x weeks.
* :func:`ttest_power` - wrapper over statsmodels ``TTestIndPower`` (effect size in sd units).
* :func:`simulated_power` - placebo/simulation power: inject a lift into random post windows
  of historical data, run the user's estimator, count detections.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _z(alpha: float, two_sided: bool) -> float:
    return stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)


def did_mde(sigma_gap: float, n_post: int, n_pre: Optional[int] = None, alpha: float = 0.10,
            power: float = 0.80, two_sided: bool = False, rho: float = 0.0) -> float:
    """MDE (same units as sigma_gap, e.g. relative gap) for a difference of mean gaps.

    SE = sigma * sqrt(1/n_post + 1/n_pre) * sqrt((1+rho)/(1-rho)) (AR(1) inflation).
    MDE = (z_{alpha} + z_{power}) * SE.
    """
    n_pre = n_post if n_pre is None else n_pre
    infl = np.sqrt((1 + rho) / (1 - rho)) if rho else 1.0
    se = sigma_gap * np.sqrt(1.0 / n_post + 1.0 / n_pre) * infl
    return float((_z(alpha, two_sided) + stats.norm.ppf(power)) * se)


def did_power(lift: float, sigma_gap: float, n_post: int, n_pre: Optional[int] = None, alpha: float = 0.10,
              two_sided: bool = False, rho: float = 0.0) -> float:
    n_pre = n_post if n_pre is None else n_pre
    infl = np.sqrt((1 + rho) / (1 - rho)) if rho else 1.0
    se = sigma_gap * np.sqrt(1.0 / n_post + 1.0 / n_pre) * infl
    zc = _z(alpha, two_sided)
    ncp = abs(lift) / se
    p = stats.norm.sf(zc - ncp)
    if two_sided:
        p += stats.norm.cdf(-zc - ncp)
    return float(p)


def weeks_for_power(lift: float, sigma_gap: float, n_pre: Optional[int] = None, alpha: float = 0.10,
                    power: float = 0.80, two_sided: bool = False, max_weeks: int = 104, rho: float = 0.0) -> Optional[int]:
    for n in range(1, max_weeks + 1):
        if did_power(lift, sigma_gap, n, n_pre if n_pre is not None else n, alpha, two_sided, rho) >= power:
            return n
    return None


def power_curve(sigma_gap: float, lifts: Sequence[float], weeks: Sequence[int], n_pre: Optional[int] = None,
                alpha: float = 0.10, two_sided: bool = False, rho: float = 0.0) -> pd.DataFrame:
    rows = [{"lift": l, "weeks": w, "power": did_power(l, sigma_gap, w, n_pre, alpha, two_sided, rho)}
            for l in lifts for w in weeks]
    return pd.DataFrame(rows)


def ttest_power(effect_size: Optional[float] = None, nobs1: Optional[int] = None, alpha: float = 0.05,
                power: Optional[float] = None, ratio: float = 1.0, alternative: str = "two-sided") -> float:
    """Solve for whichever of effect_size / nobs1 / power is None (statsmodels TTestIndPower)."""
    from statsmodels.stats.power import TTestIndPower

    return float(TTestIndPower().solve_power(effect_size=effect_size, nobs1=nobs1, alpha=alpha, power=power,
                                             ratio=ratio, alternative=alternative))


def simulated_power(series_treat: pd.Series, series_control: pd.Series, lift: float, n_post: int, n_pre: int,
                    estimator: Callable[[pd.Series, pd.Series, list, list], float],
                    threshold: Callable[[float], bool], n_sims: int = 200, seed: int = 0,
                    holdout: bool = True) -> dict:
    """Simulation-based power using historical (pre-test) data only.

    For random start points, take ``n_pre`` + ``n_post`` consecutive weeks, multiply treated post
    weeks by (1 - lift) for a holdout (or 1 + lift for a heavy-up), run ``estimator`` (returns a
    statistic, e.g. a p-value or a lift) and count ``threshold(stat)`` successes. Also returns the
    false-positive rate with no injection (empirical size).
    """
    idx = series_treat.index
    T = len(idx)
    span = n_pre + n_post
    if T < span + 1:
        raise ValueError("history too short for requested windows")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T - span + 1, n_sims)
    hit, fp = 0, 0
    for s in starts:
        pre = list(idx[s:s + n_pre])
        post = list(idx[s + n_pre:s + span])
        t = series_treat.copy()
        fp += bool(threshold(estimator(t, series_control, pre, post)))
        t.loc[post] = t.loc[post] * ((1 - lift) if holdout else (1 + lift))
        hit += bool(threshold(estimator(t, series_control, pre, post)))
    return {"power": hit / n_sims, "empirical_size": fp / n_sims, "n_sims": n_sims, "lift": lift,
            "n_pre": n_pre, "n_post": n_post}


def gap_sigma(treat: pd.Series, control: pd.Series, index: Optional[Sequence] = None, relative: bool = True) -> dict:
    """Weekly sd and AR(1) of the (relative) treat-control gap (inputs to the closed form)."""
    t, c = (treat, control) if index is None else (treat.loc[list(index)], control.loc[list(index)])
    g = ((t - c) / c if relative else (t - c)).to_numpy(float)
    rho = float(np.corrcoef(g[1:], g[:-1])[0, 1]) if len(g) > 3 else 0.0
    return {"sigma": float(np.std(g, ddof=1)), "rho": rho, "n": len(g)}
