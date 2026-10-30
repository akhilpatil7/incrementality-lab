"""Placebo / permutation inference and jackknife stability for synthetic-control designs.

* :func:`placebo_in_space` - refit the SC treating each donor as if treated (with the real
  treated unit removed from its donor pool), compare the post/pre RMSPE ratio (or the post
  effect) of the real unit to the placebo distribution. Minimum attainable p = 1/(J+1); it is
  returned so readouts can state it honestly.
* :func:`placebo_in_time` - pretend the intervention happened at earlier dates (pre data
  only) and build the distribution of "effects" with no treatment.
* :func:`jackknife_donors` - leave-one-donor-out refits; spread of the estimate shows
  dependence on any single donor.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .synth import apply_weights, fit_synth, lift_from_series


def _ratio(actual, synth, pre, post):
    e_pre = np.sqrt(np.mean((actual.loc[pre] - synth.loc[pre]) ** 2))
    e_post = np.sqrt(np.mean((actual.loc[post] - synth.loc[post]) ** 2))
    return float(e_post / max(e_pre, 1e-12))


def placebo_in_space(wide: pd.DataFrame, treated: str, pre_index: Sequence, post_index: Sequence,
                     donors: Optional[Sequence[str]] = None, stat: str = "rmspe_ratio",
                     max_pre_rmspe_mult: Optional[float] = None, fit_kw: Optional[dict] = None) -> dict:
    """In-space placebo test. ``stat``: 'rmspe_ratio' (Abadie) or 'lift' (|relative post effect|).

    ``max_pre_rmspe_mult`` drops placebos whose pre-RMSPE exceeds that multiple of the treated
    unit's (poor fits inflate the placebo distribution).
    """
    fit_kw = fit_kw or {}
    pre, post = list(pre_index), list(post_index)
    donors = [c for c in (donors if donors is not None else wide.columns) if c != treated]
    real = fit_synth(wide, treated, pre, donors=donors, **fit_kw)

    def s_of(res):
        if stat == "lift":
            return abs(lift_from_series(res.actual, res.synthetic, post)["lift"])
        return _ratio(res.actual, res.synthetic, pre, post)

    real_s = s_of(real)
    rows = []
    for d in donors:
        pool = [c for c in donors if c != d]
        r = fit_synth(wide, d, pre, donors=pool, **fit_kw)
        rows.append({"unit": d, "stat": s_of(r), "pre_rmspe": r.pre_rmspe,
                     "lift": lift_from_series(r.actual, r.synthetic, post)["lift"]})
    pl = pd.DataFrame(rows)
    if max_pre_rmspe_mult is not None and len(pl):
        pl = pl[pl["pre_rmspe"] <= max_pre_rmspe_mult * real.pre_rmspe]
    n = len(pl)
    p = (1 + int((pl["stat"] >= real_s).sum())) / (n + 1)
    return {"stat": stat, "treated_stat": real_s, "p_value": float(p), "n_placebos": n,
            "min_attainable_p": 1.0 / (n + 1), "placebos": pl, "treated_fit": real}


def placebo_in_time(wide: pd.DataFrame, treated: str, pre_index: Sequence, n_post: int, n_fit_min: int = 8,
                    donors: Optional[Sequence[str]] = None, fit_kw: Optional[dict] = None,
                    actual_lift: Optional[float] = None) -> dict:
    """Fake launch dates within the pre-period: fit on weeks before, measure 'lift' over next n_post."""
    fit_kw = fit_kw or {}
    pre = list(pre_index)
    rows = []
    for k in range(n_fit_min, len(pre) - n_post + 1):
        fpre, fpost = pre[:k], pre[k:k + n_post]
        r = fit_synth(wide.loc[pre], treated, fpre, donors=donors, **fit_kw)
        rows.append({"fake_start": pre[k], "lift": lift_from_series(r.actual, r.synthetic, fpost)["lift"],
                     "pre_rmspe": r.pre_rmspe})
    pl = pd.DataFrame(rows)
    out = {"placebos": pl, "n_placebos": len(pl), "sd_placebo_lift": float(pl["lift"].std(ddof=1)) if len(pl) > 1 else np.nan}
    if actual_lift is not None and len(pl):
        out["p_value"] = float((1 + (pl["lift"].abs() >= abs(actual_lift)).sum()) / (len(pl) + 1))
    return out


def jackknife_donors(wide: pd.DataFrame, treated: str, pre_index: Sequence, post_index: Sequence,
                     donors: Optional[Sequence[str]] = None, only_active: bool = True,
                     fit_kw: Optional[dict] = None, sign: int = 1) -> dict:
    """Leave-one-donor-out: refit without each (active) donor; report lift spread and fit change."""
    fit_kw = fit_kw or {}
    pre, post = list(pre_index), list(post_index)
    donors = [c for c in (donors if donors is not None else wide.columns) if c != treated]
    full = fit_synth(wide, treated, pre, donors=donors, **fit_kw)
    base = full.lift(post, sign=sign)["lift"]
    drop = [d for d in donors if (full.weights[d] > 0 or not only_active)]
    rows = []
    for d in drop:
        r = fit_synth(wide, treated, pre, donors=[c for c in donors if c != d], **fit_kw)
        rows.append({"dropped": d, "dropped_weight": float(full.weights[d]), "lift": r.lift(post, sign=sign)["lift"],
                     "pre_rmspe": r.pre_rmspe, "rmspe_change_pct": r.pre_rmspe / full.pre_rmspe - 1})
    jk = pd.DataFrame(rows)
    n = len(jk)
    se = float(np.sqrt((n - 1) / n * np.sum((jk["lift"] - jk["lift"].mean()) ** 2))) if n > 1 else np.nan
    return {"full_lift": base, "jackknife": jk, "jackknife_se": se,
            "lift_range": (float(jk["lift"].min()), float(jk["lift"].max())) if n else (np.nan, np.nan),
            "full_fit": full}


def permutation_pvalue(observed: float, null_draws: Sequence[float], two_sided: bool = True) -> float:
    a = np.asarray(null_draws, float)
    if two_sided:
        return float((1 + np.sum(np.abs(a) >= abs(observed))) / (len(a) + 1))
    return float((1 + np.sum(a >= observed)) / (len(a) + 1))
