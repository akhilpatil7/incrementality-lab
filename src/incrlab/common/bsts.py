"""CausalImpact-style counterfactual with statsmodels UnobservedComponents.

Fit a local-level (optionally + trend / seasonal) structural time-series on the PRE period
with control series as regressors, forecast the POST period, and compare. Uncertainty for
cumulative effects comes from simulating the predictive distribution (``n_sims`` draws of
the state-space forecast via ``simulate``; parameters are themselves drawn from their
asymptotic normal so coefficient uncertainty is included), which keeps the serial
correlation of forecast errors - unlike summing pointwise intervals.

Significance is decided ONLY by whether the interval for the cumulative relative effect
excludes 0 (a lesson from ports that print a misleading p-value). A tail-probability
``p_like`` is reported for reference, computed from the same simulated draws.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


def causal_impact(y: pd.Series, X: Optional[pd.DataFrame], pre_index: Sequence, post_index: Sequence,
                  level: str = "llevel", seasonal: Optional[int] = None, alpha: float = 0.05,
                  n_sims: int = 1000, seed: int = 0, sign: int = 1, standardize: bool = True,
                  param_draws: int = 50) -> dict:
    """Return point/cumulative effects with simulated intervals.

    y, X are time-indexed (same index). Effects are ``sign * (actual - counterfactual)``.
    """
    pre, post = list(pre_index), list(post_index)
    yv = y.astype(float)
    mu, sd = (yv.loc[pre].mean(), yv.loc[pre].std(ddof=0) or 1.0) if standardize else (0.0, 1.0)
    ys = (yv - mu) / sd
    Xs = None
    if X is not None and X.shape[1] > 0:
        Xf = X.astype(float)
        xm, xsd = Xf.loc[pre].mean(), Xf.loc[pre].std(ddof=0).replace(0, 1.0)
        Xs = (Xf - xm) / xsd
    kw = dict(level=level)
    if seasonal:
        kw["seasonal"] = seasonal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = UnobservedComponents(ys.loc[pre].to_numpy(), exog=None if Xs is None else Xs.loc[pre].to_numpy(), **kw)
        fit = mod.fit(disp=False, maxiter=500)
        fc = fit.get_forecast(steps=len(post), exog=None if Xs is None else Xs.loc[post].to_numpy())
        pred_mean = np.asarray(fc.predicted_mean) * sd + mu
        ci = np.asarray(fc.conf_int(alpha=alpha)) * sd + mu
        rng = np.random.default_rng(seed)
        exog_post = None if Xs is None else Xs.loc[post].to_numpy()
        # parameter uncertainty: draw params from the asymptotic normal (posterior-ish), then
        # simulate the state-space forecast for each draw (keeps serial correlation of errors)
        n_param = max(1, min(param_draws, n_sims))
        reps = int(np.ceil(n_sims / n_param))
        cov = np.asarray(fit.cov_params())
        ok = np.all(np.isfinite(cov)) and param_draws > 1
        sims_list = []
        for j in range(n_param):
            res_j = fit
            if ok and j > 0:
                pj = rng.multivariate_normal(np.asarray(fit.params), cov, method="eigh")
                pj = np.where(np.char.startswith(np.asarray(fit.model.param_names, dtype=str), "sigma2"), np.abs(pj), pj)
                try:
                    res_j = mod.smooth(pj)
                except Exception:  # pragma: no cover - fall back to the MLE
                    res_j = fit
            try:
                raw = res_j.simulate(len(post), anchor="end", repetitions=reps, exog=exog_post, rng=rng)
            except TypeError:  # older statsmodels
                raw = res_j.simulate(len(post), anchor="end", repetitions=reps, exog=exog_post, random_state=rng)
            sims_list.append(np.asarray(raw).reshape(len(post), -1).T)
        sims = np.vstack(sims_list)[:n_sims] * sd + mu  # (n_sims, n_post)
    actual = yv.loc[post].to_numpy()
    pointwise = pd.DataFrame({"actual": actual, "predicted": pred_mean, "pred_low": ci[:, 0], "pred_high": ci[:, 1],
                              "effect": sign * (actual - pred_mean)}, index=pd.Index(post, name=y.index.name))
    cum_actual = actual.sum()
    cum_cf = sims.sum(1)
    rel_draws = sign * (cum_actual - cum_cf) / cum_cf
    abs_draws = sign * (cum_actual - cum_cf)
    lo, hi = np.quantile(rel_draws, [alpha / 2, 1 - alpha / 2])
    alo, ahi = np.quantile(abs_draws, [alpha / 2, 1 - alpha / 2])
    point_rel = sign * (cum_actual - pred_mean.sum()) / pred_mean.sum()
    p_like = float(min(1.0, 2 * min((rel_draws <= 0).mean(), (rel_draws >= 0).mean())))
    in_sample = np.asarray(fit.fittedvalues) * sd + mu
    return {
        "rel_effect": float(point_rel), "rel_ci_low": float(lo), "rel_ci_high": float(hi),
        "abs_effect": float(sign * (cum_actual - pred_mean.sum())), "abs_ci_low": float(alo), "abs_ci_high": float(ahi),
        "significant": bool(lo > 0 or hi < 0), "ci_level": 1 - alpha, "p_like": p_like,
        "pre_mape": float(np.mean(np.abs(in_sample[1:] - yv.loc[pre].to_numpy()[1:]) / np.maximum(np.abs(yv.loc[pre].to_numpy()[1:]), 1e-9))),
        "pointwise": pointwise, "model": f"UnobservedComponents({level}{', seasonal=' + str(seasonal) if seasonal else ''}, k_exog={0 if Xs is None else Xs.shape[1]})",
        "aic": float(fit.aic),
    }
