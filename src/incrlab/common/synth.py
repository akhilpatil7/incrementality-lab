"""Synthetic control: own SLSQP implementation (+ optional pysyncon wrapper).

Conventions
-----------
* ``wide``: DataFrame indexed by time (week_idx), one column per unit, values = outcome.
* Weights are fit on the pre-period only and then LOCKED (``apply_weights``) for any
  post-period / injected-lift / readout use - never refit on post data.
* Lift is reported as ``(actual - synthetic) / synthetic`` by default (``sign=+1``). For a
  holdout, the channel contribution is ``-lift``; pass ``sign=-1`` to get it positive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

WEIGHT_EPS = 1e-4


@dataclass
class SynthResult:
    treated: str
    donors: List[str]
    weights: pd.Series  # index = donors, sums to 1, >= 0
    v_weights: Optional[pd.Series]
    pre_index: pd.Index
    synthetic: pd.Series  # full-period synthetic series (locked weights)
    actual: pd.Series
    pre_rmspe: float
    pre_mspe: float
    rel_rmse: float  # pre RMSPE / mean(actual pre)
    method: str = "slsqp"
    extra: dict = field(default_factory=dict)

    @property
    def gap(self) -> pd.Series:
        return self.actual - self.synthetic

    @property
    def n_donors_used(self) -> int:
        return int((self.weights > WEIGHT_EPS).sum())

    @property
    def max_weight(self) -> float:
        return float(self.weights.max())

    @property
    def effective_n(self) -> float:
        w = self.weights.to_numpy()
        return float(1.0 / np.sum(w ** 2))

    def lift(self, post_index: Sequence, sign: int = 1) -> dict:
        return lift_from_series(self.actual, self.synthetic, post_index, sign=sign,
                                pre_index=self.pre_index)


def _simplex_ls(X: np.ndarray, y: np.ndarray, w0: Optional[np.ndarray] = None,
                ridge: float = 0.0) -> np.ndarray:
    """min ||y - X w||^2 + ridge ||w||^2  s.t. w >= 0, sum w = 1  (SLSQP)."""
    J = X.shape[1]
    if J == 1:
        return np.ones(1)
    scale = max(np.mean(y ** 2), 1e-12)
    w0 = np.full(J, 1.0 / J) if w0 is None else w0
    XtX = X.T @ X / scale
    Xty = X.T @ y / scale

    def f(w):
        r = y - X @ w
        return float(r @ r) / scale + ridge * float(w @ w)

    def g(w):
        return 2 * (XtX @ w - Xty) + 2 * ridge * w

    res = minimize(f, w0, jac=g, method="SLSQP", bounds=[(0.0, 1.0)] * J,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(J)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    w[w < WEIGHT_EPS] = 0.0
    return w / w.sum() if w.sum() > 0 else np.full(J, 1.0 / J)


def fit_synth(wide: pd.DataFrame, treated: str, pre_index: Sequence, donors: Optional[Sequence[str]] = None,
              predictors: Optional[pd.DataFrame] = None, v: Optional[Sequence[float]] = None,
              optimize_v: bool = False, ridge: float = 0.0, standardize_predictors: bool = True) -> SynthResult:
    """Fit synthetic-control weights on the pre-period.

    Parameters
    ----------
    wide : time x unit outcome matrix.
    treated : treated column name.
    pre_index : index labels of the fitting window.
    donors : donor columns (default: all other columns).
    predictors : optional unit x feature table (rows = units incl. treated). When given, the
        loss is ``||X1 - X0 w||_V^2`` over [pre-period outcomes, predictors]; ``v`` gives
        predictor importance (outcome rows get weight 1). With ``optimize_v`` the classic
        nested problem is solved (V chosen to minimise pre-period outcome MSPE).
    ridge : L2 penalty on weights (spreads weight, AugSynth-flavoured regularisation).
    """
    donors = [c for c in (donors if donors is not None else wide.columns) if c != treated]
    pre = wide.loc[list(pre_index)]
    y = pre[treated].to_numpy(float)
    Y0 = pre[donors].to_numpy(float)
    v_out = None
    if predictors is None:
        w = _simplex_ls(Y0, y, ridge=ridge)
    else:
        P = predictors.loc[[treated] + donors].astype(float)
        if standardize_predictors:
            sd = P.std(ddof=0).replace(0, 1.0)
            P = (P - P.mean()) / sd
        p1 = P.loc[treated].to_numpy()
        p0 = P.loc[donors].to_numpy().T  # features x donors
        ysd = max(np.std(y), 1e-12)

        def solve(vv):
            sv = np.sqrt(np.asarray(vv, float))
            X = np.vstack([Y0 / ysd, (p0 * sv[:, None])])
            t = np.concatenate([y / ysd, p1 * sv])
            return _simplex_ls(X, t, ridge=ridge)

        k = P.shape[1]
        if optimize_v:
            def outer(theta):
                vv = np.exp(theta)
                vv = vv / vv.sum() * k
                ww = solve(vv)
                r = y - Y0 @ ww
                return float(r @ r)
            best = minimize(outer, np.zeros(k), method="Nelder-Mead",
                            options={"maxiter": 200 * k, "xatol": 1e-3, "fatol": 1e-10})
            vv = np.exp(best.x)
            vv = vv / vv.sum() * k
        else:
            vv = np.ones(k) if v is None else np.asarray(v, float)
        w = solve(vv)
        v_out = pd.Series(vv, index=P.columns)
    weights = pd.Series(w, index=donors)
    return _make_result(wide, treated, weights, pre_index, v_out, "slsqp")


def _make_result(wide, treated, weights, pre_index, v, method, extra=None) -> SynthResult:
    synth = apply_weights(wide, weights)
    actual = wide[treated].astype(float)
    pre = list(pre_index)
    err = (actual.loc[pre] - synth.loc[pre]).to_numpy()
    mspe = float(np.mean(err ** 2))
    rm = float(np.sqrt(mspe))
    return SynthResult(treated=treated, donors=list(weights.index), weights=weights, v_weights=v,
                       pre_index=pd.Index(pre), synthetic=synth, actual=actual, pre_rmspe=rm, pre_mspe=mspe,
                       rel_rmse=rm / max(abs(float(actual.loc[pre].mean())), 1e-12), method=method,
                       extra=extra or {})


def apply_weights(wide: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Locked-weights synthetic series for every row of ``wide``."""
    w = weights[weights > 0]
    return (wide[w.index].astype(float) * w.to_numpy()).sum(axis=1)


def rmspe(actual: pd.Series, synthetic: pd.Series, index: Optional[Sequence] = None) -> float:
    a, s = (actual, synthetic) if index is None else (actual.loc[list(index)], synthetic.loc[list(index)])
    return float(np.sqrt(np.mean((a.to_numpy(float) - s.to_numpy(float)) ** 2)))


def lift_from_series(actual: pd.Series, counterfactual: pd.Series, post_index: Sequence, sign: int = 1,
                     pre_index: Optional[Sequence] = None, bias_correct: bool = False) -> dict:
    """Relative lift over the post window: sign * (sum actual - sum cf) / sum cf.

    ``bias_correct`` subtracts the mean pre-period relative gap (natural bias) - used when
    validating an injected lift on a donor-fit that is imperfect.
    """
    post = list(post_index)
    a = float(actual.loc[post].sum())
    c = float(counterfactual.loc[post].sum())
    lift = (a - c) / c if c else np.nan
    bias = 0.0
    if bias_correct and pre_index is not None:
        pre = list(pre_index)
        cp = float(counterfactual.loc[pre].sum())
        bias = (float(actual.loc[pre].sum()) - cp) / cp if cp else 0.0
        lift = lift - bias
    return {"lift": sign * lift, "abs_effect": sign * (a - c), "actual": a, "counterfactual": c,
            "pre_bias": bias}


def injected_lift_recovery(wide: pd.DataFrame, treated: str, pre_index: Sequence, post_index: Sequence,
                           lift: float, donors: Optional[Sequence[str]] = None,
                           predictors: Optional[pd.DataFrame] = None, ridge: float = 0.0,
                           natural_bias_correction: bool = True, **fit_kw) -> dict:
    """Placebo-style validation: multiply the treated unit's post data by (1+lift), donors untouched.

    Weights are fit on pre-period only and locked. The natural (un-injected) post gap is
    subtracted so the recovered number isolates the injected effect.
    """
    res = fit_synth(wide, treated, pre_index, donors=donors, predictors=predictors, ridge=ridge, **fit_kw)
    post = list(post_index)
    inj = wide.copy()
    inj.loc[post, treated] = inj.loc[post, treated] * (1 + lift)
    nat = lift_from_series(wide[treated], res.synthetic, post)["lift"]
    det = lift_from_series(inj[treated], res.synthetic, post)["lift"]
    rec = det - nat if natural_bias_correction else det
    return {"treated": treated, "injected": lift, "detected": rec, "detected_raw": det, "natural_gap": nat,
            "error": rec - lift, "abs_error": abs(rec - lift), "pre_rmspe": res.pre_rmspe,
            "rel_rmse": res.rel_rmse, "n_donors_used": res.n_donors_used, "max_weight": res.max_weight,
            "effective_n": res.effective_n, "weights": res.weights[res.weights > 0].round(4).to_dict()}


def fit_synth_pysyncon(wide: pd.DataFrame, treated: str, pre_index: Sequence,
                       donors: Optional[Sequence[str]] = None, predictors_long: Optional[pd.DataFrame] = None,
                       predictor_cols: Optional[Sequence[str]] = None, optim_method: str = "Nelder-Mead",
                       optim_initial: str = "equal") -> SynthResult:
    """Optional wrapper around ``pysyncon.Synth`` (classic nested V/W optimisation).

    ``predictors_long`` (optional): long frame with columns unit, time, *predictor_cols*;
    averaged over the pre-period as ``predictors``. Without it, the pre-period outcome itself
    week is a special predictor (lagged-outcomes spec). Raises ImportError if pysyncon
    is missing - callers should fall back to :func:`fit_synth`.
    """
    from pysyncon import Dataprep, Synth  # noqa: WPS433

    donors = [c for c in (donors if donors is not None else wide.columns) if c != treated]
    long = wide[[treated] + donors].rename_axis("time").reset_index().melt(
        id_vars="time", var_name="unit", value_name="y")
    pre = list(pre_index)
    preds, special = None, None
    if predictors_long is not None and predictor_cols:
        long = long.merge(predictors_long, on=["unit", "time"], how="left")
        preds = list(predictor_cols)
    else:
        # classic "all lagged outcomes" specification (one special predictor per pre week)
        special = [("y", [t], "mean") for t in pre]
    dp = Dataprep(foo=long, predictors=preds or [], predictors_op="mean", time_predictors_prior=pre,
                  special_predictors=special, dependent="y", unit_variable="unit", time_variable="time",
                  treatment_identifier=treated, controls_identifier=donors, time_optimize_ssr=pre)
    s = Synth()
    s.fit(dataprep=dp, optim_method=optim_method, optim_initial=optim_initial)
    w = pd.Series(s.W, index=donors) if not isinstance(s.W, pd.Series) else s.W.reindex(donors).fillna(0)
    w = w.clip(lower=0)
    w[w < WEIGHT_EPS] = 0
    w = w / w.sum()
    v = pd.Series(np.diag(s.V) if np.ndim(s.V) == 2 else s.V) if getattr(s, "V", None) is not None else None
    return _make_result(wide, treated, w, pre_index, v, "pysyncon")


def pysyncon_available() -> bool:
    try:
        import pysyncon  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


def to_wide(panel: pd.DataFrame, unit_col: str, value_col: str, time_col: str = "week_idx") -> pd.DataFrame:
    return panel.pivot_table(index=time_col, columns=unit_col, values=value_col, aggfunc="sum").sort_index()
