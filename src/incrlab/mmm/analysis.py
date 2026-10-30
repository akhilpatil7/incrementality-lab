"""Posterior post-processing: ROI / marginal ROI, response curves, implied lift-test returns,
diagnostics, a posterior-aware budget optimizer and value-of-information next-test ranking.

Everything here works on *posterior draws* (numpy), so every business number carries uncertainty.
Ground truth (from the simulator) is used ONLY to score the model, never to fit it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .model import MMMData
from .transforms import L_MAX, adstock_batch, hill

HDI_PROB = 0.90


# --------------------------------------------------------------------------------------
# posterior draws
# --------------------------------------------------------------------------------------
@dataclass
class Draws:
    decay: np.ndarray      # (S, C)
    half_sat: np.ndarray   # (S, C)
    slope: np.ndarray      # (S, C)
    beta: np.ndarray       # (S, C)
    roi: np.ndarray        # (S, C)
    intercept: np.ndarray  # (S,)
    trend: np.ndarray      # (S,)
    gamma: np.ndarray      # (S, P)
    sigma: np.ndarray      # (S,)

    @property
    def S(self):
        return len(self.sigma)

    def subset(self, idx):
        return Draws(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})


def get_draws(idata, max_draws: int = 600, seed: int = 0) -> Draws:
    post = idata.posterior
    arr = {}
    for n in Draws.__dataclass_fields__:
        v = np.asarray(post[n].values)
        arr[n] = v.reshape((-1,) + v.shape[2:])
    S = len(arr["sigma"])
    if S > max_draws:
        idx = np.sort(np.random.default_rng(seed).choice(S, max_draws, replace=False))
        arr = {k: v[idx] for k, v in arr.items()}
    return Draws(**arr)


def hdi(x: np.ndarray, prob: float = HDI_PROB, axis: int = 0):
    """Highest-density interval along ``axis`` (narrowest window containing ``prob`` mass)."""
    x = np.moveaxis(np.asarray(x, float), axis, 0)
    xs = np.sort(x, axis=0)
    n = xs.shape[0]
    k = max(int(np.floor(prob * n)), 1)
    widths = xs[k:] - xs[: n - k]
    i = np.argmin(widths, axis=0)
    lo = np.take_along_axis(xs, i[None], 0)[0]
    hi = np.take_along_axis(xs, (i + k)[None], 0)[0]
    return lo, hi


def summarize(x: np.ndarray, prob: float = HDI_PROB) -> dict:
    lo, hi = hdi(x, prob)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "sd": float(np.std(x)),
            "hdi_low": float(lo), "hdi_high": float(hi), "hdi_prob": prob}


# --------------------------------------------------------------------------------------
# model pieces in numpy
# --------------------------------------------------------------------------------------
def baseline(dr: Draws, data: MMMData) -> np.ndarray:
    """(S, T) baseline sales (no paid media)."""
    lb = dr.intercept[:, None] + dr.trend[:, None] * data.t_std[None] + dr.gamma @ data.X_base.T
    return np.exp(lb)


def saturation(dr: Draws, data: MMMData, spend: Optional[np.ndarray] = None) -> np.ndarray:
    """(S, C, T) Hill response of each channel for a (C, T) spend matrix (default = actual)."""
    spend = data.spend if spend is None else spend
    A = adstock_batch(spend / data.spend_scale[:, None], dr.decay, L_MAX)
    return hill(A / dr.half_sat[:, :, None], dr.slope[:, :, None])


def contributions(dr: Draws, data: MMMData, spend: Optional[np.ndarray] = None, base=None) -> np.ndarray:
    """(S, C, T) incremental sales per channel per week."""
    base = baseline(dr, data) if base is None else base
    return base[:, None, :] * dr.beta[:, :, None] * saturation(dr, data, spend)


def predicted_mu(dr: Draws, data: MMMData) -> np.ndarray:
    base = baseline(dr, data)
    return base * (1 + (dr.beta[:, :, None] * saturation(dr, data)).sum(1))


def roi_draws(dr: Draws, data: MMMData, value_per_sale: float) -> np.ndarray:
    return contributions(dr, data).sum(-1) * value_per_sale / data.spend.sum(1)[None]


def marginal_roi_draws(dr: Draws, data: MMMData, value_per_sale: float, bump: float = 0.01) -> np.ndarray:
    """ROI of the next +1% of spend in every week (same definition as ground_truth marginal_roi)."""
    base = baseline(dr, data)
    c0 = contributions(dr, data, base=base).sum(-1)
    out = np.zeros_like(c0)
    for j in range(data.C):
        sp = data.spend.copy()
        sp[j] *= 1 + bump
        c1 = contributions(dr, data, sp, base=base)[:, j].sum(-1)
        out[:, j] = (c1 - c0[:, j]) * value_per_sale / (bump * data.spend[j].sum())
    return out


def response_curves(dr: Draws, data: MMMData, multipliers: Sequence[float]) -> np.ndarray:
    """(S, C, M) whole-timeline incremental sales when channel c's spend is scaled by m (others BAU)."""
    base = baseline(dr, data)
    out = np.zeros((dr.S, data.C, len(multipliers)))
    for k, m in enumerate(multipliers):
        sp = data.spend * m
        out[:, :, k] = contributions(dr, data, sp, base=base).sum(-1)
    return out


def implied_test_ipd(dr: Draws, data: MMMData, obs: List[dict]) -> np.ndarray:
    """(S, N) model-implied incremental sales per $ removed for each lift-test observation.

    Numpy mirror of the ``calib_ipd`` term inside the PyMC model: cut the arm's channels to
    ``leakage x BAU`` in the window (national, proportional), lost sales are summed over the window.
    """
    base = baseline(dr, data)
    full = contributions(dr, data, base=base)
    out = np.zeros((dr.S, len(obs)))
    for i, o in enumerate(obs):
        s, e = o["window"]
        m = (data.week_idx >= s) & (data.week_idx <= e)
        sp = data.spend.copy()
        dollars = 0.0
        for c in o["channels"]:
            j = data.channels.index(c)
            dollars += (data.spend[j] * m * (1 - o["leakage"])).sum()
            sp[j] = np.where(m, data.spend[j] * o["leakage"], data.spend[j])
        cut = contributions(dr, data, sp, base=base)
        lost = (full - cut)[:, :, m].sum((1, 2))
        out[:, i] = lost / dollars
    return out


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------
def diagnostics(idata, dr: Draws, data: MMMData, train_mask: np.ndarray, seed: int = 0) -> dict:
    import arviz as az
    summ = az.summary(idata, var_names=["roi", "decay", "half_sat", "slope", "intercept", "trend", "gamma", "sigma"])
    div = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    n_tot = int(np.asarray(idata.sample_stats["diverging"].values).size)
    mu = predicted_mu(dr, data)                           # (S, T)
    rng = np.random.default_rng(seed)
    ypp = mu * np.exp(rng.normal(0, 1, mu.shape) * dr.sigma[:, None])  # posterior predictive
    lo, hi = np.quantile(ypp, [0.05, 0.95], axis=0)
    yhat = mu.mean(0)
    y = data.y
    out = {"max_r_hat": float(summ["r_hat"].max()), "min_ess_bulk": float(summ["ess_bulk"].min()),
           "min_ess_tail": float(summ["ess_tail"].min()), "divergences": div, "n_samples": n_tot,
           "divergence_rate": div / max(n_tot, 1)}
    for name, msk in (("train", train_mask), ("holdout", ~train_mask)):
        if msk.sum() == 0:
            continue
        r = y[msk] - yhat[msk]
        out[f"{name}_mape"] = float(np.mean(np.abs(r) / y[msk]))
        out[f"{name}_r2"] = float(1 - (r ** 2).sum() / ((y[msk] - y[msk].mean()) ** 2).sum())
        out[f"{name}_ppc_coverage90"] = float(np.mean((y[msk] >= lo[msk]) & (y[msk] <= hi[msk])))
        out[f"{name}_weeks"] = int(msk.sum())
    out["_ppc"] = {"lo": lo, "hi": hi, "mean": yhat}
    return out


# --------------------------------------------------------------------------------------
# budget optimizer
# --------------------------------------------------------------------------------------
class BudgetProblem:
    """Plan horizon = the last ``horizon`` weeks of history, replayed at a new channel mix.

    Each channel's weekly flighting pattern is kept; only its level is scaled by ``m_c``. Incremental
    sales over the horizon include carry-over from pre-horizon (BAU) spend, identical across plans.
    """

    def __init__(self, dr: Draws, data: MMMData, value_per_sale: float, horizon: int = 52):
        self.dr, self.data, self.V = dr, data, value_per_sale
        T = data.T
        self.plan = np.zeros(T, bool)
        self.plan[T - horizon:] = True
        xs = data.spend / data.spend_scale[:, None]
        self.A_pre = adstock_batch(xs * ~self.plan, dr.decay)[:, :, self.plan]   # (S, C, H)
        self.A_plan = adstock_batch(xs * self.plan, dr.decay)[:, :, self.plan]
        self.base = baseline(dr, data)[:, self.plan]                            # (S, H)
        self.spend0 = data.spend[:, self.plan].sum(1)                            # (C,)
        self.budget0 = float(self.spend0.sum())

    def incremental(self, m: np.ndarray, idx=None) -> np.ndarray:
        """(S,) total incremental (media-driven) sales over the horizon at multipliers m (C,)."""
        sl = slice(None) if idx is None else idx
        A = self.A_pre[sl] + np.asarray(m)[None, :, None] * self.A_plan[sl]
        h = hill(A / self.dr.half_sat[sl][:, :, None], self.dr.slope[sl][:, :, None])
        return (self.base[sl][:, None, :] * self.dr.beta[sl][:, :, None] * h).sum((1, 2))

    def optimize(self, total_mult: float = 1.0, bounds=(0.7, 1.3), idx=None, x0=None) -> np.ndarray:
        """Maximise posterior-mean incremental sales s.t. sum_c m_c spend_c = total_mult x budget."""
        target = total_mult * self.budget0
        scale = self.incremental(np.ones(self.data.C), idx).mean()
        fun = lambda m: -self.incremental(m, idx).mean() / scale  # noqa: E731
        cons = [{"type": "eq", "fun": lambda m: (m @ self.spend0 - target) / self.budget0}]
        x0 = np.full(self.data.C, total_mult) if x0 is None else x0
        best = None
        for start in (x0, np.clip(np.full(self.data.C, total_mult) + 0.1 * np.sin(np.arange(self.data.C)),
                                  *bounds)):
            r = minimize(fun, start, method="SLSQP", bounds=[bounds] * self.data.C, constraints=cons,
                         options={"maxiter": 300, "ftol": 1e-10})
            if best is None or r.fun < best.fun:
                best = r
        return np.clip(best.x, *bounds)

    def evaluate(self, m: np.ndarray, ref: Optional[np.ndarray] = None) -> dict:
        inc = self.incremental(m)
        spend = float(np.asarray(m) @ self.spend0)
        out = {"multipliers": dict(zip(self.data.channels, np.asarray(m).tolist())),
               "spend_by_channel": dict(zip(self.data.channels, (np.asarray(m) * self.spend0).tolist())),
               "total_spend": spend, "incremental_sales": summarize(inc),
               "cpa": summarize(spend / inc), "roi": summarize(inc * self.V / spend)}
        if ref is not None:
            d = inc - self.incremental(ref)
            out["delta_vs_ref"] = summarize(d)
            out["prob_better"] = float(np.mean(d > 0))
        return out


def truth_plan_incremental(world, data: MMMData, m: Dict[str, float], horizon: int = 52) -> float:
    """TRUE expected incremental sales over the plan horizon under multipliers ``m`` (scoring only)."""
    from ..simulate import channel_effects, expected_outcomes
    T = data.T
    plan = np.zeros(T, bool)
    plan[T - horizon:] = True
    spend = {c: world.bau_spend[c] * np.where(plan, m.get(c, 1.0), 1.0)[None] for c in world.cfg.channels}
    ex = expected_outcomes(world, channel_effects(world, spend))
    return float(sum(ex["incr"]["sales"][c][:, plan].sum() for c in world.cfg.channels))


def value_of_information(bp: BudgetProblem, dr: Draws, n_bins: int = 4, bounds=(0.7, 1.3),
                         m_star: Optional[np.ndarray] = None) -> Dict[str, dict]:
    """Expected value of (partial) perfect information on each channel's ROI (EVPPI, binned).

    For channel c, split posterior draws into ``n_bins`` quantile bins of ROI_c (= "a test that pinned
    ROI_c down to this bin"). Re-optimise inside each bin and average the gain over the bin's draws
    versus the current optimum. Large EVPPI = learning this channel would change the budget decision.
    """
    m_star = bp.optimize(1.0, bounds) if m_star is None else m_star
    base = bp.incremental(m_star)
    out = {}
    for j, c in enumerate(bp.data.channels):
        r = dr.roi[:, j]
        edges = np.quantile(r, np.linspace(0, 1, n_bins + 1))
        lab = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_bins - 1)
        gain = 0.0
        for b in range(n_bins):
            idx = np.where(lab == b)[0]
            if len(idx) < 5:
                continue
            mb = bp.optimize(1.0, bounds, idx=idx, x0=m_star)
            gain += (bp.incremental(mb, idx) - base[idx]).sum()
        ev_sales = max(gain / dr.S, 0.0)
        cv = float(np.std(r) / np.mean(r))
        out[c] = {"evppi_sales": ev_sales, "evppi_value": ev_sales * bp.V, "roi_cv": cv,
                  "roi_hdi_width": float(np.subtract(*hdi(r)[::-1]))}
    return out
