"""Bayesian weekly national MMM (PyMC) with optional experiment-calibration likelihood.

Model (sales stage, weekly national)::

    log y_t ~ Normal(log mu_t, sigma)
    mu_t    = B_t * (1 + sum_c beta_c * Hill(A_c,t / K_c; s_c))
    log B_t = a + g * t_std + Fourier_K(week-of-year) + holiday dummies
    A_c,t   = normalised geometric adstock of spend_c (decay d_c), spend scaled by channel mean

The multiplicative-in-baseline form is the standard "media lifts baseline demand" structure: a channel's
incremental sales are B_t * beta_c * Hill_c,t, so ROI / marginal ROI / response curves follow directly.

Calibration (the "gap")
-----------------------
For every geo-test arm we know (from the test modules) the incremental sales per $ removed inside the
test window, ``ipd_obs +- se``. For any parameter draw, the MMM implies the same quantity: cut the arm's
channel(s) to ``leakage x BAU`` in the window, re-run adstock+Hill, and divide the lost sales in the
window by the $ removed (proportional cut => same per-capita spend point on the Hill curve as the test
geos). We add ``ipd_obs ~ Normal(ipd_model(theta), se)`` as extra likelihood terms. This is the
lift-test calibration used by PyMC-Marketing / Meridian-style workflows and is preferred over
"experiment -> informative prior on beta" because (i) a test measures a *window-average* incremental
return at the *current* spend level, a function of beta, K, slope AND decay jointly - not beta alone;
(ii) the likelihood form respects leakage, carry-over and saturation exactly; (iii) multi-channel arms
(all-search dark-out) calibrate a *sum* without pretending we observed each channel separately.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .transforms import L_MAX, lag_matrix

HOLIDAY_WEEKS = {"new_year": lambda w: w <= 3, "renewal_wave": lambda w: w == 8, "july4": lambda w: w == 27,
                 "thanksgiving": lambda w: w == 47, "year_end": lambda w: w >= 51}


@dataclass
class MMMData:
    channels: List[str]
    spend: np.ndarray            # (C, T) $
    y: np.ndarray                # (T,) sales
    week_idx: np.ndarray         # (T,)
    week_start: pd.DatetimeIndex
    X_base: np.ndarray           # (T, P) baseline design (fourier + holidays), no intercept/trend
    base_names: List[str]
    spend_scale: np.ndarray      # (C,) mean spend (for scaling)
    t_std: np.ndarray            # (T,)
    lt_sales: Dict[str, float] = field(default_factory=dict)  # last-touch claimed sales per channel

    @property
    def T(self):
        return len(self.y)

    @property
    def C(self):
        return len(self.channels)


def _woy(weeks: pd.DatetimeIndex) -> np.ndarray:
    return ((weeks.day_of_year.to_numpy() - 1) // 7) + 1


def baseline_design(weeks: pd.DatetimeIndex, n_fourier: int = 3):
    woy = _woy(weeks).astype(float)
    cols, names = [], []
    for k in range(1, n_fourier + 1):
        cols += [np.sin(2 * np.pi * k * woy / 52.18), np.cos(2 * np.pi * k * woy / 52.18)]
        names += [f"sin{k}", f"cos{k}"]
    for n, f in HOLIDAY_WEEKS.items():
        cols.append(f(woy).astype(float))
        names.append(n)
    return np.column_stack(cols), names


def prepare_data(national: pd.DataFrame, channels: Sequence[str], n_fourier: int = 3) -> MMMData:
    nat = national.sort_values("week_idx").reset_index(drop=True)
    weeks = pd.DatetimeIndex(nat["week_start"])
    spend = np.vstack([nat[f"spend_{c}"].to_numpy(float) for c in channels])
    X, names = baseline_design(weeks, n_fourier)
    t = nat["week_idx"].to_numpy(float)
    lt = {}
    for c in channels:
        if c in ("search_brand", "search_nonbrand", "social"):
            lt[c] = float(nat[f"lt_{c}_sales"].sum())
        elif c.startswith("dsp"):
            # site last-touch pools both DSP vendors; analysts split by vendor-reported conversions
            vb = nat[["vendor_conv_qs_dsp_a", "vendor_conv_qs_dsp_b"]].sum()
            lt[c] = float(nat["lt_dsp_sales"].sum() * vb[f"vendor_conv_qs_{c}"] / vb.sum())
        else:
            lt[c] = 0.0
    return MMMData(channels=list(channels), spend=spend, y=nat["sales"].to_numpy(float),
                   week_idx=nat["week_idx"].to_numpy(int), week_start=weeks, X_base=X, base_names=names,
                   spend_scale=spend.mean(1), t_std=(t - t.mean()) / t.std(), lt_sales=lt)


# --------------------------------------------------------------------------------------
# priors (documented; deliberately weakly-informative and IDENTICAL across channels - the
# point is to show what experiments add, not to smuggle the answer in through priors)
# --------------------------------------------------------------------------------------
ROI_PRIOR = (np.log(1.0), 0.9)  # LogNormal(mu, sd) on whole-period ROI ($ value / $ spend)
PRIORS = {
    "roi": "LogNormal(log 1.0, 0.9) per channel (Meridian-style ROI prior: 90% interval ~0.23-4.4x)",
    "decay": "Beta(2, 2.5) per channel (mean 0.44; anything from no carry-over to long carry-over)",
    "half_sat": "LogNormal(log 0.9, 0.5) x channel mean spend (half saturation near the operating point)",
    "slope": "LogNormal(log 1.2, 0.25) (Hill shape: concave to mildly S-shaped)",
    "baseline": "intercept N(log mean y, 0.5); trend N(0, 0.2); Fourier N(0, 0.2); holidays N(0, 0.3)",
    "sigma": "HalfNormal(0.1) on log sales",
}


@dataclass
class CalibArrays:
    pair_obs: np.ndarray       # (P,) observation index of each (obs, channel) pair
    pair_ch: np.ndarray        # (P,) channel index
    pair_weeks: np.ndarray     # (P, W) week positions (padded with the first window week)
    pair_mask: np.ndarray      # (P, W) 1 = real window week
    pair_lag_rem: np.ndarray   # (P, W, L) lagged removed (scaled) spend at those weeks
    dollars: np.ndarray        # (N,) $ removed per observation
    y: np.ndarray              # (N,) observed incremental sales per $
    se: np.ndarray             # (N,) SE used in the likelihood


def calib_arrays(data: MMMData, obs: List[dict]) -> CalibArrays:
    """For each (obs, channel) pair: removed spend (leakage-adjusted, proportional national cut) and lag stack.

    Only window weeks enter (test modules measure incremental sales *inside* the window), so the
    removed-spend adstock is needed only there: a_rem[t] = sum_k w_k rem[t-k], rem = 0 outside window.
    """
    C, T = data.C, data.T
    Wmax = max(o["window"][1] - o["window"][0] + 1 for o in obs)
    po, pc, pw, pm_, pl = [], [], [], [], []
    dollars = np.zeros(len(obs))
    for i, o in enumerate(obs):
        s, e = o["window"]
        m = (data.week_idx >= s) & (data.week_idx <= e)
        weeks = np.where(m)[0]
        for c in o["channels"]:
            j = data.channels.index(c)
            r = data.spend[j] * m * (1 - o["leakage"])
            dollars[i] += r.sum()
            lag = lag_matrix(r / data.spend_scale[j], L_MAX)[weeks]  # (W, L)
            W = len(weeks)
            pad = Wmax - W
            po.append(i)
            pc.append(j)
            pw.append(np.r_[weeks, np.full(pad, weeks[0])])
            pm_.append(np.r_[np.ones(W), np.zeros(pad)])
            pl.append(np.vstack([lag, np.zeros((pad, L_MAX))]))
    return CalibArrays(np.array(po), np.array(pc), np.array(pw, int), np.array(pm_), np.array(pl), dollars,
                       np.array([o["incremental_per_dollar"] for o in obs]),
                       np.array([o.get("se_model", o["se"]) for o in obs]))


def build_model(data: MMMData, calibration: Optional[List[dict]] = None, train_mask: Optional[np.ndarray] = None,
                value_per_sale: float = 1500.0, ipd_scale: float = 1e-3):
    """Return a PyMC model. ``calibration`` = list of validated observations (None = uncalibrated).

    Parameterisation: ROI_c (whole timeline, $ value / $) is the primitive; the Hill amplitude is derived
    beta_c = ROI_c * spend_c / (value_per_sale * sum_t B_t Hill_c,t). This puts a *business-readable*,
    channel-neutral prior on the quantity we care about and removes the beta->0 funnel of a HalfNormal beta.
    """
    import pymc as pm
    import pytensor.tensor as pt

    C, T = data.C, data.T
    xs = data.spend / data.spend_scale[:, None]
    lag = lag_matrix(xs, L_MAX)                     # (C, T, L)
    k = np.arange(L_MAX, dtype=float)
    tm = np.ones(T, bool) if train_mask is None else np.asarray(train_mask, bool)
    ly = np.log(data.y)
    tot_spend = data.spend.sum(1)
    coords = {"channel": data.channels, "base_term": data.base_names, "week": data.week_idx}
    with pm.Model(coords=coords) as model:
        roi = pm.LogNormal("roi", ROI_PRIOR[0], ROI_PRIOR[1], dims="channel")
        decay = pm.Beta("decay", 2.0, 2.5, dims="channel")
        half_sat = pm.LogNormal("half_sat", np.log(0.9), 0.5, dims="channel")
        slope = pm.LogNormal("slope", np.log(1.2), 0.25, dims="channel")
        intercept = pm.Normal("intercept", ly[tm].mean(), 0.5)
        trend = pm.Normal("trend", 0.0, 0.2)
        sd_base = np.array([0.2 if n.startswith(("sin", "cos")) else 0.3 for n in data.base_names])
        gamma = pm.Normal("gamma", 0.0, sd_base, dims="base_term")
        sigma = pm.HalfNormal("sigma", 0.1)

        w = (1 - decay)[:, None] * decay[:, None] ** k[None]          # (C, L)
        adst = (pt.as_tensor(lag) * w[:, None, :]).sum(-1)            # (C, T)
        # Hill via exp(s*log z) with a tiny floor: 0**s has a NaN gradient in s (dsp_a pre-launch weeks
        # have zero spend), which was the source of most divergences in early versions.
        zs = pt.exp(slope[:, None] * pt.log(adst / half_sat[:, None] + 1e-9))
        h = zs / (1 + zs)
        log_base = intercept + trend * data.t_std + pt.dot(data.X_base, gamma)
        base = pt.exp(log_base)
        beta = pm.Deterministic("beta", roi * tot_spend / (value_per_sale * (base[None] * h).sum(1)),
                                dims="channel")
        media = (beta[:, None] * h).sum(0)
        log_mu = log_base + pt.log1p(media)
        pm.Normal("obs", log_mu[tm], sigma, observed=ly[tm])
        pm.Deterministic("contribution", base[None] * beta[:, None] * h, dims=("channel", "week"))
        pm.Deterministic("baseline", base, dims="week")
        pm.Deterministic("mu", pt.exp(log_mu), dims="week")

        if calibration:
            ca = calib_arrays(data, calibration)
            a_full = adst[ca.pair_ch[:, None], ca.pair_weeks]                         # (P, W)
            a_rem = (pt.as_tensor(ca.pair_lag_rem) * w[ca.pair_ch][:, None, :]).sum(-1)  # (P, W)
            hs_c = half_sat[ca.pair_ch][:, None]
            sl_c = slope[ca.pair_ch][:, None]
            h_full = h[ca.pair_ch[:, None], ca.pair_weeks]
            zc = pt.exp(sl_c * pt.log(pt.maximum(a_full - a_rem, 0.0) / hs_c + 1e-9))
            h_cut = zc / (1 + zc)
            lost_pair = (base[ca.pair_weeks] * beta[ca.pair_ch][:, None] * (h_full - h_cut) * ca.pair_mask).sum(1)
            lost = pt.zeros(len(ca.y))
            lost = pt.inc_subtensor(lost[ca.pair_obs], lost_pair)
            ipd_model = pm.Deterministic("calib_ipd", lost / ca.dollars / ipd_scale)
            pm.Normal("calib_obs", ipd_model, ca.se / ipd_scale, observed=ca.y / ipd_scale)
    return model


@dataclass
class FitSettings:
    draws: int = 800
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.92
    seed: int = 20260
    method: str = "nuts"   # "nuts" | "advi"

    @classmethod
    def small(cls):
        return cls(draws=150, tune=200, chains=2, target_accept=0.85)


def fit(model, settings: FitSettings):
    import pymc as pm
    t0 = time.time()
    with model:
        if settings.method == "advi":
            approx = pm.fit(30000, method="advi", random_seed=settings.seed, progressbar=False)
            idata = approx.sample(settings.draws * settings.chains, random_seed=settings.seed)
        else:
            idata = pm.sample(settings.draws, tune=settings.tune, chains=settings.chains,
                              cores=min(settings.chains, 4), target_accept=settings.target_accept,
                              random_seed=settings.seed, progressbar=False, compute_convergence_checks=False)
    return idata, time.time() - t0


def posterior_arrays(idata, names=("decay", "half_sat", "slope", "beta", "intercept", "trend", "gamma", "sigma")):
    post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
    out = {}
    for n in names:
        v = np.asarray(post[n].values)
        out[n] = v.reshape((-1,) + v.shape[2:])
    return out
