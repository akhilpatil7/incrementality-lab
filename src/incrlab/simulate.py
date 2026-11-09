"""Single synthetic-world generator for every module.

Structure
---------
* :func:`build_world` creates the *latent* world (geography, baseline demand, BAU spend plan,
  noise draws). Everything is deterministic in ``cfg.seed``.
* :func:`realize` turns a :class:`Scenario` (world + interventions) into observed panels.
  All randomness comes from pre-drawn uniforms pushed through inverse CDFs (common random
  numbers), so a unit/week untouched by an intervention has *identical* outcomes in every
  scenario, and the untreated potential outcome of a treated unit is exactly its BAU value.
* :func:`apply_intervention` / :func:`run_scenario` let a downstream module apply a test to
  the geos *it selected* and get back observed panels plus the per-test ground truth.
* :func:`write_baseline` writes ``data/*.parquet`` and ``data/ground_truth.json``.

Outcome model (unit u = zip or rest-of-nation aggregate, week t, stage s in qs/rated/sales)::

    mu0_qs[u,t]  = pop * base_rate * propensity_u * season_t * trend_t * exp(region AR + county AR
                   + county trend + zip noise)                         # no-advertising demand
    eff_ch[u,t]  = served_u * max_effect_ch * Hill(adstock(spend per served capita) / K_ch)
    incr_ch,s    = stage_mult_ch,s * eff_ch          (paid search: * (1 - recapture_ch))
    mu_s[u,t]    = mu0_s[u,t] * (1 + sum_ch incr_ch,s)                  # additive contributions
    qs ~ NegBin(mu_qs, k);  rated ~ Bin(qs, mu_rated/mu_qs);  sales ~ Bin(rated, mu_sales/mu_rated)

Last-touch columns partition each stage total into sources; they are biased relative to the
truth (search over-claims through brand cannibalisation, social over-claims quotes and
under-claims sales, DSP is over-claimed by site last-touch and much more by the vendor).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .config import Config, get_config
from .geo import Geography, build_geography, haversine_miles

STAGE_IDX = {"qs": 0, "rated": 1, "sales": 2}
SCHEMA_VERSION = "2.0"


# ======================================================================================
# Latent world
# ======================================================================================
@dataclass
class World:
    cfg: Config
    geo: Geography
    unit_ids: np.ndarray  # zips + "ROW"
    unit_region: np.ndarray
    unit_county: np.ndarray
    pop: np.ndarray  # (U,)
    weeks: pd.DatetimeIndex
    mu0_qs: np.ndarray  # (U,T)
    rated_rate: np.ndarray  # (U,)
    close_rate: np.ndarray  # (U,)
    served: Dict[str, np.ndarray]  # ch -> (U,) served fraction
    alloc: Dict[str, np.ndarray]  # ch -> (U,T) share of national spend
    national_spend: Dict[str, np.ndarray]  # ch -> (T,)
    bau_spend: Dict[str, np.ndarray]  # ch -> (U,T) $
    half_sat: Dict[str, float]  # ch -> K ($ adstocked per served capita per week)
    cpc_index: np.ndarray  # (T,) search CPC drift
    season: np.ndarray
    seed_seq: np.random.SeedSequence = field(repr=False, default=None)
    _uniforms: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def U(self) -> int:
        return len(self.unit_ids)

    @property
    def T(self) -> int:
        return len(self.weeks)

    def uniforms(self, name: str) -> np.ndarray:
        """Named (U,T) uniform stream; each name has its own independent seed."""
        if name not in self._uniforms:
            key = int.from_bytes(name.encode()[:16].ljust(16, b"\0"), "little") % (2 ** 63)
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, 909, key]))
            self._uniforms[name] = rng.random((self.U, self.T))
        return self._uniforms[name]

    def unit_index(self, ids: Iterable[str]) -> np.ndarray:
        """Resolve zip ids and/or county ids to unit row indices."""
        ids = list(ids)
        pos = {u: i for i, u in enumerate(self.unit_ids)}
        out = set()
        for x in ids:
            if x in pos:
                out.add(pos[x])
            else:
                m = np.where(self.unit_county == x)[0]
                if len(m) == 0:
                    raise KeyError(f"unknown unit id {x!r} (expected a zip or county_id)")
                out.update(m.tolist())
        return np.array(sorted(out), dtype=int)


def _woy(weeks: pd.DatetimeIndex) -> np.ndarray:
    return ((weeks.day_of_year.to_numpy() - 1) // 7) + 1  # 1..53


def demand_season(weeks: pd.DatetimeIndex) -> np.ndarray:
    """Common multiplicative demand curve with holiday shocks."""
    w = _woy(weeks).astype(float)
    s = 1 + 0.10 * np.sin(2 * np.pi * (w - 1) / 52) + 0.04 * np.cos(4 * np.pi * w / 52)
    s = s * np.where(w <= 3, 1.12, 1.0)  # New-year shopping spike
    s = s * np.where(w == 8, 1.55, 1.0)  # late-Feb shared spike week (renewal mailing wave)
    s = s * np.where(w == 47, 0.85, 1.0)  # Thanksgiving
    s = s * np.where(w >= 51, 0.75, 1.0)  # year-end holiday dip
    s = s * np.where(w == 27, 0.90, 1.0)  # early-July holiday
    return s


def _ar1(rng, n, T, rho, sd):
    x = np.zeros((n, T))
    e = rng.normal(0, sd, (n, T))
    x[:, 0] = e[:, 0] / np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def _national_spend(cfg: Config, weeks: pd.DatetimeIndex, season: np.ndarray, rng) -> Dict[str, np.ndarray]:
    T = len(weeks)
    t = np.arange(T)
    w = _woy(weeks)
    year = t // 52
    months = weeks.month.to_numpy() + 12 * year
    out = {}

    def ar_noise(sd, rho=0.6):
        return np.exp(_ar1(rng, 1, T, rho, sd)[0])

    def monthly_levels(lo, hi):
        lv = {m: rng.uniform(lo, hi) for m in np.unique(months)}
        return np.array([lv[m] for m in months])

    tr = cfg.truth
    # TV: always-on with heavy-ups (Q1, early Q4) and a summer trough
    tv = np.where((w <= 12) | ((w >= 35) & (w <= 46)), 1.40, 1.0) * np.where((w >= 26) & (w <= 33), 0.45, 1.0)
    tv = tv * np.array([1.0, 1.12, 0.92])[year] * ar_noise(0.08)
    out["tv"] = tr["tv"].base_weekly_spend * tv
    # Social: bursts, yearly budget growth, one dark period (gives MMM identifying variation)
    soc = np.array([0.80, 1.0, 1.15])[year] * ar_noise(0.15)
    for _ in range(9):
        s0 = rng.integers(0, T - 4)
        soc[s0:s0 + 4] *= 1.6
    soc[30:33] *= 0.05
    out["social"] = tr["social"].base_weekly_spend * soc
    # Branded search: follows demand (bids on brand queries track query volume)
    out["search_brand"] = tr["search_brand"].base_weekly_spend * season ** 1.3 * \
        np.array([1.0, 1.05, 1.10])[year] * ar_noise(0.05)
    # Non-branded search: quarterly budget steps
    q = (t // 13)
    steps = rng.uniform(0.65, 1.35, q.max() + 1)
    out["search_nonbrand"] = tr["search_nonbrand"].base_weekly_spend * steps[q] * ar_noise(0.08)
    # DSP Vendor A: launches week 16; monthly flights
    a = monthly_levels(0.5, 1.5) * ar_noise(0.08) * (t >= 16)
    out["dsp_a"] = tr["dsp_a"].base_weekly_spend * a
    # DSP Vendor B line (legacy vendor until the switch, Vendor B after): monthly flights
    b = monthly_levels(0.6, 1.4) * ar_noise(0.08) * np.where(t >= C.DSP_B_VENDOR_SWITCH_WEEK, 1.15, 1.0)
    out["dsp_b"] = tr["dsp_b"].base_weekly_spend * b
    return out


def adstock(x: np.ndarray, decay: float, normalize: bool = True) -> np.ndarray:
    """Geometric adstock along the last axis: a_t = x_t + decay * a_{t-1} (times 1-decay)."""
    x = np.asarray(x, float)
    a = np.empty_like(x)
    a[..., 0] = x[..., 0]
    for t in range(1, x.shape[-1]):
        a[..., t] = x[..., t] + decay * a[..., t - 1]
    return a * (1 - decay) if normalize else a


def hill(x: np.ndarray, slope: float) -> np.ndarray:
    x = np.maximum(np.asarray(x, float), 0.0)
    xs = x ** slope
    return xs / (1.0 + xs)


def build_world(cfg: Optional[Config] = None) -> World:
    cfg = cfg or get_config()
    geo = build_geography(cfg)
    z = geo.zips
    wp = cfg.world
    T = cfg.n_weeks
    weeks = pd.date_range(cfg.start_date, periods=T, freq="7D")
    ss = np.random.SeedSequence([cfg.seed, 7])
    r_lat, r_spend, r_alloc = (np.random.default_rng(s) for s in ss.spawn(3))

    U = len(z) + 1
    unit_ids = np.append(z["zip"].to_numpy(), "ROW")
    unit_region = np.append(z["region"].to_numpy(), "ROW")
    unit_county = np.append(z["county_id"].to_numpy(), "ROW")
    pop = np.append(z["pop"].to_numpy(float), cfg.row_pop)
    prop = np.append(z["zip_propensity"].to_numpy(), 1.0)

    season = demand_season(weeks)
    t = np.arange(T)
    trend = 1 + wp.trend_per_week * t

    # region AR(1) shocks + occasional region-wide spikes / dips (storms, local news)
    regions = sorted(set(unit_region))
    reg_shock = _ar1(r_lat, len(regions), T, wp.region_ar_rho, wp.region_ar_sd)
    spikes = r_lat.random((len(regions), T)) < wp.region_spike_prob
    mag = r_lat.choice([-1, 1], size=(len(regions), T), p=[0.35, 0.65]) * r_lat.uniform(0.25, 0.35, (len(regions), T))
    for k in range(len(regions)):
        for tt in np.where(spikes[k])[0]:
            dur = r_lat.integers(1, 3)
            reg_shock[k, tt:tt + dur] += np.log1p(mag[k, tt])
    reg_map = {r: i for i, r in enumerate(regions)}
    log_noise = reg_shock[[reg_map[r] for r in unit_region]]

    counties = sorted(set(unit_county))
    cty_ar = _ar1(r_lat, len(counties), T, wp.county_ar_rho, wp.county_ar_sd)
    cty_trend = r_lat.normal(0, wp.county_trend_sd, len(counties))
    cmap = {c: i for i, c in enumerate(counties)}
    ci = np.array([cmap[c] for c in unit_county])
    is_row = unit_region == "ROW"
    log_noise = log_noise + np.where(is_row[:, None], 0.0, cty_ar[ci] + cty_trend[ci, None] * (t - T / 2))
    log_noise = log_noise + np.where(is_row[:, None], 0.0, r_lat.normal(0, wp.zip_noise_sd, (U, T)))

    mu0 = (pop * wp.base_qs_rate * prop)[:, None] * season[None] * trend[None] * np.exp(log_noise)

    rated_rate = np.clip(r_lat.normal(wp.rated_rate_mean, wp.rated_rate_sd, U), 0.45, 0.85)
    close_rate = np.clip(r_lat.normal(wp.close_rate_mean, wp.close_rate_sd, U), 0.04, 0.18)
    rated_rate[-1], close_rate[-1] = wp.rated_rate_mean, wp.close_rate_mean

    # serving footprint
    served = {}
    for ch in cfg.channels:
        pres = cfg.presence[ch]
        s = np.array([pres.get(r, 0.0) for r in unit_region], float)
        if ch == "dsp_a":
            s[:-1] = z["dsp_a_served"].to_numpy(float)
        elif ch == "dsp_b":
            s[:-1] = z["dsp_b_inclusion"].to_numpy(float)
        served[ch] = s

    national = _national_spend(cfg, weeks, season, r_spend)

    # geo allocation: served pop x static efficiency noise x regional weekly pacing
    alloc, bau = {}, {}
    pacing = {ch: np.exp(_ar1(r_alloc, len(regions), T, 0.7, 0.06)) for ch in cfg.channels}
    for ch in cfg.channels:
        w_static = served[ch] * pop * np.exp(r_alloc.normal(0, 0.15, U))
        if ch.startswith("search"):
            w_static = w_static * prop ** 0.5  # bid budgets follow query volume
        w = w_static[:, None] * pacing[ch][[reg_map[r] for r in unit_region]]
        if ch == "tv":
            w = (pop * served[ch])[:, None] * np.ones((1, T))  # national buy, no geo variation
        col = w.sum(0, keepdims=True)
        a = np.divide(w, col, out=np.zeros_like(w), where=col > 0)
        alloc[ch] = a
        bau[ch] = a * national[ch][None]

    # concurrent unrelated social test: heavy-up in a contiguous blob (design must exclude it)
    ct = C.CONCURRENT_TEST
    flag = np.append(z["concurrent_test_flag"].to_numpy(bool), False)
    bau[ct["channel"]][np.ix_(flag, np.arange(ct["start"], ct["end"] + 1))] *= ct["spend_multiplier"]

    # half-saturation: K = ratio * median national adstocked spend per served capita (on weeks)
    half_sat = {}
    for ch in cfg.channels:
        tr = cfg.truth[ch]
        served_pop = (served[ch] * pop).sum()
        pc = adstock(national[ch], tr.decay) / served_pop
        on = national[ch] > 0
        half_sat[ch] = float(tr.half_sat_ratio * np.median(pc[on]))

    cpc_index = season ** 0.8 * (1 + 0.05 * (t // 52))
    return World(cfg=cfg, geo=geo, unit_ids=unit_ids, unit_region=unit_region, unit_county=unit_county,
                 pop=pop, weeks=weeks, mu0_qs=mu0, rated_rate=rated_rate, close_rate=close_rate,
                 served=served, alloc=alloc, national_spend=national, bau_spend=bau, half_sat=half_sat,
                 cpc_index=cpc_index, season=season, seed_seq=ss)


# ======================================================================================
# Scenarios & interventions
# ======================================================================================
@dataclass
class Intervention:
    test: str
    arm: str
    channels: Tuple[str, ...]
    unit_idx: np.ndarray
    window: Tuple[int, int]  # inclusive week indices
    spend_multiplier: float = 0.0
    leakage: float = 0.0
    spillover: Optional[dict] = None

    @property
    def weeks(self) -> np.ndarray:
        return np.arange(self.window[0], self.window[1] + 1)


@dataclass
class Scenario:
    world: World
    interventions: List[Intervention] = field(default_factory=list)

    def without(self, test: str) -> "Scenario":
        return Scenario(self.world, [iv for iv in self.interventions if iv.test != test])


@dataclass
class ScenarioResult:
    panels: Dict[str, pd.DataFrame]
    truth: Dict[str, dict]
    scenario: Scenario


def apply_intervention(scenario: Union[Scenario, World], treated_units: Iterable[str],
                       window: Tuple[int, int], channel: Union[str, Sequence[str]],
                       effect_spec: Optional[dict] = None, test: str = "custom",
                       arm: str = "treatment") -> Scenario:
    """Return a new Scenario with spend for ``channel`` changed in ``treated_units`` over ``window``.

    effect_spec keys: ``spend_multiplier`` (0 = holdout/dark-out, 2 = heavy-up; default 0),
    ``leakage`` (share of BAU spend still delivered when multiplier is 0; default 0),
    ``spillover`` (dict radius_mi/decay_mi/strength/enabled; reduces the channel's effect in
    *untreated* zips near the treated set during the window - e.g. geo-targeting slippage).
    The ground-truth effect follows from the channel's response curve; nothing is "injected".
    """
    sc = scenario if isinstance(scenario, Scenario) else Scenario(scenario)
    world = sc.world
    spec = dict(effect_spec or {})
    chans = (channel,) if isinstance(channel, str) else tuple(channel)
    for ch in chans:
        if ch not in world.cfg.channels:
            raise KeyError(f"unknown channel {ch!r}")
    iv = Intervention(test=test, arm=arm, channels=chans, unit_idx=world.unit_index(treated_units),
                      window=(int(window[0]), int(window[1])),
                      spend_multiplier=float(spec.get("spend_multiplier", 0.0)),
                      leakage=float(spec.get("leakage", 0.0)), spillover=spec.get("spillover"))
    return Scenario(world, sc.interventions + [iv])


def _spend_and_effect_mult(sc: Scenario):
    """Actual spend per channel (U,T) and effect multipliers (spillover) for a scenario."""
    w = sc.world
    spend = {ch: w.bau_spend[ch].copy() for ch in w.cfg.channels}
    eff_mult = {ch: np.ones((w.U, w.T)) for ch in w.cfg.channels}
    for iv in sc.interventions:
        mult = iv.spend_multiplier if iv.spend_multiplier > 0 else iv.leakage
        rows, cols = np.ix_(iv.unit_idx, iv.weeks)
        for ch in iv.channels:
            spend[ch][rows, cols] = w.bau_spend[ch][rows, cols] * mult
        sp = iv.spillover
        if sp and sp.get("enabled", True) and iv.spend_multiplier < 1:
            d = spillover_distance(w, iv.unit_idx)
            m = (d > 0) & (d <= sp.get("radius_mi", C.BUFFER_MILES_DEFAULT))
            loss = sp.get("strength", 0.3) * np.exp(-d[m] / sp.get("decay_mi", 10.0)) * (1 - mult)
            for ch in iv.channels:
                eff_mult[ch][np.ix_(np.where(m)[0], iv.weeks)] *= (1 - loss)[:, None]
    return spend, eff_mult


def spillover_distance(world: World, unit_idx: np.ndarray) -> np.ndarray:
    """Miles from every unit to the nearest treated unit (0 = treated, inf = ROW/other region)."""
    z = world.geo.zips
    d = np.full(world.U, np.inf)
    zi = unit_idx[unit_idx < len(z)]
    if len(zi) == 0:
        return d
    lat, lon = z["lat"].to_numpy(), z["lon"].to_numpy()
    dd = haversine_miles(lat[:, None], lon[:, None], lat[zi][None], lon[zi][None]).min(1)
    d[:len(z)] = dd
    d[zi] = 0.0
    return d


def channel_effects(world: World, spend: Dict[str, np.ndarray],
                    eff_mult: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Gross effect share eff_ch[u,t] (paid search: last-touch *capture* share)."""
    cfg = world.cfg
    out = {}
    for ch in cfg.channels:
        tr = cfg.truth[ch]
        served_pop = world.served[ch] * world.pop
        pc = np.divide(spend[ch], served_pop[:, None], out=np.zeros_like(spend[ch]),
                       where=served_pop[:, None] > 0)
        a = adstock(pc, tr.decay)
        e = world.served[ch][:, None] * cfg.max_effect(ch) * hill(a / world.half_sat[ch], tr.slope)
        if eff_mult is not None:
            e = e * eff_mult[ch]
        out[ch] = e
    return out


def expected_outcomes(world: World, effects: Dict[str, np.ndarray]) -> dict:
    """Expected stage totals, incremental contributions and last-touch expectations."""
    cfg = world.cfg
    wp = cfg.world
    mu0 = {"qs": world.mu0_qs}
    mu0["rated"] = mu0["qs"] * world.rated_rate[:, None]
    mu0["sales"] = mu0["rated"] * world.close_rate[:, None]
    incr = {s: {} for s in C.STAGES}
    for ch in cfg.channels:
        tr = cfg.truth[ch]
        for s, k in STAGE_IDX.items():
            incr[s][ch] = mu0[s] * tr.stage_mult[k] * effects[ch] * (1 - tr.recapture)
    mu = {s: mu0[s] + sum(incr[s].values()) for s in C.STAGES}

    lt = {s: {} for s in C.STAGES}
    for s, k in STAGE_IDX.items():
        b = mu0[s] * cfg.truth["search_brand"].stage_mult[k] * effects["search_brand"]
        n = mu0[s] * cfg.truth["search_nonbrand"].stage_mult[k] * effects["search_nonbrand"]
        org = mu0[s] * (wp.organic_brand_share + wp.organic_generic_share) \
            - cfg.truth["search_brand"].recapture * b - cfg.truth["search_nonbrand"].recapture * n
        org = np.maximum(org, 0)
        soc = cfg.truth["social"].lt_bias[k] * incr[s]["social"]
        dsp = cfg.truth["dsp_a"].lt_bias[k] * incr[s]["dsp_a"] + cfg.truth["dsp_b"].lt_bias[k] * incr[s]["dsp_b"]
        rest = mu[s] - b - n - org
        claims = soc + dsp
        scale = np.where(claims > 0.9 * rest, 0.9 * rest / np.maximum(claims, 1e-12), 1.0)
        soc, dsp = soc * scale, dsp * scale
        rest = np.maximum(rest - soc - dsp, 0)
        d, ua, o = wp.rest_split
        lt[s] = {"search_brand": b, "search_nonbrand": n, "organic": org, "social": soc, "dsp": dsp,
                 "direct": rest * d, "unattributed": rest * ua, "other": rest * o}
    visits = mu0["qs"] * wp.visits_per_qs * (1 + sum(cfg.truth[ch].visit_mult * effects[ch]
                                                     for ch in cfg.channels))
    return {"mu0": mu0, "mu": mu, "incr": incr, "lt": lt, "visits": visits}


def _nb_ppf(u, mu, k):
    mu = np.maximum(mu, 1e-9)
    return stats.nbinom.ppf(u, k, k / (k + mu))


def _binom_ppf(u, n, p):
    p = np.clip(p, 0.0, 1.0)
    out = stats.binom.ppf(u, n, p)
    return np.where(n <= 0, 0, np.nan_to_num(out, nan=0.0))


def _draw_counts(world: World, ex: dict, rows: Optional[np.ndarray] = None) -> dict:
    """Realise counts from expectations with common random numbers (inverse-CDF)."""
    cfg = world.cfg
    sl = slice(None) if rows is None else rows
    U = lambda name: world.uniforms(name)[sl]  # noqa: E731
    is_row = (world.unit_region == "ROW")[sl][:, None]
    k = np.where(is_row, 5e4, cfg.world.nb_dispersion_zip)
    mu = {s: ex["mu"][s][sl] for s in C.STAGES}
    out = {}
    out["visits"] = _nb_ppf(U("visits"), ex["visits"][sl], np.where(is_row, 5e4, 15.0))
    out["qs"] = _nb_ppf(U("qs"), mu["qs"], k)
    out["rated"] = _binom_ppf(U("rated"), out["qs"], mu["rated"] / np.maximum(mu["qs"], 1e-12))
    out["sales"] = _binom_ppf(U("sales"), out["rated"], mu["sales"] / np.maximum(mu["rated"], 1e-12))
    for s in C.STAGES:
        N = out[s].copy()
        rem_p = np.ones_like(N, dtype=float)
        lt_s = ex["lt"][s]
        for src in C.LT_SOURCES[:-1]:
            p = lt_s[src][sl] / np.maximum(mu[s], 1e-12)
            q = np.divide(p, rem_p, out=np.ones_like(p), where=rem_p > 1e-12)
            n_k = _binom_ppf(U(f"lt_{src}_{s}"), N, np.clip(q, 0, 1))
            out[f"lt_{src}_{s}"] = n_k
            N = N - n_k
            rem_p = rem_p - p
        out[f"lt_{C.LT_SOURCES[-1]}_{s}"] = N
    return out


def realize(scenario: Union[Scenario, World], with_truth: bool = True) -> ScenarioResult:
    """Simulate observed panels (and per-test truth) for a scenario."""
    sc = scenario if isinstance(scenario, Scenario) else Scenario(scenario)
    w = sc.world
    spend, emult = _spend_and_effect_mult(sc)
    eff = channel_effects(w, spend, emult)
    ex = expected_outcomes(w, eff)
    counts = _draw_counts(w, ex)
    panels = _assemble_panels(w, spend, eff, ex, counts, sc)
    truth = _scenario_truth(sc, spend, ex, counts) if (with_truth and sc.interventions) else {}
    return ScenarioResult(panels=panels, truth=truth, scenario=sc)


# ======================================================================================
# Panel assembly
# ======================================================================================
def _media_columns(w: World, spend: Dict[str, np.ndarray], eff: Dict[str, np.ndarray], counts: dict) -> dict:
    cfg = w.cfg
    cols = {}
    for ch in cfg.channels:
        tr = cfg.truth[ch]
        sp = spend[ch]
        cols[f"spend_{ch}"] = sp
        if tr.cpc:
            cpc = tr.cpc * w.cpc_index[None]
            clicks = sp / cpc
            impr = clicks / tr.ctr
        else:
            impr = sp / tr.cpm * 1000.0 if tr.cpm > 0 else np.zeros_like(sp)
            clicks = impr * tr.ctr
        cols[f"impressions_{ch}"] = np.round(impr)
        cols[f"clicks_{ch}"] = np.round(clicks)
    # platform-reported search conversions (over-claim vs last-touch quotes)
    for ch in C.PAID_SEARCH:
        lam = cfg.world.platform_conv_overclaim * counts[f"lt_{ch}_rated"]
        cols[f"platform_conv_{ch}"] = stats.poisson.ppf(w.uniforms(f"plat_{ch}"), np.maximum(lam, 1e-9))
    # vendor-reported DSP conversion quote starts (heavily over-claimed; vendor switch + dips)
    t = np.arange(w.T)
    for ch in ("dsp_a", "dsp_b"):
        tr = cfg.truth[ch]
        share = tr.vendor_bias * cfg.truth[ch].stage_mult[0] * eff[ch] / (1 + 0.27)
        if ch == "dsp_b":
            share = share * np.where(t >= C.DSP_B_VENDOR_SWITCH_WEEK, C.DSP_B_SWITCH_ATTR_SHIFT, 1.0)[None]
            dip = np.isin(t, C.DSP_B_REPORT_DIP_WEEKS)
            share = share * np.where(dip, 0.15, 1.0)[None]
        cols[f"vendor_conv_qs_{ch}"] = _binom_ppf(w.uniforms(f"vendor_{ch}"), counts["qs"], np.clip(share, 0, 0.9))
    return cols


def _assemble_panels(w: World, spend, eff, ex, counts, sc: Scenario) -> Dict[str, pd.DataFrame]:
    cfg = w.cfg
    U, T = w.U, w.T
    media = _media_columns(w, spend, eff, counts)
    data = {"visits": counts["visits"], "qs": counts["qs"], "rated": counts["rated"], "sales": counts["sales"]}
    for s in C.STAGES:
        for src in C.LT_SOURCES:
            data[f"lt_{src}_{s}"] = counts[f"lt_{src}_{s}"]
    data.update(media)

    idx_u = np.repeat(np.arange(U), T)
    idx_t = np.tile(np.arange(T), U)
    df = pd.DataFrame({
        "week_idx": idx_t.astype(np.int16),
        "week_start": w.weeks[idx_t],
        "unit": w.unit_ids[idx_u],
        "county_id": w.unit_county[idx_u],
        "region": w.unit_region[idx_u],
    })
    for k, v in data.items():
        arr = np.asarray(v).reshape(-1)
        if k.startswith("spend_"):
            df[k] = arr.astype(np.float64)
        else:
            df[k] = np.round(arr).astype(np.int64)
    df["month_start"] = df["week_start"].dt.to_period("M").dt.to_timestamp()
    df["year"] = (df["week_idx"] // 52 + 1).astype(np.int8)

    nat = df.drop(columns=["unit", "county_id", "region", "month_start", "year"]).groupby(
        ["week_idx", "week_start"], as_index=False).sum()
    reg = df.drop(columns=["unit", "county_id", "month_start", "year"]).groupby(
        ["region", "week_idx", "week_start"], as_index=False).sum()

    zipdf = df[df["region"] != "ROW"].rename(columns={"unit": "zip"}).reset_index(drop=True)
    zs = w.geo.zips.set_index("zip")
    zipdf["pop"] = zipdf["zip"].map(zs["pop"]).astype(np.int64)
    zipdf["dsp_a_served"] = zipdf["zip"].map(zs["dsp_a_served"]).astype(np.int8)
    zipdf["dsp_b_inclusion"] = zipdf["zip"].map(zs["dsp_b_inclusion"]).astype(np.int8)
    zipdf["concurrent_test_flag"] = zipdf["zip"].map(zs["concurrent_test_flag"]).astype(np.int8)
    zipdf["dsp_b_vendor_live"] = (zipdf["week_idx"] >= C.DSP_B_VENDOR_SWITCH_WEEK).astype(np.int8)
    # arm labels per test (static per zip)
    for test in C.TESTS:
        zipdf[f"arm_{test}"] = ""
    lab = {}
    for iv in sc.interventions:
        for i in iv.unit_idx:
            if i < len(zs):
                lab.setdefault(iv.test, {})[w.unit_ids[i]] = iv.arm
    for test, m in lab.items():
        zipdf[f"arm_{test}"] = zipdf["zip"].map(m).fillna("")
    # monthly DSP media repeated on weekly rows (dedupe before summing - see geo.monthly_media_sum)
    for ch in ("dsp_a", "dsp_b"):
        g = zipdf.groupby(["zip", "month_start"])
        for col in (f"spend_{ch}", f"impressions_{ch}", f"clicks_{ch}", f"vendor_conv_qs_{ch}"):
            name = col.replace(f"_{ch}", "") + f"_{ch}_month"
            zipdf[name] = g[col].transform("sum")

    cty = zipdf.drop(columns=["zip", "month_start", "year", "dsp_b_vendor_live"]
                     + [c for c in zipdf.columns if c.endswith("_month") or c.startswith("arm_")
                        or c in ("dsp_a_served", "dsp_b_inclusion", "concurrent_test_flag")]
                     ).groupby(["county_id", "region", "week_idx", "week_start"], as_index=False).sum()
    cs = w.geo.counties.set_index("county_id")
    cty.insert(1, "county", cty["county_id"].map(cs["county"]))
    for d in (cty,):
        _add_rates(d)
    for d in (nat, reg):
        _add_rates(d, pop=False)
    return {"zip_weekly": zipdf, "county_weekly": cty, "region_weekly": reg, "national_weekly": nat}


def _add_rates(d: pd.DataFrame, pop: bool = True):
    d["rated_rate"] = np.where(d["qs"] > 0, d["rated"] / d["qs"].clip(lower=1), np.nan)
    d["close_rate"] = np.where(d["rated"] > 0, d["sales"] / d["rated"].clip(lower=1), np.nan)
    if pop and "pop" in d:
        d["qs_per_capita"] = d["qs"] / d["pop"]
    d["social_share_qs"] = np.where(d["qs"] > 0, d["lt_social_qs"] / d["qs"].clip(lower=1), np.nan)


# ======================================================================================
# Ground truth
# ======================================================================================
def _sum_rows(a: np.ndarray, rows, cols) -> float:
    return float(a[np.ix_(rows, cols)].sum())


def _scenario_truth(sc: Scenario, spend, ex, counts) -> Dict[str, dict]:
    w = sc.world
    out: Dict[str, dict] = {}
    for test in dict.fromkeys(iv.test for iv in sc.interventions):
        base = sc.without(test)
        sp0, em0 = _spend_and_effect_mult(base)
        ex0 = expected_outcomes(w, channel_effects(w, sp0, em0))
        ivs = [iv for iv in sc.interventions if iv.test == test]
        rows_all = np.unique(np.concatenate([iv.unit_idx for iv in ivs]))
        # realised counterfactual on affected rows only (CRN => identical draws)
        aff = rows_all
        for iv in ivs:
            if iv.spillover and iv.spillover.get("enabled", True):
                d = spillover_distance(w, iv.unit_idx)
                aff = np.union1d(aff, np.where(np.isfinite(d) & (d <= iv.spillover.get("radius_mi", 30)))[0])
        c0 = _draw_counts(w, ex0, rows=aff)
        pos = {u: i for i, u in enumerate(aff)}
        arms = {}
        for iv in ivs:
            cols = iv.weeks
            r = iv.unit_idx
            rr = np.array([pos[i] for i in r])
            rec = {"n_units": int(len(r)), "units": [str(u) for u in w.unit_ids[r]],
                   "pop": float(w.pop[r].sum()), "window": list(iv.window), "channels": list(iv.channels),
                   "spend_multiplier": iv.spend_multiplier, "leakage": iv.leakage}
            for s in C.STAGES:
                a = _sum_rows(ex["mu"][s], r, cols)
                cf = _sum_rows(ex0["mu"][s], r, cols)
                ra = float(counts[s][np.ix_(r, cols)].sum())
                rcf = float(c0[s][np.ix_(rr, cols)].sum())
                rec[s] = {"expected_actual": a, "expected_counterfactual": cf,
                          "true_incremental": cf - a, "true_lift_pct": (cf - a) / cf if cf else np.nan,
                          "realized_actual": ra, "realized_counterfactual": rcf,
                          "realized_lift_pct": (rcf - ra) / rcf if rcf else np.nan}
            removed = sum(_sum_rows(sp0[ch], r, cols) - _sum_rows(spend[ch], r, cols) for ch in iv.channels)
            rec["spend_removed"] = removed
            rec["bau_spend_in_window"] = sum(_sum_rows(sp0[ch], r, cols) for ch in iv.channels)
            for s in C.STAGES:
                inc = rec[s]["true_incremental"]
                rec[f"true_incremental_cost_per_{s}"] = removed / inc if inc > 0 else None
            # organic recapture (search): change in organic last-touch vs change in paid last-touch
            if any(ch in C.PAID_SEARCH for ch in iv.channels):
                for s in C.STAGES:
                    d_paid = sum(_sum_rows(ex0["lt"][s][ch], r, cols) - _sum_rows(ex["lt"][s][ch], r, cols)
                                 for ch in C.PAID_SEARCH)
                    d_org = _sum_rows(ex["lt"][s]["organic"], r, cols) - _sum_rows(ex0["lt"][s]["organic"], r, cols)
                    rec[f"true_recapture_{s}"] = d_org / d_paid if d_paid > 0 else None
            # spillover into untreated zips near the treated set
            if iv.spillover and iv.spillover.get("enabled", True):
                d = spillover_distance(w, iv.unit_idx)
                rad = iv.spillover.get("radius_mi", C.BUFFER_MILES_DEFAULT)
                buf = np.where(np.isfinite(d) & (d > 0) & (d <= rad))[0]
                rec["spillover"] = {"radius_mi": rad, "n_buffer_units": int(len(buf))}
                for s in C.STAGES:
                    cf = _sum_rows(ex0["mu"][s], buf, cols) if len(buf) else 0.0
                    a = _sum_rows(ex["mu"][s], buf, cols) if len(buf) else 0.0
                    rec["spillover"][s] = {"expected_loss": cf - a, "loss_pct": (cf - a) / cf if cf else 0.0}
            arms[iv.arm] = rec
        out[test] = arms
    return out


def channel_incremental(world: World, ch: str, spend_scale: float = 1.0,
                        weeks: Optional[np.ndarray] = None, units: Optional[np.ndarray] = None) -> dict:
    """Expected incremental visits/qs/rated/sales and spend of one channel under BAU (x scale)."""
    spend = {c: world.bau_spend[c] * (spend_scale if c == ch else 1.0) for c in world.cfg.channels}
    eff = channel_effects(world, spend)
    ex = expected_outcomes(world, {c: eff[c] for c in eff})
    rows = np.arange(world.U) if units is None else units
    cols = np.arange(world.T) if weeks is None else weeks
    out = {s: _sum_rows(ex["incr"][s][ch], rows, cols) for s in C.STAGES}
    out["spend"] = _sum_rows(spend[ch], rows, cols)
    out["lt"] = {s: _lt_for_channel(ex["lt"][s], ch, rows, cols) for s in C.STAGES}
    return out


def _lt_for_channel(lt, ch, rows, cols):
    if ch in ("search_brand", "search_nonbrand", "social"):
        return _sum_rows(lt[ch], rows, cols)
    if ch.startswith("dsp"):
        return None  # site last-touch "dsp" pools both vendors; see lt_dsp in ground truth
    return 0.0


def true_response_curve(world: World, ch: str, multipliers: Sequence[float] = (0, .25, .5, .75, 1, 1.25, 1.5, 2, 3)) -> pd.DataFrame:
    """Expected national incremental outcomes over the full timeline vs BAU spend multiplier."""
    rows = []
    for m in multipliers:
        r = channel_incremental(world, ch, spend_scale=m)
        rows.append({"channel": ch, "multiplier": m, "spend": r["spend"],
                     **{f"incr_{s}": r[s] for s in C.STAGES}})
    return pd.DataFrame(rows)


def compute_ground_truth(world: World) -> dict:
    cfg = world.cfg
    wp = cfg.world
    eff = channel_effects(world, world.bau_spend)
    ex = expected_outcomes(world, eff)
    years = {f"year{y + 1}": np.arange(y * 52, min((y + 1) * 52, world.T)) for y in range((world.T + 51) // 52)}
    allw = np.arange(world.T)
    allu = np.arange(world.U)
    tot = {s: float(ex["mu"][s].sum()) for s in C.STAGES}
    channels = {}
    for ch in cfg.channels:
        tr = cfg.truth[ch]
        spend = float(world.bau_spend[ch].sum())
        inc = {s: float(ex["incr"][s][ch].sum()) for s in C.STAGES}
        up = channel_incremental(world, ch, spend_scale=1.01)
        mroi = ((up["sales"] - inc["sales"]) * wp.value_per_sale / (up["spend"] - spend)) if spend > 0 else None
        rec = {
            "label": C.CHANNEL_LABELS[ch],
            "params": {"adstock_decay": tr.decay, "adstock_normalized": True, "hill_slope": tr.slope,
                       "half_sat_ratio": tr.half_sat_ratio,
                       "half_sat_spend_per_served_capita": world.half_sat[ch],
                       "half_sat_national_weekly_spend": world.half_sat[ch] * float((world.served[ch] * world.pop).sum()),
                       "max_effect": cfg.max_effect(ch), "stage_mult": list(tr.stage_mult),
                       "recapture": tr.recapture, "served_pop": float((world.served[ch] * world.pop).sum())},
            "total_spend": spend,
            "incremental": inc,
            "roi": inc["sales"] * wp.value_per_sale / spend if spend else None,
            "marginal_roi": mroi,
            "cost_per_incremental": {s: (spend / inc[s] if inc[s] > 0 else None) for s in C.STAGES},
            "incremental_per_1k_spend": {s: 1000 * inc[s] / spend if spend else None for s in C.STAGES},
            "contribution_share": {s: inc[s] / tot[s] for s in C.STAGES},
            "by_year": {},
        }
        for yname, cols in years.items():
            sp = _sum_rows(world.bau_spend[ch], allu, cols)
            iy = {s: _sum_rows(ex["incr"][s][ch], allu, cols) for s in C.STAGES}
            rec["by_year"][yname] = {"spend": sp, "incremental": iy,
                                     "roi": iy["sales"] * wp.value_per_sale / sp if sp else None,
                                     "cost_per_incremental_sales": sp / iy["sales"] if iy["sales"] > 0 else None}
        # last-touch claimed vs truth (site last-touch); DSP site LT pools both vendors
        lt_key = {"search_brand": "search_brand", "search_nonbrand": "search_nonbrand",
                  "social": "social"}.get(ch)
        if lt_key:
            lt = {s: float(ex["lt"][s][lt_key].sum()) for s in C.STAGES}
            rec["last_touch_claimed"] = lt
            rec["last_touch_bias"] = {s: lt[s] / inc[s] if inc[s] > 0 else None for s in C.STAGES}
            rec["last_touch_cost_per"] = {s: spend / lt[s] if lt[s] > 0 else None for s in C.STAGES}
        channels[ch] = rec
    lt_dsp = {s: float(ex["lt"][s]["dsp"].sum()) for s in C.STAGES}
    inc_dsp = {s: channels["dsp_a"]["incremental"][s] + channels["dsp_b"]["incremental"][s] for s in C.STAGES}
    # design-independent test truth: pause the channel(s) in every eligible unit of the region
    tests = {}
    for name, td in cfg.tests.items():
        rec = {"region": td.region, "unit_level": td.unit_level, "design_window": [td.design_start, td.design_end],
               "readout_pre_window": [td.readout_pre_start, td.test_start - 1],
               "test_window": [td.test_start, td.test_end], "leakage": td.leakage,
               "arms": {k: list(v) for k, v in td.channels_by_arm.items()}, "spillover": td.spillover,
               "notes": td.notes, "region_wide_holdout_lift": {}}
        for arm, chans in td.channels_by_arm.items():
            elig = np.where(world.unit_region == td.region)[0]
            sc = apply_intervention(world, world.unit_ids[elig], (td.test_start, td.test_end), chans,
                                    {"spend_multiplier": td.spend_multiplier, "leakage": td.leakage},
                                    test=name, arm=arm)
            sp, em = _spend_and_effect_mult(sc)
            ex1 = expected_outcomes(world, channel_effects(world, sp, em))
            cols = np.arange(td.test_start, td.test_end + 1)
            served = elig[np.max([world.served[c][elig] for c in chans], axis=0) > 0]
            r = {}
            for s in C.STAGES:
                cf = _sum_rows(ex["mu"][s], served, cols)
                a = _sum_rows(ex1["mu"][s], served, cols)
                r[s] = (cf - a) / cf
            removed = sum(_sum_rows(world.bau_spend[c], served, cols) - _sum_rows(sp[c], served, cols) for c in chans)
            inc_s = _sum_rows(ex["mu"]["sales"], served, cols) - _sum_rows(ex1["mu"]["sales"], served, cols)
            r["cost_per_incremental_sales"] = removed / inc_s if inc_s > 0 else None
            if any(c in C.PAID_SEARCH for c in chans):
                d_paid = sum(_sum_rows(ex["lt"]["qs"][c], served, cols) - _sum_rows(ex1["lt"]["qs"][c], served, cols)
                             for c in C.PAID_SEARCH)
                d_org = _sum_rows(ex1["lt"]["qs"]["organic"], served, cols) - _sum_rows(ex["lt"]["qs"]["organic"], served, cols)
                r["organic_recapture_qs"] = d_org / d_paid
            rec["region_wide_holdout_lift"][arm] = r
        tests[name] = rec
    gt = {
        "schema_version": SCHEMA_VERSION,
        "generator": {"seed": cfg.seed, "small": cfg.small, "n_weeks": world.T, "start_date": cfg.start_date,
                      "effect_multiplier": cfg.effect_multiplier, "n_zips": int(len(world.geo.zips)),
                      "n_counties": int(len(world.geo.counties)),
                      "regions": {r.code: r.name for r in cfg.regions}},
        "conventions": {
            "lift_sign": "positive = channel contribution = volume lost when the channel is paused",
            "roi": "incremental sales x value_per_sale / spend (expected values, whole timeline, incl. carryover)",
            "value_per_sale": wp.value_per_sale,
            "week_idx": "0-based weeks from start_date (Mondays)",
        },
        "totals": tot,
        "channels": channels,
        "search": {"true_recapture_brand": cfg.truth["search_brand"].recapture,
                   "true_recapture_nonbrand": cfg.truth["search_nonbrand"].recapture,
                   "note": "share of paid last-touch conversions that organic search would have captured anyway"},
        "dsp_site_last_touch": {"claimed": lt_dsp, "true_incremental": inc_dsp,
                                "bias": {s: lt_dsp[s] / inc_dsp[s] for s in C.STAGES}},
        "dsp_vendor_reporting": {ch: {"vendor_bias_vs_true_qs": cfg.truth[ch].vendor_bias} for ch in ("dsp_a", "dsp_b")},
        "dsp_b_vendor_switch": {"week_idx": C.DSP_B_VENDOR_SWITCH_WEEK,
                                "attribution_level_shift": C.DSP_B_SWITCH_ATTR_SHIFT - 1.0,
                                "reporting_dip_weeks": list(C.DSP_B_REPORT_DIP_WEEKS),
                                "true_effect_change": 0.0},
        "concurrent_test": C.CONCURRENT_TEST,
        "tests": tests,
        "attribution_bias_note": "last_touch_bias = last-touch claimed / true incremental (>1 over-claims)",
    }
    return _jsonable(gt)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else round(v, 8)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


# ======================================================================================
# High-level runners
# ======================================================================================
_WORLD_CACHE: Dict[tuple, World] = {}


def get_world(cfg: Optional[Config] = None) -> World:
    """Cached :func:`build_world` (keyed on seed/small/effect multiplier)."""
    cfg = cfg or get_config()
    key = (cfg.seed, cfg.small, cfg.effect_multiplier)
    if key not in _WORLD_CACHE:
        _WORLD_CACHE[key] = build_world(cfg)
    return _WORLD_CACHE[key]


def simulate(cfg: Optional[Config] = None, world: Optional[World] = None) -> Dict[str, pd.DataFrame]:
    """Baseline (BAU, no test interventions) panels."""
    world = world or get_world(cfg)
    return realize(Scenario(world), with_truth=False).panels


def run_scenario(selections: Dict[str, Dict[str, Iterable[str]]], world: Optional[World] = None,
                 cfg: Optional[Config] = None, overrides: Optional[Dict[str, dict]] = None) -> ScenarioResult:
    """Apply configured test designs to pipeline-selected units and realise the world.

    selections: {test_name: {arm: [zip or county ids]}}, e.g.
        {"meta": {"holdout": ["EM-C07"]},
         "search": {"all_search": ["NV-C03"], "branded_only": ["NV-C09"]},
         "dsp_a": {"holdout": [...zips...]}, "dsp_b": {"holdout": [...zips...]}}
    overrides: {test_name: effect_spec overrides (spend_multiplier/leakage/spillover/window)}.
    """
    world = world or get_world(cfg)
    sc = Scenario(world)
    for test, arms in selections.items():
        td = world.cfg.tests[test]
        ov = dict((overrides or {}).get(test, {}))
        window = tuple(ov.pop("window", (td.test_start, td.test_end)))
        spec = {"spend_multiplier": td.spend_multiplier, "leakage": td.leakage, "spillover": td.spillover}
        spec.update(ov)
        for arm, units in arms.items():
            if arm not in td.channels_by_arm:
                raise KeyError(f"test {test!r} has arms {list(td.channels_by_arm)}, got {arm!r}")
            sc = apply_intervention(sc, units, window, td.channels_by_arm[arm], spec, test=test, arm=arm)
    return realize(sc)


def apply_report_lag(panel: pd.DataFrame, as_of_week: int, cols: Sequence[str] = ("vendor_conv_qs_dsp_a", "vendor_conv_qs_dsp_b"),
                     lag: Sequence[float] = C.DSP_REPORT_LAG) -> pd.DataFrame:
    """Mimic vendor attribution lag: scale the last len(lag) weeks up to ``as_of_week``."""
    out = panel[panel["week_idx"] <= as_of_week].copy()
    for k, f in enumerate(reversed(lag)):
        wk = as_of_week - k
        m = out["week_idx"] == wk
        for c in cols:
            if c in out:
                out.loc[m, c] = np.round(out.loc[m, c] * f).astype(out[c].dtype)
    return out


def save_panels(panels: Dict[str, pd.DataFrame], out_dir: Path, static: Optional[Geography] = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in panels.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    if static is not None:
        static.zips.to_parquet(out_dir / "zip_static.parquet", index=False)
        static.counties.to_parquet(out_dir / "county_static.parquet", index=False)


def save_scenario(result: ScenarioResult, name: str, data_dir: Optional[Path] = None) -> Path:
    d = Path(data_dir or result.scenario.world.cfg.data_dir) / "scenarios" / name
    save_panels(result.panels, d)
    (d / "truth.json").write_text(json.dumps(_jsonable(result.truth), indent=2))
    return d


def write_baseline(cfg: Optional[Config] = None, data_dir: Optional[Path] = None) -> dict:
    cfg = cfg or get_config()
    world = get_world(cfg)
    d = Path(data_dir or cfg.data_dir)
    panels = simulate(world=world)
    save_panels(panels, d, static=world.geo)
    gt = compute_ground_truth(world)
    (d / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    return {"data_dir": str(d), "rows": {k: len(v) for k, v in panels.items()},
            "n_zips": int(len(world.geo.zips)), "n_counties": int(len(world.geo.counties))}


def load_table(name: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    d = Path(data_dir or get_config().data_dir)
    return pd.read_parquet(d / f"{name}.parquet")


def load_ground_truth(data_dir: Optional[Path] = None) -> dict:
    d = Path(data_dir or get_config().data_dir)
    return json.loads((d / "ground_truth.json").read_text())


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Generate the synthetic world (baseline panels + ground truth).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--effect-multiplier", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    t0 = time.time()
    info = write_baseline(get_config(seed=a.seed, small=a.small or None, effect_multiplier=a.effect_multiplier),
                          data_dir=Path(a.out) if a.out else None)
    info["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(info, indent=2))
