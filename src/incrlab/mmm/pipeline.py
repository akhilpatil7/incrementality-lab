"""MMM module orchestration: data -> calibration inputs -> uncalibrated + calibrated fits -> analysis.

``run()`` returns an :class:`MMMState` holding everything the figures / report need. Settings:

* full (default): 4 chains x 500 draws (700 tune) per fit, ~2-3 min for both fits on 4 cores.
* small (``small=True`` or ``INCRLAB_SMALL=1``): 2 chains x 150 draws - for tests / CI smoke only.

Training uses weeks 0..T-14; the last 13 weeks are a true holdout for MAPE / R^2. Every reported number
comes from those train-only fits (the optimizer replays the last 52 weeks, which includes the holdout
quarter - the baseline there is the model's extrapolation).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import CHANNELS, get_config
from . import analysis as A
from .calibration import CHANNEL_TEST, MIN_REL_SE, load_calibration_inputs
from .model import PRIORS, FitSettings, build_model, fit, prepare_data

log = logging.getLogger("incrlab.mmm")

HOLDOUT_WEEKS = 13
HORIZON = 52
BOUNDS = (0.7, 1.3)
CURVE_MULTS = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3])
FITS = ("uncalibrated", "calibrated")


@dataclass
class FitResult:
    name: str
    idata: object
    draws: A.Draws
    seconds: float
    diag: dict
    roi: np.ndarray
    mroi: np.ndarray
    curves: np.ndarray
    implied_ipd: np.ndarray


@dataclass
class MMMState:
    data: object
    gt: dict
    obs: List[dict]
    sources: Dict[str, str]
    stub_tests: List[str]
    settings: FitSettings
    train_mask: np.ndarray
    value_per_sale: float
    fits: Dict[str, FitResult] = field(default_factory=dict)
    truth: dict = field(default_factory=dict)
    optimizer: dict = field(default_factory=dict)
    voi: dict = field(default_factory=dict)
    seconds: float = 0.0
    small: bool = False
    figures: list = field(default_factory=list)
    powered_obs: List[dict] = field(default_factory=list)


def _truth(gt: dict, world, data, small: bool) -> dict:
    ch = data.channels
    t = {"roi": np.array([gt["channels"][c]["roi"] for c in ch]),
         "mroi": np.array([gt["channels"][c]["marginal_roi"] for c in ch]),
         "decay": np.array([gt["channels"][c]["params"]["adstock_decay"] for c in ch]),
         "incremental_sales": np.array([gt["channels"][c]["incremental"]["sales"] for c in ch]),
         "spend": np.array([gt["channels"][c]["total_spend"] for c in ch])}
    if world is not None and not small:
        from ..simulate import true_response_curve
        t["curves"] = np.array([true_response_curve(world, c, CURVE_MULTS)["incr_sales"].to_numpy() for c in ch])
    return t


def _fit_one(name, data, obs, settings, train_mask, V, seed) -> FitResult:
    m = build_model(data, obs if name != "uncalibrated" else None, train_mask=train_mask, value_per_sale=V)
    idata, sec = fit(m, settings)
    dr = A.get_draws(idata, max_draws=600, seed=seed)
    diag = A.diagnostics(idata, dr, data, train_mask, seed=seed)
    roi = A.roi_draws(dr, data, V)
    log.info("[mmm] %s fit: %.0fs, r-hat %.3f, ESS %.0f, divergences %d", name, sec, diag["max_r_hat"],
             diag["min_ess_bulk"], diag["divergences"])
    return FitResult(name, idata, dr, sec, diag, roi, A.marginal_roi_draws(dr, data, V),
                     A.response_curves(dr, data, CURVE_MULTS), A.implied_test_ipd(dr, data, obs))


def powered_observations(obs: List[dict], rel_se: float = 0.15, seed: int = 11) -> List[dict]:
    """Sensitivity ONLY: the same tests (same channels/windows/leakage) as if they had been well powered -
    one simulated readout ~ Normal(truth, rel_se x truth). Uses ground truth, so it never drives decisions;
    it isolates "is the calibration machinery right?" from "were our tests informative?"."""
    rng = np.random.default_rng(seed)
    out = []
    for o in obs:
        t = o.get("truth_incremental_per_dollar")
        if t is None:
            continue
        q = dict(o)
        se = max(rel_se * abs(float(t)), 1e-7)
        q["incremental_per_dollar"] = float(t) + float(rng.normal(0, se))
        q["se"] = se
        q["se_model"] = float(np.hypot(se, MIN_REL_SE * abs(float(t))))
        q["source"] = f"sensitivity:simulated_readout_{rel_se:.0%}_se"
        out.append(q)
    return out


def run(small: Optional[bool] = None, settings: Optional[FitSettings] = None, with_world: bool = True,
        sensitivity: Optional[bool] = None, voi_bins: int = 4) -> MMMState:
    from ..simulate import get_world, load_ground_truth, load_table

    t0 = time.time()
    cfg = get_config()
    small = cfg.small if small is None else small
    settings = settings or (FitSettings.small() if small else FitSettings(draws=500, tune=700, chains=4))
    sensitivity = (not small) if sensitivity is None else sensitivity
    world = get_world() if with_world else None
    nat = load_table("national_weekly")
    gt = load_ground_truth()
    V = float(cfg.world.value_per_sale)
    data = prepare_data(nat, CHANNELS)
    obs, sources = load_calibration_inputs(world=world)
    stub_tests = [t for t, s in sources.items() if s.startswith("stub")]
    if stub_tests:
        log.warning("[mmm] !!! STUB calibration inputs used for %s (module results missing) !!!", stub_tests)
    train = np.ones(data.T, bool)
    train[-HOLDOUT_WEEKS:] = False
    S = MMMState(data, gt, obs, sources, stub_tests, settings, train, V, small=small)
    S.truth = _truth(gt, world, data, small)

    names = list(FITS)
    pw = powered_observations(obs) if sensitivity else []
    S.powered_obs = pw
    if len(pw) == len(obs) and pw:
        names.append("calibrated_powered")
    for k, name in enumerate(names):
        o = pw if name == "calibrated_powered" else obs
        S.fits[name] = _fit_one(name, data, o, settings, train, V, settings.seed + k)
    # implied-ipd of the powered fit is evaluated on the real observation list for comparability
    if "calibrated_powered" in S.fits:
        S.fits["calibrated_powered"].implied_ipd = A.implied_test_ipd(S.fits["calibrated_powered"].draws, data, obs)

    # ---- optimizer on the calibrated posterior (+ uncalibrated for contrast)
    C = data.C
    one = np.ones(C)
    for name in S.fits:
        bp = A.BudgetProblem(S.fits[name].draws, data, V, HORIZON)
        m_opt = bp.optimize(1.0, BOUNDS)
        res = {"status_quo": bp.evaluate(one), "optimal": bp.evaluate(m_opt, ref=one), "scenarios": {}}
        for tot in (0.8, 1.2):
            ms = bp.optimize(tot, BOUNDS)
            res["scenarios"][f"{tot:.1f}x"] = {"proportional": bp.evaluate(one * tot, ref=one),
                                              "optimized": bp.evaluate(ms, ref=one)}
        if world is not None and not small:
            tq = A.truth_plan_incremental(world, data, dict(zip(data.channels, one)), HORIZON)
            to = A.truth_plan_incremental(world, data, dict(zip(data.channels, m_opt)), HORIZON)
            res["truth_check"] = {"status_quo_true_incremental": tq, "optimal_true_incremental": to,
                                  "true_gain": to - tq, "true_gain_pct": (to - tq) / tq}
            for key, sc in res["scenarios"].items():
                for kind in ("proportional", "optimized"):
                    sc[kind]["true_incremental"] = A.truth_plan_incremental(
                        world, data, sc[kind]["multipliers"], HORIZON)
        res["budget"] = bp.budget0
        res["horizon_weeks"] = HORIZON
        res["bounds"] = BOUNDS
        S.optimizer[name] = res
        if name == "calibrated":
            S.voi = A.value_of_information(bp, S.fits[name].draws, n_bins=voi_bins, bounds=BOUNDS, m_star=m_opt)
    S.seconds = time.time() - t0
    return S


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def roi_table(S: MMMState):
    import pandas as pd
    rows = []
    for j, c in enumerate(S.data.channels):
        r = {"channel": c, "true_roi": S.truth["roi"][j], "true_mroi": S.truth["mroi"][j]}
        for name, f in S.fits.items():
            sm = A.summarize(f.roi[:, j])
            mm = A.summarize(f.mroi[:, j])
            r[f"{name}_roi"] = sm["mean"]
            r[f"{name}_roi_lo"], r[f"{name}_roi_hi"] = sm["hdi_low"], sm["hdi_high"]
            r[f"{name}_roi_abs_err"] = abs(sm["mean"] - r["true_roi"])
            r[f"{name}_roi_rel_err"] = (sm["mean"] - r["true_roi"]) / r["true_roi"]
            r[f"{name}_roi_covers_truth"] = bool(sm["hdi_low"] <= r["true_roi"] <= sm["hdi_high"])
            r[f"{name}_mroi"] = mm["mean"]
            r[f"{name}_mroi_lo"], r[f"{name}_mroi_hi"] = mm["hdi_low"], mm["hdi_high"]
        rows.append(r)
    return pd.DataFrame(rows)


def triangulation_table(S: MMMState):
    """Per channel ROI view: last-touch vs experiment vs uncalibrated vs calibrated MMM vs truth."""
    import pandas as pd
    V = S.value_per_sale
    rows = []
    obs_by_ch = {}
    for o in S.obs:
        ch = o.get("label_channel") or (o["channels"][0] if len(o["channels"]) == 1 else None)
        if ch is None:
            ch = next((c for c, (t, a) in CHANNEL_TEST.items() if t == o["test"] and a == o["arm"]), None)
        if ch:
            obs_by_ch[ch] = o
    for j, c in enumerate(S.data.channels):
        o = obs_by_ch.get(c)
        rows.append({
            "channel": c,
            "last_touch_roi": S.data.lt_sales[c] * V / S.data.spend[j].sum(),
            "experiment_roi": (o["incremental_per_dollar"] * V) if o else None,
            "experiment_roi_se": (o["se"] * V) if o else None,
            "experiment_source": o["source"] if o else "no test",
            "uncalibrated_roi": float(S.fits["uncalibrated"].roi[:, j].mean()),
            "calibrated_roi": float(S.fits["calibrated"].roi[:, j].mean()),
            "true_roi": float(S.truth["roi"][j]),
        })
    return pd.DataFrame(rows)


def calibration_fit_table(S: MMMState):
    import pandas as pd
    rows = []
    for i, o in enumerate(S.obs):
        r = {"test": o["test"], "arm": o["arm"], "channels": "+".join(o["channels"]), "window": o["window"],
             "observed_ipd": o["incremental_per_dollar"], "se": o["se"], "se_model": o["se_model"],
             "truth_ipd": o.get("truth_incremental_per_dollar"), "source": o["source"],
             "se_source": o.get("se_source")}
        for name, f in S.fits.items():
            r[f"{name}_ipd"] = float(f.implied_ipd[:, i].mean())
            lo, hi = A.hdi(f.implied_ipd[:, i])
            r[f"{name}_ipd_lo"], r[f"{name}_ipd_hi"] = float(lo), float(hi)
        rows.append(r)
    return pd.DataFrame(rows)


def settings_summary(S: MMMState) -> dict:
    s = S.settings
    return {"draws": s.draws, "tune": s.tune, "chains": s.chains, "target_accept": s.target_accept,
            "holdout_weeks": HOLDOUT_WEEKS, "priors": PRIORS, "min_rel_se_transport": MIN_REL_SE,
            "small": S.small, "env_small": os.environ.get("INCRLAB_SMALL", "0")}
