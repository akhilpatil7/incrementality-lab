"""Experiment -> MMM calibration inputs.

The MMM is calibrated with one observation per geo test arm:

    incremental sales per $ of spend removed, measured inside the test window

Schema of one observation (validated by :func:`validate_observation`)::

    {"test": "meta", "arm": "holdout", "channels": ["social"], "window": [58, 65],
     "leakage": 0.25,                       # share of BAU spend that kept serving in treated geos
     "metric": "sales",
     "incremental_per_dollar": 0.00134,     # incremental sales / $ removed (positive = contribution)
     "se": 0.0005,                          # standard error of incremental_per_dollar
     "spend_removed": 412000.0,             # $ (informational)
     "incremental": 552.0,                  # incremental sales (informational)
     "source": "reports/results/meta.json", # or "stub:did"
     "truth_incremental_per_dollar": ...}   # optional, validation only - never used by the model

Sources, in order of preference (logged per test):

1. ``reports/results/{meta,search,dsp}.json`` written by the test modules. Each module populates
   ``extra.incremental_per_dollar = {channel: {"sales": x, "spend": $, "window": [s, e], ...}}``
   (optional keys: ``se``/``sales_se``, ``arm``, ``channels``, ``leakage``, ``incremental_sales``).
   If no SE is given it is derived from the module's sales-metric estimate CI for the same test/arm.
2. Fallback stub (``data/calibration_inputs.json``): a quick, self-contained geo test per arm run
   through the scenario runner with a change-ratio DiD + week bootstrap SE. Used only for tests
   whose module output is missing, so the MMM module can be developed/run in parallel with the
   test modules. The final pipeline (``make all``) runs the test modules first, so real outputs win.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..config import DATA_DIR, RESULTS_DIR, get_config

log = logging.getLogger("incrlab.mmm")

CHANNEL_TEST = {"social": ("meta", "holdout"), "search_brand": ("search", "branded_only"),
                "search_nonbrand": ("search", "all_search"), "dsp_a": ("dsp_a", "holdout"),
                "dsp_b": ("dsp_b", "holdout")}
MODULE_TESTS = {"meta": ("meta",), "search": ("search",), "dsp": ("dsp_a", "dsp_b")}
REQUIRED = {"test": str, "arm": str, "channels": list, "window": list, "leakage": float,
            "metric": str, "incremental_per_dollar": float, "se": float}
DEFAULT_REL_SE = 0.5          # used only when a module gives neither an SE nor a CI
MIN_REL_SE = 0.10             # floor: geo -> national transport / model-mismatch error
STUB_FILE = "calibration_inputs.json"


class CalibrationSchemaError(ValueError):
    pass


def _rel(path) -> str:
    """Repo-relative path for reports (never leak absolute local paths into committed JSON)."""
    from ..config import REPO_ROOT
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (ValueError, TypeError):
        return Path(str(path)).name


def validate_observation(o: dict) -> dict:
    """Check types/ranges of one calibration observation; returns a normalised copy."""
    missing = [k for k in REQUIRED if k not in o or o[k] is None]
    if missing:
        raise CalibrationSchemaError(f"calibration observation missing {missing}: {o}")
    out = dict(o)
    out["channels"] = [str(c) for c in o["channels"]]
    cfg = get_config()
    bad = [c for c in out["channels"] if c not in cfg.channels]
    if bad or not out["channels"]:
        raise CalibrationSchemaError(f"unknown channels {bad} in {o}")
    w = [int(x) for x in o["window"]]
    if len(w) != 2 or not (0 <= w[0] <= w[1] < cfg.n_weeks):
        raise CalibrationSchemaError(f"bad window {o['window']}")
    out["window"] = w
    for k in ("leakage", "incremental_per_dollar", "se"):
        out[k] = float(o[k])
        if not np.isfinite(out[k]):
            raise CalibrationSchemaError(f"{k} not finite in {o}")
    if not (0.0 <= out["leakage"] < 1.0):
        raise CalibrationSchemaError(f"leakage must be in [0,1): {o}")
    if out["se"] <= 0:
        raise CalibrationSchemaError(f"se must be > 0: {o}")
    if out["metric"] != "sales":
        raise CalibrationSchemaError(f"MMM is calibrated on sales; got metric={out['metric']}")
    return out


# --------------------------------------------------------------------------------------
# 1. module outputs
# --------------------------------------------------------------------------------------
def _se_from_estimates(doc: dict, test: str, arm: str, ipd: float) -> Tuple[Optional[float], str]:
    """Relative CI half-width of the module's sales estimate for (test, arm) -> SE on ipd."""
    best = None
    for e in doc.get("estimates", []):
        if e.get("test") != test or e.get("metric") != "sales":
            continue
        if e.get("arm") not in (arm, None):
            continue
        est, lo, hi = e.get("estimate"), e.get("ci_low"), e.get("ci_high")
        if est is None or lo is None or hi is None or abs(est) < 1e-9:
            continue
        lvl = float(e.get("ci_level") or 0.90)
        z = stats.norm.ppf(0.5 + lvl / 2)
        rel = (hi - lo) / (2 * z) / abs(est)
        pref = 0 if "did" in str(e.get("method", "")) else 1
        if best is None or pref < best[0]:
            best = (pref, rel, e.get("method"))
    if best is None:
        return None, "none"
    return abs(ipd) * best[1], f"ci_of_{best[2]}"


def _obs_from_module(module: str, doc: dict, path: Path) -> List[dict]:
    cfg = get_config()
    ipd_map = (doc.get("extra") or {}).get("incremental_per_dollar") or {}
    obs = []
    for ch, rec in ipd_map.items():
        if not isinstance(rec, dict):
            continue
        test, arm = CHANNEL_TEST.get(ch, (None, None))
        test = rec.get("test", test)
        arm = rec.get("arm", arm)
        if test not in cfg.tests:
            log.warning("skip %s/%s: unknown test %r", module, ch, test)
            continue
        td = cfg.tests[test]
        spend = rec.get("spend")
        ipd = rec.get("sales_per_dollar", rec.get("sales"))
        if ipd is None and rec.get("incremental_sales") is not None and spend:
            ipd = rec["incremental_sales"] / spend
        if ipd is None:
            continue
        ipd = float(ipd)
        if abs(ipd) > 0.05 and spend:  # value is a count of incremental sales, not per $
            log.warning("%s/%s: 'sales'=%.3g looks like a count; dividing by spend", module, ch, ipd)
            ipd = ipd / float(spend)
        se = rec.get("se", rec.get("sales_se", rec.get("se_sales")))
        se_src = "module_se"
        if se is None:
            se, se_src = _se_from_estimates(doc, test, arm, ipd)
        if se is None or not np.isfinite(se) or se <= 0:
            se, se_src = abs(ipd) * DEFAULT_REL_SE + 1e-6, f"default_{DEFAULT_REL_SE:.0%}_rel"
        chans = rec.get("channels") or [ch]
        window = rec.get("window") or [td.test_start, td.test_end]
        obs.append({"test": test, "arm": arm, "channels": list(chans), "window": list(window),
                    "leakage": float(rec.get("leakage", td.leakage)), "metric": "sales",
                    "incremental_per_dollar": ipd, "se": float(se), "se_source": se_src,
                    "spend_removed": spend, "incremental": rec.get("incremental_sales"),
                    "source": _rel(path), "label_channel": ch,
                    "truth_incremental_per_dollar": rec.get("truth_sales_per_dollar", rec.get("sales_truth", rec.get("truth")))})
    return obs


def load_module_observations(results_dir: Optional[Path] = None) -> Dict[str, List[dict]]:
    """{test_name: [obs, ...]} from whatever module result files exist."""
    d = Path(results_dir or RESULTS_DIR)
    out: Dict[str, List[dict]] = {}
    for module in MODULE_TESTS:
        p = d / f"{module}.json"
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text())
            for o in _obs_from_module(module, doc, p):
                out.setdefault(o["test"], []).append(validate_observation(o))
        except (CalibrationSchemaError, ValueError, KeyError, TypeError) as exc:
            log.warning("could not use %s (%s); will fall back to stub for its tests", p, exc)
    return out


# --------------------------------------------------------------------------------------
# 2. fallback stub: quick geo tests through the scenario runner
# --------------------------------------------------------------------------------------
def _blob(zips, eligible_mask, target_share, rng):
    """Contiguous-ish patch: grow around a random seed zip until target share of eligible pop."""
    from ..geo import haversine_miles
    el = zips[eligible_mask]
    seed = el.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
    d = haversine_miles(el["lat"].to_numpy(), el["lon"].to_numpy(), seed["lat"], seed["lon"])
    order = np.argsort(d)
    cum = np.cumsum(el["pop"].to_numpy()[order]) / el["pop"].sum()
    k = int(np.searchsorted(cum, target_share)) + 1
    return el["zip"].to_numpy()[order[:k]].tolist()


def build_stub_observations(world=None, seed: int = 7, n_boot: int = 400) -> Dict[str, List[dict]]:
    """Run one simple, pipeline-chosen geo test per arm and read it out with a change-ratio DiD."""
    from ..common.did import change_ratio_did
    from ..simulate import get_world, run_scenario, spillover_distance

    world = world or get_world()
    cfg = world.cfg
    z = world.geo.zips
    rng = np.random.default_rng(seed)
    sel: Dict[str, Dict[str, list]] = {}
    counties = world.geo.counties
    em = counties[counties["region"] == "EM"]["county_id"].tolist()
    nv = counties[counties["region"] == "NV"]["county_id"].tolist()
    pick = list(rng.permutation(em))
    sel["meta"] = {"holdout": pick[:3]}
    pick = list(rng.permutation(nv))
    sel["search"] = {"all_search": pick[:2], "branded_only": pick[2:4]}
    a_mask = ((z["region"] == "NV") & (z["dsp_a_served"] == 1)).to_numpy()
    b_mask = ((z["region"] == "EM") & (z["dsp_b_inclusion"] == 1) & (z["concurrent_test_flag"] == 0)).to_numpy()
    sel["dsp_a"] = {"holdout": _blob(z, a_mask, 0.20, rng)}
    sel["dsp_b"] = {"holdout": _blob(z, b_mask, 0.20, rng)}
    res = run_scenario(sel, world=world)
    zw = res.panels["zip_weekly"]
    out: Dict[str, List[dict]] = {}
    for test, arms in sel.items():
        td = cfg.tests[test]
        pre = list(range(td.readout_pre_start, td.test_start))
        post = list(range(td.test_start, td.test_end + 1))
        reg = zw[zw["region"] == td.region]
        treated_all = set(reg.loc[reg[f"arm_{test}"] != "", "zip"])
        if test.startswith("dsp"):
            mask = a_mask if test == "dsp_a" else b_mask
            idx = world.unit_index(sorted(treated_all))
            dist = spillover_distance(world, idx)[: len(z)]
            ctrl_zips = set(z.loc[mask & (dist > td.spillover["radius_mi"]), "zip"])
        else:
            ctrl_zips = set(reg["zip"]) - treated_all
        ctrl = reg[reg["zip"].isin(ctrl_zips)].groupby("week_idx")["sales"].sum()
        for arm in arms:
            tz = set(reg.loc[reg[f"arm_{test}"] == arm, "zip"])
            tr = reg[reg["zip"].isin(tz)].groupby("week_idx")["sales"].sum()
            r = change_ratio_did(tr, ctrl, pre, post, sign=-1, n_boot=n_boot, seed=seed)
            t = res.truth[test][arm]
            removed = float(t["spend_removed"])
            incr = float(r["abs_effect"])
            se = float(r["boot_se"]) * float(r["counterfactual_post_total"]) / removed
            ipd = incr / removed
            o = {"test": test, "arm": arm, "channels": list(td.channels_by_arm[arm]),
                 "window": [td.test_start, td.test_end], "leakage": td.leakage, "metric": "sales",
                 "incremental_per_dollar": ipd, "se": max(se, MIN_REL_SE * abs(ipd)),
                 "se_source": "stub_week_bootstrap", "spend_removed": removed, "incremental": incr,
                 "lift": r["lift"], "n_treated_units": len(tz), "n_control_units": len(ctrl_zips),
                 "source": "stub:change_ratio_did",
                 "truth_incremental_per_dollar": t["sales"]["true_incremental"] / removed}
            out.setdefault(test, []).append(validate_observation(o))
    return out


def load_stub(data_dir: Optional[Path] = None, rebuild: bool = False, world=None) -> Dict[str, List[dict]]:
    p = Path(data_dir or DATA_DIR) / STUB_FILE
    cfg = get_config()
    if p.exists() and not rebuild:
        doc = json.loads(p.read_text())
        if doc.get("seed") == cfg.seed and doc.get("small") == cfg.small:
            return {t: [validate_observation(o) for o in obs] for t, obs in doc["observations"].items()}
    obs = build_stub_observations(world=world)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"seed": cfg.seed, "small": cfg.small,
                             "note": "Fallback MMM calibration inputs (quick DiD geo tests via the scenario "
                                     "runner). Superseded by reports/results/{meta,search,dsp}.json.",
                             "observations": obs}, indent=2, default=float))
    return obs


# --------------------------------------------------------------------------------------
# public loader
# --------------------------------------------------------------------------------------
def load_calibration_inputs(results_dir: Optional[Path] = None, data_dir: Optional[Path] = None,
                            allow_stub: bool = True, world=None) -> Tuple[List[dict], Dict[str, str]]:
    """Return (observations, source_by_test). Module results win; stub fills missing tests."""
    cfg = get_config()
    mod = load_module_observations(results_dir)
    sources: Dict[str, str] = {}
    obs: List[dict] = []
    missing = []
    for test in cfg.tests:
        if mod.get(test):
            obs.extend(mod[test])
            sources[test] = mod[test][0]["source"]
        else:
            missing.append(test)
    if missing:
        if not allow_stub:
            raise FileNotFoundError(f"no module calibration inputs for tests {missing}")
        stub = load_stub(data_dir, world=world)
        for test in missing:
            obs.extend(stub.get(test, []))
            sources[test] = "stub:data/calibration_inputs.json"
    for o in obs:
        o["se_model"] = float(np.hypot(o["se"], MIN_REL_SE * abs(o["incremental_per_dollar"])))
    for test, src in sources.items():
        log.info("calibration input %-6s <- %s", test, src)
    return obs, sources
