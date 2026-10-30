"""Readout stage: realise the selected design, then triangulate four estimators per arm x metric.

Estimators (all report *contribution* = volume lost when search is dark, positive):
  * ratio DiD      - counterfactual = arm pre-mean x control post/pre ratio; week bootstrap CI.
  * OLS DiD (FE)   - unit + week fixed effects on a pre-mean-indexed per-capita outcome, population
                     weights, cluster-robust SE by unit (arm + control + all BAU donors) PLUS a
                     permutation p-value over placebo-treated donors (one treated cluster makes
                     cluster SEs optimistic - the permutation p is the honest one).
  * BSTS           - CausalImpact-style local-level model with every donor as a covariate;
                     significance ONLY from whether the interval excludes 0.
  * Synthetic ctrl - locked pre-period weights; in-space placebo p (rank of post/pre RMSPE ratio,
                     min attainable p reported) and a placebo-in-time interval.
Then: power (closed-form, TTestIndPower, simulation), cannibalisation / organic recapture,
incremental vs last-touch cost per outcome, traffic guardrail, MMM calibration inputs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..common import power as pw
from ..common.bsts import causal_impact
from ..common.did import change_ratio_did, did_ols_fe, welch_gap_test
from ..common.inference import jackknife_donors, placebo_in_space, placebo_in_time
from ..common.synth import apply_weights, fit_synth, lift_from_series
from . import panel as P
from .settings import ARMS, SOURCES, SearchSettings, wk

STAGES = ("qs", "rated", "sales")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))


def readout_units(panel: pd.DataFrame, design: dict) -> pd.DataFrame:
    """Each arm = one aggregated unit; control and BAU donors stay as counties."""
    donors = design["control"] + design["bau_donors"]
    return P.unit_panel(panel, {a: design[a] for a in ARMS}, donors)


def _donors(design: dict) -> List[str]:
    return design["control"] + design["bau_donors"]


def truth_for(truth: dict, arm: str, metric: str, kind: str = "true_lift_pct") -> Optional[float]:
    try:
        return float(truth[arm][metric][kind])
    except (KeyError, TypeError):
        return None


# --------------------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------------------
def est_ratio_did(up: pd.DataFrame, arm: str, control: str, metric: str, s: SearchSettings) -> dict:
    t = up[up["county_id"] == arm].set_index("week_idx")[f"{metric}_pc"]
    c = up[up["county_id"] == control].set_index("week_idx")[f"{metric}_pc"]
    r = change_ratio_did(t, c, wk(s.pre), wk(s.post), sign=-1, n_boot=s.n_boot, seed=s.seed, alpha=s.alpha)
    # ratio stays "contribution = (cf - actual)/cf" - the original inverted it in places
    se = r["boot_se"]
    zstat = r["lift"] / se if se > 0 else np.nan
    return {"estimate": r["lift"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "p_value": float(2 * stats.norm.sf(abs(zstat))) if np.isfinite(zstat) else None,
            "se": se, "control_ratio": r["control_ratio"],
            "notes": f"cf = arm pre-mean x control ({control}) post/pre ratio; {s.n_boot} week-bootstrap draws"}


def _ols_panel(up: pd.DataFrame, metric: str, s: SearchSettings, treated: str, exclude: Sequence[str]) -> pd.DataFrame:
    weeks = wk((s.pre[0], s.post[1]))
    d = up[up["week_idx"].isin(weeks) & ~up["county_id"].isin(list(exclude))].copy()
    pre_mean = d[d["week_idx"] <= s.pre[1]].groupby("county_id")[f"{metric}_pc"].mean()
    d["y_idx"] = d[f"{metric}_pc"] / d["county_id"].map(pre_mean)
    d["treated"] = (d["county_id"] == treated).astype(int)
    d["post"] = (d["week_idx"] >= s.post[0]).astype(int)
    return d


def est_ols_fe(up: pd.DataFrame, arm: str, design: dict, metric: str, s: SearchSettings) -> dict:
    other = [a for a in ARMS if a != arm]
    d = _ols_panel(up, metric, s, arm, other)
    r = did_ols_fe(d, "y_idx", "county_id", "week_idx", "treated", "post", cluster="unit",
                   weights="pop", alpha=s.alpha)
    # permutation (randomisation) inference: pretend each donor was treated
    null = []
    for dn in _donors(design):
        dd = _ols_panel(up, metric, s, dn, other + [arm])
        null.append(did_ols_fe(dd, "y_idx", "county_id", "week_idx", "treated", "post", cluster=None,
                               weights="pop", alpha=s.alpha)["beta"])
    null = np.asarray(null)
    p_perm = float((1 + np.sum(np.abs(null) >= abs(r["beta"]))) / (len(null) + 1))
    # With ONE treated cluster, cluster-robust SEs are badly anti-conservative (they only see
    # residual noise, not the between-unit dispersion of post-period shocks). Inference therefore
    # uses the permutation distribution: SE = sd of placebo betas, p = permutation rank.
    se_perm = float(np.std(null, ddof=1)) if len(null) > 2 else np.nan
    z = _z(s.alpha)
    tp = (d["treated"] == 1) & (d["post"] == 1)
    base = float(d.loc[tp, "y_idx"].mean() - r["beta"])  # counterfactual index level of treated-post cells
    est = -r["beta"] / base
    se_rel = se_perm / base
    return {"estimate": est, "ci_low": est - z * se_rel, "ci_high": est + z * se_rel,
            "p_value": p_perm, "significant": bool(p_perm < s.alpha), "se": se_rel,
            "p_cluster": r["p_value"], "se_cluster": r["se"],
            "ci_cluster": (-r["rel_ci_high"], -r["rel_ci_low"]), "min_attainable_p": 1 / (len(null) + 1),
            "se_type": r["se_type"], "n_units": r["n_units"], "n_obs": r["n_obs"], "beta": r["beta"],
            "null_betas": null.tolist(), "base": base,
            "notes": (f"unit+week FE, pop-weighted, y indexed to unit pre-mean; reported {r['se_type']} SE "
                      f"(p={r['p_value']:.3f}) is anti-conservative with 1 treated cluster -> CI/p from "
                      f"{len(null)} donor-placebo permutations (min attainable p {1 / (len(null) + 1):.3f})")}


def est_bsts(up: pd.DataFrame, arm: str, design: dict, metric: str, s: SearchSettings) -> dict:
    W = P.wide(up, f"{metric}_pc", wk((s.bsts_pre_start, s.post[1])))
    y, X = W[arm], W[_donors(design)]
    r = causal_impact(y, X, wk((s.bsts_pre_start, s.pre[1])), wk(s.post), alpha=s.alpha,
                      n_sims=s.bsts_sims, seed=s.seed, sign=-1)
    se = (r["rel_ci_high"] - r["rel_ci_low"]) / (2 * _z(s.alpha))
    return {"estimate": r["rel_effect"], "ci_low": r["rel_ci_low"], "ci_high": r["rel_ci_high"],
            "p_value": r["p_like"], "significant": r["significant"], "se": se, "model": r["model"],
            "pointwise": r["pointwise"], "pre_mape": r["pre_mape"],
            "notes": "significance = interval excludes 0 (p_like shown for reference only)"}


def est_sc(up: pd.DataFrame, arm: str, design: dict, metric: str, s: SearchSettings, placebo: bool = True) -> dict:
    W = P.wide(up, f"{metric}_pc", wk((s.sc_fit_start, s.post[1])))
    pre, post = wk((s.sc_fit_start, s.pre[1])), wk(s.post)
    donors = _donors(design)
    fit = fit_synth(W, arm, pre, donors=donors, ridge=s.sc_ridge)
    l = fit.lift(post, sign=-1)
    out = {"estimate": l["lift"], "fit": fit, "pre_rel_rmse": fit.rel_rmse,
           "weights": fit.weights[fit.weights > 0].round(4).to_dict()}
    if placebo:
        Wd = W[[arm] + donors]
        pis = placebo_in_space(Wd, arm, pre, post, donors=donors, stat="rmspe_ratio",
                               fit_kw={"ridge": s.sc_ridge})
        pit = placebo_in_time(Wd, arm, pre, s.n_post, n_fit_min=26, donors=donors, fit_kw={"ridge": s.sc_ridge})
        sd = pit["sd_placebo_lift"]
        z = _z(s.alpha)
        out.update({"ci_low": l["lift"] - z * sd, "ci_high": l["lift"] + z * sd, "se": sd,
                    "p_value": pis["p_value"], "significant": bool(pis["p_value"] <= s.alpha), "min_attainable_p": pis["min_attainable_p"],
                    "placebo_space": pis["placebos"], "treated_stat": pis["treated_stat"],
                    "placebo_time": pit["placebos"], "n_placebo_time": pit["n_placebos"],
                    "notes": (f"locked weights fit on weeks {s.sc_fit_start}-{s.pre[1]}; p = in-space RMSPE-ratio rank "
                              f"(min attainable {pis['min_attainable_p']:.3f}); CI = +/- z x sd of "
                              f"{pit['n_placebos']} placebo-in-time lifts")})
    return out


METHODS = {"ratio_did": est_ratio_did, "ols_fe": est_ols_fe, "bsts": est_bsts, "synth_control": est_sc}


def run_estimators(up: pd.DataFrame, design: dict, truth: dict, bau_up: pd.DataFrame, s: SearchSettings) -> dict:
    """All methods x arms x metrics. Returns rows (for the JSON) and rich objects (for charts)."""
    rows, rich = [], {}
    ctl = design["control"][0]
    for arm in ARMS:
        for m in s.readout_metrics:
            for name in METHODS:
                if name == "ratio_did":
                    r = est_ratio_did(up, arm, ctl, m, s)
                else:
                    r = METHODS[name](up, arm, design, m, s)
                rich[(arm, m, name)] = r
                sig = r.get("significant")
                if sig is None:
                    sig = bool(r["ci_low"] > 0 or r["ci_high"] < 0)
                ci_excl = bool(r["ci_low"] > 0 or r["ci_high"] < 0)
                rows.append({"test": "search", "arm": arm, "metric": m, "method": name,
                             "ci_excludes_0": ci_excl,
                             "p_ci_consistent": bool(ci_excl == ((r.get("p_value") or 1.0) < s.alpha)),
                             "estimate": r["estimate"], "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                             "ci_level": 1 - s.alpha, "p_value": r.get("p_value"),
                             "p_cluster": r.get("p_cluster"), "se": r.get("se"),
                             "significant": bool(sig),
                             "truth": truth_for(truth, arm, m), "truth_key": f"scenario.search.{arm}.{m}.true_lift_pct",
                             "truth_realized": truth_for(truth, arm, m, "realized_lift_pct"),
                             "units": "relative lift (positive = paid search contribution)",
                             "notes": r.get("notes", "")})
    est = pd.DataFrame(rows)
    est["error_vs_truth"] = est["estimate"] - est["truth"]
    est["truth_in_ci"] = (est["truth"] >= est["ci_low"]) & (est["truth"] <= est["ci_high"])
    return {"table": est, "rich": rich}


def triangulate(est: pd.DataFrame, s: SearchSettings) -> pd.DataFrame:
    """Inverse-variance combination across methods (no diversification credit: same data)."""
    rows = []
    for (arm, m), g in est.groupby(["arm", "metric"], sort=False):
        g = g.dropna(subset=["se"])
        g = g[g["se"] > 0]
        w = 1 / g["se"] ** 2
        e = float((w * g["estimate"]).sum() / w.sum())
        se = float(np.sqrt(np.mean(g["se"] ** 2)))
        z = _z(s.alpha)
        rows.append({"test": "search", "arm": arm, "metric": m, "method": "triangulated", "estimate": e,
                     "ci_low": e - z * se, "ci_high": e + z * se, "ci_level": 1 - s.alpha, "se": se,
                     "p_value": float(2 * stats.norm.sf(abs(e / se))), "significant": bool(abs(e / se) > z),
                     "truth": g["truth"].iloc[0], "truth_key": g["truth_key"].iloc[0],
                     "truth_realized": g["truth_realized"].iloc[0],
                     "units": "relative lift (positive = paid search contribution)",
                     "notes": "inverse-variance mean of the 4 methods; SE = RMS of method SEs (methods share data)"})
    t = pd.DataFrame(rows)
    t["error_vs_truth"] = t["estimate"] - t["truth"]
    t["truth_in_ci"] = (t["truth"] >= t["ci_low"]) & (t["truth"] <= t["ci_high"])
    return t


# --------------------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------------------
def power_analysis(up_design: pd.DataFrame, design: dict, observed: pd.DataFrame, s: SearchSettings) -> dict:
    """Closed-form DiD MDE / weeks-needed, TTestIndPower and simulation power (design data only)."""
    rows, curves, sims = [], [], []
    hist = wk((s.sc_fit_start, s.design_end))
    for arm in ARMS:
        for m in s.readout_metrics:
            W = P.wide(up_design, f"{m}_pc", hist)
            fit = fit_synth(W, arm, hist, donors=_donors(design), ridge=s.sc_ridge)
            gs = pw.gap_sigma(fit.actual, fit.synthetic, hist)
            sigma, rho = gs["sigma"], float(np.clip(gs["rho"], 0, 0.8))
            mde = pw.did_mde(sigma, s.n_post, s.n_pre, s.alpha, s.power, True, rho)
            obs = float(observed[(observed["arm"] == arm) & (observed["metric"] == m)]["estimate"].iloc[0])
            wneed = pw.weeks_for_power(abs(obs), sigma, None, s.alpha, s.power, True, 520, rho) if obs else None
            # TTestIndPower on weekly relative gaps: effect size d = lift / sigma
            d_mde = pw.ttest_power(nobs1=s.n_post, alpha=s.alpha, power=s.power, ratio=s.n_pre / s.n_post)
            try:
                n_t = pw.ttest_power(effect_size=abs(obs) / sigma, alpha=s.alpha, power=s.power, ratio=1.0) if obs else np.nan
            except Exception:
                n_t = np.nan
            rows.append({"arm": arm, "metric": m, "sigma_gap": sigma, "rho": rho, "mde_closed_form": mde,
                         "mde_ttest": d_mde * sigma, "observed_triangulated": obs,
                         "weeks_needed_closed_form": wneed, "weeks_needed_ttest": n_t,
                         "power_at_observed": pw.did_power(obs, sigma, s.n_post, s.n_pre, s.alpha, True, rho)})
            for lift in s.lift_grid:
                for n in range(1, s.max_power_weeks + 1):
                    curves.append({"arm": arm, "metric": m, "lift": lift, "weeks": n,
                                   "power": pw.did_power(lift, sigma, n, max(n, s.n_pre), s.alpha, True, rho)})
            if m == s.design_metric:
                treat, ctrl = fit.actual, fit.synthetic

                def estimator(t, c, pre, post):
                    return welch_gap_test(t, c, pre, post)["p_value"]

                for lift in (0.03, 0.06, 0.10, 0.15):
                    r = pw.simulated_power(treat, ctrl, lift, s.n_post, s.n_pre, estimator,
                                           lambda p: p < s.alpha, n_sims=s.n_power_sims, seed=s.seed)
                    sims.append({"arm": arm, "metric": m, **r})
    return {"table": pd.DataFrame(rows), "curves": pd.DataFrame(curves), "simulation": pd.DataFrame(sims)}


# --------------------------------------------------------------------------------------
# cannibalisation (organic recapture)
# --------------------------------------------------------------------------------------
def _source_cf(t: np.ndarray, c: np.ndarray, n_pre: int):
    """Per-week counterfactual (ratio form) and post-period delta for one source series."""
    cf = t[:n_pre].mean() * (c[n_pre:].mean() / max(c[:n_pre].mean(), 1e-12))
    return t[n_pre:].mean() - cf


def recapture(up: pd.DataFrame, design: dict, arm: str, stage: str, s: SearchSettings,
              ref: str = "synthetic") -> dict:
    """Organic recapture = Δorganic / -Δpaid within the arm (ratio-DiD per source).

    Δpaid uses only the channels PAUSED in that arm (brand for branded_only; brand + non-brand for
    all_search) - including the still-running non-brand series in the branded arm only adds noise.

    Reference series per source: the arm's locked-weight synthetic twin built on that source's
    per-capita series (``ref='synthetic'``) or the single control county (``ref='control'``).
    CI: week bootstrap (pre and post resampled within period, same draws for both sources).
    """
    weeks = wk((s.pre[0], s.post[1]))
    n_pre = s.n_pre
    series = {}
    for src in ("paid", "organic"):
        col = f"organic_{stage}_pc" if src == "organic" else None
        up2 = up.copy()
        up2["paid_pc"] = up2[f"paid_brand_{stage}_pc"] + (up2[f"paid_nonbrand_{stage}_pc"] if arm == "all_search" else 0.0)
        col = col or "paid_pc"
        W = P.wide(up2, col, wk((s.sc_fit_start, s.post[1])))
        t = W[arm]
        if ref == "synthetic":
            fit = fit_synth(W, arm, wk((s.sc_fit_start, s.pre[1])), donors=_donors(design), ridge=s.sc_ridge)
            c = fit.synthetic
        else:
            c = W[design["control"][0]]
        series[src] = (t.loc[weeks].to_numpy(float), c.loc[weeks].to_numpy(float))
    d_paid = _source_cf(*series["paid"], n_pre)
    d_org = _source_cf(*series["organic"], n_pre)
    rec = d_org / -d_paid if d_paid < 0 else np.nan
    rng = np.random.default_rng(s.seed)
    boots = []
    n_post = len(weeks) - n_pre
    for _ in range(s.n_boot):
        ip = rng.integers(0, n_pre, n_pre)
        iq = n_pre + rng.integers(0, n_post, n_post)
        idx = np.concatenate([ip, iq])
        dp = _source_cf(series["paid"][0][idx], series["paid"][1][idx], n_pre)
        do = _source_cf(series["organic"][0][idx], series["organic"][1][idx], n_pre)
        if dp < 0:
            boots.append(do / -dp)
    lo, hi = np.quantile(boots, [s.alpha / 2, 1 - s.alpha / 2]) if boots else (np.nan, np.nan)
    pop = float(up[up["county_id"] == arm]["pop"].iloc[0])
    return {"arm": arm, "stage": stage, "ref": ref, "delta_paid_per_wk": d_paid * pop / s.per_capita_scale,
            "delta_organic_per_wk": d_org * pop / s.per_capita_scale, "recapture": rec, "ci_low": float(lo),
            "ci_high": float(hi), "se": float(np.std(boots, ddof=1)) if len(boots) > 1 else np.nan}


# --------------------------------------------------------------------------------------
# economics, guardrail, MMM calibration
# --------------------------------------------------------------------------------------
def counterfactual_spend(up: pd.DataFrame, design: dict, arm: str, s: SearchSettings, channel: str) -> float:
    """Spend the arm *would* have had in the window: arm pre spend x donor-pool post/pre spend ratio.

    County spend is observed here (the original had to allocate state spend by population)."""
    col = f"spend_{channel}"
    g = up.groupby(["county_id", "week_idx"])[col].sum().unstack(0)
    pool = g[_donors(design)].sum(axis=1)
    ratio = pool.loc[wk(s.post)].mean() / pool.loc[wk(s.pre)].mean()
    return float(g[arm].loc[wk(s.pre)].mean() * ratio * s.n_post)


def economics(up: pd.DataFrame, design: dict, tri: pd.DataFrame, truth: dict, ipd_truth: dict,
              s: SearchSettings) -> dict:
    """Incremental vs last-touch cost per outcome + incremental-per-dollar for MMM calibration."""
    rows = []
    out_ipd: Dict[str, dict] = {}
    z = _z(s.alpha)
    per_arm = {}
    for arm in ARMS:
        chans = ["search_brand", "search_nonbrand"] if arm == "all_search" else ["search_brand"]
        spend = {ch: counterfactual_spend(up, design, arm, s, ch) for ch in chans}
        sp_tot = sum(spend.values())
        a = up[up["county_id"] == arm].set_index("week_idx")
        per_arm[arm] = {"spend": spend, "metrics": {}}
        for m in s.readout_metrics:
            r = tri[(tri["arm"] == arm) & (tri["metric"] == m)].iloc[0]
            actual = float(a.loc[wk(s.post), m].sum())
            cf = actual / (1 - r["estimate"]) if r["estimate"] < 1 else np.nan
            inc = cf - actual
            inc_se = r["se"] * cf  # delta method on the relative lift
            pre = a.loc[wk(s.pre)]
            lt_paid = float(sum(pre[f"paid_{'brand' if ch == 'search_brand' else 'nonbrand'}_{m}"].sum() for ch in chans))
            lt_spend = float(sum(pre[f"spend_{ch}"].sum() for ch in chans))
            t_cost = truth[arm].get(f"true_incremental_cost_per_{m}")
            rows.append({"arm": arm, "metric": m, "spend_removed_est": sp_tot,
                         "spend_removed_true": truth[arm]["spend_removed"], "incremental_est": inc,
                         "incremental_se": inc_se, "incremental_true": truth[arm][m]["true_incremental"],
                         "incr_cost_per_est": sp_tot / inc if inc > 0 else np.nan,
                         "incr_cost_per_ci_low": sp_tot / (inc + z * inc_se) if inc + z * inc_se > 0 else np.nan,
                         "incr_cost_per_ci_high": sp_tot / (inc - z * inc_se) if inc - z * inc_se > 0 else np.inf,
                         "incr_cost_per_true": t_cost, "last_touch_cost_per": lt_spend / lt_paid if lt_paid else np.nan,
                         "last_touch_share_pre": lt_paid / float(pre[m].sum()),
                         "attribution_overclaim": (lt_paid / float(pre[m].sum())) / r["estimate"] if r["estimate"] > 0 else np.nan})
            per_arm[arm]["metrics"][m] = {"inc": inc, "inc_se": inc_se}
    econ = pd.DataFrame(rows)

    # incremental per dollar: brand from branded-only arm; non-brand from all-search arm net of brand
    for m in s.readout_metrics:
        b = per_arm["branded_only"]
        b_sp = b["spend"]["search_brand"]
        ipd_b = b["metrics"][m]["inc"] / b_sp
        ipd_b_se = b["metrics"][m]["inc_se"] / b_sp
        a = per_arm["all_search"]
        a_b, a_n = a["spend"]["search_brand"], a["spend"]["search_nonbrand"]
        inc_n = a["metrics"][m]["inc"] - ipd_b * a_b
        se_n = np.sqrt(a["metrics"][m]["inc_se"] ** 2 + (ipd_b_se * a_b) ** 2)
        t_ipd_b = ipd_truth["search_brand"][m]
        t_ipd_n = ipd_truth["search_nonbrand"][m]
        for ch, v, se, sp, tv in (("search_brand", ipd_b, ipd_b_se, b_sp, t_ipd_b),
                                  ("search_nonbrand", inc_n / a_n, se_n / a_n, a_n, t_ipd_n)):
            d = out_ipd.setdefault(ch, {"spend": sp, "window": list(s.post)})
            d[m] = float(v)
            d[f"{m}_se"] = float(se)
            d[f"{m}_truth"] = float(tv)
    for ch, d in out_ipd.items():
        d["source"] = ("branded_only arm (brand paused)" if ch == "search_brand"
                       else "all_search arm minus brand effect scaled by its brand spend")
        d["notes"] = "incremental outcomes per $1 of paid search spend removed during the test window"
    return {"table": econ, "incremental_per_dollar": out_ipd}


def guardrail(up: pd.DataFrame, bau_up: pd.DataFrame, design: dict, s: SearchSettings) -> pd.DataFrame:
    """Site-traffic loss (visits) per arm via synthetic control; truth from the BAU world
    (counterfactual-consistent: the treated unit's untreated outcome equals its BAU value)."""
    rows = []
    for arm in ARMS:
        r = est_sc(up, arm, design, "visits", s, placebo=True)
        a = up[up["county_id"] == arm].set_index("week_idx").loc[wk(s.post), "visits"].sum()
        b = bau_up[bau_up["county_id"] == arm].set_index("week_idx").loc[wk(s.post), "visits"].sum()
        true = (b - a) / b
        rows.append({"arm": arm, "metric": "visits", "estimate": r["estimate"], "ci_low": r["ci_low"],
                     "ci_high": r["ci_high"], "p_value": r["p_value"], "truth": float(true),
                     "breach": bool(r["estimate"] > s.traffic_guardrail_max_loss),
                     "threshold": s.traffic_guardrail_max_loss})
    return pd.DataFrame(rows)


def source_effects(up: pd.DataFrame, bau_up: pd.DataFrame, design: dict, arm: str, stage: str,
                   s: SearchSettings) -> pd.DataFrame:
    """Per last-touch source: weekly change vs counterfactual (synthetic control on that source)."""
    tru = source_truth(up, bau_up, arm, stage, s)
    pop = float(up[up["county_id"] == arm]["pop"].iloc[0])
    rows = []
    for src in list(SOURCES) + ["total"]:
        col = f"{src}_{stage}_pc" if src != "total" else f"{stage}_pc"
        W = P.wide(up, col, wk((s.sc_fit_start, s.post[1])))
        fit = fit_synth(W, arm, wk((s.sc_fit_start, s.pre[1])), donors=_donors(design), ridge=s.sc_ridge)
        gap = float((fit.actual.loc[wk(s.post)] - fit.synthetic.loc[wk(s.post)]).mean())
        rows.append({"source": src, "estimate": gap * pop / s.per_capita_scale, "truth": tru[src]})
    return pd.DataFrame(rows).set_index("source")


def source_truth(up: pd.DataFrame, bau_up: pd.DataFrame, arm: str, stage: str, s: SearchSettings) -> dict:
    """Realised source-level deltas from the BAU twin of the same unit (for charts / checks)."""
    a = up[up["county_id"] == arm].set_index("week_idx").loc[wk(s.post)]
    b = bau_up[bau_up["county_id"] == arm].set_index("week_idx").loc[wk(s.post)]
    out = {}
    for src in SOURCES:
        out[src] = float((a[f"{src}_{stage}"].sum() - b[f"{src}_{stage}"].sum()) / s.n_post)
    out["total"] = float((a[stage].sum() - b[stage].sum()) / s.n_post)
    return out


def recommendation(tri: pd.DataFrame, econ: pd.DataFrame, rec: pd.DataFrame, guard: pd.DataFrame,
                   ipd: dict, power_table: pd.DataFrame, s: SearchSettings) -> dict:
    """Plain-language policy read. Uses rated quotes (powered) as the primary KPI and treats the
    sales read as directional unless its interval excludes zero."""
    vps = s.cfg.world.value_per_sale
    e = econ.set_index(["arm", "metric"])
    t = tri.set_index(["arm", "metric"])
    lines = []
    a_r, _b_r = t.loc[("all_search", "rated")], t.loc[("branded_only", "rated")]
    a_s = t.loc[("all_search", "sales")]
    cpr_all = e.loc[("all_search", "rated"), "incr_cost_per_est"]
    lt_all = e.loc[("all_search", "rated"), "last_touch_cost_per"]
    lines.append(f"All paid search is incremental on rated quotes: {a_r['estimate']:.1%} of rated quotes lost when dark "
                 f"(90% CI {a_r['ci_low']:.1%} to {a_r['ci_high']:.1%}); incremental cost per rated quote "
                 f"${cpr_all:,.0f} vs ${lt_all:,.0f} last-touch.")
    sig_s = a_s["ci_low"] > 0 or a_s["ci_high"] < 0
    lines.append("Sales read is " + ("significant." if sig_s else
                 f"inconclusive (est. {a_s['estimate']:+.1%}, CI {a_s['ci_low']:+.1%} to {a_s['ci_high']:+.1%}; "
                 f"6 weeks of binds cannot resolve a ~10% lift) - do not cut search on the sales number alone."))
    rb = rec[(rec["arm"] == "branded_only") & (rec["stage"] == "rated") & (rec["ref"] == "synthetic")].iloc[0]
    if rb["recapture"] > 0.25:
        lines.append(f"Branded search is heavily cannibalised: organic recaptures ~{rb['recapture']:.0%} "
                     f"(CI {rb['ci_low']:.0%}-{rb['ci_high']:.0%}) of lost branded rated quotes. Move brand to a "
                     f"defensive bidding policy (bid only where competitors are present on brand terms / lower "
                     f"caps) and re-test with a powered design.")
    ipd_b, ipd_n = ipd["search_brand"]["rated"], ipd["search_nonbrand"]["rated"]
    lines.append(f"Incremental rated quotes per $1k: branded {1000 * ipd_b:.1f} vs non-branded {1000 * ipd_n:.1f} "
                 f"(SE {1000 * ipd['search_brand']['rated_se']:.1f} / {1000 * ipd['search_nonbrand']['rated_se']:.1f})"
                 f" -> feed both, with SEs, to the MMM as calibration priors.")
    if guard["breach"].any():
        arms = ", ".join(guard.loc[guard["breach"], "arm"])
        lines.append(f"Traffic guardrail ({s.traffic_guardrail_max_loss:.0%} max visit loss) breached in: {arms}.")
    else:
        lines.append(f"Traffic guardrail held (visit loss below {s.traffic_guardrail_max_loss:.0%} in both arms).")
    mde_b = float(power_table.set_index(["arm", "metric"]).loc[("branded_only", "rated"), "mde_closed_form"])
    lines.append(f"The branded arm's rated-quote MDE ({mde_b:.0%} at 80% power) is far above "
                 f"a plausible branded lift (~3%): its lift read is directional; the cannibalisation (source-level) "
                 f"read is the decision-grade output of that arm.")
    return {"lines": lines, "value_per_sale": vps, "all_search_cost_per_rated": cpr_all,
            "all_search_last_touch_cost_per_rated": lt_all, "brand_recapture_rated": rb["recapture"]}
