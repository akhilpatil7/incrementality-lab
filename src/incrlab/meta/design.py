"""Design-time pipeline for the Social geo holdout (uses only weeks <= design_end).

Steps
-----
1. :func:`load_design_panel` - Eastmoor county panel, design history only.
2. :func:`find_candidate_units` - single counties and contiguous bundles of 2-3 adjacent counties
   whose population is near the target market size.
3. :func:`build_unit_panel` - per-unit weekly panel; counts summed, rates recomputed from sums.
4. :func:`kmeans_sanity` - demographic/behavioural clustering as a sanity lens (not fed into SC).
5. :func:`fit_design_sc` - synthetic control on the share-of-quote-starts metric, fit on the
   training window and LOCKED.
6. :func:`injected_lift_validation` - out-of-sample recovery of injected holdout losses in the
   pseudo-post window with the locked weights.
7. :func:`score_candidates` - fit / donor concentration / recovery / readout noise, hard filters
   with reasons generated from the same thresholds, ranked table.
8. :func:`power_table` - closed-form power vs test length (+ simulation cross-check).
9. :func:`jackknife_stability` - leave-one-donor-out refits.
10. :func:`select_treated_unit` - top-ranked candidate that also passes the jackknife rule.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .. import geo
from ..common import power as cpower
from ..common.synth import SynthResult, apply_weights, fit_synth, fit_synth_pysyncon, pysyncon_available
from ..simulate import load_table
from .settings import MetaSettings, get_settings

COUNT_COLS = ("qs", "rated", "sales", "lt_social_qs", "lt_social_rated", "lt_social_sales",
              "spend_social", "impressions_social", "clicks_social", "pop")
DEMO_COLS = ("median_hh_income", "pct_bachelors", "employment_rate")
PREDICTORS = ("pct_bachelors", "employment_rate", "median_hh_income", "social_share_qs", "qs_per_capita")


# ------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------
def load_design_panel(st: Optional[MetaSettings] = None, county_weekly: Optional[pd.DataFrame] = None,
                      county_static: Optional[pd.DataFrame] = None):
    """Region county panel (weeks design_start..design_end) + county static table."""
    st = st or get_settings()
    cw = load_table("county_weekly") if county_weekly is None else county_weekly
    cs = load_table("county_static") if county_static is None else county_static
    cw = cw[(cw["region"] == st.region) & cw["week_idx"].between(st.design_start, st.design_end)].copy()
    cs = cs[cs["region"] == st.region].copy().reset_index(drop=True)
    return cw, cs


def county_adjacency_graph(zip_static: Optional[pd.DataFrame] = None, region: str = "EM") -> nx.Graph:
    zs = load_table("zip_static") if zip_static is None else zip_static
    zs = zs[zs["region"] == region]
    G = nx.Graph()
    G.add_nodes_from(sorted(zs["county_id"].unique()))
    adj = geo.county_adjacency(zs)
    G.add_edges_from(adj.itertuples(index=False, name=None))
    return G


# ------------------------------------------------------------------------------------------
# Candidate units
# ------------------------------------------------------------------------------------------
def unit_id(members: Sequence[str]) -> str:
    return "+".join(sorted(members))


def find_candidate_units(county_static: pd.DataFrame, G: nx.Graph, st: Optional[MetaSettings] = None) -> pd.DataFrame:
    """Enumerate candidate test markets: counties and contiguous bundles in the population band.

    Population band = target x pop_band, target = target_pop_share x region population. A bundle
    must be a connected subgraph of the county adjacency graph (a market that can be geo-targeted
    as one area) and every member must be smaller than the band floor (otherwise the singleton
    already qualifies or the bundle is dominated by one large county). Each member set appears
    once (no duplicate singletons).
    """
    st = st or get_settings()
    pop = county_static.set_index("county_id")["pop"].astype(float)
    target = st.target_pop_share * pop.sum()
    lo, hi = target * st.pop_band[0], target * st.pop_band[1]
    rows = []
    ids = sorted(pop.index)
    for k in range(1, st.max_group_size + 1):
        for combo in itertools.combinations(ids, k):
            tot = float(pop.loc[list(combo)].sum())
            if not lo <= tot <= hi:
                continue
            if k > 1:
                if (pop.loc[list(combo)] >= lo).any():
                    continue
                if st.bundles_must_be_contiguous and not nx.is_connected(G.subgraph(combo)):
                    continue
            rows.append({"unit": unit_id(combo), "members": list(combo), "size": k, "pop": tot,
                         "is_bundle": k > 1})
    out = pd.DataFrame(rows).sort_values(["size", "unit"]).reset_index(drop=True)
    out.attrs.update({"target_pop": target, "band": (lo, hi)})
    return out


def build_unit_panel(cw: pd.DataFrame, county_static: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Weekly panel for every county (as its own unit) and every bundle; rates from sums."""
    cs = county_static.set_index("county_id")
    base = cw[["county_id", "week_idx"] + [c for c in COUNT_COLS if c in cw.columns]].copy()
    if "pop" not in base:
        base["pop"] = base["county_id"].map(cs["pop"])
    frames = []
    mapping = {c: [c] for c in cs.index}
    for r in units.itertuples():
        mapping.setdefault(r.unit, list(r.members))
    for u, members in mapping.items():
        sub = base[base["county_id"].isin(members)].groupby("week_idx", as_index=False).sum(numeric_only=True)
        sub["unit"] = u
        sub["is_bundle"] = len(members) > 1
        # demographics: population-weighted means of member counties (recomputed, not averaged rates)
        w = cs.loc[members, "pop"].astype(float)
        for c in DEMO_COLS:
            sub[c] = float((cs.loc[members, c] * w).sum() / w.sum())
        frames.append(sub)
    p = pd.concat(frames, ignore_index=True)
    p["social_share_qs"] = np.where(p["qs"] > 0, p["lt_social_qs"] / p["qs"].clip(lower=1), np.nan)
    p["qs_per_capita"] = p["qs"] / p["pop"]
    p["rated_per_capita"] = p["rated"] / p["pop"]
    p["sales_per_capita"] = p["sales"] / p["pop"]
    p["close_rate"] = np.where(p["rated"] > 0, p["sales"] / p["rated"].clip(lower=1), np.nan)
    return p


def wide(panel: pd.DataFrame, value: str) -> pd.DataFrame:
    return panel.pivot(index="week_idx", columns="unit", values=value).sort_index()


def donors_for(unit: str, county_ids: Sequence[str]) -> List[str]:
    """Donor pool = single counties that are not members of the candidate (exact match, not substring)."""
    members = set(unit.split("+"))
    return [c for c in sorted(county_ids) if c not in members]


def predictor_table(panel: pd.DataFrame, weeks: Sequence[int]) -> pd.DataFrame:
    sub = panel[panel["week_idx"].isin(list(weeks))]
    g = sub.groupby("unit")
    t = g[list(DEMO_COLS)].first()
    # share / per-capita from sums over the window (never average weekly ratios)
    s = g[["lt_social_qs", "qs", "pop"]].sum()
    t["social_share_qs"] = s["lt_social_qs"] / s["qs"]
    t["qs_per_capita"] = s["qs"] / s["pop"]
    return t[list(PREDICTORS)]


# ------------------------------------------------------------------------------------------
# K-means sanity lens
# ------------------------------------------------------------------------------------------
def kmeans_sanity(panel: pd.DataFrame, county_ids: Sequence[str], weeks: Sequence[int],
                  st: Optional[MetaSettings] = None) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import MinMaxScaler

    st = st or get_settings()
    feats = predictor_table(panel[panel["unit"].isin(county_ids)], weeks)
    pops = panel[panel["unit"].isin(county_ids)].groupby("unit")["pop"].first()
    feats = feats.assign(log_pop=np.log(pops.reindex(feats.index).astype(float)))
    X = MinMaxScaler().fit_transform(feats.to_numpy(float))
    rows = []
    n = len(feats)
    for k in st.kmeans_k_grid:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, random_state=st.kmeans_seed, n_init=10).fit(X)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(X, km.labels_))})
    k = min(st.kmeans_k, n - 1)
    km = KMeans(n_clusters=k, random_state=st.kmeans_seed, n_init=10).fit(X)
    labels = pd.Series(km.labels_, index=feats.index, name="cluster")
    return {"features": feats, "labels": labels, "k": k, "grid": pd.DataFrame(rows)}


# ------------------------------------------------------------------------------------------
# Synthetic control (design metric) with locked weights
# ------------------------------------------------------------------------------------------
@dataclass
class DesignFit:
    unit: str
    fit: SynthResult                    # share-metric fit on the training window
    donors: List[str]
    weights: pd.Series                  # locked, active only
    extra: dict = field(default_factory=dict)


def fit_design_sc(panel: pd.DataFrame, unit: str, county_ids: Sequence[str], st: Optional[MetaSettings] = None,
                  donors: Optional[Sequence[str]] = None, use_predictors: bool = True) -> DesignFit:
    """Fit SC on the share-of-QS metric over the training window; weights are then frozen.

    Predictors (demographics, mean share, QS per capita over the training window) enter the loss
    alongside the weekly share path; they are standardised so scale differences do not dominate.
    """
    st = st or get_settings()
    donors = list(donors) if donors is not None else donors_for(unit, county_ids)
    W = wide(panel, st.design_metric)
    preds = predictor_table(panel, st.train_weeks) if use_predictors else None
    # weight predictors so they matter but do not swamp the 51 weekly outcome rows
    v = np.full(len(PREDICTORS), 3.0) if use_predictors else None
    res = fit_synth(W[[unit] + donors], unit, st.train_weeks, donors=donors, predictors=preds, v=v)
    w = res.weights[res.weights > 0]
    return DesignFit(unit=unit, fit=res, donors=donors, weights=w)


def per_capita_counterfactual(panel: pd.DataFrame, unit: str, weights: pd.Series, stage: str = "qs") -> pd.DataFrame:
    """Locked-weight synthetic on the per-capita scale, re-expressed as treated-unit counts.

    Fix of the source workflow: a weighted average of donor *counts* sits at an arbitrary level;
    weighting per-capita rates and multiplying by the treated population keeps the synthetic on
    the treated unit's scale (DiD then removes only a genuine level gap).
    """
    pc = wide(panel, f"{stage}_per_capita")
    popw = wide(panel, "pop")[unit]
    synth_pc = apply_weights(pc, weights)
    return pd.DataFrame({"actual": wide(panel, stage)[unit].astype(float),
                         "synthetic": synth_pc * popw, "pop": popw})


# ------------------------------------------------------------------------------------------
# Injected-lift validation (pseudo-post, out of sample)
# ------------------------------------------------------------------------------------------
def _ratio_lift(actual: pd.Series, synth: pd.Series, pre: Sequence[int], post: Sequence[int]) -> float:
    """Contribution (positive = volume lost) with a pre-period level (ratio) adjustment:
    cf_post = synth_post x (actual_pre / synth_pre); lift = (cf - actual) / cf."""
    k = float(actual.loc[list(pre)].sum()) / float(synth.loc[list(pre)].sum())
    cf = float(synth.loc[list(post)].sum()) * k
    return (cf - float(actual.loc[list(post)].sum())) / cf


def injected_lift_validation(panel: pd.DataFrame, dfit: DesignFit, st: Optional[MetaSettings] = None,
                             lifts: Optional[Sequence[float]] = None, stage: str = "qs") -> pd.DataFrame:
    """Inject a holdout-style LOSS of ``lift`` into the treated unit's pseudo-post weeks (51-57) on
    ``stage`` volume and recover it with the LOCKED design weights.

    Recovery = ratio-adjusted lift using only the training window for the level adjustment (no
    subtraction of the un-injected pseudo-post gap - doing that makes recovery exact by
    construction, which is what made the source "MDE" look perfect). Error is therefore an honest
    out-of-sample error on 7 weeks. Also reported for the share metric itself.
    """
    st = st or get_settings()
    lifts = list(lifts or st.validation_lifts)
    cf = per_capita_counterfactual(panel, dfit.unit, dfit.weights, stage)
    post, pre = st.pseudo_post_weeks, st.train_weeks
    nat = _ratio_lift(cf["actual"], cf["synthetic"], pre, post)
    rows = []
    for L in lifts:
        a = cf["actual"].copy()
        a.loc[post] = a.loc[post] * (1 - L)
        det = _ratio_lift(a, cf["synthetic"], pre, post)
        rows.append({"unit": dfit.unit, "metric": stage, "injected": L, "detected": det, "error": det - L,
                     "abs_error": abs(det - L), "natural_gap": nat})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
# Candidate scoring
# ------------------------------------------------------------------------------------------
def readout_noise(panel: pd.DataFrame, dfit: DesignFit, st: Optional[MetaSettings] = None, stage: str = "qs") -> dict:
    """Weekly sd and AR(1) of the relative (treated - locked synthetic) QS gap over design history.

    Conservative: takes the larger of the in-sample (training) and out-of-sample (pseudo-post)
    sd, after removing the training-window level gap.
    """
    st = st or get_settings()
    cf = per_capita_counterfactual(panel, dfit.unit, dfit.weights, stage)
    k = cf.loc[st.train_weeks, "actual"].sum() / cf.loc[st.train_weeks, "synthetic"].sum()
    s = cf["synthetic"] * k
    g = (cf["actual"] - s) / s
    gi, go = g.loc[st.train_weeks], g.loc[st.pseudo_post_weeks]
    rho = float(np.corrcoef(gi.to_numpy()[1:], gi.to_numpy()[:-1])[0, 1])
    sig_in, sig_out = float(gi.std(ddof=1)), float(go.std(ddof=1))
    return {"sigma": max(sig_in, sig_out), "sigma_in": sig_in, "sigma_out": sig_out,
            "rho": max(rho, 0.0), "rho_raw": rho, "pseudo_post_bias": float(go.mean())}


def evaluate_candidate(panel: pd.DataFrame, unit: str, county_ids: Sequence[str],
                       st: Optional[MetaSettings] = None) -> dict:
    st = st or get_settings()
    dfit = fit_design_sc(panel, unit, county_ids, st)
    r = dfit.fit
    inj = injected_lift_validation(panel, dfit, st, lifts=[st.scoring_lift])
    noise = readout_noise(panel, dfit, st)
    mde = cpower.did_mde(noise["sigma"], st.n_test_weeks, n_pre=len(st.readout_pre_weeks), alpha=st.alpha,
                         power=st.power, rho=noise["rho"])
    w = dfit.weights.sort_values(ascending=False)
    return {"unit": unit, "pre_rmspe": r.pre_rmspe, "rel_rmspe": r.rel_rmse, "max_weight": float(w.max()),
            "hhi": float((w ** 2).sum()), "neff": float(1 / (w ** 2).sum()), "donor_count": int((w >= 0.01).sum()),
            "detected_10": float(inj["detected"].iloc[0]), "recovery_error": float(inj["abs_error"].iloc[0]),
            "gap_sigma": noise["sigma"], "gap_rho": noise["rho"], "mde_8wk": mde,
            "top_weights": ", ".join(f"{k}: {v:.2f}" for k, v in w.head(4).items()),
            "weights": w.round(4).to_dict()}


def rejection_reasons(row: pd.Series, st: MetaSettings, rmspe_cap: float) -> str:
    """Human-readable reasons, generated from the SAME thresholds as the filter."""
    rs = []
    if row["rel_rmspe"] > rmspe_cap:
        rs.append(f"pre-fit rel. RMSPE {row['rel_rmspe']:.3f} > Q{int(st.rmspe_quantile * 100)} cap {rmspe_cap:.3f}")
    if row["max_weight"] > st.max_weight:
        rs.append(f"max donor weight {row['max_weight']:.2f} > {st.max_weight:.2f}")
    if row["neff"] < st.min_neff:
        rs.append(f"effective donors {row['neff']:.1f} < {st.min_neff:.1f}")
    lo, hi = st.donor_count
    if not lo <= row["donor_count"] <= hi:
        rs.append(f"donor count {row['donor_count']} outside [{lo}, {hi}]")
    if row["recovery_error"] > st.max_recovery_error:
        rs.append(f"10% injection recovered as {row['detected_10']:.1%} (error > {st.max_recovery_error:.0%} pts)")
    return "; ".join(rs)


def score_candidates(evals: pd.DataFrame, st: Optional[MetaSettings] = None) -> pd.DataFrame:
    """Composite score (higher = better) + hard filters; returns the ranked table."""
    st = st or get_settings()
    d = evals.copy()
    cap = float(d["rel_rmspe"].quantile(st.rmspe_quantile))
    wf, wd, wr, wn = st.score_weights
    d["s_fit"] = 1 - d["rel_rmspe"].rank(pct=True)
    d["s_div"] = np.minimum(1.0, d["neff"] / st.neff_cap)
    d["s_rec"] = 1 - np.minimum(1.0, d["recovery_error"] / (st.scoring_lift / 2))
    d["s_noise"] = 1 - d["gap_sigma"].rank(pct=True)
    d["score"] = wf * d["s_fit"] + wd * d["s_div"] + wr * d["s_rec"] + wn * d["s_noise"]
    d["rejected_because"] = d.apply(lambda r: rejection_reasons(r, st, cap), axis=1)
    d["passes"] = d["rejected_because"] == ""
    d = d.sort_values(["passes", "score", "unit"], ascending=[False, False, True]).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    d.attrs["rmspe_cap"] = cap
    return d


# ------------------------------------------------------------------------------------------
# Power
# ------------------------------------------------------------------------------------------
def power_table(sigma: float, rho: float, st: Optional[MetaSettings] = None, n_pre: Optional[int] = None,
                lifts: Optional[Sequence[float]] = None, alpha: Optional[float] = None) -> pd.DataFrame:
    """Closed-form power vs test weeks. Unlike the source (which ignored pre-period noise and
    autocorrelation), SE = sigma * sqrt(1/n_post + 1/n_pre) * sqrt((1+rho)/(1-rho))."""
    st = st or get_settings()
    n_pre = len(st.readout_pre_weeks) if n_pre is None else n_pre
    return cpower.power_curve(sigma, list(lifts or st.power_lifts), list(range(1, st.power_max_weeks + 1)),
                              n_pre=n_pre, alpha=st.alpha if alpha is None else alpha, rho=rho)


def weeks_to_power(sigma: float, rho: float, st: Optional[MetaSettings] = None, n_pre: Optional[int] = None,
                   lifts: Optional[Sequence[float]] = None, alpha: Optional[float] = None) -> pd.DataFrame:
    st = st or get_settings()
    n_pre = len(st.readout_pre_weeks) if n_pre is None else n_pre
    rows = []
    for L in lifts or st.power_lifts:
        w = cpower.weeks_for_power(L, sigma, n_pre=n_pre, alpha=st.alpha if alpha is None else alpha,
                                   power=st.power, max_weeks=104, rho=rho)
        rows.append({"lift": L, "weeks_to_80": w})
    return pd.DataFrame(rows)


def simulated_power_check(panel: pd.DataFrame, dfit: DesignFit, lift: float, st: Optional[MetaSettings] = None,
                          n_sims: int = 300, seed: int = 0) -> dict:
    """Empirical cross-check of the closed form on design history: random (pre, post) windows of the
    readout shape, holdout injected, ratio-DiD lift > one-sided critical value."""
    from scipy import stats

    st = st or get_settings()
    cf = per_capita_counterfactual(panel, dfit.unit, dfit.weights)
    t, c = cf["actual"].loc[st.design_weeks], cf["synthetic"].loc[st.design_weeks]
    n_pre, n_post = len(st.readout_pre_weeks), st.n_test_weeks
    noise = readout_noise(panel, dfit, st)
    infl = np.sqrt((1 + noise["rho"]) / (1 - noise["rho"]))
    crit = stats.norm.ppf(1 - st.alpha) * noise["sigma"] * np.sqrt(1 / n_pre + 1 / n_post) * infl

    def est(tt, cc, pre, post):
        return (tt.loc[pre].sum() / cc.loc[pre].sum() * cc.loc[post].sum() - tt.loc[post].sum()) / (
            tt.loc[pre].sum() / cc.loc[pre].sum() * cc.loc[post].sum())

    out = cpower.simulated_power(t, c, lift, n_post, n_pre, est, lambda s: s > crit, n_sims=n_sims, seed=seed)
    out["critical_lift"] = float(crit)
    return out


# ------------------------------------------------------------------------------------------
# Jackknife
# ------------------------------------------------------------------------------------------
def jackknife_stability(panel: pd.DataFrame, unit: str, county_ids: Sequence[str],
                        st: Optional[MetaSettings] = None) -> pd.DataFrame:
    """Leave-one-donor-out on the locked design fit: refit without each active donor, re-run the
    10% injected-lift recovery and compare. Stability rule uses ABSOLUTE pp change (relative
    changes explode when the baseline is near zero)."""
    st = st or get_settings()
    base = fit_design_sc(panel, unit, county_ids, st)
    b_inj = injected_lift_validation(panel, base, st, lifts=[st.scoring_lift])["detected"].iloc[0]
    rows = [{"unit": unit, "dropped": "(none)", "dropped_weight": 0.0, "detected_10": b_inj,
             "change_pp": 0.0, "pre_rmspe": base.fit.pre_rmspe, "rmspe_ratio": 1.0,
             "neff": 1 / float((base.weights ** 2).sum())}]
    for d, wd in base.weights.sort_values(ascending=False).items():
        pool = [c for c in base.donors if c != d]
        f = fit_design_sc(panel, unit, county_ids, st, donors=pool)
        det = injected_lift_validation(panel, f, st, lifts=[st.scoring_lift])["detected"].iloc[0]
        rows.append({"unit": unit, "dropped": d, "dropped_weight": float(wd), "detected_10": det,
                     "change_pp": det - b_inj, "pre_rmspe": f.fit.pre_rmspe,
                     "rmspe_ratio": f.fit.pre_rmspe / base.fit.pre_rmspe,
                     "neff": 1 / float((f.weights ** 2).sum())})
    jk = pd.DataFrame(rows)
    jk["stable"] = (jk["change_pp"].abs() <= st.jk_max_change_pp) & (jk["rmspe_ratio"] <= st.jk_max_rmspe_ratio)
    return jk


def select_treated_unit(ranked: pd.DataFrame, panel: pd.DataFrame, county_ids: Sequence[str],
                        st: Optional[MetaSettings] = None, max_check: int = 5) -> dict:
    """Final pick = highest-scoring candidate passing filters whose leave-one-out drops are all stable.
    Deterministic (ties broken by unit id in :func:`score_candidates`)."""
    st = st or get_settings()
    passing = ranked[ranked["passes"]]
    pool = passing if len(passing) else ranked
    checked = []
    for u in pool["unit"].head(max_check):
        jk = jackknife_stability(panel, u, county_ids, st)
        ok = bool(jk["stable"].all())
        checked.append({"unit": u, "jackknife_all_stable": ok,
                        "share_stable": float(jk["stable"].iloc[1:].mean()) if len(jk) > 1 else 1.0,
                        "max_abs_change_pp": float(jk["change_pp"].abs().max()), "jackknife": jk})
        if ok:
            return {"unit": u, "checked": checked, "reason": "top-ranked candidate passing filters and jackknife"}
    best = max(checked, key=lambda c: c["share_stable"])
    return {"unit": best["unit"], "checked": checked,
            "reason": "no checked candidate fully stable; picked the most stable of the top candidates"}


# ------------------------------------------------------------------------------------------
# pysyncon agreement
# ------------------------------------------------------------------------------------------
def pysyncon_agreement(panel: pd.DataFrame, dfit: DesignFit, st: Optional[MetaSettings] = None) -> dict:
    """Cross-check the locked design fit against ``pysyncon.Synth`` (classic Abadie nested V/W with
    the five design predictors averaged over the training window, dependent = share of QS), plus
    the in-house nested-V fit on the same specification.

    SC weights are often not unique when donors are near-collinear, so agreement is judged on what
    matters for the readout: the synthetic path (correlation, pre-RMSPE) and the recovered injected
    lift - not only on weight-vector distance. Returns {'available': False} without pysyncon.
    """
    st = st or get_settings()
    if not pysyncon_available():
        return {"available": False}
    u, donors = dfit.unit, dfit.donors
    W = wide(panel, st.design_metric)[[u] + donors]
    long = panel[panel["unit"].isin([u] + donors)].rename(columns={"week_idx": "time"})
    long = long[["unit", "time"] + list(PREDICTORS)].copy()
    # rescale to comparable magnitudes (pysyncon does not standardise predictors)
    long["median_hh_income"] = long["median_hh_income"] / 1e5
    long["qs_per_capita"] = long["qs_per_capita"] * 1e4
    try:
        ps = fit_synth_pysyncon(W, u, st.train_weeks, donors=donors, predictors_long=long,
                                predictor_cols=list(PREDICTORS))
    except Exception as e:  # pragma: no cover - surfaced in the report
        return {"available": True, "error": repr(e)}
    own = fit_synth(W, u, st.train_weeks, donors=donors, predictors=predictor_table(panel, st.train_weeks),
                    optimize_v=True)
    fits = {"locked_design": dfit.fit, "in_house_nested_v": own, "pysyncon": ps}
    wv = pd.DataFrame({k: f.weights.reindex(donors).fillna(0.0) for k, f in fits.items()})
    det = {k: float(injected_lift_validation(panel, DesignFit(u, f, donors, f.weights[f.weights > 0]), st,
                                             lifts=[st.scoring_lift])["detected"].iloc[0]) for k, f in fits.items()}
    dw = st.design_weeks
    return {"available": True, "weights": wv,
            "pre_rmspe": {k: f.pre_rmspe for k, f in fits.items()},
            "detected_10": det,
            "weight_l1_pysyncon_vs_design": float((wv["pysyncon"] - wv["locked_design"]).abs().sum()),
            "weight_l1_pysyncon_vs_nested": float((wv["pysyncon"] - wv["in_house_nested_v"]).abs().sum()),
            "synthetic_corr_pysyncon_vs_design": float(np.corrcoef(ps.synthetic.loc[dw], dfit.fit.synthetic.loc[dw])[0, 1]),
            "pysyncon_v": ps.v_weights.round(4).tolist() if ps.v_weights is not None else None}
