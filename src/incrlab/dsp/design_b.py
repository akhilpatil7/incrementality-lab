"""Vendor B design: geo-weighted K-means zip clusters + synthetic-control treatment tournament.

Pipeline (design data only, ``week_idx <= design_end``; the vendor's attribution lag is
mimicked with :func:`simulate.apply_report_lag` so vendor-reported series look as they would
on the design date):

1. Universe: zips on the vendor inclusion list, minus zips in the concurrent unrelated test.
2. Vendor-switch structural-break check (Welch t-test on the weekly vendor-attributed share
   before vs after the switch; guarded for empty groups). A level shift in a *vendor-reported*
   metric is why the SC dependent is the business KPI (quote starts per capita), not the
   vendor share, and why vendor-share predictors use post-switch weeks only. Reporting-dip
   weeks are flagged and excluded from share predictors.
3. Geo-weighted K-means on standardised features with lat/lon multiplied by ``geo_weight``.
   Silhouette curve over the K range; K is a *documented business rule*: the best-silhouette
   K among those whose mean cluster population is <= ``cluster_pop_target``.
4. Tournament: every cluster takes a turn as pseudo-treatment vs all others; weights fit on
   the fit window and LOCKED; a known lift is injected in the pseudo-post weeks; the natural
   pseudo-post gap is subtracted. Score = 0.40 accuracy + 0.35 fit + 0.25 speed (percentile
   ranks). Feasibility rules are explicit; any analyst override must carry a rationale.
5. Final SC for the chosen cluster with donor-cluster series rebuilt from zips OUTSIDE the
   spillover buffer, V-table, in-space + in-time placebos, random-geography placebos (many
   more permutations than K-1), and a ridge-augmented SC (AugSynth-style) as robustness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .. import geo as G
from ..common import inference as cinf
from ..common import power as cpow
from ..common import synth as csyn
from ..config import DSP_B_REPORT_DIP_WEEKS, DSP_B_VENDOR_SWITCH_WEEK
from ..simulate import apply_report_lag
from . import features as F
from .design_a import adjacency_graph_balltree, bfs_grow_patch, donors_outside_buffer
from .params import VendorBParams

CLUSTER_FEATURES = ("dsp_share_b", "rated_rate", "close_rate", "per_capita_income", "pct_bachelors", "lfpr",
                    "qs_per_capita", "lat", "lon")
SC_PREDICTORS = ("per_capita_income", "pct_bachelors", "lfpr", "qs_per_capita", "dsp_share_b")


# ======================================================================================
# Structural break
# ======================================================================================
def check_structural_break(panel: pd.DataFrame, switch_week: int, dip_weeks: Sequence[int] = (),
                           alpha: float = 0.05) -> dict:
    """Welch t-test of the weekly vendor-attributed QS share before vs after the vendor switch."""
    g = panel.groupby("week_idx")[["vendor_conv_qs_dsp_b", "qs"]].sum()
    share = (g["vendor_conv_qs_dsp_b"] / g["qs"]).rename("share")
    clean = share.drop(index=[w for w in dip_weeks if w in share.index])
    a, b = clean[clean.index < switch_week], clean[clean.index >= switch_week]
    out = {"switch_week": switch_week, "n_before": int(len(a)), "n_after": int(len(b)), "share": share,
           "dip_weeks": list(dip_weeks)}
    if len(a) < 2 or len(b) < 2:  # guard: no pre group (the original returned NaN silently)
        out.update(t=np.nan, p_value=np.nan, shift_pct=np.nan, significant=False,
                   decision="insufficient data on one side of the switch - cannot test; keep full window")
        return out
    t, p = stats.ttest_ind(b, a, equal_var=False)
    shift = b.mean() / a.mean() - 1
    sig = bool(p < alpha)
    out.update(t=float(t), p_value=float(p), mean_before=float(a.mean()), mean_after=float(b.mean()),
               shift_pct=float(shift), significant=sig,
               decision=("level shift detected -> vendor-share predictors use post-switch weeks only; "
                         "SC dependent = quote starts per capita (unaffected by vendor reporting)") if sig else
               "no significant shift -> full window usable")
    return out


# ======================================================================================
# Clustering
# ======================================================================================
def geo_kmeans(feat: pd.DataFrame, cols: Sequence[str], geo_weight: float, k_range, seed: int, n_init: int,
               pop_target: float) -> dict:
    X = StandardScaler().fit(feat[list(cols)].to_numpy(float))
    Z = X.transform(feat[list(cols)].to_numpy(float))
    for c in ("lat", "lon"):
        if c in cols:
            Z[:, list(cols).index(c)] *= geo_weight
    rows, models = [], {}
    total_pop = float(feat["pop"].sum())
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(Z)
        models[k] = km
        rows.append({"k": k, "silhouette": float(silhouette_score(Z, km.labels_)), "inertia": float(km.inertia_),
                     "mean_cluster_pop": total_pop / k})
    sil = pd.DataFrame(rows)
    best_overall = int(sil.loc[sil["silhouette"].idxmax(), "k"])
    elig = sil[sil["mean_cluster_pop"] <= pop_target]
    k_chosen = int(elig.loc[elig["silhouette"].idxmax(), "k"]) if len(elig) else best_overall
    rule = (f"K = best silhouette among K with mean cluster pop <= {pop_target:,.0f} "
            f"(business granularity); unconstrained best K = {best_overall}")
    labels = models[k_chosen].labels_
    # relabel clusters 1..K from north to south for readable maps
    order = feat.assign(_l=labels).groupby("_l")["lat"].mean().sort_values(ascending=False).index
    remap = {old: new + 1 for new, old in enumerate(order)}
    return {"silhouette": sil, "k_best_silhouette": best_overall, "k": k_chosen, "rule": rule,
            "labels": np.array([remap[l] for l in labels]), "scaler": X, "model": models[k_chosen]}


def cluster_profile(feat: pd.DataFrame, cube: F.ZipCube, panel: pd.DataFrame, weeks: Sequence[int]) -> pd.DataFrame:
    rows = []
    for k, s in feat.groupby("cluster"):
        ids = s["zip"].tolist()
        prof = F.aggregate_profile(feat, ids, ("per_capita_income", "pct_bachelors", "lfpr", "qs_per_capita",
                                               "rated_rate", "close_rate", "dsp_share_b"))
        med = F.monthly_media(panel[panel["week_idx"].isin(list(weeks))], ids, "b").sum()
        rows.append({"cluster": int(k), "n_zips": len(ids), "pop": prof["pop"],
                     "qs": float(s["qs"].sum()), "rated": float(s["rated"].sum()), "sales": float(s["sales"].sum()),
                     "spend": float(med["spend_dsp_b_month"]), "impressions": float(med["impressions_dsp_b_month"]),
                     "clicks": float(med["clicks_dsp_b_month"]),
                     "ctr": med["clicks_dsp_b_month"] / max(med["impressions_dsp_b_month"], 1),
                     "cost_per_click": med["spend_dsp_b_month"] / max(med["clicks_dsp_b_month"], 1),
                     "cost_per_vendor_qs": med["spend_dsp_b_month"] / max(med["vendor_conv_qs_dsp_b_month"], 1),
                     **{k2: prof[k2] for k2 in ("per_capita_income", "pct_bachelors", "lfpr", "qs_per_capita",
                                                "rated_rate", "close_rate", "dsp_share_b")},
                     "lat": float(np.average(s["lat"], weights=s["pop"])),
                     "lon": float(np.average(s["lon"], weights=s["pop"]))})
    return pd.DataFrame(rows).set_index("cluster")


# ======================================================================================
# Synthetic-control helpers (cluster-level per-capita series = ratio of sums)
# ======================================================================================
def cluster_wide(cube: F.ZipCube, members: Dict[int, List[str]], col: str = "qs") -> pd.DataFrame:
    """time x cluster matrix of per-capita outcome (sum of counts / sum of pop) * 1000."""
    return pd.DataFrame({f"C{k}": cube.sum(ids, col) / max(cube.pop_of(ids), 1) * 1000
                         for k, ids in members.items()})


def predictor_table(feat: pd.DataFrame, members: Dict[int, List[str]]) -> pd.DataFrame:
    rows = {f"C{k}": F.aggregate_profile(feat, ids, SC_PREDICTORS)[list(SC_PREDICTORS)]
            for k, ids in members.items()}
    return pd.DataFrame(rows).T


def augmented_sc(wide: pd.DataFrame, treated: str, pre: Sequence, post: Sequence, weights: pd.Series,
                 lambdas: Sequence[float]) -> dict:
    """Ridge-augmented SC (Ben-Michael, Feller & Rothstein style).

    cf_t = sum_j w_j Y_jt + (X_1 - sum_j w_j X_j)' beta_t, where beta_t is a ridge regression of
    donors' Y_t on their pre-period outcomes X_j (centred). lambda chosen by leave-one-donor-out
    CV of the pre-period fit on the last pre weeks (a held-out slice of the pre window).
    """
    donors = list(weights.index)
    X0 = wide.loc[list(pre), donors].to_numpy(float).T  # J x T0
    x1 = wide.loc[list(pre), treated].to_numpy(float)
    mu = X0.mean(0)
    Xc = X0 - mu
    w = weights.to_numpy(float)
    imb = (x1 - mu) - w @ Xc

    def beta_for(Y, lam):
        A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
        return np.linalg.solve(A, Xc.T @ (Y - Y.mean()))

    # choose lambda: hold out last 3 pre weeks as pseudo-post, refit pieces on the earlier weeks
    k = min(3, len(pre) // 4)
    pre_a, pre_b = list(pre)[:-k], list(pre)[-k:]
    X0a = wide.loc[pre_a, donors].to_numpy(float).T
    mua = X0a.mean(0)
    Xca = X0a - mua
    imba = (wide.loc[pre_a, treated].to_numpy(float) - mua) - w @ Xca
    cv = {}
    for lam in lambdas:
        err = []
        for t in pre_b:
            Y = wide.loc[t, donors].to_numpy(float)
            A = Xca.T @ Xca + lam * np.eye(Xca.shape[1])
            b = np.linalg.solve(A, Xca.T @ (Y - Y.mean()))
            err.append(wide.loc[t, treated] - (w @ Y + imba @ b))
        cv[lam] = float(np.mean(np.square(err)))
    lam = min(cv, key=cv.get)
    cf = {}
    for t in wide.index:
        Y = wide.loc[t, donors].to_numpy(float)
        cf[t] = float(w @ Y + imb @ beta_for(Y, lam))
    cf = pd.Series(cf)
    return {"lambda": lam, "cv_mse": cv, "counterfactual": cf, "pre_imbalance_rmse": float(np.sqrt(np.mean(imb ** 2)))}


def score_tournament(rep: pd.DataFrame, wts=(0.40, 0.35, 0.25)) -> pd.DataFrame:
    r = rep.copy()
    ok = r["status"] == "ok"
    q = pd.DataFrame(index=r.index)
    q["q_accuracy"] = 1 - r.loc[ok, "abs_error"].rank(pct=True)
    q["q_fit"] = 1 - r.loc[ok, "rel_rmse"].rank(pct=True)
    q["q_speed"] = 1 - r.loc[ok, "weeks_to_80"].fillna(999).rank(pct=True)
    r = r.join(q)
    r["score"] = wts[0] * r["q_accuracy"] + wts[1] * r["q_fit"] + wts[2] * r["q_speed"]
    return r


@dataclass
class DesignB:
    params: VendorBParams
    feat: pd.DataFrame
    universe_excluded: Dict[str, List[str]]
    switch: dict
    kmeans: dict
    profile: pd.DataFrame
    tournament: pd.DataFrame
    treated_cluster: int
    selection_note: str
    treated: List[str]
    donor_members: Dict[int, List[str]]  # buffer-trimmed donor cluster zips
    buffer_zips: List[str]
    sc: csyn.SynthResult
    v_table: pd.DataFrame
    injected_final: dict
    placebo_space: dict
    placebo_time: dict
    placebo_geo: pd.DataFrame
    aug: dict
    power: dict
    smd: pd.DataFrame
    wide: pd.DataFrame
    fit_weeks: List[int]
    pseudo_post: List[int]
    log: List[str] = field(default_factory=list)
    tournament_series: Dict[int, pd.DataFrame] = field(default_factory=dict)
    share_compare: dict = field(default_factory=dict)
    members: Dict[int, List[str]] = field(default_factory=dict)  # full (untrimmed) cluster membership

    @property
    def weights(self) -> pd.Series:
        return self.sc.weights[self.sc.weights > 0]

    def assignment(self, zips: pd.DataFrame) -> pd.DataFrame:
        z = zips[zips["region"] == self.params.region][["zip", "county_id", "lat", "lon", "pop"]].copy()
        cl = self.feat.set_index("zip")["cluster"]
        z["cluster"] = z["zip"].map(cl)
        z["group"] = "not on inclusion list"
        z.loc[z["zip"].isin(self.universe_excluded["concurrent_test"]), "group"] = "concurrent test (excluded)"
        z.loc[z["cluster"].notna(), "group"] = "donor pool"
        w = self.weights
        active = {int(c[1:]) for c in w.index}
        z.loc[z["cluster"].isin(list(active)), "group"] = "synthetic-control donor"
        z.loc[z["zip"].isin(self.buffer_zips), "group"] = "buffer"
        z.loc[z["zip"].isin(self.treated), "group"] = "treatment"
        return z


def run_design_b(panel: pd.DataFrame, zips: pd.DataFrame, cfg, p: VendorBParams) -> DesignB:
    td = cfg.tests[p.test]
    log: List[str] = []
    reg = panel[(panel["region"] == p.region) & (panel["week_idx"] <= td.design_end)]
    reg = apply_report_lag(reg, td.design_end)  # vendor attribution lag as seen on the design date
    assert reg["week_idx"].max() <= td.design_end
    fit_start = p.fit_start if p.fit_start is not None else td.design_start
    design_weeks = list(range(fit_start, td.design_end + 1))
    zr = zips[zips["region"] == p.region]
    excluded = {"not_on_inclusion_list": zr.loc[zr["dsp_b_inclusion"] == 0, "zip"].tolist(),
                "concurrent_test": zr.loc[zr["concurrent_test_flag"] == 1, "zip"].tolist()}
    uni = zr[(zr["dsp_b_inclusion"] == 1) & (zr["concurrent_test_flag"] == 0)]["zip"].tolist()
    log.append(f"universe: {len(uni)} inclusion-list zips (excluded {len(excluded['concurrent_test'])} "
               f"concurrent-test zips, {len(excluded['not_on_inclusion_list'])} not on the list)")
    reg_u = reg[reg["zip"].isin(uni)]

    # --- 1. structural break --------------------------------------------------------------
    sw = check_structural_break(reg_u[reg_u["week_idx"].isin(design_weeks)], DSP_B_VENDOR_SWITCH_WEEK,
                                DSP_B_REPORT_DIP_WEEKS, p.switch_alpha)
    log.append(f"vendor switch Welch test: shift={sw.get('shift_pct', np.nan):+.1%} p={sw['p_value']:.3g} -> "
               f"{sw['decision']}")
    lag_weeks = list(range(td.design_end - 2, td.design_end + 1))
    share_weeks = [w for w in design_weeks if (w >= DSP_B_VENDOR_SWITCH_WEEK or not sw["significant"])
                   and w not in DSP_B_REPORT_DIP_WEEKS and w not in lag_weeks]
    # --- 2. features ----------------------------------------------------------------------
    feat = F.static_features(reg_u, zips, design_weeks, region=p.region)
    feat = feat[feat["zip"].isin(uni)].reset_index(drop=True)
    sh = F.static_features(reg_u, zips, share_weeks, region=p.region).set_index("zip")["dsp_share_b"]
    feat["dsp_share_b"] = feat["zip"].map(sh).fillna(0)
    km = geo_kmeans(feat, CLUSTER_FEATURES, p.geo_weight, p.k_range, p.kmeans_seed, p.kmeans_n_init,
                    p.cluster_pop_target)
    feat["cluster"] = km["labels"]
    K = km["k"]
    log.append(f"K-means: {km['rule']} -> K={K}")
    cube = F.ZipCube(reg_u[reg_u["week_idx"].isin(design_weeks)])
    prof = cluster_profile(feat, cube, reg_u, design_weeks)
    members = {int(k): s["zip"].tolist() for k, s in feat.groupby("cluster")}

    # --- 3. tournament --------------------------------------------------------------------
    pseudo_post = design_weeks[-p.pseudo_post_weeks:]
    fit_weeks = design_weeks[:-p.pseudo_post_weeks]
    wide = cluster_wide(cube, members)
    preds = predictor_table(feat, members)
    rows, series = [], {}
    largest = int(prof["pop"].idxmax())
    for k in members:
        name = f"C{k}"
        try:
            r = csyn.fit_synth(wide, name, fit_weeks, predictors=preds, optimize_v=False, v=[0.2] * len(SC_PREDICTORS))
            inj = wide.copy()
            inj.loc[pseudo_post, name] *= 1 + p.injected_lift
            nat = csyn.lift_from_series(wide[name], r.synthetic, pseudo_post)["lift"]
            det = csyn.lift_from_series(inj[name], r.synthetic, pseudo_post)["lift"] - nat
            resid = (r.actual - r.synthetic).loc[fit_weeks] / r.synthetic.loc[fit_weeks]
            gs = cpow.gap_sigma(r.actual.loc[fit_weeks], r.synthetic.loc[fit_weeks])
            rho = float(np.clip(gs["rho"], 0, 0.8))
            n_pre_ro = td.test_start - td.readout_pre_start
            wk = cpow.weeks_for_power(p.injected_lift, gs["sigma"], n_pre=n_pre_ro, alpha=p.alpha, power=p.power,
                                      max_weeks=200, rho=rho)
            pop_k = float(prof.loc[k, "pop"])
            feas = []
            if not (p.treat_pop_bounds[0] <= pop_k <= p.treat_pop_bounds[1]):
                feas.append("pop outside bounds")
            if p.exclude_largest_cluster and k == largest:
                feas.append("largest-volume cluster (business rule)")
            if wk is None or wk > p.max_weeks:
                feas.append(f"weeks-to-power > {p.max_weeks}")
            top = r.weights[r.weights > 0].sort_values(ascending=False)
            rows.append({"cluster": k, "status": "ok", "n_zips": len(members[k]), "pop": pop_k,
                         "detected": det, "natural_gap": nat, "abs_error": abs(det - p.injected_lift),
                         "pre_mspe": r.pre_mspe, "rel_rmse": r.rel_rmse, "noise_sd": float(resid.std(ddof=1)),
                         "sigma_gap": gs["sigma"], "rho": rho, "baseline_qs_week": float(cube.sum(members[k], "qs").loc[fit_weeks[-10:]].mean()),
                         "weeks_to_80": wk, "n_donors": r.n_donors_used,
                         "top_weights": ", ".join(f"{i}:{v:.2f}" for i, v in top.head(4).items()),
                         "feasible": not feas, "infeasible_reason": "; ".join(feas)})
            series[k] = pd.DataFrame({"actual": inj[name], "synthetic": r.synthetic})
        except Exception as e:  # logged, never silently dropped
            rows.append({"cluster": k, "status": f"ERROR: {e}", "feasible": False, "infeasible_reason": "error"})
    rep = score_tournament(pd.DataFrame(rows), p.score_weights).sort_values("score", ascending=False)
    rep["score_rank"] = np.arange(1, len(rep) + 1)
    feas = rep[rep["feasible"]]
    if p.analyst_override is not None:
        if not p.analyst_rationale:
            raise ValueError("analyst_override requires analyst_rationale")
        chosen, note = int(p.analyst_override), f"ANALYST OVERRIDE: cluster {p.analyst_override} - {p.analyst_rationale}"
    elif len(feas):
        chosen = int(feas.iloc[0]["cluster"])
        note = (f"highest composite score among feasible clusters (score rank "
                f"{int(feas.iloc[0]['score_rank'])} of {len(rep)})")
    else:
        chosen = int(rep.iloc[0]["cluster"])
        note = "no cluster met every feasibility rule; highest score used - flagged UNDER-POWERED"
    log.append(f"tournament: {int(rep['feasible'].sum())}/{len(rep)} feasible; chosen cluster {chosen} ({note})")
    treated = members[chosen]

    # --- 4. final SC with buffer-trimmed donors ---------------------------------------------
    tz = feat[feat["zip"].isin(treated)]
    others = feat[~feat["zip"].isin(treated)]
    ok = donors_outside_buffer(others, tz, p.buffer_miles)
    buffer_zips = others.loc[~ok, "zip"].tolist()
    donor_members = {k: [z for z in ids if z not in set(buffer_zips)] for k, ids in members.items() if k != chosen}
    donor_members = {k: v for k, v in donor_members.items() if cube.pop_of(v) > 0.15 * cube.pop_of(members[k])}
    fm = {chosen: treated, **donor_members}
    wide_f = cluster_wide(cube, fm)
    preds_f = predictor_table(feat, fm)
    tname = f"C{chosen}"
    sc = csyn.fit_synth(wide_f, tname, design_weeks, predictors=preds_f, v=[0.2] * len(SC_PREDICTORS))
    log.append(f"final SC: {len(buffer_zips)} donor zips inside the {p.buffer_miles:g}-mi buffer removed; "
               f"{sc.n_donors_used} active donors; rel RMSE {sc.rel_rmse:.1%}")
    vt = pd.DataFrame({"V weight": sc.v_weights, "treated": preds_f.loc[tname],
                       "synthetic": preds_f.loc[list(sc.weights.index)].T @ sc.weights,
                       "sample mean": preds_f.drop(index=tname).mean()})
    inj_final = csyn.injected_lift_recovery(wide_f, tname, fit_weeks, pseudo_post, p.injected_lift,
                                            predictors=preds_f, v=[0.2] * len(SC_PREDICTORS))
    # comparison: the vendor-share dependent used historically (fit on clean post-switch weeks)
    share_w = pd.DataFrame({f"C{k}": cube.sum(ids, "vendor_conv_qs_dsp_b") / cube.sum(ids, "qs").clip(lower=1)
                            for k, ids in fm.items()})
    sw_fit = [w for w in share_weeks if w in share_w.index]
    sh_fit = csyn.fit_synth(share_w, tname, sw_fit)
    share_compare = {"dependent": "vendor-attributed QS share", "fit_weeks": sw_fit, "rel_rmse": sh_fit.rel_rmse,
                     "weights": sh_fit.weights[sh_fit.weights > 0].round(3).to_dict(),
                     "qs_per_capita_rel_rmse": sc.rel_rmse, "wide": share_w}
    log.append(f"dependent check: vendor-share SC rel RMSE {sh_fit.rel_rmse:.1%} vs QS-per-capita SC "
               f"{sc.rel_rmse:.1%} -> QS per capita used (business KPI, immune to vendor reporting shifts)")
    fk = {"predictors": preds_f, "v": [0.2] * len(SC_PREDICTORS)}
    pl_space = cinf.placebo_in_space(wide_f, tname, fit_weeks, pseudo_post, stat="lift", fit_kw=fk)
    pl_time = cinf.placebo_in_time(wide_f, tname, design_weeks, n_post=4, n_fit_min=8, fit_kw=fk)
    pl_geo = random_geography_placebos(cube, feat, treated, donor_members, fit_weeks, pseudo_post, preds_f, p,
                                       seed=cfg.seed)
    aug = augmented_sc(wide_f, tname, fit_weeks, pseudo_post, sc.weights, p.aug_lambdas)
    # power for the final design: sigma of SC residual relative gap, by metric (counts scaled)
    n_pre_ro = td.test_start - td.readout_pre_start
    n_post = td.test_end - td.test_start + 1
    pw = {}
    for m in ("qs", "rated", "sales"):
        wm = cluster_wide(cube, fm, m)
        cf = csyn.apply_weights(wm, sc.weights)
        gs = cpow.gap_sigma(wm[tname], cf, index=design_weeks)
        rho = float(np.clip(gs["rho"], 0, 0.8))
        pw[m] = {"sigma_gap": gs["sigma"], "rho": rho,
                 "mde": cpow.did_mde(gs["sigma"], n_post, n_pre_ro, p.alpha, p.power, rho=rho),
                 "weeks_to_power": {f"{l:.2f}": cpow.weeks_for_power(l, gs["sigma"], n_pre=n_pre_ro, alpha=p.alpha,
                                                                    power=p.power, max_weeks=200, rho=rho)
                                    for l in p.lifts},
                 "curve": cpow.power_curve(gs["sigma"], p.lifts, range(1, p.max_weeks + 1), n_pre=n_pre_ro,
                                           alpha=p.alpha, rho=rho)}
    power = {"convention": f"one-sided alpha={p.alpha}, power={p.power:.0%}, sigma = sd of weekly relative "
                           f"treated-vs-synthetic gap (design window), AR(1)-inflated, n_pre={n_pre_ro}",
             "by_metric": pw, "test_weeks": n_post}
    # SMD: treated zips vs donor zips weighted by SC weights
    cw = {}
    for c, w in sc.weights.items():
        for z in fm[int(c[1:])]:
            cw[z] = float(w) / max(cube.pop_of(fm[int(c[1:])]), 1)
    smd_feats = ["per_capita_income", "pct_bachelors", "lfpr", "qs_per_capita", "rated_rate", "close_rate",
                 "dsp_share_b"]
    smd_raw = F.standardized_mean_diff(feat, treated, [z for v in donor_members.values() for z in v], smd_feats)
    smd_w = F.standardized_mean_diff(feat, treated, list(cw), smd_feats, c_weights=cw)
    smd = smd_raw[["feature", "label", "smd"]].rename(columns={"smd": "smd_unweighted_pool"}).merge(
        smd_w[["feature", "treatment_mean", "control_mean", "smd", "flag"]].rename(
            columns={"control_mean": "synthetic_mean", "smd": "smd_sc_weighted"}), on="feature")
    return DesignB(params=p, feat=feat, universe_excluded=excluded, switch=sw, kmeans=km, profile=prof,
                   tournament=rep, treated_cluster=chosen, selection_note=note, treated=treated,
                   donor_members=donor_members, buffer_zips=buffer_zips, sc=sc, v_table=vt, injected_final=inj_final,
                   placebo_space=pl_space, placebo_time=pl_time, placebo_geo=pl_geo, aug=aug, power=power, smd=smd,
                   wide=wide_f, fit_weeks=fit_weeks, pseudo_post=pseudo_post, log=log, tournament_series=series,
                   share_compare=share_compare, members=members)


def random_geography_placebos(cube: F.ZipCube, feat: pd.DataFrame, treated: List[str],
                              donor_members: Dict[int, List[str]], fit_weeks, post_weeks, preds: pd.DataFrame,
                              p: VendorBParams, n: int = 150, seed: int = 0) -> pd.DataFrame:
    """Placebo geographies: random contiguous BFS patches (pop ~ treated) drawn from donor zips.

    Each placebo is fit against the donor clusters with its own zips removed, weights locked, and
    its pseudo-post 'effect' recorded. Gives far more permutations than the K-1 cluster placebos.
    """
    rng = np.random.default_rng(seed)
    pool = feat[feat["zip"].isin([z for v in donor_members.values() for z in v])]
    if len(pool) < 10:
        return pd.DataFrame()
    g = adjacency_graph_balltree(pool, 10.0)
    popd = dict(zip(pool["zip"], pool["pop"].astype(float)))
    target = cube.pop_of(treated)
    rows, seen = [], set()
    for s in rng.choice(pool["zip"].to_numpy(), size=n * 2, replace=True):
        ids, pp, ok = bfs_grow_patch(str(s), g, popd, min(target, 0.5 * sum(popd.values())), 0.25)
        key = frozenset(ids)
        if not ok or key in seen:
            continue
        seen.add(key)
        sid = set(ids)
        mem = {k: [z for z in v if z not in sid] for k, v in donor_members.items()}
        mem = {k: v for k, v in mem.items() if len(v) >= 3}
        wide = cluster_wide(cube, {0: ids, **mem})
        r = csyn.fit_synth(wide, "C0", fit_weeks)
        rows.append({"n_zips": len(ids), "pop": pp, "lift": csyn.lift_from_series(r.actual, r.synthetic, post_weeks)["lift"],
                     "rel_rmse": r.rel_rmse})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)
