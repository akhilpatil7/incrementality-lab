"""Vendor A design: contiguous zip patch (holdout) vs a Mahalanobis-matched twin patch.

Pipeline (all on data with ``week_idx <= design_end``):

1. Eligible universe: vendor-served zips in the test region with pop >= ``min_zip_pop``.
2. Treatment counties: the ``n_treatment_counties`` most populous counties with enough
   eligible population to host a patch; seeds = their extreme N/S/E/W zips.
3. BFS growth on a 10-mile adjacency graph (BallTree-free networkx graph from ``geo``); growth
   stops at the cumulative population *closest to target* inside the +-tolerance window
   (fix of the "stop at the lower bound" behaviour).
4. Spillover buffer: haversine BallTree on the treatment centroids; a donor zip is valid only
   when no treatment centroid lies within ``buffer_miles``. The same parameter labels charts.
5. Control patches: BFS from *every* valid donor zip over the donor-only adjacency graph,
   de-duplicated; *every* unique in-window patch is scored and the top N per anchor county
   are kept (fix of the order-dependent "append only when the best improves" log).
6. Score: Mahalanobis distance on 6 pop-weighted features (standardised, ridge-regularised
   covariance; zip-level by default, bootstrapped-patch covariance as a sensitivity) +
   ``w_compact`` x spread penalty.
7. Behavioural gates on weekly series: WoW correlation, slope ratio of mean-normalised
   levels, CV. Explicit strict -> relaxed tiers, both logged; never force-passed.
8. Ranking: rank(composite) + rank(-corr) + rank(weeks to detect ``rank_lift``) + rank(|in-time
   placebo drift| over the last design weeks).
9. Power: closed form (one-sided alpha=0.10, 80% power) from the sd of the weekly relative
   treatment-control gap with AR(1) inflation, plus a design-time injected-lift check.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import BallTree

from .. import geo as G
from ..common import did as cdid
from ..common import power as cpow
from . import features as F
from .params import FEATURES_A, VendorAParams


# ======================================================================================
# Geography engineering
# ======================================================================================
def balltree(lat: np.ndarray, lon: np.ndarray) -> BallTree:
    return BallTree(np.radians(np.column_stack([lat, lon])), metric="haversine")


def adjacency_graph_balltree(zs: pd.DataFrame, miles: float) -> nx.Graph:
    """Zip adjacency graph: edge when centroids are within ``miles`` (BallTree radius query)."""
    tree = balltree(zs["lat"].to_numpy(), zs["lon"].to_numpy())
    ids = zs["zip"].to_numpy()
    nb, dist = tree.query_radius(np.radians(zs[["lat", "lon"]].to_numpy()), r=miles / G.EARTH_RADIUS_MI,
                                 return_distance=True)
    g = nx.Graph()
    g.add_nodes_from(ids)
    for i, (js, ds) in enumerate(zip(nb, dist)):
        for j, d in zip(js, ds):
            if j > i:
                g.add_edge(ids[i], ids[j], weight=float(d * G.EARTH_RADIUS_MI))
    return g


def bfs_grow_patch(seed: str, graph: nx.Graph, pop: Dict[str, float], target: float, tol: float,
                   stop: str = "closest") -> Tuple[List[str], float, bool]:
    """FIFO BFS from ``seed`` accumulating population.

    ``stop='closest'`` (default): keep growing while inside/below the window and return the
    prefix whose population is closest to ``target``. ``stop='lower'`` reproduces the original
    rule (stop as soon as the lower bound is reached). Returns (zips, pop, in_window).
    """
    lo, hi = target * (1 - tol), target * (1 + tol)
    seen, order, cum = {seed}, [], []
    q = deque([seed])
    total = 0.0
    while q and total < hi:
        n = q.popleft()
        if total + pop[n] > hi and order:
            continue  # this zip would overshoot; try the remaining frontier
        order.append(n)
        total += pop[n]
        cum.append(total)
        if stop == "lower" and total >= lo:
            break
        for m in sorted(graph.neighbors(n)):
            if m not in seen:
                seen.add(m)
                q.append(m)
    cum = np.array(cum)
    inw = (cum >= lo) & (cum <= hi)
    if inw.any():
        k = int(np.argmin(np.where(inw, np.abs(cum - target), np.inf))) if stop == "closest" else int(np.argmax(inw))
        return order[:k + 1], float(cum[k]), True
    return order, float(total), False


def donors_outside_buffer(cands: pd.DataFrame, treat: pd.DataFrame, miles: float) -> np.ndarray:
    """Boolean mask: candidate zip has NO treatment centroid within ``miles`` (BallTree count)."""
    tree = balltree(treat["lat"].to_numpy(), treat["lon"].to_numpy())
    cnt = tree.query_radius(np.radians(cands[["lat", "lon"]].to_numpy()), r=miles / G.EARTH_RADIUS_MI,
                            count_only=True)
    return cnt == 0


def mean_spread_miles(zs: pd.DataFrame) -> float:
    """Mean distance from each zip to its k=min(n-1,5) nearest neighbours in the patch."""
    n = len(zs)
    if n < 2:
        return 0.0
    D = G.distance_matrix_miles(zs["lat"], zs["lon"])
    k = min(n - 1, 5)
    return float(np.sort(D, axis=1)[:, 1:k + 1].mean())


# ======================================================================================
# Mahalanobis scoring
# ======================================================================================
@dataclass
class MahaMetric:
    features: Tuple[str, ...]
    mean: np.ndarray
    sd: np.ndarray
    VI: np.ndarray
    source: str

    def vec(self, prof: pd.Series) -> np.ndarray:
        return (prof[list(self.features)].to_numpy(float) - self.mean) / self.sd

    def distance(self, a: pd.Series, b: pd.Series) -> float:
        return float(mahalanobis(self.vec(a), self.vec(b), self.VI))


def mahalanobis_metric(X: pd.DataFrame, features: Sequence[str], ridge: float, source: str = "zip") -> MahaMetric:
    """Standardise features, then VI = inv(corr + ridge I). Equivalent to the raw-scale metric
    with a scale-aware ridge (the raw covariance mixes $ and shares, so an absolute 1e-6 ridge
    would only regularise the share dimensions)."""
    A = X[list(features)].to_numpy(float)
    mu, sd = A.mean(0), A.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (A - mu) / sd
    S = np.cov(Z.T)
    return MahaMetric(tuple(features), mu, sd, np.linalg.inv(S + ridge * np.eye(len(features))), source)


def patch_covariance_metric(feat: pd.DataFrame, graph: nx.Graph, popd: Dict[str, float], features: Sequence[str],
                            base: MahaMetric, target: float, tol: float, n: int, seed: int, ridge: float) -> MahaMetric:
    """Covariance estimated from bootstrapped random contiguous patches (patch-level scale)."""
    rng = np.random.default_rng(seed)
    nodes = sorted(graph.nodes)
    profs = []
    for s in rng.choice(nodes, size=min(n, len(nodes) * 3), replace=True):
        ids, p, ok = bfs_grow_patch(str(s), graph, popd, target, tol)
        if ok:
            profs.append(F.aggregate_profile(feat, ids, features))
    P = pd.DataFrame(profs).drop_duplicates()
    if len(P) < len(features) + 3:
        return base
    Z = (P[list(features)].to_numpy(float) - base.mean) / base.sd
    S = np.cov(Z.T)
    return MahaMetric(base.features, base.mean, base.sd, np.linalg.inv(S + ridge * np.eye(len(features))),
                      f"bootstrap patches (n={len(P)})")


# ======================================================================================
# Behavioural validation
# ======================================================================================
def validate_pair(t: pd.Series, c: pd.Series, gates: Tuple[float, float, float, float]) -> dict:
    """WoW correlation, slope ratio (mean-normalised levels) and CV gates with fail reasons."""
    min_corr, slo, shi, max_cv = gates
    common = t.index.intersection(c.index)
    t, c = t.loc[common].astype(float), c.loc[common].astype(float)
    out = {"n_weeks": int(len(common))}
    if len(common) < 4:
        out.update(passed=False, reason="fewer than 4 common weeks")
        return out
    tw, cw = F.wow(t), F.wow(c)
    corr = float(stats.pearsonr(tw, cw)[0])
    x = np.arange(len(common))
    ts = stats.linregress(x, t / t.mean()).slope
    cs = stats.linregress(x, c / c.mean()).slope
    ratio = float(ts / cs) if abs(cs) > 1e-9 else np.inf
    tcv, ccv = float(t.std(ddof=1) / t.mean()), float(c.std(ddof=1) / c.mean())
    reasons = []
    if corr < min_corr:
        reasons.append(f"wow_corr={corr:.2f}<{min_corr}")
    if not (slo <= ratio <= shi):
        reasons.append(f"slope_ratio={ratio:.2f} not in [{slo},{shi}]")
    if tcv > max_cv:
        reasons.append(f"treat_cv={tcv:.2f}>{max_cv}")
    if ccv > max_cv:
        reasons.append(f"ctrl_cv={ccv:.2f}>{max_cv}")
    out.update(wow_corr=corr, slope_treat=float(ts), slope_ctrl=float(cs), slope_ratio=ratio, treat_cv=tcv,
               ctrl_cv=ccv, passed=not reasons, reason="; ".join(reasons) or "pass")
    return out


def pair_power(t: pd.Series, c: pd.Series, p: VendorAParams, n_pre: int, lifts: Sequence[float]) -> dict:
    """Unified power convention: sigma = sd of weekly relative gap (T/C scaled), AR(1)-inflated,
    one-sided alpha, n_pre readout-pre weeks; weeks-to-80% per lift."""
    tt, cc = t / t.mean(), c / c.mean()
    gs = cpow.gap_sigma(tt, cc, relative=True)
    rho = float(np.clip(gs["rho"], 0.0, 0.8))
    wk = {f"{l:.2f}": cpow.weeks_for_power(l, gs["sigma"], n_pre=n_pre, alpha=p.alpha, power=p.power,
                                           max_weeks=200, rho=rho) for l in lifts}
    return {"sigma_gap": gs["sigma"], "rho": rho, "weeks_to_power": wk}


def spec_power_levels(t: pd.Series, c: pd.Series, lift: float, alpha: float = 0.10, power: float = 0.80) -> float:
    """The original 'sigma of weekly level differences' closed form (shown for comparison only)."""
    sd = float(np.std(t - c, ddof=1))
    mu = float(t.mean())
    z_a, z_b = stats.norm.ppf(1 - alpha), stats.norm.ppf(power)
    return float(((z_a + z_b) * sd / (mu * lift)) ** 2)


# ======================================================================================
# Main design routine
# ======================================================================================
@dataclass
class DesignA:
    params: VendorAParams
    eligible: pd.DataFrame  # zip-level features of the eligible universe (design window)
    region_zips: pd.DataFrame
    treatment_counties: List[str]
    patches: pd.DataFrame  # treatment candidate patches
    donor_pools: pd.DataFrame
    candidates: pd.DataFrame  # scored + validated + ranked
    metric: MahaMetric
    metric_patch: MahaMetric
    corr: pd.DataFrame
    winner: dict
    treated: List[str]
    control: List[str]
    buffer_zips: List[str]
    excluded: Dict[str, List[str]]
    smd: pd.DataFrame
    profile: pd.DataFrame
    power: dict
    injected: pd.DataFrame
    validation_log: List[str]
    weekly: Dict[str, pd.DataFrame] = field(default_factory=dict)  # per top pair weekly series
    sizing: pd.DataFrame = field(default_factory=pd.DataFrame)  # power-driven size ladder
    sizing_decision: dict = field(default_factory=dict)

    def assignment(self) -> pd.DataFrame:
        z = self.region_zips[["zip", "county_id", "county", "lat", "lon", "pop"]].copy()
        role = pd.Series("not eligible", index=z["zip"])
        role[self.eligible["zip"]] = "eligible (unused)"
        role[self.buffer_zips] = "buffer"
        role[self.control] = "control"
        role[self.treated] = "treatment"
        z["group"] = z["zip"].map(role)
        return z


def run_design_a(panel: pd.DataFrame, zips: pd.DataFrame, cfg, p: VendorAParams) -> DesignA:
    """Power-driven sizing (pre-registered): run the full matched-pair design at each rung of
    ``p.size_ladder`` and keep the SMALLEST holdout whose best strict-gate pair reaches
    ``p.target_mde`` on ``p.power_metric`` at 80% power over the configured test window.

    Everything uses design-window data only (``week_idx <= design_end``). If no rung reaches the
    target, the strict design with the lowest MDE is used and flagged UNDER-POWERED; a relaxed
    pair is only used if no rung has any strict pair (flagged as well). Never force-passed."""
    from dataclasses import replace as _replace
    ladder = list(p.size_ladder) or [p.pop_target]
    rows, designs, log0 = [], {}, []
    region_pop = float(zips.loc[(zips["region"] == p.region) & (zips["dsp_a_served"] == 1), "pop"].sum())
    for size in ladder:
        if size > p.max_holdout_share * region_pop * (1 + p.pop_tolerance):
            rows.append({"pop_target": size, "status": f"skipped: > {p.max_holdout_share:.0%} of served pop"})
            continue
        try:
            D = _design_at_size(panel, zips, cfg, _replace(p, pop_target=size))
        except RuntimeError as e:
            rows.append({"pop_target": size, "status": f"no design: {e}"})
            continue
        designs[size] = D
        mde = D.power["mde_at_test_length"]
        rows.append({"pop_target": size, "status": "ok", "tier": D.winner["tier"], "t_pop": float(D.winner["t_pop"]),
                     "n_strict": int(D.candidates["pass_strict"].sum()), "patch_id": D.winner["patch_id"],
                     "c_county": D.winner["c_county"], **{f"mde_{m}": v for m, v in mde.items()},
                     "meets_target": bool(D.winner["tier"] == "strict" and mde[p.power_metric] <= p.target_mde)})
    sizing = pd.DataFrame(rows)
    if not designs:
        raise RuntimeError("no feasible Vendor A design at any rung of the size ladder")
    ok = sizing[sizing["status"] == "ok"]
    meets = ok[ok["meets_target"]]
    strict = ok[ok["tier"] == "strict"]
    if len(meets):
        pick, verdict = float(meets["pop_target"].iloc[0]), "POWERED"
        why = (f"smallest holdout whose strict-gate pair detects {p.target_mde:.0%} on {p.power_metric} at "
               f"{p.power:.0%} power (MDE {meets[f'mde_{p.power_metric}'].iloc[0]:.1%})")
    elif len(strict):
        r = strict.sort_values(f"mde_{p.power_metric}").iloc[0]
        pick, verdict = float(r["pop_target"]), "UNDER-POWERED"
        why = (f"WARNING: no rung reaches the {p.target_mde:.0%} {p.power_metric} MDE target; most powerful strict design "
               f"used (MDE {r[f'mde_{p.power_metric}']:.1%}) - read sales as directional")
    else:
        r = ok.sort_values(f"mde_{p.power_metric}").iloc[0]
        pick, verdict = float(r["pop_target"]), "NOT VALIDATED"
        why = "WARNING: no strict-gate pair at any size; best relaxed/failed design used and flagged NOT VALIDATED"
    D = designs[pick]
    D.sizing = sizing
    D.sizing_decision = {"pop_target": pick, "verdict": verdict, "rationale": why, "power_metric": p.power_metric,
                         "target_mde": p.target_mde, "power": p.power, "alpha": p.alpha,
                         "test_weeks": D.power["test_weeks"],
                         "mde_selected": D.power["mde_at_test_length"][p.power_metric]}
    D.validation_log.insert(0, "size ladder: " + "; ".join(
        f"{r.pop_target/1e3:.0f}k -> " + (f"{r.tier}, {p.power_metric} MDE {getattr(r, 'mde_' + p.power_metric):.1%}"
                                          if r.status == "ok" else r.status) for r in sizing.itertuples()))
    D.validation_log.insert(1, f"sizing decision ({verdict}): {pick/1e3:.0f}k pop - {why}")
    return D


def _design_at_size(panel: pd.DataFrame, zips: pd.DataFrame, cfg, p: VendorAParams) -> DesignA:
    td = cfg.tests[p.test]
    design_weeks = list(range(td.design_start, td.design_end + 1))
    panel = panel[(panel["region"] == p.region) & (panel["week_idx"] <= td.design_end)]
    assert panel["week_idx"].max() <= td.design_end, "design must not see post-launch data"
    log: List[str] = []
    rz = zips[zips["region"] == p.region].copy()
    feat = F.static_features(panel, zips, design_weeks, region=p.region)
    excluded = {
        "not_served": feat.loc[feat["dsp_a_served"] == 0, "zip"].tolist(),
        "below_pop_floor": feat.loc[(feat["dsp_a_served"] == 1) & (feat["pop"] < p.min_zip_pop), "zip"].tolist(),
        "concurrent_test": feat.loc[feat["concurrent_test_flag"] == 1, "zip"].tolist(),
    }
    elig = feat[(feat["dsp_a_served"] == 1) & (feat["pop"] >= p.min_zip_pop)
                & (feat["concurrent_test_flag"] == 0)].reset_index(drop=True)
    log.append(f"eligible zips: {len(elig)} of {len(feat)} (served, pop >= {p.min_zip_pop:,}, not in another test)")
    popd = dict(zip(elig["zip"], elig["pop"].astype(float)))
    metric = mahalanobis_metric(elig, FEATURES_A, p.maha_ridge, "zip-level (standardised)")
    corr = elig[list(FEATURES_A)].corr()

    # --- treatment counties + seeds -----------------------------------------------------
    cpop = elig.groupby("county_id")["pop"].sum().sort_values(ascending=False)
    t_counties = [c for c in cpop.index if cpop[c] >= 1.6 * p.pop_target][:p.n_treatment_counties]
    if not t_counties:
        t_counties = list(cpop.index[:1])
    patches = []
    for cty in t_counties:
        sub = elig[elig["county_id"] == cty]
        g = adjacency_graph_balltree(sub, p.adjacency_miles)
        seeds = {"North": sub.loc[sub["lat"].idxmax(), "zip"], "South": sub.loc[sub["lat"].idxmin(), "zip"],
                 "East": sub.loc[sub["lon"].idxmax(), "zip"], "West": sub.loc[sub["lon"].idxmin(), "zip"]}
        for direction, s in seeds.items():
            ids, pp, ok = bfs_grow_patch(s, g, popd, p.pop_target, p.pop_tolerance)
            ids_lo, pp_lo, _ = bfs_grow_patch(s, g, popd, p.pop_target, p.pop_tolerance, stop="lower")
            patches.append({"patch_id": f"{cty}:{direction}", "county_id": cty,
                            "county": sub["county"].iloc[0], "direction": direction, "seed": s,
                            "zips": ids, "n_zips": len(ids), "pop": pp, "in_window": ok,
                            "pop_gap_pct": pp / p.pop_target - 1, "pop_if_stop_at_lower": pp_lo,
                            "spread_mi": mean_spread_miles(sub[sub["zip"].isin(ids)])})
    patches = pd.DataFrame(patches)
    patches = patches.drop_duplicates(subset=["pop", "n_zips"]).reset_index(drop=True)
    log.append(f"treatment patches: {len(patches)} from counties {t_counties}; in window: "
               f"{int(patches['in_window'].sum())}")

    # --- donor pools + control patches --------------------------------------------------
    rows, pools = [], []
    for _, tp in patches[patches["in_window"]].iterrows():
        tz = elig[elig["zip"].isin(tp["zips"])]
        cand = elig[elig["county_id"] != tp["county_id"]]
        ok = donors_outside_buffer(cand, tz, p.buffer_miles)
        valid = cand[ok]
        pools.append({"patch_id": tp["patch_id"], "n_valid": int(ok.sum()), "n_excluded_buffer": int((~ok).sum()),
                      "n_counties": int(valid["county_id"].nunique()), "valid_zips": valid["zip"].tolist(),
                      "buffer_zips": cand.loc[~ok, "zip"].tolist(), "valid_pop": float(valid["pop"].sum())})
        if valid.empty:
            continue
        gd = adjacency_graph_balltree(valid, p.adjacency_miles)
        t_prof = F.aggregate_profile(elig, tp["zips"], FEATURES_A)
        t_spread = tp["spread_mi"]
        seen = set()
        for s in sorted(valid["zip"]):
            ids, pp, okw = bfs_grow_patch(s, gd, popd, p.pop_target, p.pop_tolerance)
            key = frozenset(ids)
            if not okw or key in seen:
                continue
            seen.add(key)
            cz = valid[valid["zip"].isin(ids)]
            c_prof = F.aggregate_profile(elig, ids, FEATURES_A)
            if c_prof[list(FEATURES_A)].isna().any():
                continue
            c_spread = mean_spread_miles(cz)
            pen = min(c_spread / max(t_spread, 1e-6), 3.0) / 3.0
            d = metric.distance(t_prof, c_prof)
            anchor = cz.groupby("county_id")["pop"].sum().idxmax()
            rows.append({"patch_id": tp["patch_id"], "t_county": tp["county_id"], "direction": tp["direction"],
                         "c_anchor_county": anchor, "c_county": cz.loc[cz["county_id"] == anchor, "county"].iloc[0],
                         "c_n_counties": int(cz["county_id"].nunique()), "c_seed": s,
                         "t_pop": tp["pop"], "c_pop": pp, "t_n": tp["n_zips"], "c_n": len(ids),
                         "maha": d, "spread_penalty": pen, "composite": d + p.w_compact * pen,
                         "t_spread_mi": t_spread, "c_spread_mi": c_spread,
                         "t_zips": list(tp["zips"]), "c_zips": list(ids)})
    pools = pd.DataFrame(pools)
    cands = pd.DataFrame(rows)
    if cands.empty:
        raise RuntimeError("no control patch candidates found - relax population window or buffer")
    n_unique = len(cands)
    cands = (cands.sort_values("composite").groupby(["patch_id", "c_anchor_county"], group_keys=False)
             .head(p.top_per_county).sort_values("composite").reset_index(drop=True))
    log.append(f"scored {n_unique} unique control patches; kept top {p.top_per_county} per "
               f"(treatment patch, control county) -> {len(cands)} candidates")

    # patch-level covariance sensitivity
    g_all = adjacency_graph_balltree(elig, p.adjacency_miles)
    metric_patch = patch_covariance_metric(elig, g_all, popd, FEATURES_A, metric, p.pop_target, p.pop_tolerance,
                                           p.n_boot_cov, cfg.seed, p.maha_ridge)
    prof_cache: Dict[frozenset, pd.Series] = {}

    def prof(ids):
        k = frozenset(ids)
        if k not in prof_cache:
            prof_cache[k] = F.aggregate_profile(elig, ids, FEATURES_A)
        return prof_cache[k]
    cands["maha_patch_cov"] = [metric_patch.distance(prof(r.t_zips), prof(r.c_zips)) for r in cands.itertuples()]

    # --- behavioural validation (top N) ---------------------------------------------------
    cube = F.ZipCube(panel[panel["week_idx"].isin(design_weeks)])
    n_pre_readout = td.test_start - td.readout_pre_start
    top = cands.head(p.top_n_validate).copy()
    val_rows = []
    for r in top.itertuples():
        t = cube.sum(r.t_zips, p.validation_metric)
        c = cube.sum(r.c_zips, p.validation_metric)
        v_s = validate_pair(t, c, p.strict_gates)
        v_r = validate_pair(t, c, p.relaxed_gates)
        pw = pair_power(cube.sum(r.t_zips, "qs"), cube.sum(r.c_zips, "qs"), p, n_pre_readout, p.lifts)
        # in-time placebo: the change-ratio 'effect' over the last design weeks when nothing happened
        drift = cdid.change_ratio_did(cube.sum(r.t_zips, "qs"), cube.sum(r.c_zips, "qs"),
                                      design_weeks[:-p.pseudo_post_weeks], design_weeks[-p.pseudo_post_weeks:])["lift"]
        val_rows.append({**{k: v_s.get(k) for k in ("wow_corr", "slope_ratio", "treat_cv", "ctrl_cv", "n_weeks")},
                         "pass_strict": v_s["passed"], "reason_strict": v_s["reason"],
                         "pass_relaxed": v_r["passed"], "reason_relaxed": v_r["reason"],
                         "sigma_gap": pw["sigma_gap"], "rho_gap": pw["rho"], "placebo_drift": drift,
                         **{f"weeks_{k}": v for k, v in pw["weeks_to_power"].items()}})
    top = pd.concat([top.reset_index(drop=True), pd.DataFrame(val_rows)], axis=1)
    rank_col = f"weeks_{p.rank_lift:.2f}"
    top["tier"] = np.where(top["pass_strict"], "strict", np.where(top["pass_relaxed"], "relaxed", "fail"))
    w = top[rank_col].astype(float).fillna(999)
    # four equally weighted ranks: static similarity, dynamic co-movement, speed, and the
    # in-time placebo drift (a pair whose ratio already drifts with no intervention is a bad twin)
    top["rank_score"] = (top["composite"].rank(method="min") + (-top["wow_corr"]).rank(method="min")
                         + w.rank(method="min") + top["placebo_drift"].abs().rank(method="min"))
    tier_order = {"strict": 0, "relaxed": 1, "fail": 2}
    top["tier_order"] = top["tier"].map(tier_order)
    top = top.sort_values(["tier_order", "rank_score", "composite"]).reset_index(drop=True)
    top["rank"] = np.arange(1, len(top) + 1)
    n_s, n_r = int(top["pass_strict"].sum()), int((top["tier"] == "relaxed").sum())
    log.append(f"validation: {n_s}/{len(top)} pass strict gates {p.strict_gates}; {n_r} more pass relaxed "
               f"gates {p.relaxed_gates}")
    win = top.iloc[0].to_dict()
    if win["tier"] == "fail":
        log.append("WARNING: no candidate passed even the relaxed gates; the top-ranked pair is used but the "
                   "design is flagged NOT VALIDATED (no force-pass)")
    elif win["tier"] == "relaxed":
        log.append("no strict pass; winner selected from the explicitly logged RELAXED tier")
    treated, control = list(win["t_zips"]), list(win["c_zips"])
    buffer = [z for z in rz["zip"] if z not in treated]
    dmin = G.min_distance_to_set(rz, treated)
    buffer = rz.loc[(dmin > 0) & (dmin <= p.buffer_miles), "zip"].tolist()
    assert not set(buffer) & set(control), "control zip inside the spillover buffer"

    # --- balance (SMD) + profile ----------------------------------------------------------
    smd_feats = list(FEATURES_A) + ["median_hh_income", "homeowner_rate", "median_age"]
    smd = F.standardized_mean_diff(elig, treated, control, smd_feats)
    profile = pd.DataFrame({"treatment": F.aggregate_profile(elig, treated, smd_feats),
                            "control": F.aggregate_profile(elig, control, smd_feats)})
    profile["pct_diff"] = profile["treatment"] / profile["control"] - 1

    # --- power for the winner + design-time injected-lift check ---------------------------
    weekly = {}
    for r in top.head(p.top_n_review).itertuples():
        weekly[f"rank{r.rank}"] = pd.DataFrame({f"{side}_{m}": cube.sum(z, m) for side, z in
                                                (("t", r.t_zips), ("c", r.c_zips)) for m in ("qs", "rated", "sales")})
    tq, cq = cube.sum(treated, "qs"), cube.sum(control, "qs")
    pw = {m: pair_power(cube.sum(treated, m), cube.sum(control, m), p, n_pre_readout, p.lifts)
          for m in ("qs", "rated", "sales")}
    weeks_grid = list(range(1, p.max_weeks + 1))
    curves = {m: cpow.power_curve(pw[m]["sigma_gap"], p.lifts, weeks_grid, n_pre=n_pre_readout, alpha=p.alpha,
                                  rho=pw[m]["rho"]) for m in pw}
    n_post = td.test_end - td.test_start + 1
    mde = {m: cpow.did_mde(pw[m]["sigma_gap"], n_post, n_pre_readout, p.alpha, p.power, rho=pw[m]["rho"])
           for m in pw}
    power = {"convention": f"one-sided alpha={p.alpha}, power={p.power:.0%}, sigma = sd of weekly relative "
                           f"T-C gap (design window), AR(1)-inflated, n_pre={n_pre_readout} readout-pre weeks",
             "by_metric": pw, "curves": curves, "mde_at_test_length": mde, "test_weeks": n_post,
             "spec_level_formula_weeks_15pct_qs": spec_power_levels(tq, cq, 0.15, p.alpha, p.power)}
    inj = injected_check(cube, treated, control, design_weeks, p)
    log.append(f"winner: {win['patch_id']} vs {win['c_county']} | maha={win['maha']:.2f} wow_corr="
               f"{win['wow_corr']:.2f} tier={win['tier']}")
    return DesignA(params=p, eligible=elig, region_zips=rz, treatment_counties=t_counties, patches=patches,
                   donor_pools=pools, candidates=top, metric=metric, metric_patch=metric_patch, corr=corr,
                   winner=win, treated=treated, control=control, buffer_zips=buffer, excluded=excluded, smd=smd,
                   profile=profile, power=power, injected=inj, validation_log=log, weekly=weekly)


def injected_check(cube: F.ZipCube, treated, control, design_weeks: Sequence[int], p: VendorAParams,
                   lifts: Sequence[float] = (0.05, 0.10, 0.15, 0.20)) -> pd.DataFrame:
    """Design-time validation: pretend the last ``pseudo_post_weeks`` design weeks were a holdout,
    remove ``lift`` from treatment there, and recover it with the change-ratio DiD (natural
    pseudo-post bias subtracted). Nothing is tuned to make this pass."""
    pre = list(design_weeks[:-p.pseudo_post_weeks])
    post = list(design_weeks[-p.pseudo_post_weeks:])
    rows = []
    for m in ("qs", "rated", "sales"):
        t, c = cube.sum(treated, m), cube.sum(control, m)
        nat = cdid.change_ratio_did(t, c, pre, post, sign=-1)["lift"]
        for l in lifts:
            ti = t.copy()
            ti.loc[post] = ti.loc[post] * (1 - l)
            r = cdid.change_ratio_did(ti, c, pre, post, sign=-1, n_boot=400, seed=1, alpha=p.alpha * 2)
            rows.append({"metric": m, "injected": l, "detected_raw": r["lift"], "natural_bias": nat,
                         "detected": r["lift"] - nat, "error": r["lift"] - nat - l,
                         "ci_low": r["ci_low"] - nat, "ci_high": r["ci_high"] - nat})
    return pd.DataFrame(rows)
