"""Tests for the Programmatic DSP module (small geography; fast)."""
from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from incrlab import geo as G
from incrlab.config import get_config
from incrlab.dsp import features as F
from incrlab.dsp.design_a import (bfs_grow_patch, donors_outside_buffer, mahalanobis_metric, validate_pair)
from incrlab.dsp.design_b import augmented_sc, check_structural_break, geo_kmeans, score_tournament
from incrlab.dsp.params import default_params
from incrlab.dsp.readout import change_ratio_boot, welch_gap


# --------------------------------------------------------------------------------------
# granularity rules ("never average averages")
# --------------------------------------------------------------------------------------
def test_monthly_media_deduped_before_summing():
    weeks = pd.to_datetime(["2025-01-06", "2025-01-13", "2025-01-20", "2025-02-03"])
    p = pd.DataFrame({"zip": ["Z1"] * 4, "week_idx": range(4), "week_start": weeks,
                      "month_start": pd.to_datetime(["2025-01-01"] * 3 + ["2025-02-01"]),
                      "spend_dsp_a_month": [300.0, 300.0, 300.0, 50.0],
                      "impressions_dsp_a_month": [1000] * 3 + [10], "clicks_dsp_a_month": [6] * 3 + [1],
                      "vendor_conv_qs_dsp_a_month": [9] * 3 + [2]})
    m = F.monthly_media(p, ["Z1"], "a")
    assert m["spend_dsp_a_month"].sum() == pytest.approx(350.0)  # not 950
    assert m["impressions_dsp_a_month"].sum() == 1010


def test_rates_recomputed_from_sums():
    feat = pd.DataFrame({"zip": ["a", "b"], "pop": [1000, 9000], "qs": [10, 900], "rated": [9, 450],
                         "sales": [9, 45], "n_weeks": [1, 1], "per_capita_income": [1.0, 2.0]})
    prof = F.aggregate_profile(feat, ["a", "b"], ["rated_rate", "close_rate", "qs_per_capita", "per_capita_income"])
    assert prof["rated_rate"] == pytest.approx(459 / 910)  # mean of rates would be 0.70
    assert prof["close_rate"] == pytest.approx(54 / 459)
    assert prof["qs_per_capita"] == pytest.approx(910 / 10000)
    assert prof["per_capita_income"] == pytest.approx(1.9)  # population-weighted


def test_smd_zero_for_identical_groups():
    feat = pd.DataFrame({"zip": list("abcd"), "pop": [1, 2, 3, 4], "x": [1.0, 2.0, 1.0, 2.0]})
    s = F.standardized_mean_diff(feat, ["a", "b"], ["c", "d"], ["x"])
    assert abs(s["smd"].iloc[0]) < 0.25


# --------------------------------------------------------------------------------------
# geography engineering
# --------------------------------------------------------------------------------------
def _line_graph(n):
    g = nx.path_graph(n)
    return nx.relabel_nodes(g, {i: f"z{i:02d}" for i in range(n)})


def test_bfs_stops_closest_to_target_not_at_lower_bound():
    g = _line_graph(20)
    pop = {n: 100.0 for n in g}
    ids, p, ok = bfs_grow_patch("z00", g, pop, target=1000, tol=0.15)
    assert ok and p == pytest.approx(1000)
    ids_lo, p_lo, ok_lo = bfs_grow_patch("z00", g, pop, target=1000, tol=0.15, stop="lower")
    assert ok_lo and p_lo == pytest.approx(900)  # original behaviour: first value >= 850
    assert nx.is_connected(g.subgraph(ids))


def test_balltree_buffer_matches_bruteforce_haversine():
    rng = np.random.default_rng(0)
    cand = pd.DataFrame({"lat": rng.uniform(40, 41, 300), "lon": rng.uniform(-88, -87, 300)})
    treat = pd.DataFrame({"lat": [40.5, 40.55], "lon": [-87.5, -87.45]})
    miles = 15.0
    mask = donors_outside_buffer(cand, treat, miles)
    d = G.haversine_miles(cand["lat"].to_numpy()[:, None], cand["lon"].to_numpy()[:, None],
                          treat["lat"].to_numpy()[None], treat["lon"].to_numpy()[None]).min(1)
    assert np.array_equal(mask, d > miles)


def test_buffer_label_comes_from_parameter():
    p = default_params(get_config(small=True)).a
    p2 = replace(p, buffer_miles=42)
    assert "42-mi" in p2.buffer_label and "30" not in p2.buffer_label


def test_mahalanobis_metric_properties():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 3)) * [1, 1000, 0.01], columns=list("abc"))
    m = mahalanobis_metric(X, list("abc"), ridge=1e-6)
    a, b = X.iloc[0], X.iloc[1]
    assert m.distance(a, a) == pytest.approx(0, abs=1e-12)
    assert m.distance(a, b) == pytest.approx(m.distance(b, a))
    # scale-invariant: rescaling one feature does not change distances
    X2 = X.assign(b=X["b"] / 1000)
    m2 = mahalanobis_metric(X2, list("abc"), ridge=1e-6)
    assert m.distance(a, b) == pytest.approx(m2.distance(X2.iloc[0], X2.iloc[1]), rel=1e-6)


# --------------------------------------------------------------------------------------
# validation gates: explicit tiers, never force-passed
# --------------------------------------------------------------------------------------
def test_validate_pair_fails_with_reasons():
    idx = pd.RangeIndex(95, 112)
    rng = np.random.default_rng(3)
    t = pd.Series(100 + rng.normal(0, 40, len(idx)), index=idx)
    c = pd.Series(100 + rng.normal(0, 5, len(idx)), index=idx)
    r = validate_pair(t, c, (0.70, 0.5, 2.0, 0.30))
    assert not r["passed"] and "wow_corr" in r["reason"]
    r2 = validate_pair(t, t * 1.1, (0.70, 0.5, 2.0, 0.60))
    assert r2["passed"] and r2["reason"] == "pass"


def test_structural_break_guards_empty_group():
    p = pd.DataFrame({"week_idx": np.repeat(np.arange(100, 110), 2), "qs": 100, "vendor_conv_qs_dsp_b": 30})
    r = check_structural_break(p, switch_week=96)
    assert np.isnan(r["p_value"]) and not r["significant"] and "insufficient" in r["decision"]


def test_structural_break_detects_shift():
    w = np.arange(80, 110)
    p = pd.DataFrame({"week_idx": w, "qs": 1000,
                      "vendor_conv_qs_dsp_b": np.where(w < 96, 400, 300) + np.random.default_rng(0).integers(-10, 10, len(w))})
    r = check_structural_break(p, switch_week=96)
    assert r["significant"] and r["shift_pct"] == pytest.approx(-0.25, abs=0.03)


def test_kmeans_k_follows_documented_rule():
    rng = np.random.default_rng(2)
    feat = pd.DataFrame({"x": rng.normal(size=300), "lat": rng.uniform(40, 41, 300), "lon": rng.uniform(-88, -87, 300),
                         "pop": 10_000})
    km = geo_kmeans(feat, ["x", "lat", "lon"], 3.0, (3, 8), 42, 5, pop_target=600_000)
    sil = km["silhouette"]
    elig = sil[sil["mean_cluster_pop"] <= 600_000]
    assert km["k"] == int(elig.loc[elig["silhouette"].idxmax(), "k"])
    assert set(np.unique(km["labels"])) == set(range(1, km["k"] + 1))


def test_tournament_score_weights():
    rep = pd.DataFrame({"cluster": [1, 2, 3], "status": "ok", "abs_error": [0.01, 0.02, 0.03],
                        "rel_rmse": [0.05, 0.05, 0.05], "weeks_to_80": [5, 5, 5]})
    s = score_tournament(rep, (0.4, 0.35, 0.25)).set_index("cluster")["score"]
    assert s.loc[1] > s.loc[2] > s.loc[3]


def test_augmented_sc_reduces_to_sc_when_fit_is_perfect():
    t = np.arange(30)
    don = pd.DataFrame({f"C{i}": 10 + i + np.sin(t / 3 + i) for i in range(1, 5)}, index=t)
    w = pd.Series([0.5, 0.5, 0, 0], index=don.columns)
    wide = don.assign(C0=don[["C1", "C2"]].to_numpy() @ np.array([0.5, 0.5]))
    r = augmented_sc(wide, "C0", list(range(24)), list(range(24, 30)), w, (0.1, 1, 10))
    assert np.allclose(r["counterfactual"].to_numpy(), wide["C0"].to_numpy(), atol=1e-8)


# --------------------------------------------------------------------------------------
# readout estimators recover a known effect
# --------------------------------------------------------------------------------------
def test_change_ratio_and_welch_recover_holdout_loss():
    rng = np.random.default_rng(5)
    idx = pd.RangeIndex(0, 30)
    common = 1000 * (1 + 0.1 * np.sin(np.arange(30) / 4)) * np.exp(rng.normal(0, 0.01, 30))
    c = pd.Series(common * 1.2, index=idx)
    t = pd.Series(common.copy(), index=idx)
    post = list(range(22, 30))
    t.loc[post] *= 0.9  # pausing loses 10%
    r = change_ratio_boot(t, c, list(range(12, 22)), post, sign=-1, n_boot=500, seed=0, alpha=0.1)
    assert r["lift"] == pytest.approx(0.10, abs=0.02) and r["significant"]
    w = welch_gap(t, c, list(range(12, 22)), post, sign=-1, alpha=0.1)
    assert w["lift"] == pytest.approx(0.10 * 0.9 / 0.9, abs=0.03) and w["ci_low"] > 0


# --------------------------------------------------------------------------------------
# end to end on the small world
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_run():
    from incrlab.dsp.pipeline import run
    cfg = get_config(seed=123, small=True)
    return run(figures=False, cfg=cfg, write=False)


def test_end_to_end_small(small_run):
    S = small_run
    A, B, R, cfg = S.A, S.B, S.R, S.cfg
    # designs chose units from data and respect the rules
    assert len(A.treated) > 0 and len(A.control) > 0 and not set(A.treated) & set(A.control)
    assert not set(A.control) & set(A.buffer_zips)
    zs = A.region_zips.set_index("zip")
    d = G.haversine_miles(zs.loc[A.control, "lat"].to_numpy()[:, None], zs.loc[A.control, "lon"].to_numpy()[:, None],
                          zs.loc[A.treated, "lat"].to_numpy()[None], zs.loc[A.treated, "lon"].to_numpy()[None])
    assert d.min() > A.params.buffer_miles  # buffer distance in code == the parameter
    assert A.winner["tier"] in ("strict", "relaxed", "fail")
    if A.winner["tier"] != "strict":  # no silent force-pass
        assert not A.winner["pass_strict"]
    zall = R.extra["scenario"].panels["zip_weekly"].drop_duplicates("zip").set_index("zip")
    assert zall.loc[B.treated, "concurrent_test_flag"].sum() == 0  # concurrent-test zips excluded
    assert B.kmeans["k"] in set(B.kmeans["silhouette"]["k"])
    assert abs(B.weights.sum() - 1) < 1e-6 and (B.weights >= 0).all()
    # the scenario realized exactly the selected geos
    zw = R.extra["scenario"].panels["zip_weekly"]
    assert set(zw.loc[zw["arm_dsp_a"] == "holdout", "zip"]) == set(A.treated)
    assert set(zw.loc[zw["arm_dsp_b"] == "holdout", "zip"]) == set(B.treated)
    # injected-lift validation recovers the injected effect (design time; no tuning)
    assert abs(B.injected_final["detected"] - B.params.injected_lift) < 0.04
    inj = A.injected[(A.injected["metric"] == "qs") & (A.injected["injected"] == 0.10)].iloc[0]
    assert abs(inj["detected"] - 0.10) < 0.04
    # every estimate carries truth + a 90% interval; significance == interval excludes 0
    for e in R.estimates:
        assert {"test", "arm", "metric", "method", "estimate", "truth"} <= set(e)
        assert e["ci_low"] <= e["estimate"] <= e["ci_high"] or e["method"] in ("bsts",)
        assert e["significant"] == bool(e["ci_low"] > 0 or e["ci_high"] < 0)


def test_incremental_per_dollar_contract(small_run):
    from incrlab.dsp.pipeline import incremental_per_dollar
    ipd = incremental_per_dollar(small_run)
    assert set(ipd) == {"dsp_a", "dsp_b"}
    for ch, rec in ipd.items():
        assert rec["spend"] > 0 and rec["se"] > 0 and abs(rec["sales"]) < 0.05
        assert rec["window"] == [small_run.cfg.tests[ch].test_start, small_run.cfg.tests[ch].test_end]


def test_design_never_sees_post_launch_data(small_run):
    """Corrupting every post-design week must not change the selected units."""
    from incrlab.dsp.design_a import run_design_a
    from incrlab.simulate import simulate
    cfg = small_run.cfg
    zw = simulate(cfg).get("zip_weekly").copy()
    late = zw["week_idx"] > cfg.tests["dsp_a"].design_end
    zw.loc[late, "qs"] = 0
    A2 = run_design_a(zw, _zips(cfg), cfg, small_run.params.a)
    assert set(A2.treated) == set(small_run.A.treated) and set(A2.control) == set(small_run.A.control)


def _zips(cfg):
    from incrlab.simulate import get_world
    return get_world(cfg).geo.zips


# --------------------------------------------------------------------------------------
# power-driven sizing + sales-hinged decision rule
# --------------------------------------------------------------------------------------
def test_power_driven_sizing_rule(small_run):
    A = small_run.A
    sz, sd = A.sizing, A.sizing_decision
    assert len(sz) == len(A.params.size_ladder)
    assert sd["verdict"] in ("POWERED", "UNDER-POWERED", "NOT VALIDATED")
    ok = sz[sz["status"] == "ok"]
    meets = ok[ok["meets_target"]]
    if sd["verdict"] == "POWERED":  # smallest qualifying rung, strict tier, MDE under target
        assert sd["pop_target"] == meets["pop_target"].min()
        assert A.winner["tier"] == "strict" and sd["mde_selected"] <= A.params.target_mde
    else:
        assert meets.empty
    assert abs(sd["mde_selected"] - A.power["mde_at_test_length"][A.params.power_metric]) < 1e-12


def test_decision_never_go_without_sales_power(small_run):
    for v, d in small_run.R.decisions.items():
        if not d["powered_for_sales"]:
            assert d["decision"] in ("HOLD - RE-TEST POWERED ON SALES", "NO-GO")
        if d["decision"] == "GO":
            assert d["sales_significant"] and d["powered_for_sales"]
