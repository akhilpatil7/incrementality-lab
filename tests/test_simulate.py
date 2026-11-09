import json

import numpy as np
import pandas as pd

from incrlab import geo
from incrlab.config import get_config
from incrlab.simulate import (apply_intervention, build_world, compute_ground_truth, realize, run_scenario,
                              simulate, write_baseline)


def test_generator_deterministic(small_cfg):
    a = simulate(world=build_world(small_cfg))
    b = simulate(world=build_world(small_cfg))
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_generator_seed_changes_output():
    a = simulate(world=build_world(get_config(seed=1, small=True)))["national_weekly"]
    b = simulate(world=build_world(get_config(seed=2, small=True)))["national_weekly"]
    assert not a["qs"].equals(b["qs"])


def test_panel_shapes_and_accounting(small_world):
    p = simulate(world=small_world)
    z, c, n = p["zip_weekly"], p["county_weekly"], p["national_weekly"]
    assert len(n) == 156 and n["week_idx"].tolist() == list(range(156))
    assert len(z) == len(small_world.geo.zips) * 156
    assert (z["rated"] <= z["qs"]).all() and (z["sales"] <= z["rated"]).all()
    for s in ("qs", "rated", "sales"):
        lt = z[[f"lt_{src}_{s}" for src in ("search_brand", "search_nonbrand", "organic", "social", "dsp",
                                            "direct", "unattributed", "other")]].sum(1)
        assert (lt == z[s]).all(), f"last-touch partition broken at stage {s}"
        assert z[s].sum() == c[s].sum()
    assert n["qs"].sum() > z["qs"].sum()  # national includes rest-of-nation unit


def test_monthly_media_dedupe(small_world):
    z = simulate(world=small_world)["zip_weekly"]
    naive = z["spend_dsp_a_month"].sum()
    dedup = geo.monthly_media_sum(z, ["spend_dsp_a_month"])["spend_dsp_a_month"].sum()
    assert np.isclose(dedup, z["spend_dsp_a"].sum())
    assert naive > dedup * 3  # repeating monthly values on weekly rows inflates naive sums


def test_counterfactual_consistency(small_world):
    """Untreated potential outcomes are identical; only treated units in-window (+ buffer) change."""
    base = simulate(world=small_world)["zip_weekly"]
    zs = small_world.geo.zips
    treated = zs[(zs.region == "NV") & (zs.dsp_a_served == 1)].zip.tolist()[:8]
    td = small_world.cfg.tests["dsp_a"]
    sc = apply_intervention(small_world, treated, (td.test_start, td.test_end), "dsp_a",
                            {"spend_multiplier": 0.0, "leakage": 0.0}, test="dsp_a", arm="holdout")
    res = realize(sc)
    s = res.panels["zip_weekly"]
    changed = base["qs"].to_numpy() != s["qs"].to_numpy()
    outside = ~(base["zip"].isin(treated) & base["week_idx"].between(td.test_start, td.test_end + 10)).to_numpy()
    assert changed[outside].sum() == 0
    assert (base["qs"].to_numpy()[base["week_idx"] < td.test_start] ==
            s["qs"].to_numpy()[base["week_idx"] < td.test_start]).all()
    r = res.truth["dsp_a"]["holdout"]
    assert r["qs"]["realized_counterfactual"] == base.loc[base.zip.isin(treated) &
                                                          base.week_idx.between(td.test_start, td.test_end), "qs"].sum()
    assert r["qs"]["true_lift_pct"] > 0 and r["sales"]["true_lift_pct"] > r["qs"]["true_lift_pct"]


def test_run_scenario_arms(small_world):
    res = run_scenario({"search": {"all_search": ["NV-C02"], "branded_only": ["NV-C05"]},
                        "meta": {"holdout": ["EM-C03"]}}, world=small_world)
    t = res.truth
    assert set(t) == {"search", "meta"}
    a, b = t["search"]["all_search"], t["search"]["branded_only"]
    assert a["qs"]["true_lift_pct"] > b["qs"]["true_lift_pct"] > 0
    assert abs(b["true_recapture_qs"] - small_world.cfg.truth["search_brand"].recapture) < 1e-6
    z = res.panels["zip_weekly"]
    assert set(z.loc[z.county_id == "NV-C02", "arm_search"]) == {"all_search"}
    sp = z[(z.county_id == "NV-C05") & z.week_idx.between(79, 84)]["spend_search_brand"].sum()
    assert sp == 0


def test_ground_truth_schema(small_world, tmp_path):
    info = write_baseline(small_world.cfg, data_dir=tmp_path)
    gt = json.loads((tmp_path / "ground_truth.json").read_text())
    for k in ("schema_version", "generator", "conventions", "totals", "channels", "search", "tests",
              "dsp_site_last_touch", "dsp_b_vendor_switch"):
        assert k in gt
    for ch in small_world.cfg.channels:
        c = gt["channels"][ch]
        for k in ("params", "total_spend", "incremental", "roi", "marginal_roi", "cost_per_incremental",
                  "contribution_share", "by_year"):
            assert k in c, (ch, k)
        assert c["incremental"]["qs"] > 0 and c["total_spend"] > 0
        assert c["marginal_roi"] < c["roi"]  # diminishing returns
    assert gt["search"]["true_recapture_brand"] == 0.40
    assert gt["channels"]["social"]["last_touch_bias"]["qs"] > 1 > gt["channels"]["social"]["last_touch_bias"]["sales"]
    assert gt["channels"]["search_brand"]["last_touch_bias"]["qs"] > 1.5
    for name in ("meta", "search", "dsp_a", "dsp_b"):
        assert name in gt["tests"] and gt["tests"][name]["region_wide_holdout_lift"]
    for f in ("zip_weekly", "county_weekly", "region_weekly", "national_weekly", "zip_static", "county_static"):
        assert (tmp_path / f"{f}.parquet").exists()
    assert info["n_zips"] == len(small_world.geo.zips)


def test_geography(small_world):
    z = small_world.geo.zips
    assert z["zip"].is_unique and z.groupby("county_id").size().min() >= 3
    assert set(z["region"]) == {"EM", "NV"}
    d = geo.haversine_miles(40.0, -88.0, 41.0, -88.0)
    assert 68 < d < 70
    G = geo.adjacency_graph(z[z.region == "NV"], 10.0)
    import networkx as nx
    assert nx.is_connected(G)
    agg = geo.aggregate_static(z)
    assert np.isclose(agg["pop"].sum(), z["pop"].sum())
    assert z[["per_capita_income", "zip_propensity"]].corr().iloc[0, 1] < -0.2
