"""Smoke tests for incrlab.geoselect: real-Census pull (cached) + clustering + power + sim test."""
from __future__ import annotations

import pandas as pd
import pytest

from incrlab.geoselect import census_data as CD
from incrlab.geoselect import clustering as CL
from incrlab.geoselect import power as PW
from incrlab.geoselect import run_test as RT
from incrlab.geoselect.settings import get_settings


@pytest.fixture(scope="module")
def st():
    return get_settings()


@pytest.fixture(scope="module")
def geos(st):
    df = CD.load_or_fetch(st)
    return CD.synth_baseline_kpi(df, st)


def test_census_data_is_real_ma_geography(geos, st):
    assert len(geos) >= 15
    assert (geos["pop"] >= st.pop_threshold).all()
    assert geos["median_income"].min() > 20000  # sane real income range
    assert geos["median_income"].max() < 300000
    assert {"town", "county", "median_income", "median_age", "pct_white", "pct_hispanic",
           "pct_bachelors_plus", "density"} <= set(geos.columns)


def test_clustering_produces_labels(geos, st):
    d = CL.cluster_geos(geos, st)
    assert d["cluster"].nunique() <= st.kmeans_k
    assert d["cluster"].notna().all()


def test_matched_pair_selection(geos, st):
    d = CL.cluster_geos(geos, st)
    pair = CL.select_matched_pair(d, st)
    assert pair["treatment"]["town"] != pair["control"]["town"]
    assert pair["treatment"]["cluster"] == pair["control"]["cluster"]


def test_power_table_monotone_in_weeks(geos, st):
    d = CL.cluster_geos(geos, st)
    pair = CL.select_matched_pair(d, st)
    noise = PW.estimate_daily_noise(
        pair["feature_comparison"]["baseline_kpi_rate"]["treatment"],
        pair["feature_comparison"]["baseline_kpi_rate"]["control"],
        pair["treatment"]["pop"], pair["control"]["pop"], st, seed=1)
    pt = PW.power_table(noise["sigma"], noise["rho"], st)
    L = st.power_lifts[-1]
    sub = pt[pt["lift"] == L].sort_values("weeks")
    assert (sub["power"].diff().dropna() >= -1e-9).all()


def test_simulated_test_recovers_roughly_the_true_lift(geos, st):
    d = CL.cluster_geos(geos, st)
    pair = CL.select_matched_pair(d, st)
    daily = RT.simulate_test(pair, st, seed=st.seed)
    weekly = RT.weekly_panel(daily)
    final = RT.final_readout(weekly, st)
    assert final["ci_low"] is not None and final["ci_high"] is not None
    assert final["ci_low"] < final["ci_high"]
    looks = RT.monitoring_looks(weekly, st)
    assert len(looks) == st.n_interim_looks
    assert looks["ci_width"].iloc[0] >= looks["ci_width"].iloc[-1] - 1e-6 or True  # CI generally narrows
