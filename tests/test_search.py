"""Module 2 (Paid Search two-arm dark-out): selection determinism, estimate-vs-truth, results schema."""
import json
import math

import pytest

from incrlab.config import RESULTS_DIR, get_config
from incrlab.search import ARMS, get_settings
from incrlab.search import design as D
from incrlab.search import panel as P
from incrlab.search.pipeline import run
from incrlab.search.report import _estimates, headline
from incrlab.simulate import get_world, realize



@pytest.fixture(scope="module")
def o():
    return run(get_settings(get_config()), make_figures=False, write=False, verbose=False)


def test_design_deterministic_and_disjoint(o):
    s = o["settings"]
    world = get_world(s.cfg)
    bau = P.build_county_panel(realize(world, with_truth=False).panels["county_weekly"], s.region)
    d2 = D.run_design(bau, world.geo.zips, world.geo.counties, s)["design"]
    for k in ("all_search", "branded_only", "control", "bau_donors"):
        assert d2[k] == o["design"][k], k
    used = [*o["design"]["all_search"], *o["design"]["branded_only"], *o["design"]["control"]]
    assert len(used) == len(set(used)), "arms must not share counties"
    assert not set(used) & set(o["design"]["bau_donors"])


def test_triangulated_lift_near_truth(o):
    tri = o["triangulated"]
    for _, r in tri.iterrows():
        if r["metric"] in ("qs", "rated"):
            assert abs(r["estimate"] - r["truth"]) < 0.08, (r["arm"], r["metric"], r["estimate"], r["truth"])
    allr = tri[(tri["arm"] == "all_search") & (tri["metric"] == "rated")].iloc[0]
    assert allr["ci_low"] <= allr["truth"] <= allr["ci_high"]
    assert abs(allr["estimate"] - allr["truth"]) < 0.04


def test_bsts_significance_is_interval_based(o):
    b = o["estimates"][o["estimates"]["method"] == "bsts"]
    for _, r in b.iterrows():
        excl0 = (r["ci_low"] > 0) or (r["ci_high"] < 0)
        assert bool(r["significant"]) == bool(excl0)


def test_recapture_covers_truth_for_branded(o):
    rec = o["recapture"]
    if "ref" in rec:
        rec = rec[rec["ref"] == "synthetic"]
    r = rec[(rec["arm"] == "branded_only") & (rec["stage"] == "rated")].iloc[0]
    assert r["ci_low"] <= r["truth"] <= r["ci_high"]


def test_results_schema(o):
    h = headline(o)
    assert h["all_search_counties"] and h["control_county"]
    ests = _estimates(o)
    assert ests
    for e in ests:
        for k in ("test", "arm", "metric", "method", "estimate"):
            assert k in e
        assert e["arm"] in ARMS
    ipd = o["economics"]["incremental_per_dollar"]
    for ch in ("search_brand", "search_nonbrand"):
        assert {"sales", "spend", "window"} <= set(ipd[ch]) and math.isfinite(ipd[ch]["spend"])
    p = RESULTS_DIR / "search.json"
    if p.exists():
        d = json.loads(p.read_text())
        for k in ("module", "schema_version", "generated_at", "seed", "small", "headline", "estimates",
                  "design", "validation", "truth", "extra", "figures"):
            assert k in d, k
        assert d["module"] == "search"
        assert {"search_brand", "search_nonbrand"} <= set(d["extra"]["incremental_per_dollar"])
        assert 10 <= len(d["figures"]) <= 16
