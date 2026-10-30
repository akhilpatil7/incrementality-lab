"""Module 1 (Social / Meta geo holdout): selection determinism, estimate-vs-truth, results schema."""
import json
import math

import pytest

from incrlab.config import RESULTS_DIR, get_config
from incrlab.meta import pipeline as MP
from incrlab.meta import report



@pytest.fixture(scope="module")
def S():
    return MP.run(get_config(), figures=False, write=False)


def test_selection_is_deterministic_and_pipeline_chosen(S):
    S2 = MP.run_design(get_config())
    assert S2.unit == S.unit
    assert (S2.dfit.weights.round(6) == S.dfit.weights.round(6)).all()
    # the treated unit is selected from the candidate set, never hard-coded, and passes the hard filters
    assert S.unit in set(S.candidates["unit"])
    assert bool(S.ranked.set_index("unit").loc[S.unit, "passes"])
    # locked weights: simplex, donors exclude the treated counties
    w = S.dfit.weights
    assert abs(w.sum() - 1) < 1e-6 and (w >= 0).all()
    assert not set(S.unit.split("+")) & set(w[w > 0].index)


def test_injected_lift_recovered_on_selected_unit(S):
    v = S.validation[S.validation["unit"] == S.unit]
    assert set(v["injected"].round(2)) >= {0.05, 0.10, 0.15}
    assert (v["abs_error"] < 0.03).all(), v


def test_estimates_within_tolerance_of_truth(S):
    tab = MP.estimates_table(S)
    for _, r in tab.iterrows():
        assert abs(r["estimate"] - r["truth"]) < 0.06, (r["method"], r["metric"], r["estimate"], r["truth"])
    did = tab[tab["method"] == "did_week_fe_hc1"]
    assert (did["ci_low"] <= did["truth"]).all() and (did["truth"] <= did["ci_high"]).all()
    # placebo p-values are rank-based and never below 1/(J+1)
    for s, p in S.placebo.items():
        assert p["p_value_lift_one_sided"] >= p["min_attainable_p"] - 1e-12


def test_attribution_overclaims_quote_starts(S):
    ipd = MP.incremental_per_dollar(S)["social"]
    for s in ("qs", "rated", "sales"):
        assert math.isfinite(ipd[s]) and ipd[f"{s}_se"] > 0
    assert ipd["sales"] == pytest.approx(ipd["sales_truth"], rel=1.0)


def test_results_json_schema(S, tmp_path):
    p = report.write_results_json(S, results_dir=tmp_path)
    for path in [p] + ([RESULTS_DIR / "meta.json"] if (RESULTS_DIR / "meta.json").exists() else []):
        d = json.loads(path.read_text())
        for k in ("module", "schema_version", "generated_at", "seed", "small", "headline", "estimates",
                  "design", "validation", "truth", "extra"):
            assert k in d, k
        assert d["module"] == "meta"
        assert d["design"]["treated_units"]
        for e in d["estimates"]:
            for k in ("test", "arm", "metric", "method", "estimate"):
                assert k in e
            assert e["metric"] in ("qs", "rated", "sales")
        ipd = d["extra"]["incremental_per_dollar"]["social"]
        assert {"sales", "spend", "window"} <= set(ipd)
        assert {"injected_lift_recovery", "power", "jackknife", "placebo"} <= set(d["validation"])
