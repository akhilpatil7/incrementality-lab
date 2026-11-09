"""MMM module tests (small sampler settings; ~1-2 min)."""
import json

import numpy as np
import pytest

from incrlab.mmm import analysis as A
from incrlab.mmm.calibration import (CalibrationSchemaError, load_calibration_inputs, load_module_observations,
                                     validate_observation)
from incrlab.mmm.transforms import geometric_adstock, hill, lag_matrix


def test_adstock_matches_simulator():
    from incrlab.simulate import adstock
    x = np.random.default_rng(0).gamma(2, 1, 80)
    for d in (0.1, 0.5, 0.7):
        ref = adstock(x, d)
        trunc = geometric_adstock(x, d, l_max=60)
        assert np.allclose(ref, trunc, atol=1e-6 * x.max())
    assert np.allclose(geometric_adstock(np.ones(200), 0.6)[-1], 1.0)  # normalised steady state
    assert lag_matrix(np.arange(5.0), 3)[4].tolist() == [4, 3, 2]
    assert hill(1.0, 2.0) == pytest.approx(0.5)


def test_schema_validation():
    ok = {"test": "meta", "arm": "holdout", "channels": ["social"], "window": [58, 65], "leakage": 0.25,
          "metric": "sales", "incremental_per_dollar": 0.001, "se": 0.0005}
    assert validate_observation(ok)["window"] == [58, 65]
    for bad in ({**ok, "se": -1.0}, {**ok, "channels": ["radio"]}, {**ok, "metric": "qs"},
                {**ok, "window": [70, 60]}, {k: v for k, v in ok.items() if k != "se"}):
        with pytest.raises(CalibrationSchemaError):
            validate_observation(bad)


def test_module_results_parsed(tmp_path):
    doc = {"extra": {"incremental_per_dollar": {"social": {"spend": 1000.0, "window": [58, 65], "leakage": 0.25,
                                                           "sales": 0.002, "sales_se": 0.001}}}}
    (tmp_path / "meta.json").write_text(json.dumps(doc))
    obs = load_module_observations(tmp_path)
    assert list(obs) == ["meta"]
    o = obs["meta"][0]
    assert o["incremental_per_dollar"] == pytest.approx(0.002) and o["se"] == pytest.approx(0.001)
    assert o["channels"] == ["social"] and o["source"].endswith("meta.json")


def test_loader_falls_back_to_stub_and_says_so(tmp_path):
    obs, src = load_calibration_inputs(results_dir=tmp_path)  # empty dir -> stub for every test
    assert set(src) == {"meta", "search", "dsp_a", "dsp_b"}
    assert all(s.startswith("stub") for s in src.values())
    assert all(o["se_model"] >= o["se"] for o in obs)


def test_hdi():
    x = np.random.default_rng(1).normal(0, 1, 20000)
    lo, hi = A.hdi(x, 0.9)
    assert lo == pytest.approx(-1.645, abs=0.05) and hi == pytest.approx(1.645, abs=0.05)


@pytest.fixture(scope="module")
def small_state():
    from incrlab.mmm.model import FitSettings
    from incrlab.mmm.pipeline import run
    st = FitSettings(draws=100, tune=150, chains=2, target_accept=0.85)
    return run(small=True, settings=st, sensitivity=False, voi_bins=2)


def test_small_end_to_end(small_state, tmp_path):
    S = small_state
    from incrlab.mmm import report
    from incrlab.mmm.pipeline import roi_table, triangulation_table
    assert set(S.fits) == {"uncalibrated", "calibrated"}
    for f in S.fits.values():
        assert np.all(np.isfinite(f.roi)) and f.roi.shape[1] == S.data.C
        assert f.diag["train_mape"] < 0.10
        assert f.diag["holdout_mape"] < 0.15
    t = roi_table(S)
    assert len(t) == 6 and t["true_roi"].notna().all()
    tri = triangulation_table(S)
    assert {"last_touch_roi", "experiment_roi", "uncalibrated_roi", "calibrated_roi", "true_roi"} <= set(tri.columns)
    # calibrated implied test returns sit closer to the observed readouts than the uncalibrated ones (on average,
    # in SE units) - the likelihood term is doing its job
    z = lambda f: np.mean([abs(f.implied_ipd[:, i].mean() - o["incremental_per_dollar"]) / o["se_model"]  # noqa
                           for i, o in enumerate(S.obs)])
    assert z(S.fits["calibrated"]) <= z(S.fits["uncalibrated"]) + 0.05
    p = report.write_results_json(S, results_dir=tmp_path)
    doc = json.loads(p.read_text())
    assert doc["module"] == "mmm" and "roi_mae_calibrated" in doc["headline"]
    assert doc["headline"]["calibration_sources"]
    md = report.readout_md(S)
    assert "Triangulation" in md and "Budget reallocation" in md


def test_optimizer_sanity(small_state):
    S = small_state
    o = S.optimizer["calibrated"]
    bp = A.BudgetProblem(S.fits["calibrated"].draws, S.data, S.value_per_sale)
    m = np.array([o["optimal"]["multipliers"][c] for c in S.data.channels])
    assert np.all(m >= 0.7 - 1e-6) and np.all(m <= 1.3 + 1e-6)
    assert m @ bp.spend0 == pytest.approx(bp.budget0, rel=1e-4)           # budget held fixed
    assert bp.incremental(m).mean() >= bp.incremental(np.ones(S.data.C)).mean() - 1e-6
    # more budget -> more sales, less budget -> fewer
    sc = o["scenarios"]
    assert sc["1.2x"]["optimized"]["incremental_sales"]["mean"] > o["optimal"]["incremental_sales"]["mean"]
    assert sc["0.8x"]["optimized"]["incremental_sales"]["mean"] < o["status_quo"]["incremental_sales"]["mean"]
    assert set(S.voi) == set(S.data.channels) and all(v["evppi_sales"] >= 0 for v in S.voi.values())
