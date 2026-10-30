import numpy as np
import pandas as pd

from incrlab.common import bsts, did, inference, power, synth
from tests.conftest import make_toy_panel

PRE = list(range(0, 48))
POST = list(range(48, 60))


def _inject(wide, unit, lift):
    w = wide.copy()
    w.loc[POST, unit] *= (1 + lift)
    return w


def test_synth_recovers_injected_lift():
    wide = make_toy_panel(seed=1)
    true = -0.10  # holdout: volume lost
    w = _inject(wide, "u03", true)
    res = synth.fit_synth(w, "u03", PRE)
    assert abs(res.weights.sum() - 1) < 1e-8 and (res.weights >= 0).all()
    est = res.lift(POST)["lift"]
    assert abs(est - true) < 0.03, est
    assert abs(res.lift(POST, sign=-1)["lift"] + est) < 1e-12
    v = synth.injected_lift_recovery(wide, "u05", PRE, POST, 0.10)
    assert v["abs_error"] < 0.02, v


def test_synth_with_predictors_and_ridge():
    wide = make_toy_panel(seed=2)
    preds = pd.DataFrame({"mean_pre": wide.loc[PRE].mean(), "sd_pre": wide.loc[PRE].std()})
    w = _inject(wide, "u01", -0.12)
    r1 = synth.fit_synth(w, "u01", PRE, predictors=preds)
    r2 = synth.fit_synth(w, "u01", PRE, ridge=0.5)
    assert abs(r1.lift(POST)["lift"] + 0.12) < 0.04
    assert r2.effective_n >= synth.fit_synth(w, "u01", PRE).effective_n - 1e-6


def test_pysyncon_wrapper_runs():
    if not synth.pysyncon_available():
        return
    wide = make_toy_panel(seed=3, n_units=8)
    wide["tr"] = 0.6 * wide["u00"] + 0.4 * wide["u03"]  # treated inside the donor convex hull
    w = _inject(wide, "tr", -0.10)
    r = synth.fit_synth_pysyncon(w, "tr", PRE)
    own = synth.fit_synth(w, "tr", PRE)
    assert abs(r.weights.sum() - 1) < 1e-6
    assert abs(r.lift(POST)["lift"] + 0.10) < 0.03
    assert abs(r.lift(POST)["lift"] - own.lift(POST)["lift"]) < 0.02


def test_did_estimators_recover_lift():
    wide = make_toy_panel(seed=4, noise=0.02)
    true = -0.08
    w = _inject(wide, "u00", true)
    ctrl = w.drop(columns="u00").sum(1)
    cr = did.change_ratio_did(w["u00"], ctrl, PRE, POST, n_boot=200)
    assert abs(cr["lift"] - true) < 0.03 and cr["ci_low"] < cr["lift"] < cr["ci_high"]
    d2 = did.did_2x2_ratio(w["u00"], ctrl, PRE, POST)
    assert abs(d2["lift_ratio"] - true) < 0.03
    long = w.rename_axis("week_idx").reset_index().melt(id_vars="week_idx", var_name="unit", value_name="y")
    long["treated"] = (long["unit"] == "u00").astype(int)
    long["post"] = long["week_idx"].isin(POST).astype(int)
    fe = did.did_ols_fe(long, "y", "unit", "week_idx", "treated", "post", log=True)
    assert abs(fe["rel_lift"] - true) < 0.03 and fe["se_type"].startswith("cluster")
    assert fe["rel_ci_low"] < true < fe["rel_ci_high"]
    wt = did.welch_gap_test(w["u00"], ctrl * (w["u00"].loc[PRE].mean() / ctrl.loc[PRE].mean()), PRE, POST)
    assert wt["diff"] < 0 and wt["p_value"] < 0.05


def test_bsts_recovers_lift_and_uses_interval():
    rng = np.random.default_rng(5)
    wide = make_toy_panel(seed=5, noise=0.02)
    X = wide.iloc[:, :4]
    y = (X @ np.array([0.4, 0.3, 0.2, 0.1])) * np.exp(rng.normal(0, 0.02, len(X)))
    true = -0.10
    y_t = y.copy()
    y_t.loc[POST] *= (1 + true)
    ci = bsts.causal_impact(y_t, X, PRE, POST, n_sims=400, seed=1)
    assert abs(ci["rel_effect"] - true) < 0.03, ci["rel_effect"]
    assert ci["significant"] == (ci["rel_ci_high"] < 0 or ci["rel_ci_low"] > 0)
    assert ci["significant"] and ci["rel_ci_low"] < ci["rel_effect"] < ci["rel_ci_high"]
    null = bsts.causal_impact(y, X, PRE, POST, n_sims=400, seed=1)
    assert null["rel_ci_low"] < 0.0 < null["rel_ci_high"], null
    assert not null["significant"]


def test_power_functions():
    mde = power.did_mde(0.10, n_post=8, n_pre=8, alpha=0.10, power=0.8, two_sided=False)
    # closed form: (1.2816+0.8416)*0.1*sqrt(2/8)
    assert np.isclose(mde, (1.2816 + 0.8416) * 0.1 * 0.5, atol=1e-3)
    assert np.isclose(power.did_power(mde, 0.10, 8, 8, 0.10), 0.8, atol=1e-3)
    n = power.weeks_for_power(0.05, 0.10, alpha=0.10)
    assert n is not None and power.did_power(0.05, 0.10, n, n, 0.10) >= 0.8
    n_tt = power.ttest_power(effect_size=0.5, power=0.8, alpha=0.05)
    assert 60 < n_tt < 66
    curve = power.power_curve(0.1, [0.05, 0.1], [4, 8, 12])
    assert curve.groupby("lift")["power"].apply(lambda s: s.is_monotonic_increasing).all()
    wide = make_toy_panel(seed=6, n_weeks=120, noise=0.03)
    t, c = wide["u00"], wide.drop(columns="u00").sum(1)

    def est(tt, cc, pre, post):
        return did.change_ratio_did(tt, cc, pre, post)["lift"]
    sp = power.simulated_power(t, c, 0.15, 8, 12, est, lambda s: s < -0.05, n_sims=60)
    assert sp["power"] > 0.8 and sp["empirical_size"] < 0.3


def test_placebo_and_jackknife():
    wide = make_toy_panel(seed=7, n_units=10, noise=0.02)
    w = _inject(wide, "u00", -0.15)
    ps = inference.placebo_in_space(w, "u00", PRE, POST)
    assert ps["p_value"] <= 0.2 and np.isclose(ps["min_attainable_p"], 1 / 10)
    pt = inference.placebo_in_time(w, "u00", PRE, n_post=6, n_fit_min=30, actual_lift=-0.15)
    assert pt["n_placebos"] > 5 and pt["p_value"] < 0.2
    jk = inference.jackknife_donors(w, "u00", PRE, POST)
    assert len(jk["jackknife"]) >= 1 and abs(jk["full_lift"] + 0.15) < 0.04
    lo, hi = jk["lift_range"]
    assert lo <= jk["full_lift"] + 1e-9 or hi >= jk["full_lift"] - 1e-9
    assert inference.permutation_pvalue(3.0, [0.1, -0.2, 0.5]) == 0.25


def test_results_writer(tmp_path):
    from incrlab.common.results import read_results, write_results
    p = write_results("toy", {"lift_qs": 0.05}, [{"test": "t", "arm": "a", "metric": "qs", "method": "did",
                                                  "estimate": np.float64(0.05), "ci_low": np.nan}],
                      results_dir=tmp_path)
    doc = read_results("toy", results_dir=tmp_path)
    assert doc["estimates"][0]["ci_low"] is None and doc["headline"]["lift_qs"] == 0.05


def test_plotting_save(tmp_path):
    import matplotlib.pyplot as plt
    from incrlab.common import plotting
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 2, 3])
    plotting.shade_window(ax, 1, 2)
    p = plotting.save_fig(fig, "toy", "line", fig_dir=tmp_path)
    assert p.exists() and p.parent.name == "toy"
