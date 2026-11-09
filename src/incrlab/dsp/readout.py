"""Post-test readout for both DSP vendors: realize the holdouts, then triangulate.

Methods per funnel stage (qs, rated, sales); every interval is 90% and *significance is
decided only by whether the interval excludes 0* (p-values are reported alongside):

* change-ratio DiD (headline) with week-bootstrap CI;
* Welch t-test on weekly normalised gaps (pre vs post) with a Welch-Satterthwaite CI;
* two-way fixed-effects regression (zip + week FE, cluster-robust SE by zip, population
  weights; for Vendor B donors carry their locked SC weights);
* CausalImpact-style structural time-series counterfactual (statsmodels UC, long pre);
* synthetic control with LOCKED design weights + placebo inference (in-space clusters / county
  donors, random-geography placebos for many permutations, in-time placebos); a placebo-based
  interval (observed - placebo quantiles) gives the interval-based significance flag;
* Vendor B: ridge-augmented SC (AugSynth-style) robustness row.

Business translation: incremental cost per QS / rated / sale vs vendor-reported and site
last-touch cost-pers, window ROI with a bootstrap P(ROI > 1), spillover check on buffer zips
(and the bias a naive adjacent control would have had), go/no-go per vendor.

Sign: estimates are channel *contribution* (positive = volume lost when the channel is paused).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .. import geo as G
from ..common import bsts as cbsts
from ..common import did as cdid
from ..common import inference as cinf
from ..common import power as cpow
from ..common import synth as csyn
from ..simulate import run_scenario
from . import features as F
from .design_b import augmented_sc, cluster_wide, random_geography_placebos
from .params import DSPParams

STAGES = ("qs", "rated", "sales")


# ======================================================================================
# Estimators (self-contained; each takes plain series)
# ======================================================================================
def change_ratio_boot(t: pd.Series, c: pd.Series, pre, post, sign: int, n_boot: int, seed: int,
                      alpha: float) -> dict:
    """Change-ratio DiD + week bootstrap; returns the point estimate, CI and the draws."""
    base = cdid.change_ratio_did(t, c, pre, post, sign=sign)
    tp, tq = t.loc[list(pre)].to_numpy(float), t.loc[list(post)].to_numpy(float)
    cp, cq = c.loc[list(pre)].to_numpy(float), c.loc[list(post)].to_numpy(float)
    rng = np.random.default_rng(seed)
    ip = rng.integers(0, len(tp), (n_boot, len(tp)))
    iq = rng.integers(0, len(tq), (n_boot, len(tq)))
    cf = tp[ip].mean(1) * cq[iq].mean(1) / cp[ip].mean(1)
    lift = sign * (tq[iq].mean(1) - cf) / cf
    inc = sign * (tq[iq].mean(1) - cf) * len(tq)
    lo, hi = np.quantile(lift, [alpha / 2, 1 - alpha / 2])
    ilo, ihi = np.quantile(inc, [alpha / 2, 1 - alpha / 2])
    t_pre, t_post = tp.mean(), tq.mean()
    c_pre, c_post = cp.mean(), cq.mean()
    return {**base, "ci_low": float(lo), "ci_high": float(hi), "se": float(lift.std(ddof=1)),
            "incremental": base["abs_effect"], "inc_ci_low": float(ilo), "inc_ci_high": float(ihi),
            "inc_se": float(inc.std(ddof=1)), "draws_lift": lift, "draws_inc": inc,
            "t_pre": t_pre, "t_post": t_post, "c_pre": c_pre, "c_post": c_post, "t_ratio": t_post / t_pre,
            "c_ratio": c_post / c_pre, "cf_per_week": t_pre * c_post / c_pre,
            "inc_per_week": sign * (t_post - t_pre * c_post / c_pre),
            "p_value": float(2 * min((lift <= 0).mean(), (lift >= 0).mean())),
            "significant": bool(lo > 0 or hi < 0)}


def welch_gap(t: pd.Series, c: pd.Series, pre, post, sign: int, alpha: float) -> dict:
    """Welch t-test of weekly gaps of pre-mean-normalised series; CI via Welch-Satterthwaite df.

    gap_w = T_w / mean(T_pre) - C_w / mean(C_pre). DiD = mean(post gap) - mean(pre gap), expressed
    relative to the normalised control post level (~ relative lift)."""
    tn, cn = t / t.loc[list(pre)].mean(), c / c.loc[list(pre)].mean()
    g = tn - cn
    a, b = g.loc[list(pre)].to_numpy(float), g.loc[list(post)].to_numpy(float)
    tt, p = stats.ttest_ind(b, a, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = b.mean() - a.mean()
    scale = float(cn.loc[list(post)].mean())
    q = stats.t.ppf(1 - alpha / 2, df)
    est = sign * diff / scale
    lo, hi = sorted([sign * (diff - q * se) / scale, sign * (diff + q * se) / scale])
    return {"lift": float(est), "ci_low": float(lo), "ci_high": float(hi), "t": float(tt), "p_value": float(p),
            "df": float(df), "se": float(se / scale), "significant": bool(lo > 0 or hi < 0),
            "pre_gap_mean": float(a.mean()), "post_gap_mean": float(b.mean()), "gap": g}


def fe_regression(cube: F.ZipCube, t_ids, c_ids, col: str, pre, post, sign: int, alpha: float,
                  c_weights: Optional[Dict[str, float]] = None) -> dict:
    """y_zt = per-capita outcome (x1000); zip + week FE; treated x post; WLS (pop weights,
    times SC weights for donors); cluster-robust SE by zip. Relative lift = beta / weighted
    counterfactual level of treated-post cells."""
    weeks = list(pre) + list(post)
    rows = []
    for grp, ids in (("t", list(t_ids)), ("c", list(c_ids))):
        ii = cube.idx(ids)
        wk_pos = np.searchsorted(cube.weeks, weeks)
        Y = cube.arr[col][ii][:, wk_pos] / cube.pop[ii][:, None] * 1000
        for r, z in enumerate(cube.zips[ii]):
            w = cube.pop[ii][r] * (c_weights.get(z, 0.0) if (grp == "c" and c_weights is not None) else 1.0)
            for k, wkk in enumerate(weeks):
                rows.append((z, wkk, Y[r, k], int(grp == "t"), int(wkk in set(post)), w))
    d = pd.DataFrame(rows, columns=["zip", "week", "y", "treated", "post", "w"])
    d = d[d["w"] > 0]
    r = cdid.did_ols_fe(d, "y", "zip", "week", "treated", "post", cluster="unit", weights="w", alpha=alpha)
    tp = d[(d["treated"] == 1) & (d["post"] == 1)]
    base = float(np.average(tp["y"], weights=tp["w"]) - r["beta"])
    est = sign * r["beta"] / base
    lo, hi = sorted([sign * r["ci_low"] / base, sign * r["ci_high"] / base])
    return {"lift": float(est), "ci_low": float(lo), "ci_high": float(hi), "p_value": r["p_value"],
            "se": r["se"] / base, "se_type": r["se_type"], "n_obs": r["n_obs"], "n_units": r["n_units"],
            "significant": bool(lo > 0 or hi < 0)}


def bsts_readout(t: pd.Series, c: pd.Series, pre, post, sign: int, alpha: float, n_sims: int, seed: int) -> dict:
    X = pd.DataFrame({"control": c})
    r = cbsts.causal_impact(t, X, pre, post, alpha=alpha, n_sims=n_sims, seed=seed, sign=sign)
    return {"lift": r["rel_effect"], "ci_low": r["rel_ci_low"], "ci_high": r["rel_ci_high"],
            "p_value": r["p_like"], "significant": r["significant"], "abs_effect": r["abs_effect"],
            "pointwise": r["pointwise"], "model": r["model"], "pre_mape": r["pre_mape"]}


def sc_readout(wide: pd.DataFrame, treated: str, weights: pd.Series, fit_weeks, post, sign: int, alpha: float,
               placebo_lifts: Sequence[float]) -> dict:
    """Locked-weight SC lift; placebo distribution of null lifts (same windows) gives the p-value
    and a placebo-quantile interval: [obs - q_hi, obs - q_lo] in contribution units."""
    cf = csyn.apply_weights(wide, weights)
    raw = csyn.lift_from_series(wide[treated], cf, post)["lift"]
    pre_bias = csyn.lift_from_series(wide[treated], cf, list(fit_weeks)[-10:])["lift"]
    est = sign * (raw - pre_bias)  # remove the (locked-weight) late-pre level bias, as in the design check
    null = sign * (np.asarray(placebo_lifts, float))
    null = null[np.isfinite(null)]
    p = cinf.permutation_pvalue(est, null - np.mean(null)) if len(null) else np.nan
    qlo, qhi = (np.quantile(null - np.mean(null), [alpha / 2, 1 - alpha / 2]) if len(null) else (np.nan, np.nan))
    lo, hi = est - qhi, est - qlo
    return {"lift": float(est), "raw_lift": float(sign * raw), "pre_bias": float(pre_bias), "ci_low": float(lo),
            "ci_high": float(hi), "p_value": float(p), "n_placebos": int(len(null)),
            "significant": bool(lo > 0 or hi < 0), "synthetic": cf, "actual": wide[treated]}


# ======================================================================================
# Arm construction
# ======================================================================================
@dataclass
class ArmSeries:
    vendor: str
    test: str
    treated: List[str]
    control_label: str
    t: Dict[str, pd.Series]
    c: Dict[str, pd.Series]
    spend_t: pd.Series
    spend_c: pd.Series
    vendor_qs_c: pd.Series
    lt_c: Dict[str, pd.Series]
    pre: List[int]
    post: List[int]
    hist: List[int]


@dataclass
class Readout:
    params: DSPParams
    truth: dict
    series: Dict[str, ArmSeries]
    results: Dict[str, Dict[str, Dict[str, dict]]]  # vendor -> stage -> method -> result
    sc: Dict[str, dict]
    business: Dict[str, dict]
    spillover: Dict[str, dict]
    decisions: Dict[str, dict]
    estimates: List[dict]
    power: Dict[str, dict]
    extra: dict = field(default_factory=dict)


def _windows(cfg, test, hist_start):
    td = cfg.tests[test]
    pre = list(range(td.readout_pre_start, td.test_start))
    post = list(range(td.test_start, td.test_end + 1))
    hist = list(range(hist_start, td.test_start))
    return pre, post, hist


def run_readout(world, design_a, design_b, P: DSPParams, cfg=None) -> Readout:
    cfg = cfg or world.cfg
    R = P.readout
    res = run_scenario({"dsp_a": {"holdout": design_a.treated}, "dsp_b": {"holdout": design_b.treated}}, world=world)
    zw = res.panels["zip_weekly"]
    truth = res.truth
    # compliance: the scenario must label exactly our selected zips
    for test, ids in (("dsp_a", design_a.treated), ("dsp_b", design_b.treated)):
        lab = set(zw.loc[zw[f"arm_{test}"] == "holdout", "zip"].unique())
        assert lab == set(ids), f"{test}: scenario arm labels differ from the selected treatment"
    hist_start = R.bsts_history_start
    out_series, results, scs, biz, spill, power, decisions = {}, {}, {}, {}, {}, {}, {}

    for vendor, test, D in (("A", "dsp_a", design_a), ("B", "dsp_b", design_b)):
        region = D.params.region
        sub = zw[(zw["region"] == region) & (zw["week_idx"] >= min(hist_start, cfg.tests[test].design_start))
                 & (zw["week_idx"] <= cfg.tests[test].test_end + 2)]
        cube = F.ZipCube(sub)
        pre, post, hist = _windows(cfg, test, hist_start)
        td = cfg.tests[test]
        design_weeks = list(range(td.design_start, td.design_end + 1))
        if vendor == "A":
            c_ids = D.control
            t = {s: cube.sum(D.treated, s) for s in STAGES}
            c = {s: cube.sum(c_ids, s) for s in STAGES}
            spend_c = cube.sum(c_ids, "spend_dsp_a")
            vend_c = cube.sum(c_ids, "vendor_conv_qs_dsp_a")
            lt_c = {s: zw[zw["zip"].isin(c_ids)].groupby("week_idx")[f"lt_dsp_{s}"].sum() for s in STAGES}
            c_weights = None
            ctrl_label = "Matched twin patch"
            spend_t = cube.sum(D.treated, "spend_dsp_a")
        else:
            fm = {D.treated_cluster: D.treated, **D.donor_members}
            w = D.weights
            pop_t = cube.pop_of(D.treated)
            t, c = {}, {}
            for s in STAGES:
                wd = cluster_wide(cube, fm, s)
                t[s] = cube.sum(D.treated, s)
                c[s] = csyn.apply_weights(wd, w) / 1000 * pop_t  # synthetic control in treated-pop counts
            c_weights = {}
            for cl, wt in w.items():
                ids = fm[int(cl[1:])]
                for z in ids:
                    c_weights[z] = float(wt) * pop_t / cube.pop_of(ids)
            c_ids = list(c_weights)
            wts = np.array([c_weights[z] for z in c_ids])
            spend_c = cube.sum(c_ids, "spend_dsp_b", weights=wts)
            vend_c = cube.sum(c_ids, "vendor_conv_qs_dsp_b", weights=wts)
            lt_all = zw[zw["zip"].isin(c_ids)].assign(_w=lambda d: d["zip"].map(c_weights))
            lt_c = {s: (lt_all[f"lt_dsp_{s}"] * lt_all["_w"]).groupby(lt_all["week_idx"]).sum() for s in STAGES}
            ctrl_label = "Synthetic control (locked weights)"
            spend_t = cube.sum(D.treated, "spend_dsp_b")
        out_series[vendor] = ArmSeries(vendor, test, list(D.treated), ctrl_label, t, c, spend_t, spend_c, vend_c,
                                       lt_c, pre, post, hist)

        # ---- SC + placebo machinery (both vendors) ------------------------------------
        if vendor == "A":
            elig = D.eligible
            pool = elig[~elig["zip"].isin(D.treated) & ~elig["zip"].isin(D.buffer_zips)]
            groups = {i + 1: s["zip"].tolist() for i, (_, s) in enumerate(pool.groupby("county_id")) if len(s) >= 3}
            sc_members = {0: D.treated, **groups}
            feat_pool = elig
        else:
            sc_members = fm
            feat_pool = D.feat
        sc_by_stage = {}
        for s in STAGES:
            wide = cluster_wide(cube, sc_members, s)
            tname = f"C{0 if vendor == 'A' else D.treated_cluster}"
            if vendor == "A":
                fit = csyn.fit_synth(wide, tname, design_weeks)
                weights = fit.weights
            else:
                weights = D.sc.weights
            donors = {k: v for k, v in sc_members.items() if f"C{k}" != tname}
            pg = random_geography_placebos(cube, feat_pool, D.treated, donors, design_weeks, post, None, None,
                                           n=120, seed=cfg.seed + (1 if vendor == "A" else 2))
            # placebo lifts: clusters/counties in space (locked fits on design weeks)
            ps = cinf.placebo_in_space(wide, tname, design_weeks, post, stat="lift")
            null = np.concatenate([pg["lift"].to_numpy() if len(pg) else np.array([]),
                                   ps["placebos"]["lift"].to_numpy()])
            r = sc_readout(wide, tname, weights, design_weeks, post, R.sign, R.alpha, null)
            r.update({"n_space_placebos": int(ps["n_placebos"]), "n_geo_placebos": int(len(pg)),
                      "p_in_space_cluster": ps["p_value"], "min_p_in_space": ps["min_attainable_p"],
                      "weights": weights[weights > 0].round(4).to_dict(), "null": null, "wide": wide,
                      "tname": tname})
            # in-time placebo on the pre period only
            pt = cinf.placebo_in_time(wide.loc[[w_ for w_ in wide.index if w_ < td.test_start]], tname,
                                      [w_ for w_ in wide.index if w_ < td.test_start], n_post=len(post), n_fit_min=8,
                                      actual_lift=-r["lift"])
            r["placebo_in_time"] = {"n": pt["n_placebos"], "sd": pt["sd_placebo_lift"], "p_value": pt.get("p_value"),
                                    "lifts": pt["placebos"]["lift"].tolist() if pt["n_placebos"] else []}
            if vendor == "B":
                aug = augmented_sc(wide, tname, design_weeks, post, weights, P.b.aug_lambdas)
                raw = csyn.lift_from_series(wide[tname], aug["counterfactual"], post)["lift"]
                pb = csyn.lift_from_series(wide[tname], aug["counterfactual"], design_weeks[-10:])["lift"]
                est = R.sign * (raw - pb)
                nul = R.sign * (null - np.mean(null))
                qlo, qhi = np.quantile(nul, [R.alpha / 2, 1 - R.alpha / 2])
                r["aug"] = {"lift": float(est), "ci_low": float(est - qhi), "ci_high": float(est - qlo),
                            "p_value": cinf.permutation_pvalue(est, nul), "lambda": aug["lambda"],
                            "significant": bool(est - qhi > 0 or est - qlo < 0), "cf": aug["counterfactual"]}
            sc_by_stage[s] = r
        scs[vendor] = sc_by_stage

        # ---- per-stage triangulation ----------------------------------------------------
        results[vendor] = {}
        for s in STAGES:
            ts, cs_ = t[s], c[s]
            rr = {"change_ratio_did": change_ratio_boot(ts, cs_, pre, post, R.sign, R.n_boot, R.seed, R.alpha),
                  "welch_gap": welch_gap(ts, cs_, pre, post, R.sign, R.alpha),
                  "did_fe": fe_regression(cube, D.treated, c_ids, s, pre, post, R.sign, R.alpha, c_weights),
                  "bsts": bsts_readout(ts, cs_, hist, post, R.sign, R.alpha, R.bsts_sims, R.seed),
                  "synth_control": sc_by_stage[s]}
            if vendor == "B":
                rr["augsynth"] = sc_by_stage[s]["aug"]
            results[vendor][s] = rr

        # ---- readout power (pre-period noise; unified convention) -----------------------
        pw = {}
        for s in STAGES:
            gs = cpow.gap_sigma(t[s].loc[pre] / t[s].loc[pre].mean(), c[s].loc[pre] / c[s].loc[pre].mean())
            rho = float(np.clip(gs["rho"], 0, 0.8))
            pw[s] = {"sigma_gap": gs["sigma"], "rho": rho,
                     "mde": cpow.did_mde(gs["sigma"], len(post), len(pre), R.alpha, 0.8, rho=rho),
                     "weeks_90": cpow.weeks_for_power(0.10, gs["sigma"], n_pre=len(pre), alpha=R.alpha, rho=rho, max_weeks=200),
                     "weeks_95": cpow.weeks_for_power(0.10, gs["sigma"], n_pre=len(pre), alpha=0.05, rho=rho, max_weeks=200)}
        power[vendor] = pw

        # ---- business translation -----------------------------------------------------------
        sp_pre_t = spend_t.loc[pre].mean()
        sp_ratio = spend_c.loc[post].mean() / spend_c.loc[pre].mean()
        bau_spend_est = sp_pre_t * sp_ratio * len(post)
        spend_saved = bau_spend_est - spend_t.loc[post].sum()
        cr = {s: results[vendor][s]["change_ratio_did"] for s in STAGES}
        vpq = cfg.world.value_per_sale
        roi_draws = cr["sales"]["draws_inc"] * vpq / spend_saved
        tr = truth[test]["holdout"]
        b = {"spend_saved_est": float(spend_saved), "spend_saved_true": tr["spend_removed"],
             "bau_spend_est": float(bau_spend_est), "actual_spend_in_window": float(spend_t.loc[post].sum()),
             "compliance_pct_of_bau": float(spend_t.loc[post].sum() / bau_spend_est),
             "value_per_sale": vpq}
        for s in STAGES:
            inc = cr[s]["incremental"]
            b[f"incremental_{s}"] = inc
            b[f"incremental_{s}_ci"] = [cr[s]["inc_ci_low"], cr[s]["inc_ci_high"]]
            b[f"incremental_{s}_true"] = tr[s]["true_incremental"]
            b[f"cost_per_incremental_{s}"] = spend_saved / inc if inc > 0 else None
            b[f"cost_per_incremental_{s}_true"] = tr.get(f"true_incremental_cost_per_{s}")
            b[f"incremental_per_dollar_{s}"] = inc / spend_saved
            b[f"incremental_per_dollar_{s}_se"] = cr[s]["inc_se"] / spend_saved
            lt_cost = spend_c.loc[post].sum() / max(lt_c[s].loc[post].sum(), 1e-9)
            b[f"last_touch_cost_per_{s}"] = float(lt_cost)
        b["vendor_reported_cost_per_qs"] = float(spend_c.loc[post].sum() / max(vend_c.loc[post].sum(), 1e-9))
        b["vendor_overclaim_x"] = (b["cost_per_incremental_qs"] or np.nan) / b["vendor_reported_cost_per_qs"]
        b["roi_window_est"] = float(cr["sales"]["incremental"] * vpq / spend_saved)
        b["roi_window_ci"] = [float(np.quantile(roi_draws, R.alpha / 2)), float(np.quantile(roi_draws, 1 - R.alpha / 2))]
        b["p_roi_gt_1"] = float((roi_draws > 1).mean())
        b["roi_window_true"] = tr["sales"]["true_incremental"] * vpq / tr["spend_removed"]
        b["roi_draws"] = roi_draws
        biz[vendor] = b

        # ---- spillover check -------------------------------------------------------------
        buf = [z for z in D.buffer_zips if z in cube.pos]
        served_col = "dsp_a_served" if vendor == "A" else "dsp_b_inclusion"
        zs = world.geo.zips.set_index("zip")
        buf_served = [z for z in buf if zs.loc[z, served_col] == 1 and zs.loc[z, "concurrent_test_flag"] == 0]
        sp = {"radius_mi": D.params.buffer_miles, "n_buffer_zips": len(buf), "n_buffer_served": len(buf_served),
              "truth": tr.get("spillover", {})}
        for s in STAGES:
            bs = cube.sum(buf_served, s)
            est = change_ratio_boot(bs, c[s], pre, post, R.sign, R.n_boot, R.seed, R.alpha)
            naive = change_ratio_boot(t[s], bs, pre, post, R.sign, R.n_boot, R.seed, R.alpha)
            # truth for served buffer zips (expected loss) is in the scenario truth over ALL buffer zips
            sp[s] = {"buffer_loss_est": est["lift"], "ci": [est["ci_low"], est["ci_high"]],
                     "significant": est["significant"], "naive_adjacent_control_lift": naive["lift"],
                     "naive_ci": [naive["ci_low"], naive["ci_high"]],
                     "buffer_series": bs}
        spill[vendor] = sp

        # ---- decision (pre-registered; hinges on SALES-stage evidence) -----------------------
        # A GO needs a measured sales lift; top-funnel (QS / rated) lift alone can never produce a GO.
        # A NO-GO needs either P(ROI > 1) <= nogo_prob or a design powered for sales (design MDE <=
        # target) whose sales interval rules out the break-even lift.
        p_go = b["p_roi_gt_1"]
        sig_top = results[vendor]["qs"]["change_ratio_did"]["significant"] or results[vendor]["rated"]["change_ratio_did"]["significant"]
        sres = results[vendor]["sales"]["change_ratio_did"]
        sales_sig = bool(sres["significant"] and sres["lift"] > 0)
        n_methods_sales = sum(bool(r_["significant"]) for r_ in results[vendor]["sales"].values())
        cf_sales = sres["cf_per_week"] * len(post)
        breakeven = float(spend_saved / vpq / cf_sales) if cf_sales > 0 else np.nan
        target = D.params.target_mde
        design_mde_sales = (D.sizing_decision.get("mde_selected") if vendor == "A"
                            else D.power["by_metric"]["sales"]["mde"])
        powered = bool(design_mde_sales is not None and design_mde_sales <= target)
        rules_out_breakeven = bool(np.isfinite(breakeven) and sres["ci_high"] < breakeven)
        tag = (f"design powered for sales (MDE {design_mde_sales:.0%} <= {target:.0%})" if powered else
               f"design NOT powered for sales (MDE {design_mde_sales:.0%} > {target:.0%})")
        if sales_sig and powered and p_go >= R.go_prob:
            dec, why = "GO", (f"sales-stage lift significant ({n_methods_sales}/{len(results[vendor]['sales'])} methods) "
                              f"and P(window ROI > 1) = {p_go:.0%} >= {R.go_prob:.0%}; {tag}")
        elif p_go <= R.nogo_prob or (powered and rules_out_breakeven):
            dec, why = "NO-GO", (f"no sales-stage lift: 90% CI upper bound {sres['ci_high']:+.1%} vs break-even lift "
                                 f"{breakeven:+.1%}, P(window ROI > 1) = {p_go:.0%}; {tag}"
                                 + ("; top-funnel lift is real but does not reach sales" if sig_top else ""))
        elif not powered:
            # pre-registered: a sales read from a design that cannot detect the target lift is not decision-grade.
            # A 'significant' result there is exaggerated by construction (type-M error), so it cannot fund a GO.
            dec, why = "HOLD - RE-TEST POWERED ON SALES", (
                f"{tag}; sales read {sres['lift']:+.1%} [{sres['ci_low']:+.1%}, {sres['ci_high']:+.1%}] is not "
                f"decision-grade (a significant read from an under-powered design overstates the effect); "
                + ("top-funnel lift is significant but QS / rated lift alone is not a GO; " if sig_top else "")
                + "keep spend flat and re-test with a design sized for sales")
        elif sales_sig:
            dec, why = "LEAN GO - CONFIRM", (f"sales lift is significant but P(window ROI > 1) = {p_go:.0%} < "
                                             f"{R.go_prob:.0%}; confirm cost-efficiency at scaled spend; {tag}")
        else:
            dec, why = ("INCONCLUSIVE - EXTEND", f"sales interval includes 0 and does not rule out break-even "
                        f"({breakeven:+.1%}); P(window ROI > 1) = {p_go:.0%}; {tag}")
        decisions[vendor] = {"decision": dec, "rationale": why, "p_roi_gt_1": p_go, "roi_est": b["roi_window_est"],
                             "sales_significant": sales_sig, "breakeven_sales_lift": breakeven,
                             "design_mde_sales": design_mde_sales, "target_mde_sales": target,
                             "powered_for_sales": powered,
                             "roi_true": b["roi_window_true"], "top_funnel_significant": bool(sig_top)}

    estimates = build_estimates(results, truth, power)
    rd = Readout(params=P, truth=truth, series=out_series, results=results, sc=scs, business=biz, spillover=spill,
                 decisions=decisions, estimates=estimates, power=power)
    rd.extra["scenario"] = res
    return rd


METHOD_LABELS = {"change_ratio_did": "Change-ratio DiD", "welch_gap": "Welch t (weekly gaps)",
                 "did_fe": "Two-way FE regression", "bsts": "Structural TS (CausalImpact-style)",
                 "synth_control": "Synthetic control + placebos", "augsynth": "Ridge-augmented SC"}


def build_estimates(results, truth, power) -> List[dict]:
    rows = []
    for vendor, test in (("A", "dsp_a"), ("B", "dsp_b")):
        tr = truth[test]["holdout"]
        for s, methods in results[vendor].items():
            for m, r in methods.items():
                rows.append({"test": test, "arm": "holdout", "metric": s, "method": m, "estimate": r["lift"],
                             "ci_low": r["ci_low"], "ci_high": r["ci_high"], "ci_level": 0.9,
                             "p_value": r.get("p_value"), "significant": bool(r["significant"]),
                             "truth": tr[s]["true_lift_pct"], "truth_realized": tr[s]["realized_lift_pct"],
                             "truth_key": f"scenario.{test}.holdout.{s}.true_lift_pct",
                             "units": "relative lift (positive = channel contribution)",
                             "notes": (f"MDE(80% power, one-sided a=0.10) = {power[vendor][s]['mde']:.1%}; "
                                       "significance = 90% interval excludes 0")})
    return rows


def truth_table(rd: Readout) -> pd.DataFrame:
    df = pd.DataFrame(rd.estimates)
    df["vendor"] = df["test"].map({"dsp_a": "A", "dsp_b": "B"})
    df["method_label"] = df["method"].map(METHOD_LABELS)
    df["error_pp"] = (df["estimate"] - df["truth"]) * 100
    df["covers_truth"] = (df["ci_low"] <= df["truth"]) & (df["truth"] <= df["ci_high"])
    return df
