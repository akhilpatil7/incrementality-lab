# incrlab v2 — Module Builder API Contract

This is the contract between the Phase-A foundation (`config`, `geo`, `simulate`, `common/`) and the
module builders (`meta/`, `search/`, `dsp/`, `mmm/`). Build against these names; if you need a change,
add to the foundation rather than forking it.

- Python ≥ 3.11 (venv: `~/.hermes/venvs/incrlab`, Python 3.12). Install: `make venv`.
- `make data` → `python -m incrlab.simulate` writes `data/*.parquet` + `data/ground_truth.json` (~5 s).
- `INCRLAB_SMALL=1` → small geography (12 + 8 counties, 300 zips) for CI; same timeline & test windows.
- `INCRLAB_SEED`, `INCRLAB_EFFECT_MULT` (scales every channel's true max effect; use it for a "clearly
  significant" demo), `INCRLAB_DATA_DIR` env overrides.
- Each module must expose `python -m incrlab.<module>` (a `__main__.py`) — the Makefile targets
  `meta | search | dsp | mmm` call exactly that — and write `reports/results/<module>.json`.

## 1. Conventions

| Item | Convention |
|---|---|
| Time | `week_idx` int16, 0..155, 0-based weeks from `START_DATE` = 2023-01-02 (Monday). `week_start` = Monday timestamp. `month_start` = first of month. |
| Windows | Inclusive `(start, end)` week indices. |
| Stages | `qs` (quote starts) → `rated` (rated / completed quotes) → `sales` (bound policies). `visits` = site visits. |
| Lift sign | Positive = channel **contribution** = volume lost when the channel is paused. Estimators return `(actual − cf)/cf` by default; pass `sign=-1` to get the contribution for holdouts. Label charts accordingly. |
| Units | `zip` ids like `NV0123`; `county_id` like `EM-C07`; `county` = fictional display name ("Netherby County"). `region` ∈ {`EM` (Eastmoor), `NV` (Northvale)}; `ROW` = rest-of-nation aggregate (national panels only). |
| Rates | Never average averages — recompute rates from summed numerators/denominators (`geo.aggregate_static`, `geo.aggregate_zips`). |
| Monthly media | `*_month` columns on `zip_weekly` are monthly totals repeated on each week row. Dedupe with `geo.monthly_media_sum` before summing. `geo.aggregate_zips` skips them automatically. |

## 2. Regions, channels, tests (from `incrlab.config`)

`cfg = incrlab.config.get_config()` (frozen dataclass). Key fields: `cfg.seed`, `cfg.small`, `cfg.n_weeks`,
`cfg.regions`, `cfg.channels`, `cfg.truth[ch]` (`ChannelTruth`), `cfg.tests[name]` (`TestDesign`), `cfg.world`.

| Region | Counties (full / small) | Zips | Pop | Hosts |
|---|---|---|---|---|
| `EM` Eastmoor | 25 / 12 | 590 / 160 | 9.3M | Social holdout (county units); DSP Vendor B test (zip clusters; `dsp_b_inclusion` = vendor inclusion list ≈65% of zips); concurrent unrelated test zips (`concurrent_test_flag`, must be excluded from DSP-B design) |
| `NV` Northvale | 14 / 8 | 500 / 140 | 7.0M | Paid Search two-arm dark-out (county units); DSP Vendor A test (contiguous zip patches; `dsp_a_served` ≈70% of zips) |

Channels: `tv` (national, no geo variation), `social`, `search_brand`, `search_nonbrand`, `dsp_a`, `dsp_b`.
Organic / direct / unattributed / other are non-paid last-touch sources only.

| Test (`cfg.tests[...]`) | Region / unit | Arms → channels paused | Design hist. | Readout pre | Test window | Notes |
|---|---|---|---|---|---|---|
| `meta` | EM / county (or county bundle) | `holdout` → social | 0–57 (train 0–50, pseudo-post 51–57) | 51–57 | 58–65 (8 wk) | Incomplete holdout: `leakage=0.25` (25% of spend keeps serving) |
| `search` | NV / county | `all_search` → brand+nonbrand; `branded_only` → brand | 60–78 | 69–78 | 79–84 (6 wk) | Also choose a control county + BAU donors |
| `dsp_a` | NV / zip patch | `holdout` → dsp_a | 95–111 | 102–111 | 112–117 (6 wk) | `leakage=0.03`; spillover on (30 mi radius, 10 mi decay, 0.30 strength) |
| `dsp_b` | EM / zip cluster | `holdout` → dsp_b | 90–111 | 101–111 | 112–119 (8 wk) | Vendor switch at week 96 (`DSP_B_VENDOR_SWITCH_WEEK`): vendor-reported conversions ×0.8, *true* effect unchanged; reporting-dip weeks 100–101 |

Other constants: `BUFFER_MILES_DEFAULT=30`, `ADJACENCY_MILES_DEFAULT=10`, `DSP_REPORT_LAG=(0.5,0.2,0.05)`
(use `simulate.apply_report_lag(panel, as_of_week)` to mimic vendor attribution lag in design-time data).

## 3. Data files (`data/`)

| File | Grain | Rows (full) |
|---|---|---|
| `zip_static.parquet` | zip | 1,090 |
| `county_static.parquet` | county | 39 |
| `zip_weekly.parquet` | zip × week | 170,040 |
| `county_weekly.parquet` | county × week | 6,084 |
| `region_weekly.parquet` | region (EM, NV, ROW) × week | 468 |
| `national_weekly.parquet` | week (all regions + ROW) | 156 |
| `ground_truth.json` | — | — |

Load with `simulate.load_table(name)` / `simulate.load_ground_truth()`. **These are BAU panels (no test
interventions).** Test-period outcomes for your selected units come from the scenario runner (§4).

### zip_static
`zip:str, region:str, region_name:str, lat, lon:float64, pop:int64, households:int64, median_hh_income,
per_capita_income, pct_bachelors (0-1), employment_rate (0-1), median_age, homeowner_rate (0-1),
urbanicity (0-1), ses_index, lfpr (0-1), land_area_sqmi, density:float64, county:str, county_id:str,
zip_propensity (latent, pop-weighted mean 1), dsp_affinity:float64, dsp_a_served, dsp_b_inclusion,
concurrent_test_flag:int64 (0/1)`. QS per capita is negatively correlated with income/education (≈−0.55).

### county_static
`county_id, county, region, region_name:str, pop, households:int64, land_area_sqmi, median_hh_income,
per_capita_income, pct_bachelors, employment_rate, lfpr, median_age, homeowner_rate, zip_propensity,
lat, lon (pop-weighted centroid):float64, n_zips:int64, density:float64`.

### Weekly panels (common columns, all int64 unless noted)
- keys: `week_idx:int16, week_start:datetime64` (+ `zip`/`county_id`/`county`/`region` as applicable)
- funnel: `visits, qs, rated, sales`
- last-touch partition (sums exactly to the stage total in every row), for `s ∈ {qs, rated, sales}`:
  `lt_search_brand_s, lt_search_nonbrand_s, lt_organic_s, lt_social_s, lt_dsp_s (both vendors pooled),
  lt_direct_s, lt_unattributed_s, lt_other_s`
- media per channel `ch`: `spend_ch:float64, impressions_ch, clicks_ch` (search: clicks = spend/CPC with CPC
  drifting with demand; impressions = clicks/CTR)
- platform/vendor reporting: `platform_conv_search_brand, platform_conv_search_nonbrand` (Google-style
  "conversions", ~1.25× last-touch rated), `vendor_conv_qs_dsp_a, vendor_conv_qs_dsp_b` (vendor-claimed
  quote starts; heavily over-claimed vs truth)
- rates (float64, recomputed from sums): `rated_rate = rated/qs`, `close_rate = sales/rated`,
  `social_share_qs = lt_social_qs/qs` (Social test's "share of QS" design metric); county also has
  `pop:int64`, `qs_per_capita`.

### zip_weekly extras
`month_start:datetime64, year:int8 (1..3), pop:int64, dsp_a_served, dsp_b_inclusion, concurrent_test_flag,
dsp_b_vendor_live:int8, arm_meta, arm_search, arm_dsp_a, arm_dsp_b:str ("" in BAU; arm name in scenario
output), spend_dsp_{a,b}_month, impressions_dsp_{a,b}_month, clicks_dsp_{a,b}_month,
vendor_conv_qs_dsp_{a,b}_month` (monthly totals repeated weekly — dedupe!).
DSP share metric for Vendor B design: `shareof_dsp_quotes = vendor_conv_qs_dsp_b / qs` (compute on sums).

## 4. Scenario runner (apply YOUR selected geos)

The world is counterfactual-consistent: all noise is pre-drawn per (unit, week, stream) and pushed through
inverse CDFs. Any unit/week not affected by an intervention has **identical** counts in every scenario; a
treated unit's untreated potential outcome equals its BAU value. Pre-period data is never changed.

```python
from incrlab.config import get_config
from incrlab.simulate import get_world, run_scenario, apply_intervention, realize, save_scenario

world = get_world()                       # cached; get_world(get_config(small=True)) for CI
res = run_scenario({
    "meta":   {"holdout": ["EM-C07"]},                         # county ids or zips (counties expand to zips)
    "search": {"all_search": ["NV-C10"], "branded_only": ["NV-C13"]},
    "dsp_a":  {"holdout": treated_zip_list},
    "dsp_b":  {"holdout": cluster_zip_list},
}, world=world, overrides={"dsp_a": {"spillover": {"enabled": False}}})   # optional per-test overrides
panels = res.panels      # same 4 weekly tables as data/, now with interventions + arm_* labels
truth  = res.truth       # per test -> per arm ground truth (below)
save_scenario(res, "dsp_a_readout")  # optional: data/scenarios/<name>/*.parquet + truth.json
```
Run one test per call if you only need yours (`{"dsp_a": {...}}`); tests are in different weeks/regions so
combining them is also fine. Lower-level: `apply_intervention(world_or_scenario, units, (start, end),
channel_or_channels, {"spend_multiplier": 0.0, "leakage": 0.03, "spillover": {...}}, test=..., arm=...)`
then `realize(scenario)`. `spend_multiplier > 1` = heavy-up test. The effect is NOT injected: it follows from
the channel's true adstock+Hill response to the changed spend (carryover continues after the window).

`res.truth[test][arm]` keys: `n_units, units, pop, window, channels, spend_multiplier, leakage,
spend_removed, bau_spend_in_window, true_incremental_cost_per_{qs,rated,sales}`, and for each stage
`s`: `{expected_actual, expected_counterfactual, true_incremental, true_lift_pct, realized_actual,
realized_counterfactual, realized_lift_pct}` (window only; realized = with the actual noise draws, i.e. the
best possible estimate). Search arms add `true_recapture_{qs,rated,sales}`. Tests with spillover add
`spillover: {radius_mi, n_buffer_units, qs/rated/sales: {expected_loss, loss_pct}}` for untreated zips
within the radius. Compare your estimates to `true_lift_pct` (and say which you compare to).

Module-design note: designs must use only data available at design time (`week_idx <= design_end`).

## 5. ground_truth.json (top level)
- `generator` (seed, small, n_weeks, n_zips, n_counties, regions), `conventions`, `totals` (expected stage totals)
- `channels[ch]`: `params` (adstock_decay [normalized geometric], hill_slope, half_sat_ratio,
  half_sat_spend_per_served_capita, half_sat_national_weekly_spend, max_effect, stage_mult, recapture,
  served_pop), `total_spend`, `incremental{qs,rated,sales}`, `roi` (= incr sales × value_per_sale / spend),
  `marginal_roi` (+1% spend), `cost_per_incremental{s}`, `incremental_per_1k_spend{s}`,
  `contribution_share{s}`, `by_year.yearN{spend, incremental, roi, cost_per_incremental_sales}`;
  social/search add `last_touch_claimed{s}`, `last_touch_bias{s}` (claimed/true), `last_touch_cost_per{s}`.
- `search.true_recapture_brand` (0.40), `true_recapture_nonbrand` (0.15)
- `dsp_site_last_touch` (pooled DSP last-touch claimed vs true), `dsp_vendor_reporting`, `dsp_b_vendor_switch`
- `tests[name]`: windows/arms/leakage/spillover + `region_wide_holdout_lift[arm]{qs, rated, sales,
  cost_per_incremental_sales, organic_recapture_qs}` — design-independent reference lift if the channel were
  paused in all served units of the region. Your selected units' truth comes from the scenario runner.
- For MMM: true response curves via `simulate.true_response_curve(world, ch)`; `simulate.adstock`, `simulate.hill`
  are the exact transforms used.

## 6. `incrlab.common` signatures

```python
# synth.py  (wide = time x unit DataFrame; use synth.to_wide(panel, unit_col, value_col, time_col="week_idx"))
fit_synth(wide, treated, pre_index, donors=None, predictors=None, v=None, optimize_v=False,
          ridge=0.0, standardize_predictors=True) -> SynthResult
    # SynthResult: weights (Series, sum 1, >=0, <1e-4 zeroed), synthetic, actual, gap, pre_rmspe, pre_mspe,
    #   rel_rmse, n_donors_used, max_weight, effective_n, v_weights, .lift(post_index, sign=1)
apply_weights(wide, weights) -> Series            # LOCKED weights applied to any period
rmspe(actual, synthetic, index=None) -> float
lift_from_series(actual, cf, post_index, sign=1, pre_index=None, bias_correct=False) -> dict
injected_lift_recovery(wide, treated, pre_index, post_index, lift, donors=None, predictors=None,
          ridge=0.0, natural_bias_correction=True) -> dict(injected, detected, error, abs_error, ...)
fit_synth_pysyncon(wide, treated, pre_index, donors=None, predictors_long=None, predictor_cols=None) -> SynthResult
pysyncon_available() -> bool

# did.py
did_2x2_ratio(treat, control, pre_index, post_index, sign=1) -> dict
change_ratio_did(treat, control, pre_index, post_index, sign=1, n_boot=0, seed=0, alpha=0.10) -> dict
did_ols_fe(panel_long, y, unit, time, treated_col, post_col, cluster="unit"|"time"|None, log=False,
           weights=None, alpha=0.05) -> dict(beta, se, p_value, ci_*, rel_lift, rel_ci_*, se_type, few_clusters_warning)
welch_gap_test(treat, control, pre_index, post_index, relative=True) -> dict

# bsts.py
causal_impact(y, X, pre_index, post_index, level="llevel", seasonal=None, alpha=0.05, n_sims=1000,
              seed=0, sign=1, param_draws=50) -> dict(rel_effect, rel_ci_low, rel_ci_high, abs_effect,
              abs_ci_*, significant [= CI excludes 0 — the ONLY significance rule], p_like, pointwise DataFrame, ...)

# power.py   (sigma = sd of weekly relative treat-control gap; defaults one-sided alpha=0.10, power 0.8 — always state)
did_mde(sigma_gap, n_post, n_pre=None, alpha=0.10, power=0.80, two_sided=False, rho=0.0) -> float
did_power(lift, sigma_gap, n_post, n_pre=None, alpha=0.10, two_sided=False, rho=0.0) -> float
weeks_for_power(lift, sigma_gap, n_pre=None, ..., max_weeks=104) -> int | None
power_curve(sigma_gap, lifts, weeks, ...) -> DataFrame(lift, weeks, power)
ttest_power(effect_size=None, nobs1=None, alpha=0.05, power=None, ratio=1.0, alternative="two-sided") -> float
simulated_power(treat, control, lift, n_post, n_pre, estimator(t,c,pre,post)->stat, threshold(stat)->bool,
                n_sims=200, seed=0, holdout=True) -> dict(power, empirical_size, ...)
gap_sigma(treat, control, index=None, relative=True) -> dict(sigma, rho, n)

# inference.py
placebo_in_space(wide, treated, pre_index, post_index, donors=None, stat="rmspe_ratio"|"lift",
                 max_pre_rmspe_mult=None, fit_kw=None) -> dict(p_value, min_attainable_p, placebos, treated_fit)
placebo_in_time(wide, treated, pre_index, n_post, n_fit_min=8, donors=None, fit_kw=None, actual_lift=None) -> dict
jackknife_donors(wide, treated, pre_index, post_index, donors=None, only_active=True, fit_kw=None, sign=1) -> dict
permutation_pvalue(observed, null_draws, two_sided=True) -> float

# plotting.py  (importing sets the theme; Agg backend)
PALETTE, SERIES, ROLE (treatment/control/synthetic/buffer/excluded/truth/estimate/attribution/donor), CHANNEL_COLORS
set_theme(); shade_window(ax, start, end, label="Test window"); mark_line(ax, x, label); subtitle(ax, text)
pct_axis(ax); money_axis(ax); channel_label(ch); source_note(fig)
save_fig(fig, module, name) -> Path      # reports/figures/{module}/{name}.png @150 dpi
plot_actual_vs_counterfactual(index, actual, cf, window, title, ylabel, band=None); bar_compare(labels, series, title, ylabel, pct=False)

# geo.py
build_geography(cfg) -> Geography(zips, counties); haversine_miles(lat1, lon1, lat2, lon2) (vectorised)
distance_matrix_miles(lat, lon, lat2=None, lon2=None); min_distance_to_set(zips, ids)
adjacency_graph(zips, threshold_mi=10, connect_components=True) -> networkx.Graph (edge weight = miles)
county_adjacency(zips, threshold_mi=10) -> DataFrame(county_a, county_b)
voronoi_polygons(lon, lat, bbox=None) -> list[(k,2) arrays]   # choropleth-style maps
aggregate_static(zips, by="county_id") ; aggregate_zips(panel, mapping=None, by="county_id") ; monthly_media_sum(panel, cols)
zips_to_counties(zips, ids); counties_to_zips(zips, ids)
```
For BallTree buffers use `sklearn.neighbors.BallTree(np.radians(latlon), metric="haversine")` with
radius `miles / geo.EARTH_RADIUS_MI`.

## 7. Registering outputs: `reports/results/{module}.json`

Use `incrlab.common.results.write_results(module, headline, estimates, design=None, validation=None,
truth=None, figures=None, extra=None)`; read with `read_results(module)`. Schema:

```json
{
  "module": "dsp", "schema_version": "1.0", "generated_at": "ISO-8601", "seed": 20260, "small": false,
  "headline": {"<short_key>": <number|string>, "...": "numbers quoted in README must come from here"},
  "estimates": [
    {"test": "dsp_a", "arm": "holdout", "metric": "sales", "method": "change_ratio_did",
     "estimate": 0.142, "ci_low": 0.01, "ci_high": 0.27, "ci_level": 0.9, "p_value": 0.21,
     "significant": false, "truth": 0.146, "truth_key": "scenario.dsp_a.holdout.sales.true_lift_pct",
     "units": "relative lift (positive = channel contribution)", "notes": ""}
  ],
  "design": {"treated_units": [...], "control_units": [...], "weights": {...}, "...": "..."},
  "validation": {"injected_lift_recovery": [...], "placebo": {...}, "power": {...}},
  "truth": {"...": "copy of the scenario truth you compared against"},
  "figures": ["reports/figures/dsp/01_map.png"], "extra": {}
}
```
Required per estimate: `test, arm, metric, method, estimate`. NaN/inf are written as `null`.
**MMM calibration input**: MMM reads every module's `estimates` (use `metric` ∈ {qs, rated, sales}, plus
`extra.incremental_per_dollar = {"<channel>": {"sales": x, "spend": $, "window": [s, e]}}` from each test
module) — Meta/Search/DSP builders: please populate `extra.incremental_per_dollar` keyed by channel
(`social`, `search_brand`, `search_nonbrand`, `dsp_a`, `dsp_b`).
