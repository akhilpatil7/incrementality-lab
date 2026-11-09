# Methodology overview

incrementality-lab v2 is one measurement program run on one synthetic world. Everything below is implemented
in `src/incrlab/` and reproduced by `make all` (`python -m incrlab.pipeline all`). Every headline number is
written to `reports/results/*.json`, and `reports/results/summary.json` condenses them next to the truth.

```
config.py (truth) -> simulate.py (world + scenario runner) -> meta -> search -> dsp -> mmm -> summary.json
```

## 0. The synthetic world (`config.py`, `geo.py`, `simulate.py`)

- **Geography.** Two synthetic regions ("Eastmoor", "Northvale"), 39 counties, ~1,100 zips with lat/lon and
  ACS-like features (population, income, education, labour-force participation, age, homeownership,
  density), plus a rest-of-nation aggregate for the national panel. All units are generated in code.
- **Time.** 156 weeks with trend, seasonality, holidays, regional AR shocks and negative-binomial noise.
- **Funnel.** Impressions -> visits -> quote starts -> rated quotes -> sales (binds).
- **Channels.** TV (national), Social, Paid Search (branded / non-branded, with organic recapture), DSP
  Vendor A (full-funnel), DSP Vendor B (top-funnel; little effect on sales).
- **Truth.** Each channel's response is geometric adstock + Hill saturation; `data/ground_truth.json` holds the
  true ROI, marginal ROI and cost per incremental conversion of every channel, and the scenario runner returns
  the true lift of every test on the units the pipeline chose.
- **Attribution.** Last-touch and vendor-reported conversions are deliberately biased (Social over-claims
  quotes and under-claims sales; branded search over-claims through cannibalisation; DSP vendors over-claim
  by multiples).
- **Interventions.** `simulate.run_scenario({test: {arm: units}})` pauses the channel(s) in exactly the
  selected geos (with leakage and spillover), so outcomes reflect the design that was chosen.

## 1. Social geo holdout

`src/incrlab/meta/` · notebook `01_meta.ipynb` · readout `reports/readouts/meta.md`

- Candidate units: single counties and adjacent bundles inside a population band; k-means sanity check.
- Synthetic control (own SLSQP solver; pysyncon cross-check) on the share-of-quotes metric with weights
  **locked** before launch.
- Injected-lift validation (5/10/15%), candidate scoring (RMSPE, donor concentration, recovery), closed-form
  and simulated power, leave-one-donor-out jackknife.
- Readout: DiD with week fixed effects and HC1 robust SE, ratio DiD with bootstrap CI, synthetic-control gap with
  in-space placebo p-value, across quote starts / rated quotes / sales; attribution vs incrementality.
- Deliberately the "first, rough" test: 8 weeks, leaky holdout, so the honest label is **directional**.

## 2. Paid Search two-arm dark-out

`src/incrlab/search/` · notebook `02_search.ipynb` · readout `reports/readouts/search.md`

- Two arms (all paid search paused; branded only paused), a matched control and BAU donors, per-capita
  normalisation, county MDE, injected-lift validation of the synthetic control.
- DiD with unit + week fixed effects and **cluster-robust SE**; BSTS / CausalImpact-style counterfactual with
  significance taken from the credible interval; power via `TTestIndPower` and simulation.
- Organic **recapture** (cannibalisation) of lost branded conversions, incremental vs last-touch cost per
  rated quote / sale, site-traffic guardrail.

## 3. Programmatic DSP (flagship)

`src/incrlab/dsp/` · notebook `03_dsp.ipynb` · readout `reports/readouts/dsp.md` · **full detail in
[dsp_pipeline.md](dsp_pipeline.md)**

- Vendor A: 10-mi zip adjacency graph, BFS contiguous patches, haversine spillover buffer (one parameter drives
  code and charts), 6-feature Mahalanobis matching with compactness penalty, strict/relaxed pre-trend gates
  (never force-passed), SMD balance table, injected-lift check.
- **Power drives the design**: the holdout size is chosen by a pre-registered ladder on design-window data, the
  smallest strict-gate design that detects a 20% **sales** contribution at 80% power.
- Vendor B: vendor-switch structural-break test, geo-weighted K-means (silhouette + business rule for K),
  synthetic-control tournament, placebo inference (in-space, in-time, ~100 random geographies), ridge-augmented SC.
- Readout triangulation: change-ratio DiD (headline), Welch on weekly gaps, two-way FE DiD with clustered SE,
  structural time series, synthetic control + placebo p-value; funnel-stage lifts, incremental CPA, spillover check.
- Decision rule hinges on sales: GO needs a significant sales lift from a design powered for sales.

## 4. Experiment-calibrated MMM

`src/incrlab/mmm/` · notebook `04_mmm_calibration.ipynb` · readout `reports/readouts/mmm.md`

- Weekly national Bayesian MMM in PyMC: geometric adstock + Hill saturation per channel, trend, Fourier
  seasonality, holidays; 13-week true holdout for out-of-sample checks.
- Fit (a) uncalibrated and (b) calibrated: every test arm adds a likelihood term
  `observed incremental sales per $ ~ Normal(model-implied, sqrt(SE^2 + transport^2))`, where the model replays
  the test (channel cut to its leakage level in the window) for each posterior draw.
- Sensitivity fit with well-powered (simulated, truth-centred) readouts separates "is the machinery right?" from
  "were the tests informative?".
- Posterior-aware budget optimizer (SLSQP, total fixed, +-30% per channel), scenarios, response curves,
  contribution waterfall, triangulation (last-touch vs experiment vs uncalibrated vs calibrated vs truth), and a
  value-of-information (EVPPI) **next test** recommendation.

## Conventions used everywhere

- Lift sign: positive = channel **contribution** (volume lost when paused).
- Intervals are 90%; "significant" means the interval excludes 0, nothing else.
- Power: one-sided alpha = 0.10, 80% power, sigma = sd of the weekly relative treatment-control gap with AR(1)
  inflation.
- Designs see only pre-launch data (`tests/` blank post-launch weeks and check the selection does not change).
- Rates are recomputed from summed numerators / denominators, never averaged.

## Honest caveats

- The data are synthetic and the simulator's response family matches the MMM's; real programs also face
  misspecification, measurement error and organisational constraints that this lab does not model.
- Magnitudes depend on the design and the noise draw; different seeds give different (but similarly honest) reads.
- Test windows for the DSP program were lengthened after its pre-period power analysis showed the first plan could
  not resolve sales; true effects were never changed.
