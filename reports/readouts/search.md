# Paid Search geo dark-out — executive readout (synthetic)

*Module 2 of incrementality-lab. Synthetic data with known ground truth; magnitudes are illustrative.
All intervals are two-sided 90%. Lift = share of volume lost while paid search was dark (positive = paid search contribution).*

## Question
How much of paid search is truly incremental, and how much branded search merely buys conversions that organic
search would have delivered anyway (cannibalisation)?

## Design (selected by the pipeline from 20 candidate units, data through week 78)
| Arm | Counties | Paused |
|---|---|---|
| All paid search off | Pinecrest County + Hearthstone County | branded + non-branded |
| Branded search off | Fernhollow County + Northcote County | branded only |
| Control | Ironwood County | nothing (ratio-DiD comparator) |
| BAU donors | 9 counties | nothing (synthetic control / BSTS / FE-DiD pool) |

Weeks 79–84 dark (6 wk); readout pre-period weeks 69–78; synthetic control fit on weeks
27–78. Arms were chosen to maximise expected power (rated quotes + sales) under a recovery gate
on +5/10/15% injected lifts measured on the **same** metric used for fitting.

## Results — triangulated lift vs truth
| Arm | Metric | Lift | 90% CI | Methods significant | Truth | MDE (80% power) |
|---|---|---|---|---|---|---|
| All paid search off | quote starts | **9.3%** | [1.3%, 17.4%] | 3/4 | 8.1% | 9% |
| All paid search off | rated quotes | **9.1%** | [0.2%, 18.0%] | 2/4 | 9.3% | 10% |
| All paid search off | sales | **-4.5%** | [-24.0%, 15.1%] | 0/4 | 11.1% | 30% |
| Branded search off | quote starts | **8.0%** | [-1.0%, 17.0%] | 1/4 | 2.8% | 13% |
| Branded search off | rated quotes | **9.8%** | [-1.1%, 20.7%] | 0/4 | 3.2% | 13% |
| Branded search off | sales | **6.5%** | [-17.6%, 30.5%] | 0/4 | 4.0% | 35% |

Methods: ratio DiD vs control, OLS DiD with unit + week fixed effects (inference by donor permutation — cluster-robust
SE with one treated cluster is anti-conservative), BSTS counterfactual on all donors (significance only when its interval
excludes 0), synthetic control with in-space placebo p (never below 1/(J+1)).

## Cannibalisation — organic recapture
| Arm | Stage | Δ paid LT / wk | Δ organic LT / wk | Recapture [90% CI] | Truth |
|---|---|---|---|---|---|
| Branded search off | quote starts | -11.5 | +2.8 | **24%** [2%, 51%] | 40% |
| Branded search off | rated quotes | -8.7 | +3.0 | **35%** [11%, 63%] | 40% |
| All paid search off | quote starts | -34.8 | +6.6 | **19%** [7%, 30%] | 26% |
| All paid search off | rated quotes | -26.3 | +4.3 | **16%** [3%, 30%] | 26% |

## Efficiency — incremental vs last-touch
| Arm | Per | Last-touch | Incremental (est.) [90% CI] | Incremental (truth) |
|---|---|---|---|---|
| All paid search off | rated quote | $98 | **$137** [$69, $6,096] | $140 |
| Branded search off | rated quote | $77 | **$37** [$17, n/a] | $117 |
| All paid search off | sale | $1,178 | **n/a** [$843, n/a] | $1,119 |
| Branded search off | sale | $774 | **$607** [$129, n/a] | $916 |

MMM calibration (`extra.incremental_per_dollar`): branded **27.2** incremental rated quotes per $1k
(SE 18.4; truth 8.6); non-branded
**0.4** (SE 8.7; truth 6.3).

Traffic guardrail (≤10% visit loss): all-search 12.7% (truth 13.9%),
branded-only 9.7% (truth 6.0%).

## Recommendation
- All paid search is incremental on rated quotes: 9.1% of rated quotes lost when dark (90% CI 0.2% to 18.0%); incremental cost per rated quote $137 vs $98 last-touch.
- Sales read is inconclusive (est. -4.5%, CI -24.0% to +15.1%; 6 weeks of binds cannot resolve a ~10% lift) - do not cut search on the sales number alone.
- Branded search is heavily cannibalised: organic recaptures ~35% (CI 11%-63%) of lost branded rated quotes. Move brand to a defensive bidding policy (bid only where competitors are present on brand terms / lower caps) and re-test with a powered design.
- Incremental rated quotes per $1k: branded 27.2 vs non-branded 0.4 (SE 18.4 / 8.7) -> feed both, with SEs, to the MMM as calibration priors.
- Traffic guardrail (10% max visit loss) breached in: all_search.
- The branded arm's rated-quote MDE (13% at 80% power) is far above a plausible branded lift (~3%): its lift read is directional; the cannibalisation (source-level) read is the decision-grade output of that arm.

## Caveats
- Synthetic world; true effects come from the generator's adstock + Hill response, not from this analysis.
- One treated unit per arm: every p-value is rank-based and bounded below by 1/(donors+1).
- Six weeks of binds cannot resolve sales lifts near 10% — sales reads are directional; rated quotes are the decision KPI.
- 100% of the 20 headline quantities have the truth inside their 90% interval (see `reports/figures/search/14_truth_table.png`).
