"""Design constants for the geo-experiment-selection module (module 5).

Unlike meta/search/dsp (which run on the repo's synthetic world), geoselect runs on REAL
Massachusetts geography pulled from the US Census ACS, clustered demographically, matched into
a treatment/control pair, power-sized and then put through a *simulated* test window (the
simulation step is the explicitly synthetic part of this module - see README §5 and
``run_test.py``). All week/day indices below are 0-based.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GeoselectSettings:
    module: str = "geoselect"
    state: str = "MA"
    state_fips: str = "25"

    # --- real geography filter -------------------------------------------------------------
    pop_threshold: float = 40_000.0       # ACS 5yr population floor for a county subdivision
    acs_vintage: int = 2022               # ACS 5-year end year (covers 2018-2022)

    # --- demographic feature set used for both clustering and matching ---------------------
    features: Tuple[str, ...] = (
        "median_income", "median_age", "pct_bachelors_plus", "pct_white", "pct_hispanic",
        "log_density", "log_pop", "baseline_kpi_rate",
    )
    kmeans_k: int = 5
    kmeans_seed: int = 7

    # --- synthetic baseline KPI (documented assumption; see census_data.synth_baseline_kpi) -
    # weekly "conversions" per 1,000 population, built from real demographics + noise so that
    # matching has a performance proxy beyond pure demographics. NOT real business data.
    kpi_name: str = "weekly_conversions_per_1k"
    kpi_seed: int = 2026
    kpi_noise_sd: float = 0.06            # relative noise sd on the demographic-driven baseline

    # --- test design -------------------------------------------------------------------------
    pre_period_days: int = 70             # 10 weeks of daily history used to estimate sigma/rho
    test_weeks: int = 8
    test_days: int = 56
    true_lift: float = 0.12               # ground truth lift injected into the treatment market
    daily_noise_sd: float = 0.09          # relative daily noise sd (geo x day)
    day_of_week_amp: float = 0.08         # weekly seasonality amplitude

    # --- power ---------------------------------------------------------------------------------
    alpha: float = 0.10                   # one-sided (direction of the lift is known/hypothesised)
    power_target: float = 0.80
    power_lifts: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)
    power_weeks_grid: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16, 20, 26)

    # --- monitoring: interim looks each week of the test window ------------------------------
    n_interim_looks: int = 8              # one look per test week
    did_alpha: float = 0.10               # 90% CI on the final + interim DiD reads
    n_boot: int = 2000

    seed: int = 20260

    @property
    def pre_period_weeks(self) -> float:
        return self.pre_period_days / 7.0


def get_settings() -> GeoselectSettings:
    return GeoselectSettings()
