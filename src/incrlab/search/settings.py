"""Single configuration object for the Paid Search two-arm dark-out (module 2).

Every analysis choice lives here so the notebook, CLI and tests run the same pipeline.
Test windows come from :mod:`incrlab.config` (``cfg.tests["search"]``); only analysis
choices (metrics, alphas, selection weights) are defined in this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ..config import Config, get_config

MODULE = "search"
TEST = "search"
ARMS = ("all_search", "branded_only")
ARM_LABELS = {"all_search": "All paid search off", "branded_only": "Branded search off",
              "control": "Control (BAU)", "bau": "BAU donor"}
# last-touch source buckets used in the readout (others = social + dsp + direct + other)
SOURCES = ("paid_brand", "paid_nonbrand", "organic", "unattributed", "other")
SOURCE_LABELS = {"paid_brand": "Paid search - branded", "paid_nonbrand": "Paid search - non-branded",
                 "organic": "Organic search", "unattributed": "Unattributed", "other": "Other channels"}


@dataclass(frozen=True)
class SearchSettings:
    # --- metrics -------------------------------------------------------------------
    # ONE design metric, used for synth fitting, injected-lift validation and MDE alike
    # (the original fit on one ratio and injected into another).
    design_metric: str = "rated"
    readout_metrics: Tuple[str, ...] = ("qs", "rated", "sales")
    guardrail_metric: str = "visits"
    per_capita_scale: float = 1e4  # outcomes per 10k residents
    # --- windows (inclusive week_idx); None -> taken from cfg.tests["search"] -------------
    sc_fit_start: int = 27  # long design history (1 year) for the synthetic-control fit
    validation_train: Tuple[int, int] = (27, 72)  # pseudo-launch validation inside design data
    validation_post: Tuple[int, int] = (73, 78)
    bsts_pre_start: int = 27
    # --- design selection -----------------------------------------------------------
    max_treat_pop_share: float = 0.12  # business risk cap per arm (share of region pop)
    min_weekly_rated: float = 60.0  # treated county needs volume to read anything
    injected_lifts: Tuple[float, ...] = (0.05, 0.10, 0.15)
    # planning priors (from platform data / earlier tests, NOT the simulator truth) used to
    # trade off which candidate goes to which arm: maximise expected power across both arms
    prior_lift_all: float = 0.08
    prior_lift_branded: float = 0.03
    sales_lift_ratio: float = 1.2  # planning prior: search lifts sales a bit more than rated quotes
    w_sales_power: float = 0.5  # weight of sales power in the arm-assignment score
    recovery_gate: float = 0.04  # max |detected - injected| at 10% injected lift to be eligible
    max_units_per_arm: int = 2  # arms may be a single county or a contiguous 2-county bundle
    w_recovery: float = 2.0  # penalty weight on injected-lift recovery error in the pairing score
    sc_ridge: float = 0.0
    balance_covariates: Tuple[str, ...] = ("median_hh_income", "pct_bachelors", "density", "homeowner_rate",
                                           "median_age", "pre_rated_pc", "pre_sales_pc", "pre_close_rate",
                                           "pre_paid_share_rated", "pre_brand_spend_pc")
    # --- inference ------------------------------------------------------------------
    alpha: float = 0.10  # two-sided 90% intervals (stated everywhere)
    power: float = 0.80
    n_boot: int = 2000
    bsts_sims: int = 1000
    n_power_sims: int = 300
    lift_grid: Tuple[float, ...] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)
    max_power_weeks: int = 52
    # --- guardrail / economics -------------------------------------------------------
    traffic_guardrail_max_loss: float = 0.10  # max acceptable site-visit loss when paused
    seed: int = 7
    cfg: Config = field(default_factory=get_config)

    @property
    def td(self):
        return self.cfg.tests[TEST]

    @property
    def region(self) -> str:
        return self.td.region

    @property
    def design_end(self) -> int:
        return self.td.design_end

    @property
    def pre(self) -> Tuple[int, int]:
        return (self.td.readout_pre_start, self.td.design_end)

    @property
    def post(self) -> Tuple[int, int]:
        return (self.td.test_start, self.td.test_end)

    @property
    def n_post(self) -> int:
        return self.post[1] - self.post[0] + 1

    @property
    def n_pre(self) -> int:
        return self.pre[1] - self.pre[0] + 1


def wk(window: Tuple[int, int]) -> list:
    """Inclusive window -> list of week indices."""
    return list(range(int(window[0]), int(window[1]) + 1))


def get_settings(cfg: Config | None = None, **kw) -> SearchSettings:
    cfg = cfg or get_config()
    if cfg.small:
        kw.setdefault("max_treat_pop_share", 0.25)
        kw.setdefault("min_weekly_rated", 30.0)
        kw.setdefault("n_boot", 500)
        kw.setdefault("bsts_sims", 400)
        kw.setdefault("n_power_sims", 100)
    return SearchSettings(cfg=cfg, **kw)
