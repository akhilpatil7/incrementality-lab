"""Every design / readout parameter of the DSP program in one place.

The parameters are *design choices* (not results). Labels on charts and in the readout are
generated from these values (e.g. the spillover buffer distance), so code and charts can
never disagree about what was actually used.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

from ..config import ADJACENCY_MILES_DEFAULT, BUFFER_MILES_DEFAULT, Config, get_config

FEATURES_A: Tuple[str, ...] = ("per_capita_income", "pct_bachelors", "lfpr",
                               "qs_per_capita", "rated_rate", "close_rate")
FEATURE_LABELS = {
    "per_capita_income": "Per-capita income", "pct_bachelors": "Bachelor's share", "lfpr": "Labor-force part.",
    "qs_per_capita": "Quote starts / capita", "rated_rate": "Rated-quote rate", "close_rate": "Close rate",
    "dsp_share": "DSP-attributed QS share", "dsp_share_b": "Vendor-attributed QS share", "lat": "Latitude", "lon": "Longitude",
    "median_hh_income": "Median HH income", "homeowner_rate": "Homeowner rate", "median_age": "Median age",
}
STAGE_LABELS = {"qs": "Quote starts", "rated": "Rated quotes", "sales": "Sales"}


@dataclass(frozen=True)
class VendorAParams:
    """Vendor A (contiguous zip patch, Mahalanobis-matched twin patch)."""
    test: str = "dsp_a"
    region: str = "NV"
    pop_target: float = 500_000  # current rung of the size ladder (set by the power-driven sizing step)
    pop_tolerance: float = 0.15
    # Pre-registered, power-driven sizing (decided BEFORE launch, from design-window data only):
    # walk the holdout-size ladder from small to large and keep the smallest size whose best
    # STRICT-gate pair detects ``target_mde`` on ``power_metric`` (sales) at 80% power over the
    # configured test window. If no rung qualifies, use the most powerful strict design and say
    # so loudly (never fall back to a relaxed pair silently).
    size_ladder: Tuple[float, ...] = (500_000, 750_000, 1_000_000, 1_250_000)
    max_holdout_share: float = 0.25  # business cap: share of the region's eligible population held out
    power_metric: str = "sales"
    target_mde: float = 0.20  # pre-registered: detect a 20% sales contribution at 80% power
    require_strict: bool = True
    min_zip_pop: int = 2_500
    adjacency_miles: float = ADJACENCY_MILES_DEFAULT
    buffer_miles: float = BUFFER_MILES_DEFAULT
    n_treatment_counties: int = 3  # largest eligible counties used to seed treatment patches
    w_compact: float = 0.20
    maha_ridge: float = 1e-6  # added to the covariance of *standardised* features
    covariance: str = "zip"  # "zip" (zip-level, as designed) or "patch" (bootstrap random patches)
    top_per_county: int = 5  # candidates kept per (treatment patch, control county)
    top_n_validate: int = 30
    top_n_review: int = 5
    strict_gates: Tuple[float, float, float, float] = (0.70, 0.5, 2.0, 0.50)  # corr, slope lo, hi, max CV
    relaxed_gates: Tuple[float, float, float, float] = (0.65, 0.4, 2.5, 0.60)
    validation_metric: str = "rated"  # weekly series used for the behavioural gates
    rank_lift: float = 0.15  # lift whose weeks-to-power enters the ranking
    lifts: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    max_weeks: int = 20
    alpha: float = 0.10  # one-sided (unified power convention)
    power: float = 0.80
    pseudo_post_weeks: int = 6  # design-time injected-lift check
    n_boot_cov: int = 300

    @property
    def pop_min(self) -> float:
        return self.pop_target * (1 - self.pop_tolerance)

    @property
    def pop_max(self) -> float:
        return self.pop_target * (1 + self.pop_tolerance)

    @property
    def buffer_label(self) -> str:
        return f"{self.buffer_miles:g}-mi spillover buffer"


@dataclass(frozen=True)
class VendorBParams:
    """Vendor B (geo-weighted K-means clusters, synthetic-control tournament)."""
    test: str = "dsp_b"
    region: str = "EM"
    k_range: Tuple[int, int] = (3, 14)  # inclusive
    cluster_pop_target: float = 750_000  # business granularity -> minimum K
    geo_weight: float = 3.0
    kmeans_seed: int = 42
    kmeans_n_init: int = 20
    fit_start: Optional[int] = None  # default: design_start
    pseudo_post_weeks: int = 8
    injected_lift: float = 0.10
    score_weights: Tuple[float, float, float] = (0.40, 0.35, 0.25)  # accuracy, fit, speed
    max_weeks: int = 20
    treat_pop_bounds: Tuple[float, float] = (300_000, 900_000)  # holdout size the business accepts
    exclude_largest_cluster: bool = True
    analyst_override: Optional[int] = None  # cluster id; must come with a rationale
    analyst_rationale: str = ""
    target_mde: float = 0.20  # same pre-registered sales-stage bar as Vendor A (used by the decision rule)
    switch_alpha: float = 0.05
    alpha: float = 0.10
    power: float = 0.80
    lifts: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    aug_lambdas: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    buffer_miles: float = BUFFER_MILES_DEFAULT  # for the spillover check (radius from treated zips)


@dataclass(frozen=True)
class ReadoutParams:
    sign: int = -1  # holdout: report channel contribution (positive = volume lost when paused)
    alpha: float = 0.10  # 90% intervals everywhere; significance = interval excludes 0
    n_boot: int = 2000
    bsts_history_start: int = 86  # long pre-period for the structural time-series model
    bsts_sims: int = 600
    go_prob: float = 0.80  # P(ROI > 1) needed for GO
    nogo_prob: float = 0.20
    seed: int = 7


@dataclass(frozen=True)
class DSPParams:
    a: VendorAParams = field(default_factory=VendorAParams)
    b: VendorBParams = field(default_factory=VendorBParams)
    readout: ReadoutParams = field(default_factory=ReadoutParams)
    module: str = "dsp"


def default_params(cfg: Optional[Config] = None) -> DSPParams:
    """Parameters scaled to the active geography (small/CI mode shrinks population targets)."""
    cfg = cfg or get_config()
    p = DSPParams()
    if cfg.small:
        p = replace(p, a=replace(p.a, pop_target=200_000, size_ladder=(200_000, 300_000), min_zip_pop=1_500,
                                 n_treatment_counties=2, max_holdout_share=0.35,
                                 top_n_validate=15, n_boot_cov=80),
                    b=replace(p.b, cluster_pop_target=450_000, k_range=(3, 8),
                              treat_pop_bounds=(120_000, 700_000)),
                    readout=replace(p.readout, n_boot=400, bsts_sims=200))
    return p
