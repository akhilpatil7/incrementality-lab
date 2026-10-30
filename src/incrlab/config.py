"""Central configuration: seeds, paths, timeline, channels, TRUE effect parameters, test designs.

Everything that defines the synthetic world's *ground truth* lives here, so that every
module (Social / Paid Search / DSP / MMM) reads one consistent definition.

Small mode
----------
Set the environment variable ``INCRLAB_SMALL=1`` to shrink the geography (fewer counties and
zips) while keeping the 156-week timeline and all test windows intact. Used by CI.

All week indices are 0-based offsets from ``START_DATE`` (a Monday).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent.parent
DATA_DIR = Path(os.environ.get("INCRLAB_DATA_DIR", REPO_ROOT / "data"))
REPORTS_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

SEED = int(os.environ.get("INCRLAB_SEED", 20260))
START_DATE = "2023-01-02"  # Monday
N_WEEKS = 156

CHANNELS: Tuple[str, ...] = (
    "tv", "social", "search_brand", "search_nonbrand", "dsp_a", "dsp_b",
)
PAID_SEARCH = ("search_brand", "search_nonbrand")
STAGES: Tuple[str, ...] = ("qs", "rated", "sales")  # quote starts -> rated quotes -> sales (binds)
# Last-touch source partition (sums to the stage total in every geo-week)
LT_SOURCES: Tuple[str, ...] = (
    "search_brand", "search_nonbrand", "organic", "social", "dsp",
    "direct", "unattributed", "other",
)

CHANNEL_LABELS: Dict[str, str] = {
    "tv": "TV / Brand",
    "social": "Social",
    "search_brand": "Paid Search - Branded",
    "search_nonbrand": "Paid Search - Non-branded",
    "dsp_a": "Programmatic DSP - Vendor A",
    "dsp_b": "Programmatic DSP - Vendor B",
}


# --------------------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionSpec:
    code: str
    name: str
    n_counties: int
    n_zips: int
    pop_total: float
    center_lat: float
    center_lon: float
    width_mi: float
    height_mi: float
    n_metros: int
    ses_shift: float = 0.0  # region-level socio-economic shift (income/education)


REGIONS_FULL = (
    # Region A: hosts the Social holdout (county units) + DSP Vendor B test (zip clusters)
    RegionSpec("EM", "Eastmoor", 25, 590, 9.3e6, 40.10, -87.60, 115, 165, 4, 0.15),
    # Region B: hosts the Paid Search dark-out (county units) + DSP Vendor A test (zip patches)
    RegionSpec("NV", "Northvale", 14, 500, 7.0e6, 43.40, -91.20, 150, 110, 3, 0.05),
)
REGIONS_SMALL = (
    RegionSpec("EM", "Eastmoor", 12, 160, 3.2e6, 40.10, -87.60, 80, 110, 2, 0.15),
    RegionSpec("NV", "Northvale", 8, 140, 2.6e6, 43.40, -91.20, 95, 70, 2, 0.05),
)
ROW_POP = 50.0e6  # rest-of-nation aggregate unit (only feeds the national panel / MMM)


# --------------------------------------------------------------------------------------
# TRUE channel response (ground truth)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelTruth:
    """Geometric adstock + Hill saturation on adstocked spend per *served* capita.

    effect_qs(g,t) = served_frac_g * max_effect * Hill(adstock_pc; K, slope)
    K = half_sat_ratio * (median BAU adstock-per-capita over served geo-weeks)  [computed in simulate]

    For paid search, ``max_effect`` is the max *last-touch capture* share of baseline quote
    starts; the TRUE incremental effect is ``(1 - recapture) * capture`` (the rest would have
    arrived through organic search anyway).
    stage_mult scales the quote-start effect at the rated and sales stage (full-funnel vs
    top-funnel channels). lt_bias = last-touch-claimed / true incremental at each stage.
    """
    decay: float
    half_sat_ratio: float
    slope: float
    max_effect: float
    stage_mult: Tuple[float, float, float]  # (qs, rated, sales)
    visit_mult: float
    lt_bias: Tuple[float, float, float]  # last-touch claimed / true, per stage (non-search)
    base_weekly_spend: float
    cpm: float
    ctr: float
    cpc: Optional[float] = None  # search: clicks = spend / cpc
    recapture: float = 0.0  # search only: share of paid LT conversions organic recaptures
    vendor_bias: float = 0.0  # DSP only: vendor-reported conv quotes / true incremental QS


CHANNEL_TRUTH: Dict[str, ChannelTruth] = {
    # TV: national, always-on, no geo variation, no last-touch credit (shows up as direct/unattributed)
    "tv": ChannelTruth(0.70, 1.00, 1.0, 0.22, (1.0, 1.0, 1.0), 0.8, (0.0, 0.0, 0.0),
                       600_000, 22.0, 0.0),
    # Social: over-claims quotes, under-claims sales in last-touch
    "social": ChannelTruth(0.40, 0.90, 1.3, 0.22, (1.0, 1.0, 1.15), 1.3, (2.8, 2.4, 0.8),
                           200_000, 9.0, 0.009),
    # Branded search: near-saturated, heavily cannibalizes organic (TRUE recapture 40%)
    "search_brand": ChannelTruth(0.10, 0.35, 1.5, 0.085, (1.0, 1.2, 1.7), 1.4, (0, 0, 0),
                                 110_000, 0.0, 0.30, cpc=12.0, recapture=0.40),
    "search_nonbrand": ChannelTruth(0.20, 0.60, 1.2, 0.145, (1.0, 1.2, 1.6), 1.4, (0, 0, 0),
                                    300_000, 0.0, 0.06, cpc=88.0, recapture=0.15),
    # DSP Vendor A: full-funnel (strongest at sales)
    "dsp_a": ChannelTruth(0.50, 1.20, 1.0, 0.29, (1.0, 1.43, 2.9), 1.2, (1.5, 1.5, 1.5),
                          140_000, 4.0, 0.006, vendor_bias=4.5),
    # DSP Vendor B: top-funnel only (little effect on sales)
    "dsp_b": ChannelTruth(0.50, 1.00, 1.0, 0.20, (1.0, 0.67, 0.05), 1.2, (1.5, 1.5, 1.5),
                          90_000, 6.0, 0.0006, vendor_bias=5.5),
}

# Channel presence (served share of a geo's population). Zips are 0/1; ROW is fractional.
PRESENCE = {
    "tv": {"EM": 1.0, "NV": 1.0, "ROW": 1.0},
    "social": {"EM": 1.0, "NV": 1.0, "ROW": 1.0},
    "search_brand": {"EM": 1.0, "NV": 1.0, "ROW": 1.0},
    "search_nonbrand": {"EM": 1.0, "NV": 1.0, "ROW": 1.0},
    "dsp_a": {"EM": 0.0, "NV": 0.70, "ROW": 0.60},  # NV: ~70% of zips served
    "dsp_b": {"EM": 0.65, "NV": 0.0, "ROW": 0.50},  # EM: vendor inclusion list ~65% of zips
}


@dataclass(frozen=True)
class WorldParams:
    base_qs_rate: float = 4.5e-4  # baseline quote starts / person / week (no ads)
    propensity_sigma: float = 0.40
    propensity_income_slope: float = -0.35  # QS per capita falls with SES (corr ~ -0.5)
    trend_per_week: float = 0.0015
    county_trend_sd: float = 0.0006
    region_ar_rho: float = 0.6
    region_ar_sd: float = 0.03
    region_spike_prob: float = 0.012
    county_ar_rho: float = 0.5
    county_ar_sd: float = 0.035
    zip_noise_sd: float = 0.03
    nb_dispersion_zip: float = 20.0
    rated_rate_mean: float = 0.64
    rated_rate_sd: float = 0.03
    close_rate_mean: float = 0.09
    close_rate_sd: float = 0.012
    visits_per_qs: float = 12.0
    # organic pools (share of baseline QS arriving via organic search with NO paid search)
    organic_brand_share: float = 0.08
    organic_generic_share: float = 0.05
    # remainder split of non-search, non-claimed volume
    rest_split: Tuple[float, float, float] = (0.30, 0.25, 0.45)  # direct, unattributed, other
    value_per_sale: float = 1500.0  # $ margin-ish value of a bound policy (for ROI)
    platform_conv_overclaim: float = 1.25  # search platform conversions vs LT quotes


# --------------------------------------------------------------------------------------
# Test designs (windows are inclusive 0-based week indices)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TestDesign:
    name: str
    channels_by_arm: Dict[str, Tuple[str, ...]]
    region: str
    unit_level: str  # "county" or "zip"
    design_start: int  # first week of design/selection history
    design_end: int  # last week available at design time (pre-launch)
    readout_pre_start: int
    test_start: int
    test_end: int
    spend_multiplier: float = 0.0  # holdout = 0
    leakage: float = 0.03  # share of spend that keeps serving in treated geos
    spillover: Optional[dict] = None
    notes: str = ""


TESTS: Dict[str, TestDesign] = {
    # Year 2 Q1: first, rough test. Incomplete holdout (residual ~35% delivery).
    "meta": TestDesign(
        "meta", {"holdout": ("social",)}, "EM", "county",
        design_start=0, design_end=57, readout_pre_start=51, test_start=58, test_end=65,
        leakage=0.25,
        notes="Social geo holdout; 8 test weeks; 57-week design history (train 1-50, pseudo-post 51-57)."),
    # Year 2 Q3: two-arm dark-out.
    "search": TestDesign(
        "search", {"all_search": PAID_SEARCH, "branded_only": ("search_brand",)}, "NV", "county",
        design_start=60, design_end=78, readout_pre_start=69, test_start=79, test_end=84,
        leakage=0.0,
        notes="Two treatment arms + control + BAU donors; 10 pre / 6 test weeks."),
    # Year 3 Q1: flagship DSP tests.
    "dsp_a": TestDesign(
        "dsp_a", {"holdout": ("dsp_a",)}, "NV", "zip",
        design_start=95, design_end=111, readout_pre_start=102, test_start=112, test_end=117,
        leakage=0.03,
        spillover={"enabled": True, "radius_mi": 30.0, "decay_mi": 10.0, "strength": 0.30},
        notes="Contiguous zip patch (~0.45M pop) vs Mahalanobis-matched twin patch; 6 test weeks."),
    "dsp_b": TestDesign(
        "dsp_b", {"holdout": ("dsp_b",)}, "EM", "zip",
        design_start=90, design_end=111, readout_pre_start=101, test_start=112, test_end=119,
        leakage=0.03,
        spillover={"enabled": True, "radius_mi": 30.0, "decay_mi": 10.0, "strength": 0.30},
        notes="K-means zip cluster treated vs synthetic control of other clusters; 8 test weeks."),
}

DSP_B_VENDOR_SWITCH_WEEK = 96  # legacy vendor -> Vendor B (attribution level shift)
DSP_B_SWITCH_ATTR_SHIFT = 0.80  # vendor-reported conversions x0.8 after the switch
DSP_B_REPORT_DIP_WEEKS = (100, 101)  # vendor reporting outage-like dips
DSP_REPORT_LAG = (0.5, 0.2, 0.05)  # attribution lag multipliers on the final 3 weeks of data
BUFFER_MILES_DEFAULT = 30.0
ADJACENCY_MILES_DEFAULT = 10.0
CONCURRENT_TEST = {"region": "EM", "share_of_zips": 0.07, "channel": "social",
                   "spend_multiplier": 2.0, "start": 106, "end": 121}


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    small: bool = False
    n_weeks: int = N_WEEKS
    start_date: str = START_DATE
    regions: Tuple[RegionSpec, ...] = REGIONS_FULL
    row_pop: float = ROW_POP
    channels: Tuple[str, ...] = CHANNELS
    truth: Dict[str, ChannelTruth] = field(default_factory=lambda: dict(CHANNEL_TRUTH))
    presence: Dict[str, Dict[str, float]] = field(default_factory=lambda: dict(PRESENCE))
    world: WorldParams = WorldParams()
    tests: Dict[str, TestDesign] = field(default_factory=lambda: dict(TESTS))
    effect_multiplier: float = 1.0  # scales every channel's max_effect (for "clearly significant" demos)
    data_dir: Path = DATA_DIR

    def max_effect(self, ch: str) -> float:
        return self.truth[ch].max_effect * self.effect_multiplier


def get_config(seed: Optional[int] = None, small: Optional[bool] = None,
               effect_multiplier: Optional[float] = None) -> Config:
    """Return the active config. ``INCRLAB_SMALL=1`` selects the small geography."""
    if small is None:
        small = os.environ.get("INCRLAB_SMALL", "0") not in ("", "0", "false", "False")
    if effect_multiplier is None:
        effect_multiplier = float(os.environ.get("INCRLAB_EFFECT_MULT", 1.0))
    cfg = Config(seed=SEED if seed is None else int(seed), small=bool(small),
                 regions=REGIONS_SMALL if small else REGIONS_FULL,
                 row_pop=ROW_POP * (0.4 if small else 1.0),
                 effect_multiplier=float(effect_multiplier))
    return cfg


def with_overrides(cfg: Config, **kw) -> Config:
    return replace(cfg, **kw)
