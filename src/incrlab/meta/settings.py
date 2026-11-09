"""Design constants for the Social (Meta) geo-holdout module.

All week indices are 0-based (``incrlab`` convention). The 58-week design history
(weeks 0..57) is split into a synthetic-control *training* window (0..50) and a 7-week
*pseudo-post* window (51..57) used to validate the design with injected lifts. The live
readout uses weeks 51..57 as its DiD pre-period and the configured test window (58..65).

Thresholds are defined once here and used by BOTH the hard filters and the human-readable
rejection reasons (the source workflow used different numbers in the two places).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ..config import Config, get_config


@dataclass(frozen=True)
class MetaSettings:
    test: str = "meta"
    arm: str = "holdout"
    channel: str = "social"
    region: str = "EM"
    design_start: int = 0
    train_end: int = 50        # SC training window = design_start..train_end
    design_end: int = 57       # last week visible at design time (pseudo-post = train_end+1..design_end)
    readout_pre_start: int = 51
    test_start: int = 58
    test_end: int = 65

    # candidate units: target population as a share of the region (≈0.5M of 9.3M at full size)
    target_pop_share: float = 0.054
    pop_band: Tuple[float, float] = (0.80, 1.20)  # x target
    max_group_size: int = 3
    bundles_must_be_contiguous: bool = True       # bundles = adjacent counties only

    design_metric: str = "social_share_qs"        # SC dependent for unit selection
    readout_metric: str = "qs_per_capita"         # SC dependent for volume readout weights

    # k-means sanity lens
    kmeans_k_grid: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    kmeans_k: int = 5
    kmeans_seed: int = 42

    # injected-lift validation (holdout => volume LOSS of `lift` in the pseudo-post window)
    validation_lifts: Tuple[float, ...] = (0.05, 0.10, 0.15)
    scoring_lift: float = 0.10

    # candidate filters (one source of truth for filter + reason text)
    max_weight: float = 0.50
    min_neff: float = 2.5
    donor_count: Tuple[int, int] = (3, 12)
    max_recovery_error: float = 0.04              # |raw detected - injected| (out-of-sample)
    rmspe_quantile: float = 0.75
    # score = weights . (fit, diversification, recovery, readout-noise)
    score_weights: Tuple[float, float, float, float] = (0.35, 0.20, 0.25, 0.20)
    neff_cap: float = 6.0
    shortlist_n: int = 10

    # power
    alpha: float = 0.10               # one-sided (holdout direction known)
    power: float = 0.80
    power_lifts: Tuple[float, ...] = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
    power_max_weeks: int = 20
    alpha_grid: Tuple[float, ...] = (0.025, 0.05, 0.10, 0.20)

    # jackknife stability rule (absolute pp change avoids the divide-by-~0 blow-up of % rules)
    jk_max_change_pp: float = 0.03
    jk_max_rmspe_ratio: float = 2.0

    # readout
    did_alpha: float = 0.10           # CIs reported at 90% (two-sided) alongside p-values
    n_boot: int = 2000
    directional_p: float = 0.10

    @property
    def train_weeks(self) -> list:
        return list(range(self.design_start, self.train_end + 1))

    @property
    def pseudo_post_weeks(self) -> list:
        return list(range(self.train_end + 1, self.design_end + 1))

    @property
    def design_weeks(self) -> list:
        return list(range(self.design_start, self.design_end + 1))

    @property
    def readout_pre_weeks(self) -> list:
        return list(range(self.readout_pre_start, self.test_start))

    @property
    def test_weeks(self) -> list:
        return list(range(self.test_start, self.test_end + 1))

    @property
    def n_test_weeks(self) -> int:
        return self.test_end - self.test_start + 1


def get_settings(cfg: Config | None = None) -> MetaSettings:
    """Settings aligned with the configured ``meta`` TestDesign (windows come from config)."""
    cfg = cfg or get_config()
    td = cfg.tests["meta"]
    return MetaSettings(region=td.region, design_start=td.design_start, design_end=td.design_end,
                        readout_pre_start=td.readout_pre_start, test_start=td.test_start,
                        test_end=td.test_end,
                        train_end=td.readout_pre_start - 1)
