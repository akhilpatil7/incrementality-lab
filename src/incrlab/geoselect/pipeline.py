"""End-to-end orchestration of the geoselect module: real-Census pull -> cluster -> match ->
power -> simulate test + monitor -> final DiD readout -> figures/results/readout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import census_data as CD
from . import clustering as CL
from . import power as PW
from . import run_test as RT
from .settings import GeoselectSettings, get_settings


@dataclass
class GeoselectState:
    st: GeoselectSettings
    geos: Optional[pd.DataFrame] = None
    clustered: Optional[pd.DataFrame] = None
    cluster_summary: Optional[pd.DataFrame] = None
    pair: Optional[dict] = None
    noise: Optional[dict] = None
    power_table: Optional[pd.DataFrame] = None
    mde: Optional[float] = None
    weeks_to_power_at_true_lift: Optional[int] = None
    daily: Optional[pd.DataFrame] = None
    weekly: Optional[pd.DataFrame] = None
    monitoring: Optional[pd.DataFrame] = None
    final: Optional[dict] = None
    verdict: Optional[dict] = None
    figures: list = field(default_factory=list)


def run(st: Optional[GeoselectSettings] = None, figures: bool = True, write: bool = True,
       force_refresh_census: bool = False) -> GeoselectState:
    st = st or get_settings()
    S = GeoselectState(st=st)

    # (a) real Census pull
    S.geos = CD.load_or_fetch(st, force_refresh=force_refresh_census)
    S.geos = CD.synth_baseline_kpi(S.geos, st)

    # (b) cluster
    S.clustered = CL.cluster_geos(S.geos, st)
    S.cluster_summary = CL.cluster_summary(S.clustered, st)

    # (c) match -> treatment/control pair + power analysis
    S.pair = CL.select_matched_pair(S.clustered, st)
    S.noise = PW.estimate_daily_noise(
        S.pair["feature_comparison"]["baseline_kpi_rate"]["treatment"],
        S.pair["feature_comparison"]["baseline_kpi_rate"]["control"],
        S.pair["treatment"]["pop"], S.pair["control"]["pop"], st, seed=st.seed)
    S.power_table = PW.power_table(S.noise["sigma"], S.noise["rho"], st)
    S.mde = PW.mde_for_design(S.noise["sigma"], S.noise["rho"], st)
    S.weeks_to_power_at_true_lift = PW.weeks_for_power(S.noise["sigma"], S.noise["rho"], st.true_lift, st)

    # (d) run simulated test + monitor
    S.daily = RT.simulate_test(S.pair, st, seed=st.seed)
    S.weekly = RT.weekly_panel(S.daily)
    S.monitoring = RT.monitoring_looks(S.weekly, st)

    # (e) final readout
    S.final = RT.final_readout(S.weekly, st)
    from . import readout as R
    S.verdict = R.verdict(S.final, S.mde, st)

    if figures:
        from . import figures as F
        S.figures = F.make_all(S)
    if write:
        from . import report
        report.write_results_json(S)
        report.write_readout_md(S)
    return S
