"""Weekly county panel for the Paid Search test: last-touch source buckets + per-capita outcomes.

The world generator gives every county-week a last-touch partition over eight sources. The
readout groups them into the buckets a search team actually reports on: paid brand, paid
non-brand, organic, unattributed and "other" (social + DSP + direct + other). All rates are
recomputed from summed counts (never averaged), and per-capita values use the unit's
population so county size never drives synthetic-control weights.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .settings import SOURCES, SearchSettings

STAGES = ("qs", "rated", "sales")
_SRC_MAP = {"paid_brand": ("search_brand",), "paid_nonbrand": ("search_nonbrand",), "organic": ("organic",),
            "unattributed": ("unattributed",), "other": ("social", "dsp", "direct", "other")}
COUNT_COLS = (["visits", "pop", "spend_search_brand", "spend_search_nonbrand", "clicks_search_brand",
               "clicks_search_nonbrand", "impressions_search_brand", "impressions_search_nonbrand",
               "platform_conv_search_brand", "platform_conv_search_nonbrand"]
              + list(STAGES) + [f"{src}_{s}" for src in SOURCES for s in STAGES])


def build_county_panel(county_weekly: pd.DataFrame, region: str) -> pd.DataFrame:
    """Long county x week panel for one region with source buckets (counts) attached."""
    d = county_weekly[county_weekly["region"] == region].copy()
    for s in STAGES:
        for b, srcs in _SRC_MAP.items():
            d[f"{b}_{s}"] = sum(d[f"lt_{x}_{s}"] for x in srcs)
        chk = sum(d[f"{b}_{s}"] for b in SOURCES)
        if not np.array_equal(chk.to_numpy(), d[s].to_numpy()):
            raise AssertionError(f"last-touch buckets do not sum to {s}")
    keep = ["county_id", "county", "week_idx", "week_start"] + list(COUNT_COLS)
    return add_rates(d[keep].sort_values(["county_id", "week_idx"]).reset_index(drop=True))


def add_rates(d: pd.DataFrame, scale: float = 1e4) -> pd.DataFrame:
    """Per-capita (per ``scale`` residents) outcomes and ratio diagnostics from summed counts."""
    d = d.copy()
    for s in STAGES:
        d[f"{s}_pc"] = d[s] / d["pop"] * scale
        for b in SOURCES:
            d[f"{b}_{s}_pc"] = d[f"{b}_{s}"] / d["pop"] * scale
        paid = d[f"paid_brand_{s}"] + d[f"paid_nonbrand_{s}"]
        d[f"paid_share_{s}"] = np.where(d[s] > 0, paid / d[s].clip(lower=1), np.nan)
    d["visits_pc"] = d["visits"] / d["pop"] * scale
    d["close_rate"] = np.where(d["rated"] > 0, d["sales"] / d["rated"].clip(lower=1), np.nan)
    d["rated_rate"] = np.where(d["qs"] > 0, d["rated"] / d["qs"].clip(lower=1), np.nan)
    d["spend_search"] = d["spend_search_brand"] + d["spend_search_nonbrand"]
    d["spend_search_pc"] = d["spend_search"] / d["pop"] * scale
    d["brand_spend_pc"] = d["spend_search_brand"] / d["pop"] * scale
    for ch in ("brand", "nonbrand"):
        clicks = d[f"clicks_search_{ch}"]
        d[f"cpc_{ch}"] = np.where(clicks > 0, d[f"spend_search_{ch}"] / clicks.clip(lower=1), np.nan)
        d[f"ctr_{ch}"] = np.where(d[f"impressions_search_{ch}"] > 0,
                                  clicks / d[f"impressions_search_{ch}"].clip(lower=1), np.nan)
    return d


def aggregate_units(panel: pd.DataFrame, groups: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Sum counts of county groups into pseudo-units (e.g. a 2-county arm), then recompute rates."""
    rows = []
    for name, ids in groups.items():
        g = panel[panel["county_id"].isin(list(ids))]
        s = g.groupby(["week_idx", "week_start"], as_index=False)[list(COUNT_COLS)].sum()
        s.insert(0, "county", " + ".join(sorted(panel.loc[panel["county_id"].isin(ids), "county"].unique())))
        s.insert(0, "county_id", name)
        rows.append(s)
    return add_rates(pd.concat(rows, ignore_index=True))


def unit_panel(panel: pd.DataFrame, arm_units: Mapping[str, Sequence[str]], donors: Iterable[str]) -> pd.DataFrame:
    """Panel where each arm is one aggregated unit and each donor county is its own unit."""
    donors = list(donors)
    arm = aggregate_units(panel, arm_units) if arm_units else panel.iloc[:0]
    don = panel[panel["county_id"].isin(donors)]
    return pd.concat([arm, don], ignore_index=True)


def wide(panel: pd.DataFrame, value: str, weeks: Optional[Sequence[int]] = None) -> pd.DataFrame:
    w = panel.pivot_table(index="week_idx", columns="county_id", values=value, aggfunc="sum").sort_index()
    return w if weeks is None else w.loc[list(weeks)]


def pooled_series(panel: pd.DataFrame, units: Sequence[str], value_count: str, scale: float = 1e4) -> pd.Series:
    """Population-weighted (sum counts / sum pop) per-capita series for a set of units."""
    g = panel[panel["county_id"].isin(list(units))].groupby("week_idx")[[value_count, "pop"]].sum()
    return g[value_count] / g["pop"] * scale


def static_features(panel: pd.DataFrame, county_static: pd.DataFrame, pre_weeks: Sequence[int],
                    groups: Optional[Mapping[str, Sequence[str]]] = None) -> pd.DataFrame:
    """Unit-level covariates for balance checks (static ACS-like + pre-period behaviour).

    For grouped units the static rates are population-weighted means of member counties.
    """
    cs = county_static.set_index("county_id")
    pre = panel[panel["week_idx"].isin(list(pre_weeks))]
    out = {}
    units = {u: [u] for u in pre["county_id"].unique() if u in cs.index}
    if groups:
        units.update({k: list(v) for k, v in groups.items()})
    for u, ids in units.items():
        c = cs.loc[ids]
        w = c["pop"] / c["pop"].sum()
        g = pre[pre["county_id"].isin(ids)]
        tot = g[["rated", "sales", "qs", "pop", "spend_search_brand", "paid_brand_rated",
                 "paid_nonbrand_rated"]].sum()
        nw = g["week_idx"].nunique()
        pop = c["pop"].sum()
        out[u] = {"pop": float(pop),
                  "median_hh_income": float((c["median_hh_income"] * w).sum()),
                  "pct_bachelors": float((c["pct_bachelors"] * w).sum()),
                  "density": float(pop / c["land_area_sqmi"].sum()),
                  "homeowner_rate": float((c["homeowner_rate"] * w).sum()),
                  "median_age": float((c["median_age"] * w).sum()),
                  "pre_rated_pc": float(tot["rated"] / nw / pop * 1e4),
                  "pre_sales_pc": float(tot["sales"] / nw / pop * 1e4),
                  "pre_close_rate": float(tot["sales"] / max(tot["rated"], 1)),
                  "pre_paid_share_rated": float((tot["paid_brand_rated"] + tot["paid_nonbrand_rated"])
                                                / max(tot["rated"], 1)),
                  "pre_brand_spend_pc": float(tot["spend_search_brand"] / nw / pop * 1e4)}
    return pd.DataFrame(out).T


def summarize_sources(panel: pd.DataFrame, weeks: Sequence[int], stage: str = "rated") -> pd.DataFrame:
    """Share of each last-touch source per county over a window (sum-based)."""
    g = panel[panel["week_idx"].isin(list(weeks))].groupby("county_id")[[f"{b}_{stage}" for b in SOURCES] + [stage]].sum()
    return g[[f"{b}_{stage}" for b in SOURCES]].div(g[stage], axis=0).rename(columns=lambda c: c.replace(f"_{stage}", ""))
