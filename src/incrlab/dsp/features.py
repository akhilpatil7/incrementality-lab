"""Granularity-safe data preparation shared by both DSP designs and the readout.

Three rules ("never average averages"):
1. zip-static columns (ACS features, lat/lon, pop) are de-duplicated per zip, then aggregated
   with population (or household) weights;
2. monthly media columns (``*_month``) repeat on every weekly row -> de-duplicate on
   (month_start, zip) before summing;
3. weekly outcomes are summed directly; every rate is recomputed from summed numerators and
   denominators at the target grain.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .. import geo as G

COUNT_COLS = ("visits", "qs", "rated", "sales", "lt_dsp_qs", "vendor_conv_qs_dsp_a", "vendor_conv_qs_dsp_b",
              "spend_dsp_a", "spend_dsp_b", "impressions_dsp_a", "impressions_dsp_b", "clicks_dsp_a",
              "clicks_dsp_b", "pop")


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute rates from summed columns (in place-safe copy)."""
    d = df.copy()
    qs = d["qs"].astype(float)
    d["rated_rate"] = np.where(qs > 0, d["rated"] / qs.where(qs > 0, 1), np.nan)
    d["close_rate"] = np.where(d["rated"] > 0, d["sales"] / d["rated"].where(d["rated"] > 0, 1), np.nan)
    if "pop" in d:
        d["qs_per_capita"] = d["qs"] / d["pop"].where(d["pop"] > 0, np.nan)
    for v in ("a", "b"):
        c = f"vendor_conv_qs_dsp_{v}"
        if c in d:
            d[f"dsp_share_{v}"] = np.where(qs > 0, d[c] / qs.where(qs > 0, 1), np.nan)
    return d


def static_features(panel: pd.DataFrame, zips: pd.DataFrame, weeks: Sequence[int],
                    region: Optional[str] = None) -> pd.DataFrame:
    """One row per zip: ACS features (rule 1) + design-window totals (rule 3) + recomputed rates."""
    z = zips.drop_duplicates("zip").set_index("zip")
    if region is not None:
        z = z[z["region"] == region]
    p = panel[panel["week_idx"].isin(list(weeks)) & panel["zip"].isin(z.index)]
    cols = [c for c in COUNT_COLS if c in p.columns and c != "pop"]
    tot = p.groupby("zip")[cols].sum()
    out = z.join(tot, how="left").fillna({c: 0 for c in cols})
    out["n_weeks"] = len(list(weeks))
    out = add_rates(out.reset_index())
    out["qs_per_capita"] = out["qs"] / out["pop"] / out["n_weeks"]  # weekly per capita
    return out


def aggregate_profile(feat: pd.DataFrame, ids: Iterable[str], features: Sequence[str]) -> pd.Series:
    """Population-weighted means of ACS features; rates recomputed from summed totals."""
    s = feat[feat["zip"].isin(set(ids))]
    w = s["pop"].astype(float)
    out = {}
    for f in features:
        if f == "qs_per_capita":
            out[f] = s["qs"].sum() / w.sum() / float(s["n_weeks"].iloc[0])
        elif f == "rated_rate":
            out[f] = s["rated"].sum() / max(s["qs"].sum(), 1)
        elif f == "close_rate":
            out[f] = s["sales"].sum() / max(s["rated"].sum(), 1)
        elif f.startswith("dsp_share"):
            v = f.split("_")[-1]
            out[f] = s[f"vendor_conv_qs_dsp_{v}"].sum() / max(s["qs"].sum(), 1)
        else:
            out[f] = float(np.average(s[f], weights=w))
    out["pop"] = float(w.sum())
    out["n_zips"] = int(len(s))
    return pd.Series(out)


def weekly_by_group(panel: pd.DataFrame, mapping: Mapping[str, str], weeks: Optional[Sequence[int]] = None,
                    cols: Sequence[str] = ("qs", "rated", "sales", "visits", "vendor_conv_qs_dsp_a",
                                           "vendor_conv_qs_dsp_b", "lt_dsp_qs", "spend_dsp_a",
                                           "spend_dsp_b", "pop")) -> pd.DataFrame:
    """Sum weekly zip outcomes to groups (patch / cluster / arm) -> long frame with rates."""
    p = panel if weeks is None else panel[panel["week_idx"].isin(list(weeks))]
    p = p[p["zip"].isin(mapping.keys())]
    cols = [c for c in cols if c in p.columns]
    g = G.aggregate_zips(p, mapping=pd.Series(mapping), by="group", time_col="week_idx", sum_cols=cols)
    return add_rates(g)


def series(panel: pd.DataFrame, ids: Iterable[str], col: str, weeks: Optional[Sequence[int]] = None) -> pd.Series:
    """Weekly sum of ``col`` over a zip set (index = week_idx)."""
    ids = set(ids)
    p = panel[panel["zip"].isin(ids)]
    if weeks is not None:
        p = p[p["week_idx"].isin(list(weeks))]
    s = p.groupby("week_idx")[col].sum().astype(float)
    s.index = s.index.astype(int)
    return s


def monthly_media(panel: pd.DataFrame, ids: Iterable[str], vendor: str) -> pd.DataFrame:
    """Monthly media totals for a zip set via the de-duplication rule (rule 2)."""
    cols = [f"spend_dsp_{vendor}_month", f"impressions_dsp_{vendor}_month", f"clicks_dsp_{vendor}_month",
            f"vendor_conv_qs_dsp_{vendor}_month"]
    p = panel[panel["zip"].isin(set(ids))]
    m = G.monthly_media_sum(p, cols)
    return m.groupby("month_start")[cols].sum()


def wow(s: pd.Series) -> pd.Series:
    return s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def standardized_mean_diff(feat: pd.DataFrame, t_ids: Iterable[str], c_ids: Iterable[str],
                           features: Sequence[str], c_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """SMD balance table: (mean_T - mean_C) / sqrt((var_T + var_C)/2) on zip-level, pop-weighted.

    ``c_weights`` (zip -> weight) lets a synthetic control's donor weights define the control
    distribution (weights are multiplied by population).
    """
    rows = []
    t = feat[feat["zip"].isin(set(t_ids))]
    c = feat[feat["zip"].isin(set(c_ids))]
    wt = t["pop"].astype(float).to_numpy()
    wc = c["pop"].astype(float).to_numpy()
    if c_weights is not None:
        wc = wc * c["zip"].map(c_weights).fillna(0).to_numpy()
    for f in features:
        xt, xc = t[f].astype(float).to_numpy(), c[f].astype(float).to_numpy()
        mt, mc = np.average(xt, weights=wt), np.average(xc, weights=wc)
        vt = np.average((xt - mt) ** 2, weights=wt)
        vc = np.average((xc - mc) ** 2, weights=wc)
        sd = np.sqrt((vt + vc) / 2)
        smd = (mt - mc) / sd if sd > 0 else 0.0
        rows.append({"feature": f, "label": _label(f), "treatment_mean": mt, "control_mean": mc,
                     "pct_diff": mt / mc - 1 if mc else np.nan, "smd": smd, "abs_smd": abs(smd),
                     "flag": "ok (<0.10)" if abs(smd) < 0.10 else ("caution (<0.25)" if abs(smd) < 0.25
                                                                   else "imbalanced (>=0.25)")})
    return pd.DataFrame(rows)


class ZipCube:
    """Dense (zip x week) arrays of additive columns for fast subset sums.

    Only additive weekly columns are stored (rule 3). ``sum(ids, col)`` returns the weekly
    series of the subset; rates must be formed from two such sums.
    """

    def __init__(self, panel: pd.DataFrame, cols: Sequence[str] = ("qs", "rated", "sales", "visits",
                                                                  "vendor_conv_qs_dsp_a", "vendor_conv_qs_dsp_b",
                                                                  "lt_dsp_qs", "spend_dsp_a", "spend_dsp_b",
                                                                  "impressions_dsp_a", "impressions_dsp_b")):
        p = panel.sort_values(["zip", "week_idx"])
        self.zips = np.array(sorted(p["zip"].unique()))
        self.weeks = np.array(sorted(p["week_idx"].unique()), dtype=int)
        self.pos = {z: i for i, z in enumerate(self.zips)}
        nz, nw = len(self.zips), len(self.weeks)
        if len(p) != nz * nw:
            raise ValueError("ZipCube needs a balanced zip x week panel")
        self.arr = {c: p[c].to_numpy(float).reshape(nz, nw) for c in cols if c in p.columns}
        pop = p.drop_duplicates("zip").set_index("zip")["pop"].reindex(self.zips)
        self.pop = pop.to_numpy(float)

    def idx(self, ids: Iterable[str]) -> np.ndarray:
        return np.array([self.pos[z] for z in ids if z in self.pos], dtype=int)

    def sum(self, ids: Iterable[str], col: str, weights: Optional[np.ndarray] = None) -> pd.Series:
        ii = self.idx(ids)
        a = self.arr[col][ii]
        v = a.sum(0) if weights is None else (a * np.asarray(weights, float)[:, None]).sum(0)
        return pd.Series(v, index=pd.Index(self.weeks, name="week_idx"))

    def pop_of(self, ids: Iterable[str]) -> float:
        return float(self.pop[self.idx(ids)].sum())

    def per_capita(self, ids: Iterable[str], col: str) -> pd.Series:
        return self.sum(ids, col) / max(self.pop_of(ids), 1.0)


def _label(f: str) -> str:
    from .params import FEATURE_LABELS
    return FEATURE_LABELS.get(f, f)
