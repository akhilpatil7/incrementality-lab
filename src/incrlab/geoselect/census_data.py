"""Real US Census ACS pull for Massachusetts county subdivisions (towns/cities), with a
synthetic baseline-KPI overlay used only for market matching (see module docstring below).

Data source & API note
-----------------------
``api.census.gov``'s tabular Data API (the usual ``requests`` + ``?key=...`` pattern) now
**requires a registered API key for every data query** (metadata-only queries are exempt);
without one every request 302-redirects to ``missing_key.html``. We do not have a key and the
task says not to block indefinitely on that, so instead we pull the same underlying estimates
from the Census Bureau's public, key-free bulk distribution:

* ACS 2018-2022 5-Year *table-based Summary File* (pipe-delimited ``.dat``, one file per table) from
  ``www2.census.gov/programs-surveys/acs/summary_file/2022/table-based-SF/data/5YRData/`` - this is
  the exact same ACS 5-year estimates the Data API serves, just pre-joined per table instead of
  queried ad hoc.
* The matching geography header file (``Geos20225YR.txt``) to find Massachusetts county
  subdivisions (SUMLEVEL ``060``) and their ``GEO_ID`` / ``TL_GEO_ID``.
* The 2023 Census **Gazetteer** county-subdivisions file (``2023_Gaz_cousubs_national.zip``) for
  land area (``ALAND_SQMI``), used to build a density proxy for urbanicity.

These bulk files require no key and are not request-rate-limited the way the Data API's
metered endpoint is; they are the real ACS 5-year (2018-2022) estimates for real Massachusetts
geographies. Tables pulled: B01003 (population), B19013 (median household income), B01002
(median age), B02001 (race), B03002 (Hispanic/Latino ethnicity), B15003 (educational attainment).

Caching
-------
The merged, filtered extract is cached to ``data/census_raw/ma_towns_acs2022_cache.json`` so
repeat runs of the pipeline do not re-download ~380MB of national summary-file tables. Delete
the cache file (or pass ``force_refresh=True``) to re-pull.

Synthetic baseline KPI (clearly NOT Census data)
-------------------------------------------------
Real-world "geo experiment" matching uses a *business* KPI (e.g. conversion rate, revenue per
capita) in addition to demographics - we don't have Akhil's real marketing KPI data for every MA
town, so :func:`synth_baseline_kpi` manufactures a plausible one: a demographic-driven baseline
(negatively sloped in income/education per common conversion-rate patterns, scaled by urbanicity)
plus i.i.d. lognormal noise, seeded deterministically. This keeps clustering/matching realistic
without pretending we have real conversion data for every geo - it is used ONLY as one more
matching feature, never reported as a Census statistic.
"""
from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from ..config import REPO_ROOT
from .settings import GeoselectSettings, get_settings

CENSUS_RAW_DIR = REPO_ROOT / "data" / "census_raw"
CACHE_PATH = CENSUS_RAW_DIR / "ma_towns_acs2022_cache.json"

SF_BASE = "https://www2.census.gov/programs-surveys/acs/summary_file/{vintage}/table-based-SF/data/5YRData"
GEO_HEADER_URL = "https://www2.census.gov/programs-surveys/acs/summary_file/{vintage}/table-based-SF/documentation/Geos{vintage}5YR.txt"
GAZ_COUSUB_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_cousubs_national.zip"

TABLES = ("b01003", "b19013", "b01002", "b02001", "b03002", "b15003")
API_KEY_NOTE = (
    "api.census.gov Data API now requires a registered key for data queries (302 -> "
    "missing_key.html on every request without one); used the key-free bulk ACS Summary File + "
    "Gazetteer distribution on www2.census.gov instead (same underlying ACS 5-year estimates)."
)


def _download(url: str, dest: Path, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def fetch_raw_files(vintage: int = 2022, force: bool = False) -> dict:
    """Download the ACS summary-file tables, geo header, and cousub gazetteer if not cached."""
    CENSUS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    geo_header = CENSUS_RAW_DIR / f"Geos{vintage}5YR.txt"
    if force or not geo_header.exists():
        _download(GEO_HEADER_URL.format(vintage=vintage), geo_header)
    paths["geo_header"] = geo_header
    for t in TABLES:
        p = CENSUS_RAW_DIR / f"{t}.dat"
        if force or not p.exists():
            _download(f"{SF_BASE.format(vintage=vintage)}/acsdt5y{vintage}-{t}.dat", p)
        paths[t] = p
    gaz = CENSUS_RAW_DIR / "2023_Gaz_cousubs_national.txt"
    if force or not gaz.exists():
        r = requests.get(GAZ_COUSUB_URL, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(BytesIO(r.content)) as zf:
            name = [n for n in zf.namelist() if n.endswith(".txt")][0]
            gaz.write_bytes(zf.read(name))
    paths["gazetteer"] = gaz
    return paths


def build_ma_towns(st: Optional[GeoselectSettings] = None, force_refresh: bool = False) -> pd.DataFrame:
    """Real ACS 5yr (2018-2022) demographics for MA county subdivisions, pop >= threshold."""
    st = st or get_settings()
    paths = fetch_raw_files(st.acs_vintage, force=force_refresh)

    geos = pd.read_csv(paths["geo_header"], sep="|", dtype=str, encoding="utf-8-sig")
    ma = geos[(geos.STUSAB == st.state) & (geos.SUMLEVEL == "060") & (geos.COMPONENT == "00")]
    ma = ma[["GEO_ID", "NAME", "TL_GEO_ID"]].copy()

    def tbl(name, cols):
        d = pd.read_csv(paths[name], sep="|", dtype=str)
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        return d[["GEO_ID"] + cols]

    pop = tbl("b01003", ["B01003_E001"])
    inc = tbl("b19013", ["B19013_E001"])
    age = tbl("b01002", ["B01002_E001"])
    race = tbl("b02001", ["B02001_E001", "B02001_E002"])
    eth = tbl("b03002", ["B03002_E001", "B03002_E003", "B03002_E012"])
    edu = tbl("b15003", ["B15003_E001", "B15003_E022", "B15003_E023", "B15003_E024", "B15003_E025"])

    d = ma.merge(pop, on="GEO_ID", how="left").merge(inc, on="GEO_ID", how="left") \
          .merge(age, on="GEO_ID", how="left").merge(race, on="GEO_ID", how="left") \
          .merge(eth, on="GEO_ID", how="left").merge(edu, on="GEO_ID", how="left")

    d["pop"] = d["B01003_E001"]
    d["median_income"] = d["B19013_E001"]
    d["median_age"] = d["B01002_E001"]
    d["pct_white"] = d["B02001_E002"] / d["B02001_E001"]
    d["pct_hispanic"] = d["B03002_E012"] / d["B03002_E001"]
    d["pct_bachelors_plus"] = (d["B15003_E022"] + d["B15003_E023"] + d["B15003_E024"]
                               + d["B15003_E025"]) / d["B15003_E001"]

    gaz = pd.read_csv(paths["gazetteer"], sep="\t", dtype=str)
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz[gaz.USPS == st.state].copy()
    gaz["GEOID"] = gaz["GEOID"].str.strip()
    gaz["ALAND_SQMI"] = pd.to_numeric(gaz["ALAND_SQMI"], errors="coerce")
    gaz["lat"] = pd.to_numeric(gaz["INTPTLAT"], errors="coerce")
    gaz["lon"] = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")

    d["TL_GEO_ID"] = d["TL_GEO_ID"].str.strip()
    m = d.merge(gaz[["GEOID", "ALAND_SQMI", "lat", "lon"]], left_on="TL_GEO_ID", right_on="GEOID", how="left")
    m["density"] = m["pop"] / m["ALAND_SQMI"]
    m["town"] = m["NAME"].str.split(",").str[0].str.replace(" city", "", regex=False) \
                          .str.replace(" Town city", " Town", regex=False).str.replace(" town", "", regex=False)
    m["county"] = m["NAME"].str.split(",").str[1].str.strip()

    cols = ["TL_GEO_ID", "town", "county", "pop", "median_income", "median_age", "pct_white",
            "pct_hispanic", "pct_bachelors_plus", "density", "ALAND_SQMI", "lat", "lon"]
    f = m[cols].rename(columns={"TL_GEO_ID": "geoid", "ALAND_SQMI": "land_sq_mi"})
    f = f[f["pop"] >= st.pop_threshold].dropna(
        subset=["pop", "median_income", "median_age", "pct_white", "pct_hispanic",
                "pct_bachelors_plus", "density"]).sort_values("pop", ascending=False).reset_index(drop=True)
    return f


def load_or_fetch(st: Optional[GeoselectSettings] = None, force_refresh: bool = False) -> pd.DataFrame:
    """Load the cached MA-towns extract, or pull + cache it if missing / forced."""
    st = st or get_settings()
    if CACHE_PATH.exists() and not force_refresh:
        doc = json.loads(CACHE_PATH.read_text())
        df = pd.DataFrame(doc["geos"])
        return df
    df = build_ma_towns(st, force_refresh=force_refresh)
    doc = {
        "source": ("US Census ACS 2018-2022 5-Year Summary File (table-based, www2.census.gov, no API "
                   "key needed) + 2023 Census Gazetteer county-subdivisions file (land area). Geography: "
                   f"Massachusetts county subdivisions (towns/cities), ACS 5yr population >= "
                   f"{int(st.pop_threshold):,} (n={len(df)})."),
        "api_key_note": API_KEY_NOTE,
        "tables": {"pop": "B01003", "median_income": "B19013", "median_age": "B01002",
                  "race_total": "B02001_001", "race_white": "B02001_002",
                  "ethnicity_total": "B03002_001", "hispanic": "B03002_012",
                  "education_total": "B15003_001", "bachelors_or_higher": "B15003_022-025"},
        "acs_vintage": st.acs_vintage, "pop_threshold": st.pop_threshold,
        "n_geos": len(df), "geos": df.to_dict(orient="records"),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(doc, indent=2))
    return df


# ------------------------------------------------------------------------------------------
# Synthetic baseline KPI overlay (explicitly NOT Census data - see module docstring)
# ------------------------------------------------------------------------------------------
def synth_baseline_kpi(df: pd.DataFrame, st: Optional[GeoselectSettings] = None) -> pd.DataFrame:
    """Add a synthetic 'weekly_conversions_per_1k' baseline KPI correlated with real demographics.

    rate = base * (1 - 0.30 * z(income)) * (1 + 0.20 * z(bachelors)) * (1 + 0.15 * z(log density))
           * lognormal(0, kpi_noise_sd)
    This is a plausible, clearly-labelled-synthetic performance proxy, not a real business metric.
    """
    st = st or get_settings()
    rng = np.random.default_rng(st.kpi_seed)
    d = df.copy()
    d["log_pop"] = np.log(d["pop"])
    d["log_density"] = np.log(d["density"])

    def z(s):
        return (s - s.mean()) / s.std(ddof=0)

    base = 5.5  # conversions per 1k population per week, baseline
    rate = (base * (1 - 0.30 * z(d["median_income"])) * (1 + 0.20 * z(d["pct_bachelors_plus"]))
            * (1 + 0.15 * z(d["log_density"])))
    rate = rate.clip(lower=0.5)
    noise = np.exp(rng.normal(0, st.kpi_noise_sd, len(d)))
    d[st.kpi_name] = rate * noise
    d["baseline_kpi_rate"] = d[st.kpi_name]
    return d
