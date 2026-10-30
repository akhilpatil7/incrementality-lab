"""Synthetic geography: two fictional regions of counties with nested zips.

No real place names or real boundaries are used. Zips are points with ACS-like attributes;
counties are contiguous (Voronoi-style) groups of zips. Voronoi cell polygons are available
for choropleth-like maps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ADJACENCY_MILES_DEFAULT, Config, RegionSpec, get_config

EARTH_RADIUS_MI = 3958.7613
MI_PER_DEG_LAT = 69.0

_NAME_STEMS = [
    "Ashcombe", "Briarford", "Coldwater Vale", "Dunmere", "Elderbrook", "Fallowmere",
    "Glenhollow", "Harrowgate", "Ironwood", "Juniper Ridge", "Kestrel", "Larkspur",
    "Millbourne", "Northcote", "Oakhaven", "Pinecrest", "Quarry Hill", "Ravensholm",
    "Stillwater", "Thornbury", "Upland", "Valebrook", "Willowmere", "Yarrow", "Amberlane",
    "Bramblewick", "Cinderford", "Deepdale", "Eastwick Moor", "Foxmoor", "Greywell",
    "Hollowbrook", "Ivybridge Vale", "Kingsmere", "Lowmoor", "Marshfield Downs", "Netherby",
    "Orchard Vale", "Pebblecreek", "Redfern", "Silverbirch", "Tamberlin", "Underhill",
    "Westerholt", "Wrenfield", "Alderbank", "Blackthorn", "Copperleaf", "Driftwood",
    "Emberly", "Fernhollow", "Goldmoor", "Hearthstone", "Lanternfield", "Mossgrove",
]


@dataclass
class Geography:
    zips: pd.DataFrame  # one row per zip (static)
    counties: pd.DataFrame  # one row per county (static, aggregated from zips)


# --------------------------------------------------------------------------------------
# Distance helpers
# --------------------------------------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in miles (broadcasts)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_matrix_miles(lat: Sequence[float], lon: Sequence[float],
                          lat2: Optional[Sequence[float]] = None,
                          lon2: Optional[Sequence[float]] = None) -> np.ndarray:
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lat2 = lat if lat2 is None else np.asarray(lat2, float)
    lon2 = lon if lon2 is None else np.asarray(lon2, float)
    return haversine_miles(lat[:, None], lon[:, None], lat2[None, :], lon2[None, :])


def min_distance_to_set(zips: pd.DataFrame, target_ids: Iterable[str], id_col: str = "zip") -> pd.Series:
    """Miles from every zip to the nearest zip in ``target_ids`` (0 for targets)."""
    tgt = zips[zips[id_col].isin(set(target_ids))]
    if tgt.empty:
        return pd.Series(np.inf, index=zips.index)
    d = distance_matrix_miles(zips["lat"], zips["lon"], tgt["lat"], tgt["lon"]).min(axis=1)
    return pd.Series(d, index=zips.index)


def adjacency_graph(zips: pd.DataFrame, threshold_mi: float = ADJACENCY_MILES_DEFAULT,
                    id_col: str = "zip", connect_components: bool = True):
    """networkx Graph of zips; edge when centroid distance <= threshold (weight = miles).

    If ``connect_components`` the nearest pair between disconnected components is linked so
    BFS growth never dead-ends on an isolated rural zip.
    """
    import networkx as nx

    ids = zips[id_col].to_numpy()
    D = distance_matrix_miles(zips["lat"], zips["lon"])
    G = nx.Graph()
    G.add_nodes_from(ids)
    ii, jj = np.where(np.triu(D <= threshold_mi, k=1))
    G.add_weighted_edges_from((ids[i], ids[j], float(D[i, j])) for i, j in zip(ii, jj))
    if connect_components:
        idx = {z: k for k, z in enumerate(ids)}
        comps = [list(c) for c in nx.connected_components(G)]
        while len(comps) > 1:
            a = [idx[z] for z in comps[0]]
            b = [idx[z] for c in comps[1:] for z in c]
            sub = D[np.ix_(a, b)]
            i, j = np.unravel_index(np.argmin(sub), sub.shape)
            G.add_edge(ids[a[i]], ids[b[j]], weight=float(sub[i, j]))
            comps = [list(c) for c in nx.connected_components(G)]
    return G


def county_adjacency(zips: pd.DataFrame, threshold_mi: float = ADJACENCY_MILES_DEFAULT) -> pd.DataFrame:
    """Pairs of counties that have at least one zip pair within ``threshold_mi``."""
    D = distance_matrix_miles(zips["lat"], zips["lon"])
    c = zips["county_id"].to_numpy()
    ii, jj = np.where(np.triu(D <= threshold_mi, k=1))
    pairs = {tuple(sorted((c[i], c[j]))) for i, j in zip(ii, jj) if c[i] != c[j]}
    return pd.DataFrame(sorted(pairs), columns=["county_a", "county_b"])


# --------------------------------------------------------------------------------------
# Voronoi polygons (bonus: choropleth-like maps)
# --------------------------------------------------------------------------------------
def voronoi_polygons(lon: Sequence[float], lat: Sequence[float],
                     bbox: Optional[tuple] = None, pad: float = 0.05) -> List[np.ndarray]:
    """Bounded Voronoi cells (list of (k,2) lon/lat arrays) via the mirror trick.

    Points are reflected across the four bbox edges so every original cell is finite and
    clipped (approximately) to the bbox.
    """
    from scipy.spatial import Voronoi

    pts = np.column_stack([np.asarray(lon, float), np.asarray(lat, float)])
    if bbox is None:
        x0, y0 = pts.min(0) - pad
        x1, y1 = pts.max(0) + pad
    else:
        x0, y0, x1, y1 = bbox
    mirrors = [pts,
               np.column_stack([2 * x0 - pts[:, 0], pts[:, 1]]),
               np.column_stack([2 * x1 - pts[:, 0], pts[:, 1]]),
               np.column_stack([pts[:, 0], 2 * y0 - pts[:, 1]]),
               np.column_stack([pts[:, 0], 2 * y1 - pts[:, 1]])]
    vor = Voronoi(np.vstack(mirrors))
    polys = []
    for i in range(len(pts)):
        region = vor.regions[vor.point_region[i]]
        polys.append(vor.vertices[region] if region and -1 not in region else np.empty((0, 2)))
    return polys


def polygon_area_sq_miles(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return np.nan
    lat0 = np.radians(poly[:, 1].mean())
    x = poly[:, 0] * MI_PER_DEG_LAT * np.cos(lat0)
    y = poly[:, 1] * MI_PER_DEG_LAT
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


# --------------------------------------------------------------------------------------
# Granularity-safe aggregation ("never average averages")
# --------------------------------------------------------------------------------------
ACS_WEIGHTED = {  # feature -> weight column for population-weighted means
    "median_hh_income": "households", "per_capita_income": "pop", "pct_bachelors": "pop",
    "employment_rate": "pop", "lfpr": "pop", "median_age": "pop", "homeowner_rate": "households",
    "zip_propensity": "pop",
}
ACS_SUMMED = ("pop", "households", "land_area_sqmi")


def aggregate_static(zips: pd.DataFrame, by: str = "county_id",
                     weighted: Optional[Dict[str, str]] = None,
                     summed: Sequence[str] = ACS_SUMMED) -> pd.DataFrame:
    """Aggregate zip-static attributes to ``by`` units: sums for counts, weighted means for rates.

    Lat/lon become population-weighted centroids. Zip-static columns are de-duplicated first
    (safe to pass a weekly panel).
    """
    weighted = dict(ACS_WEIGHTED if weighted is None else weighted)
    z = zips.drop_duplicates("zip") if "zip" in zips.columns else zips
    weighted.update({"lat": "pop", "lon": "pop"})
    cols = [c for c in summed if c in z.columns]
    out = z.groupby(by)[cols].sum()
    for col, w in weighted.items():
        if col in z.columns and w in z.columns:
            num = (z[col] * z[w]).groupby(z[by]).sum()
            out[col] = num / z.groupby(by)[w].sum()
    out["n_zips"] = z.groupby(by).size()
    if "pop" in out and "land_area_sqmi" in out:
        out["density"] = out["pop"] / out["land_area_sqmi"]
    return out.reset_index()


def aggregate_zips(panel: pd.DataFrame, mapping: Optional[pd.Series] = None, by: str = "county_id",
                   time_col: str = "week_start", sum_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Sum weekly zip outcomes to a coarser unit (county, patch, cluster).

    ``mapping`` (index=zip, value=unit) overrides ``by`` - e.g. a patch / cluster assignment.
    Only additive count/spend columns should be summed; recompute rates afterwards.
    Columns ending in ``_month`` (monthly values repeated on weekly rows) are excluded.
    """
    df = panel
    if mapping is not None:
        df = df.assign(**{by: df["zip"].map(mapping)}).dropna(subset=[by])
    if sum_cols is None:
        sum_cols = [c for c in df.select_dtypes("number").columns
                    if c not in ("week_idx", "lat", "lon") and not c.endswith("_month")
                    and not c.endswith("_rate") and not c.startswith("is_")]
    return df.groupby([by, time_col], as_index=False)[list(sum_cols)].sum()


def monthly_media_sum(panel: pd.DataFrame, cols: Sequence[str], by: str = "zip",
                      month_col: str = "month_start") -> pd.DataFrame:
    """Sum monthly media columns that are repeated on each weekly row (dedupe first)."""
    m = panel.drop_duplicates([month_col, "zip"])
    return m.groupby([by, month_col], as_index=False)[list(cols)].sum()


# --------------------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------------------
def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def _region_geo(spec: RegionSpec, rng: np.random.Generator, name_pool: List[str]) -> pd.DataFrame:
    lat_span = spec.height_mi / MI_PER_DEG_LAT
    lon_span = spec.width_mi / (MI_PER_DEG_LAT * np.cos(np.radians(spec.center_lat)))
    lat0, lon0 = spec.center_lat - lat_span / 2, spec.center_lon - lon_span / 2

    def to_mi(lat, lon):
        return ((lon - lon0) * MI_PER_DEG_LAT * np.cos(np.radians(spec.center_lat)),
                (lat - lat0) * MI_PER_DEG_LAT)

    # metro cores (one dominant)
    m_x = rng.uniform(0.15, 0.85, spec.n_metros) * spec.width_mi
    m_y = rng.uniform(0.15, 0.85, spec.n_metros) * spec.height_mi
    m_w = np.sort(rng.dirichlet(np.ones(spec.n_metros) * 2))[::-1]
    m_sd = rng.uniform(9, 16, spec.n_metros)

    # zip points: denser around metros
    n_urban = int(spec.n_zips * 0.55)
    k = rng.choice(spec.n_metros, n_urban, p=m_w)
    xs, ys = [], []
    for i in range(spec.n_zips):
        while True:
            if i < n_urban:
                x = rng.normal(m_x[k[i]], m_sd[k[i]])
                y = rng.normal(m_y[k[i]], m_sd[k[i]])
            else:
                x = rng.uniform(0, spec.width_mi)
                y = rng.uniform(0, spec.height_mi)
            if 0 <= x <= spec.width_mi and 0 <= y <= spec.height_mi:
                break
        xs.append(x)
        ys.append(y)
    x = np.array(xs)
    y = np.array(ys)
    d_metro = np.sqrt((x[:, None] - m_x) ** 2 + (y[:, None] - m_y) ** 2)
    urban = (np.exp(-0.5 * (d_metro / (m_sd * 1.4)) ** 2) * (0.5 + m_w)).max(1)
    urban = np.clip(urban / urban.max(), 0, 1)

    # county seeds: roughly uniform in space (Lloyd-relaxed) -> urban counties hold many people
    cand = np.column_stack([rng.uniform(0, spec.width_mi, 4000), rng.uniform(0, spec.height_mi, 4000)])
    seeds = cand[rng.choice(len(cand), spec.n_counties, replace=False)]
    for _ in range(6):
        lab = np.argmin(((cand[:, None, :] - seeds[None]) ** 2).sum(-1), 1)
        seeds = np.array([cand[lab == j].mean(0) if np.any(lab == j) else seeds[j]
                          for j in range(spec.n_counties)])
    pts = np.column_stack([x, y])
    county = np.argmin(((pts[:, None, :] - seeds[None]) ** 2).sum(-1), 1)
    # every county needs >= 3 zips: reassign nearest zips from big counties
    for j in range(spec.n_counties):
        while (county == j).sum() < 3:
            d = ((pts - seeds[j]) ** 2).sum(1)
            donors = np.bincount(county, minlength=spec.n_counties)
            order = np.argsort(d)
            for z in order:
                if county[z] != j and donors[county[z]] > 4:
                    county[z] = j
                    break

    # socio-economic latent: county component + metro gradient + zip noise
    county_ses = rng.normal(0, 0.55, spec.n_counties)
    ses = spec.ses_shift + county_ses[county] + 0.6 * urban + rng.normal(0, 0.45, spec.n_zips)
    ses = (ses - ses.mean()) / ses.std() + spec.ses_shift

    # population: lognormal, bigger in urban zips; scaled to the region total
    pop = np.exp(rng.normal(0, 1.0, spec.n_zips) + 1.3 * urban)
    pop = pop / pop.sum() * spec.pop_total
    pop = np.maximum(np.round(pop), 150).astype(int)

    lon = lon0 + x / (MI_PER_DEG_LAT * np.cos(np.radians(spec.center_lat)))
    lat = lat0 + y / MI_PER_DEG_LAT

    hh_size = np.clip(2.75 - 0.45 * urban + rng.normal(0, 0.12, spec.n_zips), 1.8, 3.4)
    income = np.clip(76_000 * np.exp(0.24 * ses + rng.normal(0, 0.08, spec.n_zips)), 32_000, 240_000)
    df = pd.DataFrame({
        "region": spec.code,
        "region_name": spec.name,
        "county_idx": county,
        "lat": lat, "lon": lon,
        "pop": pop,
        "households": np.round(pop / hh_size).astype(int),
        "median_hh_income": np.round(income, -2),
        "per_capita_income": np.round(income * np.clip(0.44 + 0.03 * ses, 0.3, 0.6) / (hh_size / 2.5), -1),
        "pct_bachelors": np.clip(_logistic(-0.55 + 0.75 * ses + 0.35 * urban
                                           + rng.normal(0, 0.12, spec.n_zips)), 0.08, 0.85),
        "employment_rate": np.clip(0.605 + 0.022 * ses + rng.normal(0, 0.014, spec.n_zips), 0.50, 0.72),
        "median_age": np.clip(41 - 4.5 * urban + 1.2 * ses + rng.normal(0, 2.6, spec.n_zips), 26, 58),
        "homeowner_rate": np.clip(_logistic(1.1 - 1.9 * urban + 0.45 * ses
                                            + rng.normal(0, 0.25, spec.n_zips)), 0.15, 0.95),
        "urbanicity": urban,
        "ses_index": ses,
    })
    df["lfpr"] = np.clip(df["employment_rate"] / 0.955 + rng.normal(0, 0.01, spec.n_zips), 0.5, 0.8)

    # land area from Voronoi cells (miles), density
    polys = voronoi_polygons(df["lon"], df["lat"],
                             bbox=(lon0, lat0, lon0 + lon_span, lat0 + lat_span))
    area = np.array([polygon_area_sq_miles(p) for p in polys])
    area = np.where(np.isfinite(area) & (area > 0.2), area, np.nanmedian(area))
    df["land_area_sqmi"] = area
    df["density"] = df["pop"] / df["land_area_sqmi"]

    names = [name_pool.pop(0) for _ in range(spec.n_counties)]
    df["county"] = [f"{names[c]} County" for c in county]
    df["county_id"] = [f"{spec.code}-C{c + 1:02d}" for c in county]
    df = df.sort_values(["county_id", "lat"]).reset_index(drop=True)
    df.insert(0, "zip", [f"{spec.code}{i + 1:04d}" for i in range(len(df))])
    return df.drop(columns=["county_idx"])


def build_geography(cfg: Optional[Config] = None) -> Geography:
    """Deterministic synthetic geography for all configured regions."""
    cfg = cfg or get_config()
    ss = np.random.SeedSequence([cfg.seed, 101])
    pool = list(_NAME_STEMS)
    np.random.default_rng(ss.spawn(1)[0]).shuffle(pool)
    frames = []
    for k, spec in enumerate(cfg.regions):
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 202, k]))
        frames.append(_region_geo(spec, rng, pool))
    zips = pd.concat(frames, ignore_index=True)

    # latent quote propensity: lognormal, negatively related to SES (income/education)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 303]))
    wp = cfg.world
    logp = wp.propensity_income_slope * zips["ses_index"].to_numpy() \
        + rng.normal(0, wp.propensity_sigma, len(zips))
    prop = np.exp(logp)
    zips["zip_propensity"] = prop / np.average(prop, weights=zips["pop"])
    zips["dsp_affinity"] = np.exp(rng.normal(0, 0.25, len(zips)))

    # channel serving flags
    zips["dsp_a_served"] = 0
    zips["dsp_b_inclusion"] = 0
    for reg, col, ch in (("NV", "dsp_a_served", "dsp_a"), ("EM", "dsp_b_inclusion", "dsp_b")):
        m = (zips["region"] == reg).to_numpy()
        share = cfg.presence[ch][reg]
        n_in = int(round(share * m.sum()))
        # vendor serves the "quality" zips: score = propensity x affinity x pop^0.3 x noise
        score = (zips.loc[m, "zip_propensity"] * zips.loc[m, "dsp_affinity"]
                 * zips.loc[m, "pop"] ** 0.3 * np.exp(rng.normal(0, 0.5, m.sum())))
        chosen = score.sort_values(ascending=False).index[:n_in]
        zips.loc[chosen, col] = 1

    # concurrent (unrelated) test zips: a contiguous blob in the concurrent-test region
    from .config import CONCURRENT_TEST
    zips["concurrent_test_flag"] = 0
    m = zips["region"] == CONCURRENT_TEST["region"]
    sub = zips[m]
    centre = sub.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
    d = haversine_miles(sub["lat"], sub["lon"], centre["lat"], centre["lon"])
    n_c = max(3, int(round(CONCURRENT_TEST["share_of_zips"] * len(sub))))
    zips.loc[d.sort_values().index[:n_c], "concurrent_test_flag"] = 1

    counties = aggregate_static(zips, by="county_id")
    meta = zips.drop_duplicates("county_id")[["county_id", "county", "region", "region_name"]]
    counties = meta.merge(counties, on="county_id").sort_values("county_id").reset_index(drop=True)
    return Geography(zips=zips, counties=counties)


def zips_to_counties(zips: pd.DataFrame, zip_ids: Iterable[str]) -> List[str]:
    return sorted(zips.loc[zips["zip"].isin(set(zip_ids)), "county_id"].unique())


def counties_to_zips(zips: pd.DataFrame, county_ids: Iterable[str]) -> List[str]:
    return zips.loc[zips["county_id"].isin(set(county_ids)), "zip"].tolist()
