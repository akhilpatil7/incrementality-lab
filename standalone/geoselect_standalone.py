#!/usr/bin/env python3
"""
geoselect_standalone.py - a self-contained, end-to-end geo-experiment lifecycle tool.

Does NOT import incrlab. Runnable on its own with just: requests, numpy, pandas, scipy,
scikit-learn, matplotlib (all pip-installable; no project-specific package needed).

Pipeline (same five stages as incrlab.geoselect, logic duplicated so this file is portable):
  1. Pull REAL US Census ACS 2018-2022 5-year demographics for Massachusetts county
     subdivisions (towns/cities) with population >= POP_THRESHOLD, from the Census Bureau's
     public, key-free bulk ACS Summary File + Gazetteer distribution (api.census.gov's tabular
     Data API now requires a registered key for data queries - see NOTE below).
  2. K-means cluster the real geos on demographics + a documented SYNTHETIC baseline-KPI proxy.
  3. Select the closest treatment/control pair within a cluster and run a closed-form DiD power
     analysis (MDE, power-vs-weeks curve).
  4. Simulate a daily test window for both markets with a known injected true lift, and take
     weekly sequential-monitoring DiD looks.
  5. Final DiD readout (estimate + CI vs the known ground-truth lift).

Usage:
    python geoselect_standalone.py [--out-dir OUT_DIR] [--pop-threshold 40000]
                                   [--true-lift 0.12] [--test-weeks 8] [--force-refresh]

Outputs (under --out-dir, default ./geoselect_standalone_out/):
    census_raw/ma_towns_acs_cache.json   - cached real Census extract
    results.json                         - full numeric results (headline + design + validation)
    readout.md                           - markdown readout (same spirit as incrlab's reports/readouts)
    figures/01_clusters.png
    figures/02_power_ladder.png
    figures/03_monitoring.png
    figures/04_final_readout.png

NOTE on Census API access
--------------------------
api.census.gov's tabular Data API (``?get=...&for=...&key=...``) now requires a registered API
key for every *data* query; without one every request 302-redirects to
``api.census.gov/data/missing_key.html``. This script was built and run without a key, so it
pulls the identical underlying ACS 2018-2022 5-year estimates from the Census Bureau's public,
key-free bulk distribution instead:
  * ACS Summary File (table-based):
    https://www2.census.gov/programs-surveys/acs/summary_file/2022/table-based-SF/data/5YRData/
  * Matching geography header: .../table-based-SF/documentation/Geos20225YR.txt
  * 2023 Gazetteer county-subdivisions file (land area, for a density/urbanicity proxy):
    https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_cousubs_national.zip
No rate limit or missing-data issue was hit on this bulk path; all demographic figures below are
real public ACS data for real Massachusetts towns/cities, not synthetic geography.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# =====================================================================================
# Config (all knobs in one place, standalone style)
# =====================================================================================
GEO_HEADER_URL = "https://www2.census.gov/programs-surveys/acs/summary_file/{v}/table-based-SF/documentation/Geos{v}5YR.txt"
SF_BASE = "https://www2.census.gov/programs-surveys/acs/summary_file/{v}/table-based-SF/data/5YRData"
GAZ_COUSUB_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_cousubs_national.zip"
TABLES = ("b01003", "b19013", "b01002", "b02001", "b03002", "b15003")
ACS_VINTAGE = 2022

FEATURES = ("median_income", "median_age", "pct_bachelors_plus", "pct_white", "pct_hispanic",
           "log_density", "log_pop", "baseline_kpi_rate")

PALETTE = {"navy": "#1F3A5F", "orange": "#E76F51", "teal": "#2A9D8F", "green": "#2E8B57", "slate": "#5C677D"}
SERIES = ["#1F3A5F", "#E76F51", "#2A9D8F", "#E9C46A", "#6C5B7B", "#5C677D", "#7FB3D5", "#C0392B"]


# =====================================================================================
# 1. Real Census ACS pull (duplicated minimal logic, no incrlab import)
# =====================================================================================
def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def fetch_census_raw(raw_dir: Path, force: bool = False) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    geo_header = raw_dir / f"Geos{ACS_VINTAGE}5YR.txt"
    if force or not geo_header.exists():
        _download(GEO_HEADER_URL.format(v=ACS_VINTAGE), geo_header)
    paths["geo_header"] = geo_header
    for t in TABLES:
        p = raw_dir / f"{t}.dat"
        if force or not p.exists():
            _download(f"{SF_BASE.format(v=ACS_VINTAGE)}/acsdt5y{ACS_VINTAGE}-{t}.dat", p)
        paths[t] = p
    gaz = raw_dir / "2023_Gaz_cousubs_national.txt"
    if force or not gaz.exists():
        r = requests.get(GAZ_COUSUB_URL, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(BytesIO(r.content)) as zf:
            name = [n for n in zf.namelist() if n.endswith(".txt")][0]
            gaz.write_bytes(zf.read(name))
    paths["gazetteer"] = gaz
    return paths


def build_ma_towns(raw_dir: Path, pop_threshold: float, force: bool = False) -> pd.DataFrame:
    """Real ACS 5yr (2018-2022) demographics for MA county subdivisions, pop >= threshold."""
    paths = fetch_census_raw(raw_dir, force=force)
    geos = pd.read_csv(paths["geo_header"], sep="|", dtype=str, encoding="utf-8-sig")
    ma = geos[(geos.STUSAB == "MA") & (geos.SUMLEVEL == "060") & (geos.COMPONENT == "00")]
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

    d = (ma.merge(pop, on="GEO_ID", how="left").merge(inc, on="GEO_ID", how="left")
          .merge(age, on="GEO_ID", how="left").merge(race, on="GEO_ID", how="left")
          .merge(eth, on="GEO_ID", how="left").merge(edu, on="GEO_ID", how="left"))

    d["pop"] = d["B01003_E001"]
    d["median_income"] = d["B19013_E001"]
    d["median_age"] = d["B01002_E001"]
    d["pct_white"] = d["B02001_E002"] / d["B02001_E001"]
    d["pct_hispanic"] = d["B03002_E012"] / d["B03002_E001"]
    d["pct_bachelors_plus"] = (d["B15003_E022"] + d["B15003_E023"] + d["B15003_E024"]
                               + d["B15003_E025"]) / d["B15003_E001"]

    gaz = pd.read_csv(paths["gazetteer"], sep="\t", dtype=str)
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz[gaz.USPS == "MA"].copy()
    gaz["GEOID"] = gaz["GEOID"].str.strip()
    gaz["ALAND_SQMI"] = pd.to_numeric(gaz["ALAND_SQMI"], errors="coerce")
    gaz["lat"] = pd.to_numeric(gaz["INTPTLAT"], errors="coerce")
    gaz["lon"] = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")

    d["TL_GEO_ID"] = d["TL_GEO_ID"].str.strip()
    m = d.merge(gaz[["GEOID", "ALAND_SQMI", "lat", "lon"]], left_on="TL_GEO_ID", right_on="GEOID", how="left")
    m["density"] = m["pop"] / m["ALAND_SQMI"]
    m["town"] = (m["NAME"].str.split(",").str[0].str.replace(" city", "", regex=False)
                .str.replace(" Town city", " Town", regex=False).str.replace(" town", "", regex=False))
    m["county"] = m["NAME"].str.split(",").str[1].str.strip()

    cols = ["TL_GEO_ID", "town", "county", "pop", "median_income", "median_age", "pct_white",
            "pct_hispanic", "pct_bachelors_plus", "density", "ALAND_SQMI", "lat", "lon"]
    f = m[cols].rename(columns={"TL_GEO_ID": "geoid", "ALAND_SQMI": "land_sq_mi"})
    f = f[f["pop"] >= pop_threshold].dropna(
        subset=["pop", "median_income", "median_age", "pct_white", "pct_hispanic",
               "pct_bachelors_plus", "density"]).sort_values("pop", ascending=False).reset_index(drop=True)
    return f


def load_or_fetch_census(raw_dir: Path, pop_threshold: float, force: bool = False) -> pd.DataFrame:
    cache_path = raw_dir / "ma_towns_acs_cache.json"
    if cache_path.exists() and not force:
        return pd.DataFrame(json.loads(cache_path.read_text())["geos"])
    df = build_ma_towns(raw_dir, pop_threshold, force=force)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "source": ("US Census ACS 2018-2022 5-Year Summary File (bulk, no API key) + 2023 Census "
                  f"Gazetteer cousubs file. MA county subdivisions, pop >= {int(pop_threshold):,} "
                  f"(n={len(df)})."),
        "pop_threshold": pop_threshold, "n_geos": len(df), "geos": df.to_dict(orient="records"),
    }, indent=2))
    return df


def add_synthetic_baseline_kpi(df: pd.DataFrame, seed: int = 2026, noise_sd: float = 0.06) -> pd.DataFrame:
    """SYNTHETIC performance proxy correlated with real demographics (NOT Census data) -
    see module docstring. Used only so matching has a performance dimension beyond demographics."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["log_pop"] = np.log(d["pop"])
    d["log_density"] = np.log(d["density"])

    def z(s):
        return (s - s.mean()) / s.std(ddof=0)

    base = 5.5
    rate = (base * (1 - 0.30 * z(d["median_income"])) * (1 + 0.20 * z(d["pct_bachelors_plus"]))
            * (1 + 0.15 * z(d["log_density"]))).clip(lower=0.5)
    d["baseline_kpi_rate"] = rate * np.exp(rng.normal(0, noise_sd, len(d)))
    return d


# =====================================================================================
# 2. Clustering + matched-pair selection
# =====================================================================================
def cluster_geos(df: pd.DataFrame, k: int = 5, seed: int = 7) -> pd.DataFrame:
    d = df.copy()
    X = d[list(FEATURES)].to_numpy(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    d["cluster"] = km.fit_predict(Xs)
    d.attrs["Xs"] = Xs
    return d


def select_matched_pair(d: pd.DataFrame) -> dict:
    Xs = d.attrs["Xs"]
    sizes = d["cluster"].value_counts()
    eligible = sizes[sizes >= 2].index.tolist()
    if not eligible:
        raise RuntimeError("no cluster with >=2 members - widen pop_threshold or lower k")
    best = None
    for c in eligible:
        idx = d.index[d["cluster"] == c].tolist()
        sub = Xs[idx]
        dmat = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)
        i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
        dist = float(dmat[i, j])
        if best is None or dist < best["dist"]:
            best = {"cluster": int(c), "dist": dist, "i": idx[i], "j": idx[j], "cluster_size": len(idx)}
    ti, cj = best["i"], best["j"]
    if d.loc[ti, "pop"] < d.loc[cj, "pop"]:
        ti, cj = cj, ti
    treat, ctrl = d.loc[ti], d.loc[cj]
    return {
        "cluster": best["cluster"], "cluster_size": best["cluster_size"], "standardized_distance": best["dist"],
        "treatment": {"town": treat["town"], "geoid": treat["geoid"], "pop": float(treat["pop"])},
        "control": {"town": ctrl["town"], "geoid": ctrl["geoid"], "pop": float(ctrl["pop"])},
        "feature_comparison": {f: {"treatment": float(treat[f]), "control": float(ctrl[f])} for f in FEATURES},
    }


# =====================================================================================
# 3. Power analysis (closed-form AR(1)-inflated DiD, same formula as incrlab.common.power)
# =====================================================================================
def _z(alpha, two_sided):
    return stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)


def did_mde(sigma, n_post, n_pre=None, alpha=0.10, power=0.80, rho=0.0):
    n_pre = n_post if n_pre is None else n_pre
    infl = np.sqrt((1 + rho) / (1 - rho)) if rho else 1.0
    se = sigma * np.sqrt(1.0 / n_post + 1.0 / n_pre) * infl
    return float((_z(alpha, False) + stats.norm.ppf(power)) * se)


def did_power(lift, sigma, n_post, n_pre=None, alpha=0.10, rho=0.0):
    n_pre = n_post if n_pre is None else n_pre
    infl = np.sqrt((1 + rho) / (1 - rho)) if rho else 1.0
    se = sigma * np.sqrt(1.0 / n_post + 1.0 / n_pre) * infl
    zc = _z(alpha, False)
    return float(stats.norm.sf(zc - abs(lift) / se))


def weeks_for_power(lift, sigma, n_pre=None, alpha=0.10, power=0.80, rho=0.0, max_weeks=104):
    for n in range(1, max_weeks + 1):
        if did_power(lift, sigma, n, n_pre if n_pre is not None else n, alpha, rho) >= power:
            return n
    return None


def estimate_daily_noise(t_kpi, c_kpi, t_pop, c_pop, pre_days, daily_noise_sd, dow_amp, seed=0):
    rng = np.random.default_rng(seed)
    n = pre_days
    t_mean, c_mean = t_pop / 1000.0 * t_kpi / 7.0, c_pop / 1000.0 * c_kpi / 7.0
    dow = 1 + dow_amp * np.sin(2 * np.pi * np.arange(n) / 7.0)
    t = t_mean * dow * np.exp(rng.normal(0, daily_noise_sd, n))
    c = c_mean * dow * np.exp(rng.normal(0, daily_noise_sd, n))
    gap = (t - c) / c
    rho = float(np.corrcoef(gap[1:], gap[:-1])[0, 1]) if n > 3 else 0.0
    return {"sigma": float(np.std(gap, ddof=1)), "rho": rho}


# =====================================================================================
# 4. Simulated test run + monitoring
# =====================================================================================
def change_ratio_did(treat, control, pre_idx, post_idx, n_boot=2000, seed=0, alpha=0.10):
    pre, post = list(pre_idx), list(post_idx)

    def est(tp, tq, cp, cq):
        cf = tp.mean() * (cq.mean() / cp.mean())
        return (tq.mean() - cf) / cf

    tp, tq = treat.loc[pre].to_numpy(float), treat.loc[post].to_numpy(float)
    cp, cq = control.loc[pre].to_numpy(float), control.loc[post].to_numpy(float)
    lift = est(tp, tq, cp, cq)
    rng = np.random.default_rng(seed)
    ip = rng.integers(0, len(pre), (n_boot, len(pre)))
    iq = rng.integers(0, len(post), (n_boot, len(post)))
    b = np.array([est(tp[a], tq[c], cp[a], cq[c]) for a, c in zip(ip, iq)])
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    return {"lift": float(lift), "ci_low": float(lo), "ci_high": float(hi)}


def simulate_test(pair, pre_days, test_days, true_lift, daily_noise_sd, dow_amp, seed=0):
    rng = np.random.default_rng(seed)
    n = pre_days + test_days
    t_pop, c_pop = pair["treatment"]["pop"], pair["control"]["pop"]
    t_kpi = pair["feature_comparison"]["baseline_kpi_rate"]["treatment"]
    c_kpi = pair["feature_comparison"]["baseline_kpi_rate"]["control"]
    t_mean, c_mean = t_pop / 1000.0 * t_kpi / 7.0, c_pop / 1000.0 * c_kpi / 7.0

    def series(mean_daily, lift_from=None):
        dow = 1 + dow_amp * np.sin(2 * np.pi * np.arange(n) / 7.0)
        trend = 1 + 0.0005 * np.arange(n)
        mult = np.ones(n)
        if lift_from is not None:
            mult[lift_from:] *= (1 + true_lift)
        mu = mean_daily * dow * trend * mult
        return rng.poisson(mu * np.exp(rng.normal(0, daily_noise_sd, n) - daily_noise_sd ** 2 / 2)).astype(float)

    t = series(t_mean, lift_from=pre_days)
    c = series(c_mean, lift_from=None)
    day_idx = np.arange(n)
    phase = np.where(day_idx < pre_days, "pre", "test")
    df = pd.DataFrame({"day_idx": day_idx, "phase": phase, "treat": t, "control": c})
    df["week_idx"] = df["day_idx"] // 7
    return df


def weekly_panel(daily):
    return daily.groupby("week_idx").agg(treat=("treat", "sum"), control=("control", "sum"),
                                         phase=("phase", "first")).reset_index()


def monitoring_looks(weekly, n_boot=2000, seed=0, alpha=0.10):
    pre = weekly[weekly["phase"] == "pre"].set_index("week_idx")
    test_weeks = weekly[weekly["phase"] == "test"]["week_idx"].tolist()
    rows = []
    for k in range(1, len(test_weeks) + 1):
        seen = test_weeks[:k]
        post = weekly[weekly["week_idx"].isin(seen)].set_index("week_idx")
        treat = pd.concat([pre["treat"], post["treat"]])
        control = pd.concat([pre["control"], post["control"]])
        res = change_ratio_did(treat, control, pre.index.tolist(), post.index.tolist(), n_boot, seed, alpha)
        rows.append({"look": k, "weeks_elapsed": k, **res})
    return pd.DataFrame(rows)


def final_readout(weekly, true_lift, n_boot=2000, seed=0, alpha=0.10):
    pre = weekly[weekly["phase"] == "pre"].set_index("week_idx")
    post = weekly[weekly["phase"] == "test"].set_index("week_idx")
    treat = pd.concat([pre["treat"], post["treat"]])
    control = pd.concat([pre["control"], post["control"]])
    res = change_ratio_did(treat, control, pre.index.tolist(), post.index.tolist(), n_boot, seed, alpha)
    res["true_lift"] = true_lift
    res["ci_covers_truth"] = bool(res["ci_low"] <= true_lift <= res["ci_high"])
    return res


# =====================================================================================
# Figures
# =====================================================================================
def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_clusters(d, pair, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in sorted(d["cluster"].unique()):
        sub = d[d["cluster"] == c]
        ax.scatter(sub["median_income"], sub["pct_bachelors_plus"], s=np.sqrt(sub["pop"]) * 1.1,
                  color=SERIES[c % len(SERIES)], alpha=0.65, label=f"Cluster {c} (n={len(sub)})",
                  edgecolor="white", linewidth=0.6)
    for role, name, color in (("Treatment", pair["treatment"]["town"], PALETTE["orange"]),
                              ("Control", pair["control"]["town"], PALETTE["navy"])):
        r = d[d["town"] == name].iloc[0]
        ax.scatter([r["median_income"]], [r["pct_bachelors_plus"]], marker="*", s=550, color=color,
                  edgecolor="black", linewidth=1.0, zorder=5, label=f"{role}: {name}")
    ax.set_xlabel("Median household income ($, real ACS 2018-2022 5yr)")
    ax.set_ylabel("Share with bachelor's degree or higher (real ACS)")
    ax.set_title("Massachusetts towns clustered on real Census demographics (standalone)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    fig.text(0.01, -0.02, "Standalone script; real Census geography, synthetic baseline KPI/test outcomes.",
            fontsize=8, color="#777777")
    _save(fig, path)


def fig_power(power_df, mde, test_weeks, target_power, true_lift, path):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for i, L in enumerate(sorted(power_df["lift"].unique())):
        sub = power_df[power_df["lift"] == L].sort_values("weeks")
        ax.plot(sub["weeks"], sub["power"], marker="o", ms=4, color=SERIES[i % len(SERIES)], label=f"lift={L:.0%}")
    ax.axhline(target_power, color=PALETTE["slate"], ls="--", lw=1.1)
    ax.axvline(test_weeks, color=PALETTE["orange"], ls="--", lw=1.2)
    ax.set_xlabel("Test length (weeks)")
    ax.set_ylabel("Power (closed-form, one-sided)")
    ax.set_title(f"Power-sizing ladder (MDE @ {test_weeks}wk = {mde:.1%}; true lift = {true_lift:.0%})")
    ax.legend(ncol=2, fontsize=8.5)
    fig.text(0.01, -0.02, "Closed-form AR(1)-inflated DiD power.", fontsize=8, color="#777777")
    _save(fig, path)


def fig_monitoring(looks, true_lift, path):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(looks["weeks_elapsed"], looks["lift"], color=PALETTE["navy"], marker="o", label="DiD lift estimate")
    ax.fill_between(looks["weeks_elapsed"], looks["ci_low"], looks["ci_high"], color=PALETTE["navy"], alpha=0.18,
                   label="90% CI")
    ax.axhline(true_lift, color=PALETTE["green"], ls="--", lw=1.4, label=f"true lift ({true_lift:.0%})")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("Test weeks elapsed (interim look)")
    ax.set_ylabel("Cumulative DiD lift estimate")
    ax.set_title("Sequential monitoring: interim DiD reads across the test window")
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.01, -0.02, "Simulated test outcomes.", fontsize=8, color="#777777")
    _save(fig, path)


def fig_final(final, pair, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = [0, 1]
    vals = [final["lift"], final["true_lift"]]
    bars = ax.bar(x, vals, color=[PALETTE["navy"], PALETTE["green"]], width=0.5)
    ax.errorbar([0], [final["lift"]], yerr=[[final["lift"] - final["ci_low"]], [final["ci_high"] - final["lift"]]],
               fmt="none", ecolor="black", capsize=6, lw=1.6)
    ax.set_xticks(x, ["DiD estimate\n(90% CI)", "True injected lift\n(ground truth)"])
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_title(f"Final readout: {pair['treatment']['town']} vs {pair['control']['town']}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1%}", ha="center",
               va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    fig.text(0.01, -0.02, "Simulated test outcomes vs known ground truth.", fontsize=8, color="#777777")
    _save(fig, path)


# =====================================================================================
# Main
# =====================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Standalone real-Census geo-experiment lifecycle tool")
    ap.add_argument("--out-dir", default="./geoselect_standalone_out")
    ap.add_argument("--pop-threshold", type=float, default=40000)
    ap.add_argument("--true-lift", type=float, default=0.12)
    ap.add_argument("--test-weeks", type=int, default=8)
    ap.add_argument("--pre-period-days", type=int, default=70)
    ap.add_argument("--kmeans-k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--power-target", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=20260)
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    raw_dir = out / "census_raw"
    fig_dir = out / "figures"
    test_days = args.test_weeks * 7

    print("[1/5] Pulling real Census ACS demographics (MA towns, key-free bulk ACS Summary File)...")
    geos = load_or_fetch_census(raw_dir, args.pop_threshold, force=args.force_refresh)
    geos = add_synthetic_baseline_kpi(geos, seed=args.seed)
    print(f"      {len(geos)} MA county subdivisions, pop >= {int(args.pop_threshold):,}")

    print("[2/5] Clustering + matched-pair selection...")
    d = cluster_geos(geos, k=args.kmeans_k, seed=7)
    pair = select_matched_pair(d)
    print(f"      treatment={pair['treatment']['town']}  control={pair['control']['town']}  "
         f"cluster={pair['cluster']} dist={pair['standardized_distance']:.3f}")

    print("[3/5] Power analysis...")
    noise = estimate_daily_noise(
        pair["feature_comparison"]["baseline_kpi_rate"]["treatment"],
        pair["feature_comparison"]["baseline_kpi_rate"]["control"],
        pair["treatment"]["pop"], pair["control"]["pop"], args.pre_period_days, 0.09, 0.08, seed=args.seed)
    lifts = (0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)
    weeks_grid = (2, 4, 6, 8, 10, 12, 16, 20, 26)
    pre_weeks = args.pre_period_days / 7.0
    rows = [{"lift": L, "weeks": w, "power": did_power(L, noise["sigma"], w, pre_weeks, args.alpha, noise["rho"])}
           for L in lifts for w in weeks_grid]
    power_df = pd.DataFrame(rows)
    mde = did_mde(noise["sigma"], args.test_weeks, pre_weeks, args.alpha, args.power_target, noise["rho"])
    weeks_needed = weeks_for_power(args.true_lift, noise["sigma"], pre_weeks, args.alpha, args.power_target,
                                   noise["rho"])
    print(f"      sigma_gap={noise['sigma']:.3f} rho={noise['rho']:.2f} MDE@{args.test_weeks}wk={mde:.1%}")

    print("[4/5] Simulating test run + monitoring...")
    daily = simulate_test(pair, args.pre_period_days, test_days, args.true_lift, 0.09, 0.08, seed=args.seed)
    weekly = weekly_panel(daily)
    looks = monitoring_looks(weekly, n_boot=2000, seed=args.seed, alpha=args.alpha)

    print("[5/5] Final DiD readout...")
    final = final_readout(weekly, args.true_lift, n_boot=2000, seed=args.seed, alpha=args.alpha)
    print(f"      estimate={final['lift']:+.1%}  CI=[{final['ci_low']:+.1%}, {final['ci_high']:+.1%}]  "
         f"true={args.true_lift:+.1%}  covers={final['ci_covers_truth']}")

    fig_clusters(d, pair, fig_dir / "01_clusters.png")
    fig_power(power_df, mde, args.test_weeks, args.power_target, args.true_lift, fig_dir / "02_power_ladder.png")
    fig_monitoring(looks, args.true_lift, fig_dir / "03_monitoring.png")
    fig_final(final, pair, fig_dir / "04_final_readout.png")

    results = {
        "census_data_source": ("US Census ACS 2018-2022 5-Year Summary File (bulk, key-free) + 2023 Gazetteer "
                              "cousubs file - api.census.gov Data API requires a key for data queries "
                              "(302->missing_key.html), so this standalone script uses the key-free bulk path."),
        "synthetic_assumptions": "baseline_kpi_rate + test-window outcomes are simulated; geography/demographics are real.",
        "n_geos": len(geos), "pop_threshold": args.pop_threshold, "kmeans_k": args.kmeans_k,
        "pair": pair, "power": {"sigma_gap": noise["sigma"], "rho": noise["rho"], "mde_at_test_weeks": mde,
                                "test_weeks": args.test_weeks, "weeks_to_target_power_at_true_lift": weeks_needed,
                                "power_curve": power_df.to_dict(orient="records")},
        "monitoring_interim_looks": looks.to_dict(orient="records"),
        "final_readout": final, "true_lift": args.true_lift,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))

    md = f"""# Standalone geo-experiment readout (Massachusetts, real Census demographics)

*Geography/demographics real (US Census ACS 2018-2022 5yr); baseline KPI, test outcomes and
monitoring are SIMULATED (true lift known by construction). Generated by
`geoselect_standalone.py`; full numbers in `results.json`.*

**Treatment:** {pair['treatment']['town']} (pop {pair['treatment']['pop']:,.0f}) vs
**control:** {pair['control']['town']} (pop {pair['control']['pop']:,.0f}), cluster
{pair['cluster']} (size {pair['cluster_size']}, standardized distance {pair['standardized_distance']:.3f}).

- Design MDE @ {args.test_weeks}wk / {args.power_target:.0%} power: **{mde:.1%}**
- Weeks to {args.power_target:.0%} power at the true {args.true_lift:.0%} lift: **{weeks_needed or 'n/a'}**
- Final DiD lift: **{final['lift']:+.1%}** [90% CI {final['ci_low']:+.1%}, {final['ci_high']:+.1%}] vs
  true **{args.true_lift:+.1%}** — interval **{'covers' if final['ci_covers_truth'] else 'misses'}** the truth.

## Census data source note
api.census.gov's tabular Data API now requires a registered key for every data query (we don't
have one); this script instead pulls the same ACS 2018-2022 5-year estimates from the public,
key-free bulk ACS Summary File + Gazetteer distribution on www2.census.gov. Real MA geography,
no rate-limit issue encountered.

## Figures
![Clusters](figures/01_clusters.png)
![Power ladder](figures/02_power_ladder.png)
![Monitoring](figures/03_monitoring.png)
![Final readout](figures/04_final_readout.png)
"""
    (out / "readout.md").write_text(md)
    print(f"\nDone. Outputs in {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
