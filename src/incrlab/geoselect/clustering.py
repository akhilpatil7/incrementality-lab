"""K-means clustering of MA towns on real demographics + synthetic baseline KPI, then
matched-pair selection (treatment + control) within the best cluster.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .settings import GeoselectSettings, get_settings


def build_feature_matrix(df: pd.DataFrame, st: Optional[GeoselectSettings] = None) -> Tuple[pd.DataFrame, list]:
    st = st or get_settings()
    d = df.copy()
    if "log_pop" not in d:
        d["log_pop"] = np.log(d["pop"])
    if "log_density" not in d:
        d["log_density"] = np.log(d["density"])
    feats = list(st.features)
    X = d[feats].to_numpy(float)
    return d, feats


def cluster_geos(df: pd.DataFrame, st: Optional[GeoselectSettings] = None) -> pd.DataFrame:
    """Standardize features, run K-means, attach cluster labels + distance to own centroid."""
    st = st or get_settings()
    d, feats = build_feature_matrix(df, st)
    X = d[feats].to_numpy(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=st.kmeans_k, random_state=st.kmeans_seed, n_init=10)
    labels = km.fit_predict(Xs)
    d["cluster"] = labels
    centers = km.cluster_centers_
    d["dist_to_centroid"] = np.linalg.norm(Xs - centers[labels], axis=1)
    d.attrs["scaler"] = scaler
    d.attrs["kmeans"] = km
    d.attrs["features"] = feats
    d.attrs["Xs"] = Xs
    return d


def cluster_summary(d: pd.DataFrame, st: Optional[GeoselectSettings] = None) -> pd.DataFrame:
    st = st or get_settings()
    feats = d.attrs.get("features", list(st.features))
    agg = d.groupby("cluster").agg(n=("town", "count"), **{f"{c}_mean": (c, "mean") for c in feats})
    return agg.reset_index()


def select_matched_pair(d: pd.DataFrame, st: Optional[GeoselectSettings] = None) -> dict:
    """Pick the treatment/control pair: the two closest geos (by standardized distance)
    within the largest sufficiently-populous cluster, breaking ties by population similarity.
    """
    st = st or get_settings()
    feats = d.attrs.get("features", list(st.features))
    Xs = d.attrs.get("Xs")
    if Xs is None:
        _, feats = build_feature_matrix(d, st)
        Xs = d[feats].to_numpy(float)

    sizes = d["cluster"].value_counts()
    eligible = sizes[sizes >= 2].index.tolist()
    if not eligible:
        raise RuntimeError("no cluster with >=2 members - widen pop_threshold or lower kmeans_k")
    best = None
    for c in eligible:
        idx = d.index[d["cluster"] == c].tolist()
        sub = Xs[idx]
        n = len(idx)
        dmat = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)
        i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
        gi, gj = idx[i], idx[j]
        dist = float(dmat[i, j])
        if best is None or dist < best["dist"]:
            best = {"cluster": int(c), "dist": dist, "i": gi, "j": gj, "cluster_size": n}
    ti, cj = best["i"], best["j"]
    # orient treatment = higher pop (arbitrary, documented), control = the match
    if d.loc[ti, "pop"] < d.loc[cj, "pop"]:
        ti, cj = cj, ti
    treat, ctrl = d.loc[ti], d.loc[cj]
    return {
        "cluster": best["cluster"], "cluster_size": best["cluster_size"],
        "standardized_distance": best["dist"],
        "treatment": {"town": treat["town"], "geoid": treat["geoid"], "pop": float(treat["pop"]),
                      "cluster": int(treat["cluster"])},
        "control": {"town": ctrl["town"], "geoid": ctrl["geoid"], "pop": float(ctrl["pop"]),
                   "cluster": int(ctrl["cluster"])},
        "feature_comparison": {
            f: {"treatment": float(treat[f]), "control": float(ctrl[f])} for f in feats
        },
    }
