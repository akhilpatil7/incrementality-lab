"""
Small Marketing Mix Model (MMM).

Transforms media spend with two canonical effects and fits a linear response:
  1. Adstock (geometric carryover): today's effective media carries into the
     following days at decay rate lambda.
  2. Saturation (Hill function): diminishing returns as spend grows.

We fit conversions ~ baseline_controls + beta * hill(adstock(spend)) via OLS on
the pooled panel (with geo/time controls), then plot the response curve. Because
only treated geos have spend, the coefficient captures media-driven conversions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def adstock(x: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """Geometric adstock: y_t = x_t + decay * y_{t-1}."""
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = x[i] + (decay * out[i - 1] if i > 0 else 0.0)
    return out


def hill(x: np.ndarray, half_sat: float, shape: float = 1.5) -> np.ndarray:
    """Hill saturation, output in [0,1). half_sat = spend at 50% saturation."""
    x = np.clip(x, 0, None)
    return x ** shape / (x ** shape + half_sat ** shape)


def build_features(df: pd.DataFrame, decay: float = 0.5) -> pd.DataFrame:
    d = df.sort_values(["geo", "day"]).copy()
    d["adstock"] = d.groupby("geo")["spend"].transform(lambda s: adstock(s.values, decay))
    return d


def fit_mmm(df: pd.DataFrame, decay: float = 0.5, shape: float = 1.5):
    d = build_features(df, decay)
    # half-saturation = median of positive adstocked spend
    pos = d.loc[d["adstock"] > 0, "adstock"]
    half_sat = float(pos.median()) if len(pos) else 1.0
    d["media_sat"] = hill(d["adstock"].values, half_sat, shape)

    # Response model with geo + weekday controls.
    d["dow"] = d["day"] % 7
    res = smf.ols("conversions ~ media_sat + C(geo) + C(dow)", data=d).fit()

    beta = res.params.get("media_sat", np.nan)

    # Response curve: incremental conversions vs spend level (single geo scale).
    spend_grid = np.linspace(0, d["spend"].max() * 1.2, 60)
    # steady-state adstock of a constant spend s = s / (1 - decay)
    adstk = spend_grid / (1 - decay)
    resp = beta * hill(adstk, half_sat, shape)

    return {
        "decay": decay,
        "shape": shape,
        "half_sat": half_sat,
        "beta_media": float(beta),
        "r2": float(res.rsquared),
        "spend_grid": spend_grid,
        "response": resp,
        "model": res,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/geo_panel.csv")
    r = fit_mmm(df)
    print(f"MMM fit: decay={r['decay']} shape={r['shape']} half_sat={r['half_sat']:.2f}")
    print(f"beta(media_sat)={r['beta_media']:.2f}  R^2={r['r2']:.3f}")
    print(f"Peak modeled incremental conv/day at high spend ~ {r['response'].max():.1f}")
