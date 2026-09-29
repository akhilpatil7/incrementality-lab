"""
Synthetic Control validation layer.

Builds a "virtual control" for the pooled treated market as a convex,
non-negative weighted combination of donor (control) geos, with weights fit to
match the treated series in the PRE period only. The post-period gap between
the observed treated series and its synthetic counterpart estimates the
incremental effect - an independent cross-check on DiD.

Weights are solved as a constrained least squares problem (weights >= 0,
sum to 1) via SLSQP, the standard Abadie et al. formulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Return day x geo matrix of conversions."""
    return df.pivot_table(index="day", columns="geo", values="conversions", aggfunc="mean")


def fit_synthetic_control(df: pd.DataFrame, campaign_start_day: int | None = None):
    if campaign_start_day is None:
        campaign_start_day = int(df.loc[df["post"] == 1, "day"].min())

    treated_geos = df.loc[df["treated"] == 1, "geo"].unique()
    control_geos = df.loc[df["treated"] == 0, "geo"].unique()

    mat = _pivot(df)
    # Pooled treated target = mean across treated geos (aggregate market).
    target = mat[treated_geos].mean(axis=1)
    donors = mat[control_geos]

    pre = mat.index < campaign_start_day
    Y_pre = target[pre].values
    X_pre = donors[pre].values          # (T_pre, n_donors)

    n = X_pre.shape[1]

    def loss(w):
        return np.mean((Y_pre - X_pre @ w) ** 2)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    w0 = np.full(n, 1.0 / n)
    sol = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-10})
    w = sol.x

    synthetic = donors.values @ w
    synth_series = pd.Series(synthetic, index=mat.index)

    post = mat.index >= campaign_start_day
    obs_post = target[post].sum()
    syn_post = synth_series[post].sum()
    lift = obs_post / syn_post - 1

    pre_rmse = np.sqrt(np.mean((target[pre].values - synth_series[pre].values) ** 2))

    return {
        "weights": dict(zip(control_geos, np.round(w, 4))),
        "lift": float(lift),
        "pre_rmse": float(pre_rmse),
        "target": target,
        "synthetic": synth_series,
        "campaign_start_day": campaign_start_day,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/geo_panel.csv")
    r = fit_synthetic_control(df)
    print(f"Synthetic control lift (post) = {r['lift']:+.2%}  "
          f"pre-fit RMSE={r['pre_rmse']:.2f}")
    top = sorted(r["weights"].items(), key=lambda kv: -kv[1])[:5]
    print("Top donor weights:", top)
