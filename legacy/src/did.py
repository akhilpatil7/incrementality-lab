"""
Geo-split Difference-in-Differences (primary incrementality estimator).

Design: treated geos receive media in the post period; control geos never do.
DiD removes both time-invariant geo differences AND the common national shock,
so the interaction term (treated x post) identifies the incremental effect.

We estimate on a log outcome so the coefficient is interpretable directly as a
percentage lift, matching the multiplicative way the ground-truth lift was
injected. Standard errors are clustered by geo (the unit of treatment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def run_did(df: pd.DataFrame):
    """Two-way DiD on log(conversions) with geo-clustered SEs.

    Returns a dict with the point estimate (as % lift), CI, p-value, and the
    fitted model.
    """
    d = df.copy()
    d["log_conv"] = np.log(d["conversions"].clip(lower=1e-6))
    d["treated_post"] = d["treated"] * d["post"]

    # geo and time fixed effects via categoricals; cluster SE by geo.
    model = smf.ols("log_conv ~ treated_post + C(geo) + C(day)", data=d)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": d["geo"]})

    beta = res.params["treated_post"]
    ci_low, ci_high = res.conf_int().loc["treated_post"]
    pval = res.pvalues["treated_post"]

    # Convert log-difference to percentage lift: exp(beta) - 1
    lift = np.exp(beta) - 1
    lift_ci = (np.exp(ci_low) - 1, np.exp(ci_high) - 1)

    return {
        "beta_log": float(beta),
        "lift": float(lift),
        "lift_ci": (float(lift_ci[0]), float(lift_ci[1])),
        "pvalue": float(pval),
        "model": res,
    }


def naive_pre_post(df: pd.DataFrame) -> float:
    """Naive treated-only pre/post % change (biased by national shock) for
    contrast with DiD."""
    t = df[df["treated"] == 1]
    pre = t[t["post"] == 0]["conversions"].mean()
    post = t[t["post"] == 1]["conversions"].mean()
    return float(post / pre - 1)


if __name__ == "__main__":
    df = pd.read_csv("data/geo_panel.csv")
    r = run_did(df)
    print(f"DiD lift = {r['lift']:+.2%}  "
          f"95% CI [{r['lift_ci'][0]:+.2%}, {r['lift_ci'][1]:+.2%}]  p={r['pvalue']:.2e}")
    print(f"Naive pre/post (biased) = {naive_pre_post(df):+.2%}")
