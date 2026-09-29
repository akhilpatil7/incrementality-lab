"""
CUPED variance reduction demo.

CUPED (Controlled experiment Using Pre-Existing Data) reduces the variance of
an experiment's outcome by regressing out a pre-period covariate that is
correlated with the outcome but unaffected by treatment. Lower variance => a
smaller detectable effect (MDE) for the same sample size.

Here the pre-period mean conversions per geo is the covariate. We compare the
variance and the effective standard error of the treatment effect estimate
with and without CUPED adjustment, at the geo level (post-period mean per geo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def geo_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per geo: pre-mean covariate, post-mean outcome, treatment flag."""
    g = df.groupby("geo")
    pre = df[df["post"] == 0].groupby("geo")["conversions"].mean()
    post = df[df["post"] == 1].groupby("geo")["conversions"].mean()
    treated = g["treated"].first()
    out = pd.DataFrame({"pre": pre, "post": post, "treated": treated}).reset_index()
    # Work in log space to match multiplicative lift, keeps geos comparable.
    out["y"] = np.log(out["post"].clip(lower=1e-6))
    out["x"] = np.log(out["pre"].clip(lower=1e-6))
    return out


def cuped_adjust(frame: pd.DataFrame):
    """Return naive vs CUPED-adjusted treatment effect + variance stats."""
    y = frame["y"].values
    x = frame["x"].values
    t = frame["treated"].values.astype(float)

    # theta = Cov(y, x) / Var(x)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_cv = y - theta * (x - x.mean())

    def effect_and_se(vals):
        a = vals[t == 1]
        b = vals[t == 0]
        eff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return eff, se, a.var(ddof=1)

    naive_eff, naive_se, naive_var = effect_and_se(y)
    cuped_eff, cuped_se, cuped_var = effect_and_se(y_cv)

    var_reduction = 1 - cuped_var / naive_var
    se_reduction = 1 - cuped_se / naive_se

    return {
        "theta": float(theta),
        "corr_xy": float(np.corrcoef(x, y)[0, 1]),
        "naive": {"lift": float(np.exp(naive_eff) - 1), "se": float(naive_se), "var": float(naive_var)},
        "cuped": {"lift": float(np.exp(cuped_eff) - 1), "se": float(cuped_se), "var": float(cuped_var)},
        "var_reduction": float(var_reduction),
        "se_reduction": float(se_reduction),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/geo_panel.csv")
    frame = geo_level_frame(df)
    r = cuped_adjust(frame)
    print(f"corr(pre, post) = {r['corr_xy']:.3f}")
    print(f"Naive  lift={r['naive']['lift']:+.2%}  SE={r['naive']['se']:.4f}")
    print(f"CUPED  lift={r['cuped']['lift']:+.2%}  SE={r['cuped']['se']:.4f}")
    print(f"Variance reduction = {r['var_reduction']:.1%}  |  SE reduction = {r['se_reduction']:.1%}")
