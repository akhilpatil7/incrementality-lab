"""
Power analysis / Minimum Detectable Effect (MDE) calculator.

For a geo experiment we care about: given the between-geo variance of the
outcome, the number of treated/control geos, and a target power, what is the
smallest relative lift we can reliably detect? And conversely, what power do we
have for a given assumed lift?

Uses the standard two-sample MDE formula on the log outcome (so effects read as
% lift):  MDE = (z_alpha/2 + z_power) * sqrt(var*(1/n_t + 1/n_c))
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mde(var: float, n_treat: int, n_ctrl: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect on the log scale; returned as % lift."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    log_mde = (z_a + z_b) * np.sqrt(var * (1.0 / n_treat + 1.0 / n_ctrl))
    return float(np.exp(log_mde) - 1)


def power_for_lift(lift: float, var: float, n_treat: int, n_ctrl: int, alpha: float = 0.05) -> float:
    """Statistical power to detect a given % lift."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    effect = np.log(1 + lift)
    se = np.sqrt(var * (1.0 / n_treat + 1.0 / n_ctrl))
    z = effect / se
    return float(stats.norm.cdf(z - z_a) + stats.norm.cdf(-z - z_a))


def variance_from_data(df: pd.DataFrame) -> float:
    """Between-geo variance of the log post-period mean (the experiment unit)."""
    post = df[df["post"] == 1].groupby("geo")["conversions"].mean()
    return float(np.var(np.log(post.clip(lower=1e-6)), ddof=1))


def mde_curve(var: float, n_ctrl: int, n_treat_grid, **kw):
    return np.array([mde(var, int(n), n_ctrl, **kw) for n in n_treat_grid])


if __name__ == "__main__":
    df = pd.read_csv("data/geo_panel.csv")
    var = variance_from_data(df)
    n_t = int(df.groupby("geo")["treated"].first().sum())
    n_c = int((df.groupby("geo")["treated"].first() == 0).sum())
    print(f"Between-geo log-variance = {var:.4f}  ({n_t} treated / {n_c} control)")
    print(f"MDE @80% power = {mde(var, n_t, n_c):+.2%}")
    print(f"Power to detect +12% lift = {power_for_lift(0.12, var, n_t, n_c):.1%}")
