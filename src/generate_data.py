"""
Synthetic multi-geo marketing dataset generator.

Produces a realistic DMA-level daily panel with:
  - baseline conversions driven by geo-specific size, weekly seasonality,
    an annual trend, and an underlying market shock component,
  - media spend that is only nonzero for TREATED geos during the campaign
    window (a classic geo holdout / geo-split experiment design),
  - a KNOWN, injected incremental lift applied to treated geos in the
    post-period, expressed as a percentage of counterfactual conversions.

Because the ground-truth lift is injected explicitly, every method in this
repo can be validated: a correct causal estimator should recover TRUE_LIFT.

Run:  python -m src.generate_data   (writes data/geo_panel.csv)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import os

import numpy as np
import pandas as pd

# ---- Ground-truth experiment configuration -------------------------------
TRUE_LIFT = 0.12          # +12% incremental conversions in treated geos, post
DEFAULT_SEED = 42
N_GEOS = 40               # total DMAs
FRAC_TREATED = 0.45       # ~18 treated, ~22 control
N_DAYS = 180              # ~6 months daily
CAMPAIGN_START_DAY = 120  # pre-period = first 120 days, post = last 60
DATA_PATH = os.path.join("data", "geo_panel.csv")


@dataclass
class GenConfig:
    seed: int = DEFAULT_SEED
    n_geos: int = N_GEOS
    frac_treated: float = FRAC_TREATED
    n_days: int = N_DAYS
    campaign_start_day: int = CAMPAIGN_START_DAY
    true_lift: float = TRUE_LIFT


def generate(cfg: GenConfig = GenConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    geos = [f"DMA_{i:02d}" for i in range(cfg.n_geos)]
    dates = pd.date_range("2025-01-01", periods=cfg.n_days, freq="D")
    day_idx = np.arange(cfg.n_days)

    # Geo-level structural characteristics.
    geo_size = rng.lognormal(mean=8.0, sigma=0.4, size=cfg.n_geos)       # market size
    geo_intercept = geo_size * rng.uniform(0.9, 1.1, cfg.n_geos)
    geo_noise_sd = geo_size * 0.05                                        # heteroskedastic

    # Assign treatment. Treated geos are picked to span the size distribution
    # (stratified) so control donors are comparable for synthetic control.
    order = np.argsort(geo_size)
    n_treated = int(round(cfg.n_geos * cfg.frac_treated))
    treated_positions = np.linspace(0, cfg.n_geos - 1, n_treated).round().astype(int)
    treated_mask = np.zeros(cfg.n_geos, dtype=bool)
    treated_mask[order[treated_positions]] = True

    # Shared, national market shock (common time trend + shocks). This is the
    # confounder that DiD/synthetic control must remove.
    trend = 0.0006 * day_idx
    national_shock = rng.normal(0, 0.03, cfg.n_days).cumsum() * 0.15
    weekly = 0.10 * np.sin(2 * np.pi * (day_idx % 7) / 7.0)

    rows = []
    for g in range(cfg.n_geos):
        base = geo_intercept[g]
        # geo-specific weekly amplitude & phase for realism
        w_amp = rng.uniform(0.6, 1.4)
        seasonal = 1.0 + w_amp * weekly + trend + national_shock
        counterfactual = base * seasonal
        counterfactual = np.clip(counterfactual, 1.0, None)

        # Poisson-like conversions around the counterfactual mean.
        noise = rng.normal(0, geo_noise_sd[g], cfg.n_days)
        conversions_cf = np.clip(counterfactual + noise, 0, None)

        is_treated = bool(treated_mask[g])
        post = day_idx >= cfg.campaign_start_day

        # Media spend: only treated geos spend, only during post window.
        spend = np.zeros(cfg.n_days)
        if is_treated:
            ramp = np.linspace(0.4, 1.0, cfg.n_days - cfg.campaign_start_day)
            spend[cfg.campaign_start_day:] = base * 0.05 * ramp * rng.uniform(0.9, 1.1, ramp.size)

        # Inject the known incremental lift on treated geos in the post period.
        conversions = conversions_cf.copy()
        if is_treated:
            conversions[post] = conversions_cf[post] * (1.0 + cfg.true_lift)

        for t in range(cfg.n_days):
            rows.append({
                "geo": geos[g],
                "date": dates[t],
                "day": int(day_idx[t]),
                "treated": int(is_treated),
                "post": int(post[t]),
                "spend": round(float(spend[t]), 2),
                "conversions": round(float(conversions[t]), 3),
                "conversions_counterfactual": round(float(conversions_cf[t]), 3),
            })

    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=DATA_PATH)
    args = ap.parse_args()

    cfg = GenConfig(seed=args.seed)
    df = generate(cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    meta = asdict(cfg)
    meta["n_rows"] = len(df)
    meta["n_treated_geos"] = int(df.groupby("geo")["treated"].first().sum())
    meta["n_control_geos"] = int((df.groupby("geo")["treated"].first() == 0).sum())
    with open(os.path.join(os.path.dirname(args.out), "ground_truth.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {args.out}  ({len(df):,} rows)")
    print(f"TRUE_LIFT = {cfg.true_lift:+.1%}  |  "
          f"{meta['n_treated_geos']} treated / {meta['n_control_geos']} control geos")


if __name__ == "__main__":
    main()
