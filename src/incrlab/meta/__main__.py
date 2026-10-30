"""`python -m incrlab.meta` - run the full Social geo-holdout module (design, readout, figures, results)."""
from __future__ import annotations

import argparse
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Social (Meta) geo holdout: design -> holdout -> readout")
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = ap.parse_args(argv)
    from .pipeline import estimates_table, run

    t0 = time.time()
    S = run(figures=not args.no_figures)
    tab = estimates_table(S)
    print(f"[meta] treated unit: {S.unit}  ({S.selection['reason']})")
    print(f"[meta] donors: {S.dfit.weights.round(3).to_dict()}")
    cols = ["method", "metric", "estimate", "ci_low", "ci_high", "p_value", "truth", "error_pp"]
    print(tab[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[meta] verdict: {S.verdict}")
    print(f"[meta] figures: {len(S.figures)}; results -> reports/results/meta.json; readout -> reports/readouts/meta.md")
    print(f"[meta] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
