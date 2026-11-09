"""`python -m incrlab.geoselect` - run the full geo-experiment-selection module end to end."""
from __future__ import annotations

import argparse
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Geoselect: real-Census clustering -> matched pair -> "
                                             "power -> simulated test/monitoring -> DiD readout")
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation")
    ap.add_argument("--force-refresh-census", action="store_true", help="re-pull Census data, ignore cache")
    args = ap.parse_args(argv)
    from .pipeline import run

    t0 = time.time()
    S = run(figures=not args.no_figures, force_refresh_census=args.force_refresh_census)
    print(f"[geoselect] geos considered: {len(S.geos)} MA towns (pop >= {int(S.st.pop_threshold):,}, real ACS)")
    print(f"[geoselect] treatment: {S.pair['treatment']['town']}  control: {S.pair['control']['town']}  "
         f"(cluster {S.pair['cluster']}, dist {S.pair['standardized_distance']:.3f})")
    print(f"[geoselect] design MDE @ {S.st.test_weeks}wk: {S.mde:.1%}  true lift: {S.st.true_lift:+.1%}")
    print(f"[geoselect] final DiD lift: {S.final['lift']:+.1%}  "
         f"CI [{S.final.get('ci_low', float('nan')):+.1%}, {S.final.get('ci_high', float('nan')):+.1%}]  "
         f"verdict: {S.verdict['label']}")
    print(f"[geoselect] figures: {len(S.figures)}; results -> reports/results/geoselect.json; "
         f"readout -> reports/readouts/geoselect.md")
    print(f"[geoselect] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
