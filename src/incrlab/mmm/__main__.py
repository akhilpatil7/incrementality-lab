"""`python -m incrlab.mmm` - fit uncalibrated + experiment-calibrated MMMs, optimise budget, write reports."""
from __future__ import annotations

import argparse
import logging
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Experiment-calibrated Bayesian MMM + budget optimizer")
    ap.add_argument("--small", action="store_true", help="fast smoke settings (2 chains x 150 draws)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the well-powered-tests sensitivity fit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for n in ("pymc", "pytensor", "arviz", "matplotlib"):
        logging.getLogger(n).setLevel(logging.ERROR)
    from . import figures as F
    from . import report
    from .pipeline import roi_table, run

    t0 = time.time()
    S = run(small=True if args.small else None, sensitivity=False if args.no_sensitivity else None)
    if not args.no_figures:
        S.figures = F.make_all(S)
    S.seconds = time.time() - t0
    report.write_results_json(S)
    report.write_readout_md(S)
    t = roi_table(S)
    print(t[["channel", "true_roi", "uncalibrated_roi", "calibrated_roi", "uncalibrated_roi_abs_err",
             "calibrated_roi_abs_err"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"[mmm] calibration sources: {S.sources}")
    if S.stub_tests:
        print(f"[mmm] !!! STUB calibration inputs for {S.stub_tests} !!!")
    o = S.optimizer["calibrated"]["optimal"]
    print(f"[mmm] optimizer gain {o['delta_vs_ref']['mean']:+.0f} sales (P>0 {o['prob_better']:.0%}); next test: "
          f"{max(S.voi, key=lambda c: S.voi[c]['evppi_value'])}")
    print(f"[mmm] {len(S.figures)} figures; results -> reports/results/mmm.json; readout -> reports/readouts/mmm.md; "
          f"{S.seconds:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
