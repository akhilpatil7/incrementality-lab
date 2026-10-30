"""`python -m incrlab.dsp` - run the Programmatic DSP program end to end (both vendors)."""
from __future__ import annotations

import argparse
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Programmatic DSP geo experiments (Vendor A twin patch, Vendor B "
                                             "cluster SC): design -> holdout -> triangulated readout")
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = ap.parse_args(argv)
    from .pipeline import run
    from .readout import truth_table

    t0 = time.time()
    S = run(figures=not args.no_figures)
    A, B, R = S.A, S.B, S.R
    print("[dsp] Vendor A design:")
    for line in A.validation_log:
        print("   ", line)
    print("[dsp] Vendor B design:")
    for line in B.log:
        print("   ", line)
    tt = truth_table(R)
    cols = ["vendor", "metric", "method", "estimate", "ci_low", "ci_high", "p_value", "significant", "truth"]
    print(tt[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for v in "AB":
        print(f"[dsp] Vendor {v}: {R.decisions[v]['decision']} - {R.decisions[v]['rationale']}")
    print(f"[dsp] figures: {len(S.figures)}; results -> reports/results/dsp.json; readout -> reports/readouts/dsp.md")
    print(f"[dsp] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
