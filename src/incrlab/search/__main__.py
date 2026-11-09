"""`python -m incrlab.search` - run the full Paid Search module (design, readout, figures, JSON, readout md)."""
from .pipeline import run


def main() -> None:
    out = run()
    tt = out["truth_table"]
    print("\nTruth vs estimate")
    print(tt[["quantity", "estimate_fmt", "ci_fmt", "truth_fmt", "error_fmt"]].to_string(index=False))
    print("\nRecommendation:")
    for line in out["recommendation"]["lines"]:
        print(" -", line)


if __name__ == "__main__":
    main()
