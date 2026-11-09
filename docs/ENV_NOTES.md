# Environment notes

- Python 3.12 (3.11+ works). Create the environment with `make venv` (uv) or
  `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .`.
- Pinned versions are in `requirements.txt`, including:
  - **pymc 6.3.2** (+ pytensor 3.3.5, arviz 1.3.0) for the Bayesian MMM (NUTS).
  - **pysyncon 1.7.0**, wrapped by `incrlab.common.synth.fit_synth_pysyncon` (validated in tests to match
    the in-house SLSQP fit). The in-house `fit_synth` is the default and has no optional dependency.
- statsmodels 0.15: `MLEResults.simulate` takes `rng=` (not `random_state=` together); `common/bsts.py`
  handles both.
- pandas 3.x: use `DatetimeIndex.day_of_year` (not `dayofyear`); string columns load as `str` dtype.
