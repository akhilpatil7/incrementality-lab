# Environment notes

- Venv: `~/.hermes/venvs/incrlab` (created with `uv venv --python 3.12` → Python 3.12.14). Host has no
  system pip; use `uv pip install --python ~/.hermes/venvs/incrlab/bin/python ...`.
- All requested packages installed OK (pinned in `requirements.txt`), including:
  - **pymc 6.3.2** (+ pytensor 3.3.5, arviz 1.3.0) — smoke-tested: 2 chains × 200 draws sampled in < 1 s,
    posterior recovered. MMM builder can use PyMC (NUTS) directly; keep a scipy MAP fallback for CI speed.
  - **pysyncon 1.7.0** — wrapped by `incrlab.common.synth.fit_synth_pysyncon` (validated in tests to match
    the in-house SLSQP fit). The in-house `fit_synth` is the default and has no optional dependency.
- statsmodels 0.15: `MLEResults.simulate` takes `rng=` (not `random_state=` together); `common/bsts.py`
  handles both.
- pandas 3.x: use `DatetimeIndex.day_of_year` (not `dayofyear`); string columns load as `str` dtype.
