# incrlab v2 - `make all` = data -> meta -> search -> dsp -> mmm -> summary.json (+ README numbers) -> notebooks
# Override the interpreter with `make all VENV=/path/to/venv` or `make all PY=python`.
VENV ?= .venv
PY   ?= $(VENV)/bin/python
NB   := notebooks/00_overview.ipynb notebooks/01_meta.ipynb notebooks/02_search.ipynb \
        notebooks/03_dsp.ipynb notebooks/04_mmm_calibration.ipynb

.PHONY: venv data meta search dsp mmm pipeline summary all test test-small notebooks clean small

venv:
	@command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)"; exit 1; }
	@test -x $(PY) || uv venv $(VENV) --python 3.12 || uv venv $(VENV) --python 3.13
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -e .

data:
	$(PY) -m incrlab.simulate

small:
	INCRLAB_SMALL=1 $(PY) -m incrlab.simulate

meta: data
	$(PY) -m incrlab.meta

search: data
	$(PY) -m incrlab.search

dsp: data
	$(PY) -m incrlab.dsp

geoselect:
	$(PY) -m incrlab.geoselect

mmm: meta search dsp
	$(PY) -m incrlab.mmm
	$(PY) -m incrlab.pipeline summary

# whole program, each stage once; writes reports/results/summary.json and refreshes the README scoreboard
pipeline:
	$(PY) -m incrlab.pipeline all

summary:
	$(PY) -m incrlab.pipeline summary

all: pipeline notebooks

test:
	$(PY) -m pytest

test-small:
	INCRLAB_SMALL=1 $(PY) -m pytest -q

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=1800 $(NB)

clean:
	rm -f data/*.parquet
	rm -rf data/scenarios reports/results/*.json
