# incrementality-lab v2
VENV ?= $(HOME)/.hermes/venvs/incrlab
PY   ?= $(VENV)/bin/python
NB   := $(wildcard notebooks/*.ipynb)

.PHONY: venv data meta search dsp mmm all test notebooks clean small

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

mmm: meta search dsp
	$(PY) -m incrlab.mmm

all: data meta search dsp mmm notebooks

test:
	$(PY) -m pytest

notebooks:
	@if [ -n "$(NB)" ]; then $(PY) -m jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=1800 $(NB); else echo "no notebooks yet"; fi

clean:
	rm -f data/*.parquet
	rm -rf data/scenarios reports/results/*.json
