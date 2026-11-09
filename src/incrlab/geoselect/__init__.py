"""Module 5 - Geo-experiment selection: real Census ACS clustering -> matched-market power
analysis -> simulated test + monitoring -> DiD readout.

Entry points: ``python -m incrlab.geoselect`` or :func:`incrlab.geoselect.pipeline.run`.
"""
from .settings import GeoselectSettings, get_settings  # noqa: F401


def run(*args, **kw):
    from .pipeline import run as _run

    return _run(*args, **kw)
