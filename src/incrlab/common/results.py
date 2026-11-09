"""Register module outputs: reports/results/{module}.json (one schema for every module)."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR, get_config

REQUIRED_KEYS = ("module", "schema_version", "generated_at", "seed", "small", "headline", "estimates")


def _clean(o: Any):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return _clean(o.to_dict())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (pd.Timestamp, _dt.date)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_results(module: str, headline: Dict[str, Any], estimates: list, design: Optional[dict] = None,
                  validation: Optional[dict] = None, truth: Optional[dict] = None, figures: Optional[list] = None,
                  extra: Optional[dict] = None, results_dir: Optional[Path] = None) -> Path:
    """Write the module result file. ``estimates`` is a list of dicts with at least
    {test, arm, metric, method, estimate} and optionally ci_low/ci_high/p_value/significant/truth/notes.
    """
    cfg = get_config()
    for e in estimates:
        missing = {"test", "arm", "metric", "method", "estimate"} - set(e)
        if missing:
            raise ValueError(f"estimate missing keys {missing}: {e}")
    doc = {"module": module, "schema_version": "1.0",
           "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
           "seed": cfg.seed, "small": cfg.small, "headline": headline, "estimates": estimates,
           "design": design or {}, "validation": validation or {}, "truth": truth or {},
           "figures": [str(f) for f in (figures or [])], "extra": extra or {}}
    d = Path(results_dir or RESULTS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{module}.json"
    p.write_text(json.dumps(_clean(doc), indent=2))
    return p


def read_results(module: str, results_dir: Optional[Path] = None) -> dict:
    return json.loads((Path(results_dir or RESULTS_DIR) / f"{module}.json").read_text())
