"""Readout helpers: headline dict + markdown builder for the geoselect module."""
from __future__ import annotations

from typing import Optional

import numpy as np

STAGE_LABEL = {"conversions": "Conversions"}


def verdict(final: dict, mde: float, st) -> dict:
    lift_in_range = final["ci_low"] <= final["lift"] <= final["ci_high"] if final.get("ci_low") is not None else None
    detected = bool(final.get("ci_low", -np.inf) > 0) if final.get("ci_low") is not None else False
    underpowered = mde > st.true_lift
    label = "powered_and_detected" if (detected and not underpowered) else (
        "detected_but_underpowered_design" if detected else "not_detected")
    return {"label": label, "detected_significant": detected, "underpowered": bool(underpowered),
           "design_mde": mde, "true_lift": st.true_lift, "measured_lift": final["lift"],
           "ci_covers_truth": final.get("ci_covers_truth")}
