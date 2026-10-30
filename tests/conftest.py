import numpy as np
import pandas as pd
import pytest

from incrlab.config import get_config
from incrlab.simulate import build_world


@pytest.fixture(scope="session")
def small_cfg():
    return get_config(seed=123, small=True)


@pytest.fixture(scope="session")
def small_world(small_cfg):
    return build_world(small_cfg)


def make_toy_panel(n_units=12, n_weeks=60, seed=0, noise=0.03, base=500.0):
    """Toy panel with a common factor + unit loadings; returns time x unit wide frame."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    f1 = 1 + 0.15 * np.sin(2 * np.pi * t / 52) + 0.002 * t
    f2 = np.cumsum(rng.normal(0, 0.02, n_weeks))
    load = rng.uniform(0.5, 1.5, n_units)
    scale = base * rng.uniform(0.6, 1.6, n_units)
    Y = scale[None] * (f1[:, None] + load[None] * f2[:, None]) * np.exp(rng.normal(0, noise, (n_weeks, n_units)))
    return pd.DataFrame(Y, index=pd.Index(t, name="week_idx"), columns=[f"u{i:02d}" for i in range(n_units)])
