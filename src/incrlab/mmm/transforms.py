"""Media transforms shared by the MMM (numpy reference + vectorised forms used inside PyMC).

Conventions match :mod:`incrlab.simulate` exactly:

* geometric adstock, *normalised*:  a_t = (1 - d) * sum_{k>=0} d^k x_{t-k}
  (a constant spend level x therefore has steady-state adstock x)
* Hill saturation on the half-saturation-scaled adstock:  hill(z; s) = z^s / (1 + z^s)

Inside the PyMC model adstock is computed as ``lag_matrix(x) @ (1-d) d^k`` (a truncated
convolution, L_MAX weeks) instead of a scan - this is exact up to d^L_MAX and an order of magnitude
faster to sample.
"""
from __future__ import annotations

import numpy as np

L_MAX = 30  # adstock truncation (0.8**30 = 0.1%)


def geometric_adstock(x, decay: float, normalize: bool = True, l_max: int | None = None) -> np.ndarray:
    """Geometric adstock along the last axis. ``l_max=None`` = full recursion (as in the simulator)."""
    x = np.asarray(x, float)
    if l_max is None:
        a = np.empty_like(x)
        a[..., 0] = x[..., 0]
        for t in range(1, x.shape[-1]):
            a[..., t] = x[..., t] + decay * a[..., t - 1]
        return a * (1 - decay) if normalize else a
    w = decay ** np.arange(l_max)
    if normalize:
        w = w * (1 - decay)
    lag = lag_matrix(x, l_max)  # (..., T, L)
    return lag @ w


def hill(z, slope: float) -> np.ndarray:
    """Hill saturation z^s / (1 + z^s) for z >= 0 (z = adstock / half-saturation)."""
    z = np.maximum(np.asarray(z, float), 0.0)
    zs = z ** slope
    return zs / (1.0 + zs)


def hill_derivative(z, slope: float) -> np.ndarray:
    z = np.maximum(np.asarray(z, float), 1e-12)
    zs = z ** slope
    return slope * zs / (z * (1 + zs) ** 2)


def lag_matrix(x, l_max: int = L_MAX) -> np.ndarray:
    """Stack lags: out[..., t, k] = x[..., t-k] (0 before the series starts)."""
    x = np.asarray(x, float)
    T = x.shape[-1]
    out = np.zeros(x.shape[:-1] + (T, l_max))
    for k in range(l_max):
        out[..., k:, k] = x[..., : T - k]
    return out


def adstock_batch(x: np.ndarray, decays: np.ndarray, l_max: int = L_MAX) -> np.ndarray:
    """Posterior-draw-aware adstock. x: (C, T) spend; decays: (S, C) -> (S, C, T)."""
    lag = lag_matrix(x, l_max)  # (C, T, L)
    k = np.arange(l_max)
    w = (1 - decays)[..., None] * decays[..., None] ** k  # (S, C, L)
    return np.einsum("ctl,scl->sct", lag, w)
