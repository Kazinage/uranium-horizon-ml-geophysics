"""Simple along-profile cubic-spline helpers."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def cubic_profile(x, values, x_new):
    """Interpolate one property along a profile using a natural cubic spline."""
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    order = np.argsort(x)
    x = x[order]
    values = values[order]

    unique, idx = np.unique(x, return_index=True)
    values = values[idx]
    if len(unique) < 3:
        raise ValueError("At least three unique profile positions are required")
    spline = CubicSpline(unique, values, bc_type="natural", extrapolate=False)
    return spline(x_new)
