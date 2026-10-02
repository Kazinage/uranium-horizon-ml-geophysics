"""Well-log feature engineering for uranium-horizon classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_gamma_features(
    df: pd.DataFrame,
    *,
    well_col="well_id",
    depth_col="depth_m",
    gamma_col="gamma",
):
    """Add gamma-rate-of-change and squared-gamma features within each well."""
    out = df.copy().sort_values([well_col, depth_col])
    dz = out.groupby(well_col)[depth_col].diff()
    dg = out.groupby(well_col)[gamma_col].diff()
    rate = dg / dz.replace(0, np.nan)
    out["gamma_rate"] = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["gamma_sq"] = out[gamma_col].astype(float) ** 2
    return out


def assign_weak_productive_label(gamma, threshold):
    """Create a transparent weak label from a user-supplied gamma threshold."""
    return (np.asarray(gamma, dtype=float) >= float(threshold)).astype(int)
