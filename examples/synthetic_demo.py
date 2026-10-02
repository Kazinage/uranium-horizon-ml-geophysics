"""Synthetic well-log demonstration with grouped validation."""

import numpy as np
import pandas as pd

from uranium_horizon.features import engineer_gamma_features
from uranium_horizon.modeling import DEFAULT_FEATURES, binary_metrics, grouped_oof_binary


rng = np.random.default_rng(12)
frames = []
for well in range(20):
    depth = np.arange(0, 80, 0.5)
    baseline = 18 + 2 * np.sin(depth / 11) + rng.normal(0, 1.5, len(depth))
    centre = rng.uniform(25, 60)
    anomaly = 28 * np.exp(-0.5 * ((depth - centre) / 3.5) ** 2)
    gamma = baseline + anomaly
    resistivity = 45 - 0.45 * anomaly + rng.normal(0, 3, len(depth))
    sp = rng.normal(0, 5, len(depth)) + 0.2 * anomaly
    productive = (anomaly > 13).astype(int)
    frames.append(pd.DataFrame({
        "well_id": f"W{well:02d}",
        "depth_m": depth,
        "gamma": gamma,
        "resistivity": resistivity,
        "sp": sp,
        "productive": productive,
    }))

df = engineer_gamma_features(pd.concat(frames, ignore_index=True))
X = df[DEFAULT_FEATURES].to_numpy()
y = df["productive"].to_numpy()
groups = df["well_id"].to_numpy()

result = grouped_oof_binary(X, y, groups, n_splits=5)
print(binary_metrics(y, result))
