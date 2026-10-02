import numpy as np
import pandas as pd
from uranium_horizon.features import engineer_gamma_features
from uranium_horizon.modeling import DEFAULT_FEATURES, grouped_oof_binary

def test_grouped_oof_runs():
    rng = np.random.default_rng(2)
    frames=[]
    for w in range(10):
        depth=np.arange(20, dtype=float)
        gamma=rng.normal(10,2,20)+(depth>10)*8
        frames.append(pd.DataFrame({
            "well_id":f"W{w}","depth_m":depth,"gamma":gamma,
            "resistivity":rng.normal(30,3,20),"sp":rng.normal(0,2,20),
            "productive":(gamma>15).astype(int)
        }))
    df=engineer_gamma_features(pd.concat(frames,ignore_index=True))
    result=grouped_oof_binary(df[DEFAULT_FEATURES],df["productive"],df["well_id"],n_splits=5)
    assert result.validation_mask.sum() == len(df)
    assert np.isfinite(result.probability[result.validation_mask]).all()
