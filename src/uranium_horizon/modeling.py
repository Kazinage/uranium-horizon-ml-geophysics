"""Grouped cross-validation utilities for borehole classification."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold


DEFAULT_FEATURES = [
    "depth_m",
    "gamma",
    "resistivity",
    "sp",
    "gamma_rate",
    "gamma_sq",
]


def default_random_forest(random_state=42):
    """Reference RF configuration matching the tuned manuscript settings."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )


@dataclass
class OOFResult:
    prediction: np.ndarray
    probability: np.ndarray
    validation_mask: np.ndarray
    fold: np.ndarray


def grouped_oof_binary(
    X,
    y,
    groups,
    *,
    estimator=None,
    weak_label_mask=None,
    n_splits=5,
):
    """OOF prediction with entire wells held out.

    Weak-labelled rows may be retained in training but excluded from validation,
    mirroring the leakage-control rule described in the manuscript.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    weak = (
        np.zeros(len(y), dtype=bool)
        if weak_label_mask is None
        else np.asarray(weak_label_mask, dtype=bool)
    )

    estimator = default_random_forest() if estimator is None else estimator
    gkf = GroupKFold(n_splits=n_splits)

    pred = np.full(len(y), -1, dtype=int)
    proba = np.full(len(y), np.nan, dtype=float)
    fold_id = np.full(len(y), -1, dtype=int)
    valid = np.zeros(len(y), dtype=bool)

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
        eval_va = va[~weak[va]]
        model = clone(estimator).fit(X[tr], y[tr])
        pred[eval_va] = model.predict(X[eval_va])
        proba[eval_va] = model.predict_proba(X[eval_va])[:, 1]
        fold_id[eval_va] = fold
        valid[eval_va] = True

    return OOFResult(pred, proba, valid, fold_id)


def binary_metrics(y, result: OOFResult):
    m = result.validation_mask
    yt = np.asarray(y, dtype=int)[m]
    yp = result.prediction[m]
    pp = result.probability[m]
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision_positive": float(precision_score(yt, yp, zero_division=0)),
        "recall_positive": float(recall_score(yt, yp, zero_division=0)),
        "f1_positive": float(f1_score(yt, yp, zero_division=0)),
        "average_precision": float(average_precision_score(yt, pp)),
        "brier": float(brier_score_loss(yt, pp)),
    }
