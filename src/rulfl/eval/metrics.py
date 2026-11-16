"""Evaluation metrics for RUL prediction (DECISIONS E1, E3, E4, E9-E14).

Every score in the project goes through this file. Predictions always come from `predict_rul` (already clamped to
[0, cap], M4), so nothing here clamps.
"""
from __future__ import annotations

import math

import numpy as np


def _validate_inputs(pred, true) -> tuple[np.ndarray, np.ndarray]:
    """1-D, same length, non-empty and finite, or ValueError (E10, E12)."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim != 1 or true.ndim != 1:
        raise ValueError("pred and true must be 1-D arrays")
    if pred.size == 0 or true.size == 0:
        raise ValueError("pred and true must not be empty")
    if len(pred) != len(true):
        raise ValueError(f"pred and true must have equal lengths, got {len(pred)} and {len(true)}")
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(true))):
        raise ValueError("pred and true must contain only finite values")
    return pred, true


def nasa_score(pred, true) -> float:
    """NASA PHM08 score, summed over engines (E1). d = pred - true; late (d >= 0) exp(d/10)-1, early exp(-d/13)-1."""
    pred, true = _validate_inputs(pred, true)
    d = pred - true
    return float(np.sum(np.where(d >= 0, np.exp(d / 10.0) - 1.0, np.exp(-d / 13.0) - 1.0)))


def metrics(pred, true) -> dict:
    """{"mae", "rmse", "nasa"} as plain Python floats, so they can be written to final.json (E1, E8)."""
    pred, true = _validate_inputs(pred, true)
    d = pred - true
    return {"mae": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "nasa": nasa_score(pred, true)}


def life_stage_mae(pred, true, edges=(50, 125)) -> dict:
    """MAE per life stage by TRUE RUL (E4): near failure < edges[0] <= mid < edges[1] <= healthy.

    Returns the three MAEs and their group sizes (E11). An empty group gives NaN, never 0 (E9).
    """
    pred, true = _validate_inputs(pred, true)
    if len(edges) != 2 or not edges[0] < edges[1]:
        raise ValueError(f"edges must be two strictly increasing values, got {edges}")
    lo, hi = edges
    groups = {"near_failure": true < lo,
              "mid": (true >= lo) & (true < hi),
              "healthy": true >= hi}
    out = {}
    for name, mask in groups.items():
        out[name] = float(np.mean(np.abs(pred[mask] - true[mask]))) if mask.any() else float("nan")
    for name, mask in groups.items():
        out["n_" + name] = int(mask.sum())
    return out


def rounds_to_threshold(val_curve, threshold_value: float) -> tuple[int | None, bool]:
    """First round whose validation MAE is at or below the threshold (E3, E14).

    `val_curve` is laid out as `run_federated` returns `val_mae`: list index = round number, index 0 = round 0
    (the untrained start weights, F10). Missing entries (None or NaN) are skipped. Returns (round, True), or
    (None, False) when the threshold is never reached: a censored run, which is recorded, never dropped (E3).
    The threshold itself (1.10 x mean best centralized validation MAE) is computed by the caller.
    """
    if not (math.isfinite(threshold_value) and threshold_value > 0):
        raise ValueError(f"threshold_value must be finite and positive, got {threshold_value}")
    for rnd, v in enumerate(val_curve):
        if v is not None and math.isfinite(v) and v <= threshold_value:
            return rnd, True
    return None, False
