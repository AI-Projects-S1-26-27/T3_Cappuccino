"""Metrics (DECISIONS E1, E3, E4, E9-E14). Pure numpy; hand-checkable numbers."""
import math

import numpy as np
import pytest

from rulfl.eval.metrics import life_stage_mae, metrics, nasa_score, rounds_to_threshold


# ---------- NASA score (E1) ----------

def test_nasa_reference_costs_late_and_early():
    # d = +10 (late) and d = -13 (early) each cost e - 1
    assert nasa_score([10.0], [0.0]) == pytest.approx(math.e - 1, rel=1e-12)
    assert nasa_score([20.0], [33.0]) == pytest.approx(math.e - 1, rel=1e-12)
    assert nasa_score([10.0, 20.0], [0.0, 33.0]) == pytest.approx(2 * (math.e - 1), rel=1e-12)


def test_late_costs_more_than_early():
    late, early = nasa_score([5.0], [0.0]), nasa_score([0.0], [5.0])
    assert late == pytest.approx(math.exp(0.5) - 1, rel=1e-12)          # 0.64872
    assert early == pytest.approx(math.exp(5 / 13) - 1, rel=1e-12)      # 0.46905
    assert late > early


def test_nasa_is_a_sum_and_zero_when_exact():
    assert nasa_score([7.0, 50.0], [7.0, 50.0]) == 0.0
    one = nasa_score([3.0], [0.0])
    assert nasa_score([3.0, 3.0], [0.0, 0.0]) == pytest.approx(2 * one, rel=1e-12)


# ---------- MAE / RMSE (E1) ----------

def test_mae_rmse_hand_calculated():
    r = metrics([10, 20], [0, 33])
    assert r["mae"] == 11.5
    assert r["rmse"] == pytest.approx(math.sqrt((100 + 169) / 2), rel=1e-12)   # 11.59741

    r = metrics([1, 2, 3], [2, 2, 5])
    assert r["mae"] == 1.0
    assert r["rmse"] == pytest.approx(math.sqrt(5 / 3), rel=1e-12)             # 1.29099


def test_metrics_returns_plain_floats_with_exact_keys():
    r = metrics(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    assert set(r) == {"mae", "rmse", "nasa"}
    assert all(type(v) is float for v in r.values())      # JSON-safe for final.json (E8)


# ---------- Input validation (E10, E12) ----------

@pytest.mark.parametrize("pred,true", [([1, 2], [1]), ([], []), ([[1, 2]], [[1, 2]])])
def test_bad_shapes_raise(pred, true):
    for fn in (metrics, nasa_score, life_stage_mae):
        with pytest.raises(ValueError):
            fn(pred, true)


@pytest.mark.parametrize("pred,true", [([np.nan], [10]), ([np.inf], [10]), ([10], [np.nan])])
def test_non_finite_raise(pred, true):
    with pytest.raises(ValueError):
        metrics(pred, true)


# ---------- Life-stage breakdown (E4, E9, E11) ----------

def test_life_stage_boundaries_and_values():
    r = life_stage_mae(pred=[50, 60, 130, 140], true=[49, 50, 124, 125])
    assert (r["n_near_failure"], r["n_mid"], r["n_healthy"]) == (1, 2, 1)   # 49 | 50, 124 | 125
    assert r["near_failure"] == 1.0
    assert r["mid"] == 8.0                                                  # (10 + 6) / 2
    assert r["healthy"] == 15.0


def test_life_stage_groups_partition_random_labels():
    rng = np.random.default_rng(0)
    true = rng.integers(0, 126, size=1000).astype(float)
    pred = np.clip(true + rng.normal(0, 10, size=1000), 0, 125)
    r = life_stage_mae(pred, true)
    assert r["n_near_failure"] + r["n_mid"] + r["n_healthy"] == len(true)
    # The size-weighted group MAEs recombine into the overall MAE only if every engine is counted exactly once
    total = sum(r[g] * r["n_" + g] for g in ("near_failure", "mid", "healthy"))
    assert total / len(true) == pytest.approx(metrics(pred, true)["mae"], rel=1e-12)


def test_empty_group_is_nan_not_zero():
    r = life_stage_mae(pred=[60, 70], true=[55, 80])
    assert math.isnan(r["near_failure"]) and r["n_near_failure"] == 0
    assert math.isnan(r["healthy"]) and r["n_healthy"] == 0


def test_edges_argument_is_used(slice_cfg):
    true = [40, 75, 110, 125]
    pred = [40, 75, 110, 125]
    default = life_stage_mae(pred, true, edges=tuple(slice_cfg["eval"]["life_stage_edges"]))
    assert (default["n_near_failure"], default["n_mid"], default["n_healthy"]) == (1, 2, 1)
    # Different edges must move engines between groups
    other = life_stage_mae(pred, true, edges=(30, 100))
    assert (other["n_near_failure"], other["n_mid"], other["n_healthy"]) == (0, 2, 2)


def test_bad_edges_raise():
    for edges in [(125, 50), (50, 50), (50,), (10, 50, 125)]:
        with pytest.raises(ValueError):
            life_stage_mae([1.0], [1.0], edges=edges)


def test_on_real_slice(slice_ds):
    pred = np.full_like(slice_ds.y_val, slice_ds.y_train.mean(), dtype=np.float64)
    r = metrics(pred, slice_ds.y_val)
    ls = life_stage_mae(pred, slice_ds.y_val)
    assert all(np.isfinite(v) for v in r.values())
    assert ls["n_near_failure"] + ls["n_mid"] + ls["n_healthy"] == len(slice_ds.y_val)


# ---------- Rounds to threshold (E3, E14) ----------

def test_rounds_to_threshold_uses_run_federated_indexing():
    # val_mae as returned by run_federated: index = round number, index 0 = untrained round 0 (F10)
    assert rounds_to_threshold([50.0, 30.0, 20.0], 30.0) == (1, True)     # first round at or below: round 1
    assert rounds_to_threshold([50.0, 35.0, 29.9], 30.0) == (2, True)


def test_rounds_to_threshold_censored_when_never_reached():
    assert rounds_to_threshold([50.0, 40.0, 31.0], 30.0) == (None, False)


def test_rounds_to_threshold_skips_missing_and_rejects_bad_threshold():
    assert rounds_to_threshold([50.0, float("nan"), None, 20.0], 30.0) == (3, True)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            rounds_to_threshold([50.0, 20.0], bad)
