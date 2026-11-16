"""Gate 2: Flower reproduces the reference simulator for FedAvg, FedProx and FedLA (DECISIONS F11).

Both engines run the SAME `client_update`, so any difference comes from the server adding the client updates in a
different order: Flower aggregates replies as they arrive from its workers, the simulator always goes client
0..K-1, and float32 addition is not associative. F11 fixes how close they must be and over how many rounds.

Needs Ray (Flower's simulation engine) and so is the slowest test file in the repo: one simulation per arm.
"""
import numpy as np
import pytest
import torch

from rulfl.fl.simulate import run_federated
from rulfl.flower_app.rul import run_flower_federated

# F11 tolerances. Validation MAE is in cycles, so this is an absolute tolerance on a number around 40-70.
VAL_MAE_ATOL = 1e-4
WEIGHT_RTOL = 1e-5
WEIGHT_ATOL = 1e-7
ROUNDS = 2


@pytest.fixture()
def cfg(slice_cfg):
    slice_cfg["model"] = {"hidden": 16, "layers": 1, "lr": 0.003, "dropout": 0.0}
    slice_cfg["fl"].update(E=1, R=ROUNDS, lr=0.003, mu=0.01)
    return slice_cfg


@pytest.fixture()
def manifest(slice_ds, slice_cfg):
    """Engines dealt round-robin to K clients, as tests/test_simulate.py does. The slice has too few engines for
    PT4's minimum of 10 per client, so Gate 2 uses a hand-made assignment rather than a real manifest."""
    K = slice_cfg["fl"]["K"]
    deal = lambda engines: {str(e): i % K for i, e in enumerate(sorted(set(engines.tolist())))}
    return {"train": deal(slice_ds.engine_train), "val": deal(slice_ds.engine_val)}


@pytest.fixture(scope="module")
def runs(request):
    """One reference run and one Flower run per arm, computed once and shared (a simulation costs ~30 s)."""
    return {}


def both(cfg, slice_ds, manifest, algo, seed=3):
    ref = run_federated(cfg, slice_ds, manifest, algo, seed)
    flw = run_flower_federated(cfg, slice_ds, manifest, algo, seed)
    return ref, flw


@pytest.mark.parametrize("algo", ["fedavg", "fedprox", "fedla"])
def test_flower_reproduces_the_reference_simulator(cfg, slice_ds, manifest, algo):
    ref, flw = both(cfg, slice_ds, manifest, algo)

    # Same shape of log: one entry per round, round 0 first (F10, F12)
    assert len(flw["val_mae"]) == len(ref["val_mae"]) == ROUNDS + 1
    assert flw["bytes"] == ref["bytes"]                                  # F8 (+ LA6 for fedla); exact integers

    # Round 0 is the untrained start weights: identical seeds must give identical numbers, not merely close
    assert flw["val_mae"][0] == pytest.approx(ref["val_mae"][0], abs=1e-12)

    # Trained rounds: equal within the F11 tolerance
    assert flw["val_mae"] == pytest.approx(ref["val_mae"], abs=VAL_MAE_ATOL)
    for k, v in ref["best_state"].items():
        assert torch.allclose(flw["best_state"][k], v, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL), k

    # Per-client validation MAE (E2 feeds on this)
    assert np.allclose(np.array(flw["client_val_mae"], dtype=float),
                       np.array(ref["client_val_mae"], dtype=float), atol=VAL_MAE_ATOL, equal_nan=True)

    # best_round may legitimately differ when two rounds are within tolerance of each other
    if abs(ref["val_mae"][ref["best_round"]] - sorted(ref["val_mae"])[1]) > VAL_MAE_ATOL:
        assert flw["best_round"] == ref["best_round"]


def test_the_tolerance_is_not_hiding_a_real_difference(cfg, slice_ds, manifest):
    """Guards the test above: a wrong engine must fail it. FedProx with mu = 0 is FedAvg, so the gap between the
    two arms shows what a real bug looks like next to the tolerance Gate 2 allows."""
    ref = run_federated(cfg, slice_ds, manifest, "fedavg", seed=3)
    cfg["fl"]["mu"] = 0.5                                                 # a different algorithm, not noise
    other = run_federated(cfg, slice_ds, manifest, "fedprox", seed=3)
    gap = max(abs(a - b) for a, b in zip(ref["val_mae"][1:], other["val_mae"][1:]))
    assert gap > 10 * VAL_MAE_ATOL, f"arms differ by only {gap}; the Gate 2 tolerance would hide a real bug"


def test_unknown_algo_is_rejected_before_starting_ray(cfg, slice_ds, manifest):
    with pytest.raises(ValueError):
        run_flower_federated(cfg, slice_ds, manifest, "fedsomething", seed=1)
