"""The real Flower client: weight conversion, per-worker setup and one round through Flower (F9, F11-F13).

Gate 2 (Flower equals the reference simulator over several rounds) lives in tests/test_gate2.py.
"""
import numpy as np
import pytest
import torch

from rulfl.fl.client import client_seed, client_update
from rulfl.fl.simulate import client_rows
from rulfl.flower_app.rul import (algo_mu, arrays_to_state, dataset, fedla_sample_weights, make_train_fn,
                                  new_model, state_keys, state_to_arrays)


@pytest.fixture()
def cfg(slice_cfg):
    """Small, fast settings, as tests/test_simulate.py uses: `model:` exists only after tuning (M2)."""
    slice_cfg["model"] = {"hidden": 16, "layers": 1, "lr": 0.003, "dropout": 0.0}
    slice_cfg["fl"].update(E=1, R=1, lr=0.003, mu=0.01)
    return slice_cfg


def manifest_for(ds, K):
    """Engines dealt round-robin to K clients, exactly as tests/test_simulate.py does."""
    deal = lambda engines: {str(e): i % K for i, e in enumerate(sorted(set(engines.tolist())))}
    return {"train": deal(ds.engine_train), "val": deal(ds.engine_val)}


# ---------- weights <-> arrays (F12) ----------

def test_state_round_trip_is_exact(cfg, slice_ds):
    keys = state_keys(cfg, slice_ds.X_train.shape[2])
    state = new_model(cfg, slice_ds.X_train.shape[2]).state_dict()
    assert set(keys) == set(state)
    back = arrays_to_state(state_to_arrays(state, keys), keys)
    assert list(back) == keys                                   # order preserved, so Flower's keys line up
    for k in keys:
        assert back[k].dtype == state[k].dtype == torch.float32  # float32 both ways, as FedAvg aggregates in
        assert torch.equal(back[k], state[k])                    # bit-for-bit, not just close


def test_state_key_order_is_the_same_for_every_worker(cfg, slice_ds):
    """Workers are separate processes that each rebuild the model, so the order must come from the config alone."""
    n = slice_ds.X_train.shape[2]
    assert state_keys(cfg, n) == state_keys(dict(cfg), n)
    assert len(state_keys(cfg, n)) == len(set(state_keys(cfg, n)))


def test_arrays_to_state_rejects_the_wrong_count(cfg, slice_ds):
    keys = state_keys(cfg, slice_ds.X_train.shape[2])
    with pytest.raises(ValueError):
        arrays_to_state([np.zeros(1)], keys)


# ---------- per-worker setup ----------

def test_threads_come_from_the_config(cfg):
    """M5: every registered run uses the same thread count, and a Flower worker must set it itself."""
    assert cfg["train"]["threads"] == 3
    from rulfl.flower_app import rul
    rul._THREADS_SET = False
    rul.set_threads(cfg)
    assert torch.get_num_threads() == 3


def test_dataset_is_built_once_per_process(cfg):
    assert dataset(cfg) is dataset(cfg)                          # same object: rebuilt per worker, not per round


def test_algo_mu(cfg):
    assert algo_mu(cfg, "fedavg") == 0.0
    assert algo_mu(cfg, "fedla") == 0.0                          # FedLA is FedAvg plus sample weights (LA5)
    assert algo_mu(cfg, "fedprox") == cfg["fl"]["mu"]
    with pytest.raises(ValueError):
        algo_mu(cfg, "fedsomething")


def test_fedla_weights_match_the_simulator(cfg, slice_ds):
    """The worker's global histogram is the sum of every client's own histogram, so the weights are the ones
    run_federated computes (LA4-LA6)."""
    man = manifest_for(slice_ds, cfg["fl"]["K"])
    got = fedla_sample_weights(cfg, slice_ds, man)
    rows = client_rows(slice_ds.engine_train, man["train"], cfg["fl"]["K"])
    assert [len(w) for w in got] == [len(r) for r in rows]
    for w in got:
        assert w.mean() == pytest.approx(1.0, rel=1e-9)


# ---------- the training function (F9) ----------

def test_train_fn_equals_calling_client_update_directly(cfg, slice_ds):
    """What Flower runs in a worker must be exactly what the simulator runs in-process: same weights, same count."""
    man = manifest_for(slice_ds, cfg["fl"]["K"])
    n_features, client, rnd, seed = slice_ds.X_train.shape[2], 2, 1, 7
    keys = state_keys(cfg, n_features)

    torch.manual_seed(seed)
    start = new_model(cfg, n_features).state_dict()
    got_arrays, got_n = make_train_fn(cfg, man, "fedavg", seed)(state_to_arrays(start, keys), client,
                                                               {"server-round": rnd})

    rows = client_rows(slice_ds.engine_train, man["train"], cfg["fl"]["K"])[client]
    want, want_n = client_update(new_model(cfg, n_features), start,
                                 slice_ds.X_train[rows], slice_ds.y_train[rows], cfg,
                                 mu=0.0, seed=client_seed(seed, rnd, client), sample_weight=None)
    assert got_n == want_n == len(rows)
    for arr, k in zip(got_arrays, keys):
        assert torch.equal(torch.from_numpy(arr), want[k])


def test_train_fn_uses_the_round_number_for_the_seed(cfg, slice_ds):
    """client_seed(seed, round, client): a different round must train differently, or every round would repeat."""
    man = manifest_for(slice_ds, cfg["fl"]["K"])
    keys = state_keys(cfg, slice_ds.X_train.shape[2])
    torch.manual_seed(1)
    start = state_to_arrays(new_model(cfg, slice_ds.X_train.shape[2]).state_dict(), keys)
    fn = make_train_fn(cfg, man, "fedavg", seed=1)
    cfg["fl"]["E"] = 2                                           # two epochs, so the batch order can differ
    r1, _ = fn(start, 0, {"server-round": 1})
    r2, _ = fn(start, 0, {"server-round": 2})
    assert any(not np.array_equal(a, b) for a, b in zip(r1, r2))


def test_train_fn_does_not_modify_the_weights_it_was_given(cfg, slice_ds):
    man = manifest_for(slice_ds, cfg["fl"]["K"])
    keys = state_keys(cfg, slice_ds.X_train.shape[2])
    torch.manual_seed(1)
    arrays = state_to_arrays(new_model(cfg, slice_ds.X_train.shape[2]).state_dict(), keys)
    before = [a.copy() for a in arrays]
    make_train_fn(cfg, man, "fedprox", seed=1)(arrays, 0, {"server-round": 1})
    for a, b in zip(arrays, before):
        assert np.array_equal(a, b)
