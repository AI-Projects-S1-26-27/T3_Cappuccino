"""The real Flower client: the RUL model trained through `client_update` inside Flower (DECISIONS F9-F13).

`app.py` holds the wiring (plain FedAvg server, stateless client); this file plugs the project's own training into
it, so Flower and `run_federated` execute the SAME code and Gate 2 compares engines, not implementations.

    run_flower_federated(cfg, ds, manifest, algo, seed) -> the same dict as run_federated(...)

Clients run in separate worker processes, so each one rebuilds the dataset from `cfg` (cached per process) and
sets its own thread count (M5); nothing large is shipped between processes.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from rulfl.data.pipeline import Dataset, build_dataset
from rulfl.fl.client import client_seed, client_update
from rulfl.fl.fedla import fedla_weights, label_histogram
from rulfl.fl.simulate import client_rows, mae
from rulfl.flower_app.app import run_flower
from rulfl.model.gru import RULModel

_DATASET_CACHE: dict[tuple, Dataset] = {}
_THREADS_SET = False


# ---------- per-worker setup ----------

def set_threads(cfg: dict) -> None:
    """All registered runs use the same CPU thread count (M5). The runner sets it in its own process; a Flower
    worker is a fresh process, so the client sets it there too. Once per process: torch warns if it is changed
    after work has started."""
    global _THREADS_SET
    if not _THREADS_SET:
        torch.set_num_threads(int(cfg["train"]["threads"]))
        _THREADS_SET = True


def dataset(cfg: dict) -> Dataset:
    """The dataset for this config, built once per process and reused (F13)."""
    d = cfg["data"]
    key = tuple(sorted((k, str(v)) for k, v in d.items()))
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = build_dataset(cfg)
    return _DATASET_CACHE[key]


# ---------- weights <-> arrays ----------

def new_model(cfg: dict, n_features: int) -> RULModel:
    m = cfg["model"]
    return RULModel(n_features, m["hidden"], m["layers"], m["dropout"])


def state_keys(cfg: dict, n_features: int) -> list[str]:
    """The model's parameter names in a fixed order. Every process builds the same architecture from the same
    config, so this order is identical everywhere; it is what turns a state dict into the ordered array list
    Flower sends, and back (F12)."""
    return list(new_model(cfg, n_features).state_dict().keys())


def state_to_arrays(state: dict, keys: list[str]) -> list[np.ndarray]:
    return [state[k].detach().cpu().numpy() for k in keys]


def arrays_to_state(arrays: list[np.ndarray], keys: list[str]) -> dict:
    if len(arrays) != len(keys):
        raise ValueError(f"expected {len(keys)} arrays, got {len(arrays)}")
    return {k: torch.from_numpy(np.asarray(a)) for k, a in zip(keys, arrays)}


# ---------- FedLA ----------

def fedla_sample_weights(cfg: dict, ds: Dataset, manifest: dict) -> list[np.ndarray]:
    """Every client's per-window weights (LA4, LA5), computed once before round 1.

    Each client's own histogram is counted from its own windows; the global histogram is their sum, which is what
    the server would send back (LA6). Here that sum is computed in the worker from the manifest instead of being
    exchanged over the network: the numbers are identical to `run_federated`, and no FL result depends on the
    exchange happening for real. Same kind of simulation shortcut as pooled scaling statistics (D5); say so in
    the report (F13).
    """
    edges, clip, cap = cfg["fedla"]["bin_edges"], cfg["fedla"]["clip"], cfg["data"]["rul_cap"]
    rows = client_rows(ds.engine_train, manifest["train"], cfg["fl"]["K"])
    local = [label_histogram(ds.y_train[r], edges, cap) for r in rows]
    glob = sum(local)
    return [fedla_weights(ds.y_train[r], h, glob, edges, cap, clip) for r, h in zip(rows, local)]


def algo_mu(cfg: dict, algo: str) -> float:
    """FedAvg and FedLA train with mu = 0; FedProx with the selected mu (F4, F5, LA5)."""
    if algo in ("fedavg", "fedla"):
        return 0.0
    if algo == "fedprox":
        return float(cfg["fl"]["mu"])
    raise ValueError(f"unknown algo {algo!r}; expected 'fedavg', 'fedprox' or 'fedla'")


# ---------- the pluggable training function ----------

def make_train_fn(cfg: dict, manifest: dict, algo: str, seed: int):
    """Build the `train_fn` that app.py calls inside each Flower client (F9).

    Only small values are captured (config, manifest, algo, seed): the dataset and the FedLA weights are rebuilt
    inside the worker. The round number comes from Flower's own `server-round` config key, so client k in round r
    uses exactly the seed `run_federated` gives it: client_seed(seed, r, k).
    """
    mu = algo_mu(cfg, algo)
    cache: dict[str, list[np.ndarray]] = {}              # FedLA weights: computed once, then reused (LA4)

    def train_fn(arrays, client_id: int, config: dict):
        set_threads(cfg)
        ds = dataset(cfg)
        rnd = int(config["server-round"])
        rows = client_rows(ds.engine_train, manifest["train"], cfg["fl"]["K"])[client_id]
        weights = None
        if algo == "fedla":
            weights = cache.setdefault("w", fedla_sample_weights(cfg, ds, manifest))[client_id]

        keys = state_keys(cfg, ds.X_train.shape[2])
        model = new_model(cfg, ds.X_train.shape[2])
        state, n = client_update(model, arrays_to_state(arrays, keys),
                                 ds.X_train[rows], ds.y_train[rows], cfg,
                                 mu=mu, seed=client_seed(seed, rnd, client_id), sample_weight=weights)
        return state_to_arrays(state, keys), n

    return train_fn


# ---------- the run ----------

def run_flower_federated(cfg: dict, ds: Dataset, manifest: dict, algo: str, seed: int) -> dict:
    """One federated training through Flower, returning exactly what `run_federated` returns (F12), so the two
    engines can be compared field by field at Gate 2 and either can fill a run folder (E8)."""
    algo_mu(cfg, algo)                                   # reject an unknown algo before starting Ray
    set_threads(cfg)
    K, R, cap = cfg["fl"]["K"], cfg["fl"]["R"], cfg["data"]["rul_cap"]
    n_features = ds.X_train.shape[2]
    keys = state_keys(cfg, n_features)

    torch.use_deterministic_algorithms(True)             # M5, as run_federated does
    torch.manual_seed(seed)                              # identical start weights to run_federated(seed)
    model = new_model(cfg, n_features)
    initial = state_to_arrays(model.state_dict(), keys)
    n_params = sum(a.size for a in initial)

    val_rows = client_rows(ds.engine_val, manifest["val"], K)
    hist_bytes = 2 * K * len(cfg["fedla"]["bin_edges"]) * 4 if algo == "fedla" else 0   # LA6
    scorer = new_model(cfg, n_features)
    per_client: list[list[float]] = []

    def evaluate_fn(server_round: int, arrays) -> dict:
        """Server-side, on every validation engine, after every round including round 0 (F10)."""
        scorer.load_state_dict(arrays_to_state(arrays, keys))
        per_client.append([mae(scorer, ds.X_val[r], ds.y_val[r], cap) for r in val_rows])
        return {"val_mae": mae(scorer, ds.X_val, ds.y_val, cap)}

    res = run_flower(train_fn=make_train_fn(cfg, manifest, algo, seed), initial=initial, K=K, num_rounds=R,
                     train_config={"algo": algo, "seed": int(seed)}, evaluate_fn=evaluate_fn, best_key="val_mae",
                     # Reserve a client's threads so Ray does not oversubscribe the machine, but never ask for
                     # more CPUs than it has: Ray would wait forever for a client it can never place.
                     cpus_per_client=min(int(cfg["train"]["threads"]), max(1, os.cpu_count() or 1)))

    rounds = [h["round"] for h in res.history]
    if rounds != list(range(R + 1)):
        raise RuntimeError(f"expected rounds 0..{R} from Flower, got {rounds}")
    return {"val_mae": [h["metrics"]["val_mae"] for h in res.history],
            "client_val_mae": per_client,
            "bytes": [0] + [2 * K * n_params * 4 + (hist_bytes if r == 1 else 0) for r in range(1, R + 1)],
            "best_round": res.best_round,
            "best_state": arrays_to_state(res.best_arrays, keys)}
