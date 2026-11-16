"""GPU runs (DECISIONS M8): same code path as CPU, model and data moved to cfg['train']['device'].
Skipped on machines without CUDA (CI, CPU-only laptops)."""
import numpy as np
import pytest
import torch

from rulfl.fl.simulate import run_federated
from rulfl.model.gru import RULModel, predict_rul
from rulfl.model.train import train_centralized

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU on this machine")


@pytest.fixture()
def gcfg(slice_cfg):
    slice_cfg["model"] = {"hidden": 16, "layers": 2, "lr": 0.003, "dropout": 0.2}   # 2 layers: dropout on, like M7
    slice_cfg["train"].update(device="cuda", max_epochs=2)
    slice_cfg["fl"].update(K=2, E=1, R=2, lr=0.003, mu=0.01)
    return slice_cfg


def two_clients(ds):
    deal = lambda engines: {int(e): i % 2 for i, e in enumerate(sorted(set(engines.tolist())))}
    return {"train": deal(ds.engine_train), "val": deal(ds.engine_val)}


def same(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def test_centralized_training_runs_on_the_gpu_and_repeats_exactly(gcfg, slice_ds):
    m1, h1 = train_centralized(gcfg, slice_ds, seed=1)
    m2, h2 = train_centralized(gcfg, slice_ds, seed=1)
    assert next(m1.parameters()).is_cuda
    assert h1 == h2 and same(m1.state_dict(), m2.state_dict())          # M5: same seed, same run, also on GPU


def test_federated_training_runs_on_the_gpu_and_repeats_exactly(gcfg, slice_ds):
    for algo in ("fedavg", "fedprox", "fedla"):
        a = run_federated(gcfg, slice_ds, two_clients(slice_ds), algo, seed=1)
        b = run_federated(gcfg, slice_ds, two_clients(slice_ds), algo, seed=1)
        assert all(v.is_cuda for v in a["best_state"].values())
        assert a["val_mae"] == b["val_mae"] and same(a["best_state"], b["best_state"])


def test_gpu_starts_from_the_same_weights_as_cpu(gcfg, slice_ds):
    """Built on CPU from the seed, then moved: round 0 (untrained) scores the same on both devices."""
    gpu = run_federated(gcfg, slice_ds, two_clients(slice_ds), "fedavg", seed=3)
    gcfg["train"]["device"] = "cpu"
    cpu = run_federated(gcfg, slice_ds, two_clients(slice_ds), "fedavg", seed=3)
    assert gpu["val_mae"][0] == pytest.approx(cpu["val_mae"][0], abs=1e-3)


def test_predict_rul_on_a_gpu_model_returns_numpy_on_cpu(slice_ds):
    m = RULModel(slice_ds.X_train.shape[2], 16, 1, 0.0)
    cpu_pred = predict_rul(m, slice_ds.X_test, 125)
    gpu_pred = predict_rul(m.to("cuda"), slice_ds.X_test, 125)
    assert isinstance(gpu_pred, np.ndarray) and np.allclose(gpu_pred, cpu_pred, atol=1e-3)
