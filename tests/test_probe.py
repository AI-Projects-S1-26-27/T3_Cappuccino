"""Head-refit probe (DECISIONS E6): GRU frozen, fresh head trained on pooled windows. Diagnostic only."""
import numpy as np
import pytest
import torch

from rulfl.model.gru import RULModel, predict_rul
from rulfl.probe import gru_features, refit_head


@pytest.fixture()
def pcfg(slice_cfg):
    slice_cfg["model"] = {"hidden": 16, "layers": 1, "lr": 0.003, "dropout": 0.0}
    slice_cfg["train"]["max_epochs"] = 3
    return slice_cfg


def broken_head_model(ds):
    """A model whose GRU is fine (random) but whose head outputs ~0 for every window: an extreme 'damaged head'."""
    torch.manual_seed(0)
    m = RULModel(ds.X_train.shape[2], 16, 1, 0.0)
    with torch.no_grad():
        for p in m.head.parameters():
            p.zero_()
    return m


def test_features_are_the_grus_last_step(pcfg, slice_ds):
    m = broken_head_model(slice_ds)
    with torch.no_grad():
        ref = m.gru(torch.from_numpy(slice_ds.X_val[:50]))[0][:, -1]
    assert torch.allclose(gru_features(m, slice_ds.X_val[:50]), ref)


def test_gru_is_untouched_and_only_the_head_changes(pcfg, slice_ds):
    m = broken_head_model(slice_ds)
    gru_before = {k: v.clone() for k, v in m.gru.state_dict().items()}
    refit, hist = refit_head(m, pcfg, slice_ds, seed=1)
    assert all(torch.equal(refit.gru.state_dict()[k], gru_before[k]) for k in gru_before)
    assert all(torch.equal(m.gru.state_dict()[k], gru_before[k]) for k in gru_before)      # original not modified
    assert not torch.equal(refit.head[0].weight, m.head[0].weight)
    assert len(hist["val_mae"]) >= 1 and 0 <= hist["best_epoch"] < len(hist["val_mae"])


def test_refitting_a_damaged_head_lowers_validation_error(pcfg, slice_ds):
    m = broken_head_model(slice_ds)
    before = np.abs(predict_rul(m, slice_ds.X_val, 125) - slice_ds.y_val).mean()
    refit, hist = refit_head(m, pcfg, slice_ds, seed=1)
    after = np.abs(predict_rul(refit, slice_ds.X_val, 125) - slice_ds.y_val).mean()
    assert after < before
    assert after == pytest.approx(hist["val_mae"][hist["best_epoch"]], abs=1e-4)          # returns the best epoch


def test_same_seed_same_refit(pcfg, slice_ds):
    a, _ = refit_head(broken_head_model(slice_ds), pcfg, slice_ds, seed=2)
    b, _ = refit_head(broken_head_model(slice_ds), pcfg, slice_ds, seed=2)
    assert all(torch.equal(a.head.state_dict()[k], b.head.state_dict()[k]) for k in a.head.state_dict())
